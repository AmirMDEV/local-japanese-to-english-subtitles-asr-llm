from __future__ import annotations

import gc
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import requests

from .domain import ChunkPlan, Cue
from .utils import atomic_write_json, no_window_creationflags, read_json


class ExternalToolError(RuntimeError):
    pass


def run_command(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
        creationflags=no_window_creationflags(),
    )
    return completed.stdout


class FFmpegClient:
    def __init__(self, ffmpeg_path: str, ffprobe_path: str) -> None:
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
        self.ffprobe_path = ffprobe_path or "ffprobe"

    def probe_duration(self, source_path: Path) -> float:
        output = run_command(
            [
                self.ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source_path),
            ]
        )
        return float(output.strip())

    def create_chunk_plan(
        self,
        source_path: Path,
        chunks_dir: Path,
        chunk_seconds: int,
        overlap_seconds: int,
        progress_callback: Callable[[dict[str, float | int]], None] | None = None,
    ) -> list[ChunkPlan]:
        chunks_dir.mkdir(parents=True, exist_ok=True)
        duration = self.probe_duration(source_path)
        step = max(chunk_seconds - overlap_seconds, 1)
        total_chunks = self._estimate_chunk_count(duration, step)
        plans: list[ChunkPlan] = []
        index = 0
        start = 0.0
        while start < duration - 0.05:
            index += 1
            end = min(start + chunk_seconds, duration)
            chunk_path = chunks_dir / f"chunk_{index:04d}.wav"
            plans.append(ChunkPlan(index=index, start=start, end=end, path=str(chunk_path)))
            if progress_callback is not None:
                progress_callback(
                    {
                        "current_chunk": index,
                        "total_chunks": total_chunks,
                        "covered_seconds": end,
                        "total_seconds": duration,
                    }
                )
            start += step
        return plans

    def _estimate_chunk_count(self, duration: float, step: int) -> int:
        count = 0
        current = 0.0
        while current < duration - 0.05:
            count += 1
            current += step
        return max(count, 1)

    def extract_chunk(
        self,
        *,
        source_path: Path,
        chunk_path: Path,
        start: float,
        duration: float,
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        args = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-progress",
            "pipe:1",
            "-nostats",
            str(chunk_path),
        ]
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=no_window_creationflags(),
        )
        latest_progress = 0.0
        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key == "out_time":
                    latest_progress = min(self._parse_ffmpeg_timecode(value), duration)
                    if progress_callback is not None:
                        progress_callback(latest_progress)
                elif key in {"out_time_ms", "out_time_us"}:
                    latest_progress = min(self._parse_ffmpeg_progress_value(value), duration)
                    if progress_callback is not None:
                        progress_callback(latest_progress)
                elif key == "progress" and value == "end":
                    latest_progress = duration
                    if progress_callback is not None:
                        progress_callback(duration)
        finally:
            if process.stdout is not None:
                process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, args)

    def _parse_ffmpeg_timecode(self, value: str) -> float:
        parts = value.strip().split(":")
        if len(parts) != 3:
            return 0.0
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    def _parse_ffmpeg_progress_value(self, value: str) -> float:
        raw_value = max(float(value.strip() or 0.0), 0.0)
        if raw_value >= 10_000_000:
            return raw_value / 1_000_000
        if raw_value >= 10_000:
            return raw_value / 1000
        return raw_value


class TransformersASRClient:
    def __init__(self, model_id: str, cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir or None
        self._pipe: Any | None = None
        self._device: str | None = None

    def _load(self, device: str) -> Any:
        if self._pipe is not None and self._device == device:
            return self._pipe

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=self.cache_dir,
        )
        if device == "cuda":
            model = model.to("cuda")
        processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )
        self._device = device
        return self._pipe

    def transcribe_chunk(self, chunk_path: Path, batch_size: int, device: str) -> list[Cue]:
        pipe = self._load(device=device)
        result = pipe(
            str(chunk_path),
            return_timestamps=True,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs={"language": "ja", "task": "transcribe"},
        )
        raw_chunks = list(result.get("chunks", []))
        cues: list[Cue] = []
        for index, item in enumerate(raw_chunks, start=1):
            timestamps = item.get("timestamp") or (0.0, 0.0)
            start, end = timestamps
            if start is None:
                start = 0.0
            if end is None:
                end = start + 0.8
            cues.append(
                Cue(
                    index=index,
                    start=float(start),
                    end=float(end),
                    text=str(item.get("text", "")).strip(),
                )
            )
        return cues

    def close(self) -> None:
        self._pipe = None
        self._device = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", keep_alive: str = "5m") -> None:
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive

    def list_models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=30)
        response.raise_for_status()
        payload = response.json()
        return [item["name"] for item in payload.get("models", [])]

    def generate_json(self, model: str, prompt: str, temperature: float) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": self.keep_alive,
                "options": {"temperature": temperature},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        body = payload.get("response", "{}")
        return json.loads(body)


class SubtitleEditClient:
    def __init__(self, executable_path: str) -> None:
        self.executable_path = executable_path

    def open_files(self, paths: list[Path]) -> None:
        if not self.executable_path:
            raise ExternalToolError("Subtitle Edit path is not configured.")
        subprocess.Popen([self.executable_path, *[str(path) for path in paths]])


def save_cues(path: Path, cues: list[Cue]) -> None:
    atomic_write_json(
        path,
        [
            {"index": cue.index, "start": cue.start, "end": cue.end, "text": cue.text}
            for cue in cues
        ],
    )


def load_cues(path: Path) -> list[Cue]:
    rows = read_json(path, default=[]) or []
    return [Cue.from_dict(item) for item in rows]
