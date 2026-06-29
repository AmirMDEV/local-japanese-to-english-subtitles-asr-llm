from __future__ import annotations

import gc
import math
import os
import json
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

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

    def extract_audio(
        self,
        *,
        source_path: Path,
        audio_path: Path,
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-progress",
            "pipe:1",
            "-nostats",
            str(audio_path),
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
                    latest_progress = self._parse_ffmpeg_timecode(value)
                    if progress_callback is not None:
                        progress_callback(latest_progress)
                elif key in {"out_time_ms", "out_time_us"}:
                    latest_progress = self._parse_ffmpeg_progress_value(value)
                    if progress_callback is not None:
                        progress_callback(latest_progress)
                elif key == "progress" and value == "end":
                    if progress_callback is not None:
                        progress_callback(latest_progress)
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
        cleaned = value.strip()
        if not cleaned or cleaned.upper() == "N/A":
            return 0.0
        raw_value = max(float(cleaned), 0.0)
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


class Qwen3ASRClient:
    ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
    INSTALL_HINT = (
        "Qwen3-ASR is optional. Install it with `py -3.11 -m pip install qwen-asr` "
        "and restart the app. Timestamps use Qwen/Qwen3-ForcedAligner-0.6B."
    )

    def __init__(self, model_id: str, cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir or None
        self._model: Any | None = None
        self._device: str | None = None

    def _load(self, device: str, batch_size: int) -> Any:
        if self._model is not None and self._device == device:
            return self._model
        if self.cache_dir:
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ModuleNotFoundError as exc:
            missing = exc.name or "qwen-asr"
            raise ExternalToolError(f"{self.INSTALL_HINT} Missing Python package: {missing}") from exc

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        device_map = "cuda:0" if device == "cuda" else "cpu"
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=max(int(batch_size or 1), 1),
            max_new_tokens=512,
            forced_aligner=self.ALIGNER_MODEL_ID,
            forced_aligner_kwargs={"dtype": dtype, "device_map": device_map},
        )
        self._device = device
        return self._model

    def transcribe_chunk(self, chunk_path: Path, batch_size: int, device: str) -> list[Cue]:
        model = self._load(device=device, batch_size=batch_size)
        results = model.transcribe(
            audio=str(chunk_path),
            language="Japanese",
            return_time_stamps=True,
        )
        result = results[0] if isinstance(results, list) and results else results
        return self._result_to_cues(result, self._wave_duration_seconds(chunk_path))

    def _wave_duration_seconds(self, chunk_path: Path) -> float:
        try:
            with wave.open(str(chunk_path), "rb") as audio:
                rate = audio.getframerate()
                if rate > 0:
                    return audio.getnframes() / float(rate)
        except (OSError, wave.Error):
            return 0.0
        return 0.0

    def _result_to_cues(self, result: Any, chunk_duration: float) -> list[Cue]:
        text = str(getattr(result, "text", "") or "").strip()
        stamps = list(getattr(result, "time_stamps", []) or [])
        cues: list[Cue] = []
        for index, stamp in enumerate(stamps, start=1):
            stamp_text = self._stamp_value(stamp, "text", "").strip()
            if not stamp_text:
                continue
            start = float(self._stamp_value(stamp, "start_time", self._stamp_value(stamp, "start", 0.0)) or 0.0)
            end = float(self._stamp_value(stamp, "end_time", self._stamp_value(stamp, "end", start + 0.8)) or start + 0.8)
            cues.append(Cue(index=index, start=max(start, 0.0), end=max(end, start + 0.2), text=stamp_text))
        if cues:
            return cues
        if not text:
            return []
        end = chunk_duration if chunk_duration > 0 else 3.0
        return [Cue(index=1, start=0.0, end=max(end, 0.5), text=text)]

    def _stamp_value(self, stamp: Any, key: str, default: Any) -> Any:
        if isinstance(stamp, dict):
            return stamp.get(key, default)
        return getattr(stamp, key, default)

    def close(self) -> None:
        self._model = None
        self._device = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()


class ReazonSpeechK2ASRClient:
    INSTALL_HINT = (
        "ReazonSpeech k2 is optional. Install it with "
        "`py -3.11 -m pip install "
        "git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/k2-asr` "
        "and restart the app."
    )

    def __init__(self, model_id: str, cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir or None
        self._model: Any | None = None
        self._device: str | None = None
        self.language, self.precision = self._parse_model_options(model_id)

    def _parse_model_options(self, model_id: str) -> tuple[str, str]:
        normalized = model_id.strip().lower()
        if "ja-en-mls-5k" in normalized:
            language = "ja-en-mls-5k"
        elif "ja-en" in normalized:
            language = "ja-en"
        else:
            language = "ja"
        if "int8-fp32" in normalized:
            precision = "int8-fp32"
        elif "int8" in normalized:
            precision = "int8"
        else:
            precision = "fp32"
        return language, precision

    def _load(self, device: str) -> Any:
        if self._model is not None and self._device == device:
            return self._model
        if self.cache_dir:
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)
        try:
            from reazonspeech.k2.asr import load_model
        except ModuleNotFoundError as exc:
            missing = exc.name or "reazonspeech-k2-asr"
            raise ExternalToolError(f"{self.INSTALL_HINT} Missing Python package: {missing}") from exc
        provider = device if device in {"cpu", "cuda", "coreml"} else "cpu"
        self._model = load_model(device=provider, precision=self.precision, language=self.language)
        self._device = provider
        return self._model

    def transcribe_chunk(self, chunk_path: Path, batch_size: int, device: str) -> list[Cue]:
        del batch_size
        try:
            from reazonspeech.k2.asr import TranscribeConfig, audio_from_path, transcribe
        except ModuleNotFoundError as exc:
            missing = exc.name or "reazonspeech-k2-asr"
            raise ExternalToolError(f"{self.INSTALL_HINT} Missing Python package: {missing}") from exc
        model = self._load(device=device)
        result = transcribe(model, audio_from_path(str(chunk_path)), TranscribeConfig(verbose=False))
        return self._result_to_cues(result, self._wave_duration_seconds(chunk_path))

    def _wave_duration_seconds(self, chunk_path: Path) -> float:
        try:
            with wave.open(str(chunk_path), "rb") as audio:
                frames = audio.getnframes()
                rate = audio.getframerate()
                if rate > 0:
                    return frames / float(rate)
        except (OSError, wave.Error):
            return 0.0
        return 0.0

    def _result_to_cues(self, result: Any, chunk_duration: float) -> list[Cue]:
        text = str(getattr(result, "text", "") or "").strip()
        subwords = list(getattr(result, "subwords", []) or [])
        if not text:
            return []
        if not subwords:
            end = chunk_duration if chunk_duration > 0 else 3.0
            return [Cue(index=1, start=0.0, end=max(end, 0.5), text=text)]

        cues: list[Cue] = []
        current_tokens: list[str] = []
        current_start = max(float(getattr(subwords[0], "seconds", 0.0) or 0.0) - 0.08, 0.0)
        for index, subword in enumerate(subwords):
            token = self._clean_token(str(getattr(subword, "token", "") or ""))
            if not token:
                continue
            seconds = max(float(getattr(subword, "seconds", 0.0) or 0.0), 0.0)
            current_tokens.append(token)
            current_text = "".join(current_tokens).strip()
            current_duration = seconds - current_start
            should_split = (
                (token in "。！？!?、," and len(current_text) >= 8)
                or (len(current_text) >= 52 and current_duration >= 1.2)
                or current_duration >= 11.0
            )
            is_last = index == len(subwords) - 1
            if should_split or is_last:
                next_seconds = self._next_subword_seconds(subwords, index, fallback=chunk_duration)
                end = self._cue_end(seconds, next_seconds, current_start, chunk_duration)
                cues.append(
                    Cue(
                        index=len(cues) + 1,
                        start=current_start,
                        end=end,
                        text=current_text,
                    )
                )
                current_tokens = []
                current_start = end
        return cues

    def _clean_token(self, token: str) -> str:
        return token.replace("<blk>", "").replace("▁", " ").strip()

    def _next_subword_seconds(self, subwords: list[Any], index: int, fallback: float) -> float:
        for item in subwords[index + 1 :]:
            seconds = float(getattr(item, "seconds", 0.0) or 0.0)
            if seconds > 0:
                return seconds
        last_seconds = float(getattr(subwords[index], "seconds", 0.0) or 0.0)
        if fallback > 0 and last_seconds >= fallback - 0.7:
            return fallback
        return last_seconds + 0.7

    def _cue_end(self, seconds: float, next_seconds: float, start: float, chunk_duration: float) -> float:
        candidate = max(seconds + 0.35, next_seconds, start + 0.5)
        if chunk_duration > 0:
            candidate = min(candidate, chunk_duration)
        if not math.isfinite(candidate):
            candidate = start + 0.5
        return max(candidate, start + 0.5)

    def close(self) -> None:
        self._model = None
        self._device = None
        gc.collect()


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        keep_alive: str = "5m",
        executable_path: str = "",
        startup_timeout_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.executable_path = executable_path or "ollama"
        self.startup_timeout_seconds = startup_timeout_seconds
        self._recent_events: list[dict[str, str]] = []

    def _record_event(self, level: str, message: str) -> None:
        self._recent_events.append({"level": level, "message": message})
        if len(self._recent_events) > 40:
            self._recent_events = self._recent_events[-40:]

    def pop_recent_events(self) -> list[dict[str, str]]:
        events = list(self._recent_events)
        self._recent_events.clear()
        return events

    def _request(self, method: str, path: str, *, timeout: float, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
        except requests.ConnectionError:
            if not self._can_auto_start():
                raise
            self.ensure_available()
            self._record_event("warning", "Ollama was restarted automatically so English translation could continue.")
            response = requests.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response

    def _can_auto_start(self) -> bool:
        hostname = (urlparse(self.base_url).hostname or "").strip().lower()
        return hostname in {"127.0.0.1", "localhost", "::1"}

    def is_available(self, timeout: float = 2.0) -> bool:
        try:
            response = requests.request("GET", f"{self.base_url}/api/tags", timeout=timeout)
            response.raise_for_status()
        except requests.RequestException:
            return False
        return True

    def ensure_available(self) -> None:
        if self.is_available():
            return
        if not self._can_auto_start():
            return
        self._start_server()

    def _start_server(self) -> None:
        try:
            subprocess.Popen(
                [self.executable_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=no_window_creationflags(),
            )
        except OSError as exc:
            raise ExternalToolError(
                f"Could not start Ollama from '{self.executable_path}': {exc}"
            ) from exc
        self._record_event("warning", "Ollama was not running and the app started it automatically.")

        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.is_available(timeout=1.0):
                self._record_event("info", "Ollama is ready again.")
                return
            time.sleep(0.5)
        raise ExternalToolError(
            f"Ollama did not become ready within {self.startup_timeout_seconds:.0f} seconds."
        )

    def list_models(self) -> list[str]:
        response = self._request("GET", "/api/tags", timeout=30)
        payload = response.json()
        return [item["name"] for item in payload.get("models", [])]

    def model_storage_root(self) -> str:
        hostname = (urlparse(self.base_url).hostname or "").strip().lower()
        if hostname not in {"127.0.0.1", "localhost", "::1"}:
            return f"Remote Ollama host: {hostname or self.base_url}"
        return os.environ.get("OLLAMA_MODELS") or str(Path.home() / ".ollama" / "models")

    def list_model_details(self) -> dict[str, dict[str, Any]]:
        response = self._request("GET", "/api/tags", timeout=30)
        payload = response.json()
        return {
            item["name"]: {
                "size": item.get("size"),
                "digest": item.get("digest", ""),
                "modified_at": item.get("modified_at", ""),
            }
            for item in payload.get("models", [])
            if item.get("name")
        }

    def pull_model(self, model: str) -> None:
        response = self._request(
            "POST",
            "/api/pull",
            json={"name": model, "stream": False},
            timeout=3600,
        )
        response.json()

    def generate_json(self, model: str, prompt: str, temperature: float) -> dict[str, Any]:
        response = self._request(
            "POST",
            "/api/generate",
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
