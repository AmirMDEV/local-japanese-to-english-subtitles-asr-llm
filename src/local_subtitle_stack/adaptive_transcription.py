from __future__ import annotations

import gc
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .domain import Cue
from .guards import ResourceSnapshot, capture_snapshot
from .integrations import FFmpegClient
from .pipeline import format_srt_timestamp, write_srt
from .utils import atomic_write_json, atomic_write_text, list_video_sources, now_iso, safe_slug


class TranscriptionError(RuntimeError):
    pass


@dataclass(slots=True)
class TranscriptionProfile:
    name: str
    model_id: str
    device: str
    compute_type: str
    batch_size: int
    beam_size: int
    min_free_ram_mb: int
    min_free_vram_mb: int


@dataclass(slots=True)
class TranscriptionResult:
    source_path: str
    relative_output_dir: str
    output_dir: str
    requested_profile: str
    chosen_profile: str
    language: str
    segment_count: int
    duration_seconds: float | None
    resource_snapshot: dict[str, int]
    profile: dict[str, Any]
    backend: dict[str, Any]
    outputs: dict[str, str]
    selection_note: str
    transcript_text: str

    def to_manifest_entry(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("transcript_text", None)
        return data


TRANSCRIPTION_PROFILE_LADDER: tuple[TranscriptionProfile, ...] = (
    TranscriptionProfile(
        name="high",
        model_id="large-v3",
        device="cuda",
        compute_type="float16",
        batch_size=8,
        beam_size=5,
        min_free_ram_mb=12_000,
        min_free_vram_mb=6_000,
    ),
    TranscriptionProfile(
        name="balanced",
        model_id="large-v3",
        device="cuda",
        compute_type="int8_float16",
        batch_size=6,
        beam_size=5,
        min_free_ram_mb=10_000,
        min_free_vram_mb=3_500,
    ),
    TranscriptionProfile(
        name="low_gpu",
        model_id="large-v3",
        device="cuda",
        compute_type="int8_float16",
        batch_size=4,
        beam_size=4,
        min_free_ram_mb=7_000,
        min_free_vram_mb=1_500,
    ),
    TranscriptionProfile(
        name="cpu_fallback",
        model_id="large-v3",
        device="cpu",
        compute_type="int8",
        batch_size=2,
        beam_size=4,
        min_free_ram_mb=4_000,
        min_free_vram_mb=0,
    ),
)

SENTENCE_ENDING_PATTERN = re.compile(r"([^。？！!?]+[。？！!?]?)")


def _profile_map() -> dict[str, TranscriptionProfile]:
    return {profile.name: profile for profile in TRANSCRIPTION_PROFILE_LADDER}


def _is_memory_failure(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "out of memory",
        "cuda",
        "cudnn",
        "cublas",
        "memoryerror",
        "cannot allocate memory",
        "not enough memory",
    )
    return any(token in message for token in needles)


def _resource_snapshot_dict(snapshot: ResourceSnapshot) -> dict[str, int]:
    return {
        "free_ram_mb": snapshot.free_ram_mb,
        "process_rss_mb": snapshot.process_rss_mb,
        "gpu_free_mb": snapshot.gpu_free_mb,
        "gpu_total_mb": snapshot.gpu_total_mb,
        "gpu_used_mb": snapshot.gpu_used_mb,
    }


def profile_fits(snapshot: ResourceSnapshot, profile: TranscriptionProfile) -> bool:
    if snapshot.free_ram_mb < profile.min_free_ram_mb:
        return False
    if profile.device == "cuda":
        if snapshot.gpu_total_mb <= 0:
            return False
        if snapshot.gpu_free_mb < profile.min_free_vram_mb:
            return False
    return True


def ordered_profile_candidates(
    snapshot: ResourceSnapshot,
    requested_profile: str,
    low_memory_policy: str,
) -> tuple[list[TranscriptionProfile], str]:
    profile_map = _profile_map()
    if requested_profile == "auto":
        candidates = [profile for profile in TRANSCRIPTION_PROFILE_LADDER if profile_fits(snapshot, profile)]
        if candidates:
            return candidates, "auto"
        if low_memory_policy == "wait":
            return [TRANSCRIPTION_PROFILE_LADDER[-1]], "auto-wait"
        return [TRANSCRIPTION_PROFILE_LADDER[-1]], "auto-fallback"

    if requested_profile not in profile_map:
        available = ", ".join(sorted(profile_map))
        raise TranscriptionError(
            f"Unknown transcription profile '{requested_profile}'. Available profiles: auto, {available}"
        )

    requested_index = next(
        index for index, profile in enumerate(TRANSCRIPTION_PROFILE_LADDER) if profile.name == requested_profile
    )
    requested = TRANSCRIPTION_PROFILE_LADDER[requested_index]
    if profile_fits(snapshot, requested):
        return [requested], "requested"
    if low_memory_policy == "wait":
        return [requested], "requested-wait"

    candidates = [
        profile
        for profile in TRANSCRIPTION_PROFILE_LADDER[requested_index:]
        if profile_fits(snapshot, profile)
    ]
    if candidates:
        return candidates, "requested-downgrade"
    return [TRANSCRIPTION_PROFILE_LADDER[-1]], "requested-fallback"


class FasterWhisperBackend:
    def __init__(self, profile: TranscriptionProfile, cache_dir: str | None = None) -> None:
        self.profile = profile
        self.cache_dir = cache_dir or None
        self._model: Any | None = None
        self._is_batched = False

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
        except ModuleNotFoundError as exc:
            raise TranscriptionError(
                "faster-whisper is not installed. Install the project dependencies first."
            ) from exc

        kwargs: dict[str, Any] = {
            "device": self.profile.device,
            "compute_type": self.profile.compute_type,
            "download_root": self.cache_dir,
        }
        if self.profile.device == "cpu":
            kwargs["cpu_threads"] = max(os.cpu_count() or 1, 1)
            kwargs["num_workers"] = 1

        model = WhisperModel(self.profile.model_id, **kwargs)
        if self.profile.batch_size > 1:
            self._model = BatchedInferencePipeline(model=model)
            self._is_batched = True
        else:
            self._model = model
            self._is_batched = False
        return self._model

    def transcribe(self, audio_path: Path, language: str) -> tuple[list[Cue], dict[str, Any]]:
        model = self._load()
        transcribe_kwargs: dict[str, Any] = {
            "language": language,
            "task": "transcribe",
            "beam_size": self.profile.beam_size,
            "vad_filter": True,
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "temperature": 0.0,
        }
        if self._is_batched:
            transcribe_kwargs["batch_size"] = self.profile.batch_size
        segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)

        cues: list[Cue] = []
        for segment in segments:
            for cue in self._segment_to_cues(segment):
                cue.index = len(cues) + 1
                cues.append(cue)

        metadata = {
            "detected_language": getattr(info, "language", language),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
        }
        return cues, metadata

    def _segment_to_cues(self, segment: Any) -> list[Cue]:
        text = " ".join(str(getattr(segment, "text", "")).split()).strip()
        if not text:
            return []

        words = list(getattr(segment, "words", []) or [])
        if not words:
            return self._split_text_proportionally(
                text,
                float(getattr(segment, "start", 0.0) or 0.0),
                float(getattr(segment, "end", 0.0) or 0.0),
            )

        cues: list[Cue] = []
        current: list[str] = []
        current_start: float | None = None
        current_end: float | None = None
        for word in words:
            token = str(getattr(word, "word", "")).strip()
            if not token:
                continue
            start = float(getattr(word, "start", 0.0) or 0.0)
            end = max(float(getattr(word, "end", start + 0.2) or start + 0.2), start + 0.05)
            if current_start is None:
                current_start = start
            current.append(token)
            current_end = end
            joined = "".join(current).strip()
            duration = (current_end or end) - current_start
            should_split = (
                token.endswith(("\u3002", "\uff1f", "\uff01", "?", "!"))
                or len(joined) >= 42
                or duration >= 7.0
            )
            if should_split and joined:
                cues.append(
                    Cue(
                        index=0,
                        start=current_start,
                        end=max(current_end or end, current_start + 0.5),
                        text=joined,
                    )
                )
                current = []
                current_start = None
                current_end = None

        joined = "".join(current).strip()
        if joined and current_start is not None:
            cues.append(
                Cue(
                    index=0,
                    start=current_start,
                    end=max(current_end or current_start + 0.5, current_start + 0.5),
                    text=joined,
                )
            )
        return cues or self._split_text_proportionally(
            text,
            float(getattr(segment, "start", 0.0) or 0.0),
            float(getattr(segment, "end", 0.0) or 0.0),
        )

    def _split_text_proportionally(self, text: str, start: float, end: float) -> list[Cue]:
        end = max(end, start + 0.5)
        parts = [match.group(1).strip() for match in SENTENCE_ENDING_PATTERN.finditer(text)]
        parts = [part for part in parts if part]
        if len(parts) <= 1:
            return [Cue(index=0, start=start, end=end, text=text)]

        total_chars = max(sum(len(part) for part in parts), 1)
        duration = end - start
        cues: list[Cue] = []
        cursor = start
        for part in parts:
            share = len(part) / total_chars
            part_duration = max(duration * share, 0.5)
            part_end = min(cursor + part_duration, end)
            cues.append(Cue(index=0, start=cursor, end=max(part_end, cursor + 0.5), text=part))
            cursor = part_end
        cues[-1].end = end
        return cues

    def close(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class CourseTranscriptionRunner:
    def __init__(
        self,
        *,
        ffmpeg: FFmpegClient,
        cache_dir: str | None = None,
        requested_profile: str = "auto",
        low_memory_policy: str = "downgrade",
        snapshot_provider: Callable[[], ResourceSnapshot] = capture_snapshot,
        backend_factory: Callable[[TranscriptionProfile, str | None], FasterWhisperBackend] | None = None,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.cache_dir = cache_dir or None
        self.requested_profile = requested_profile
        self.low_memory_policy = low_memory_policy
        self.snapshot_provider = snapshot_provider
        self.backend_factory = backend_factory or self._default_backend_factory
        self._backend_cache: dict[str, FasterWhisperBackend] = {}

    def _default_backend_factory(
        self,
        profile: TranscriptionProfile,
        cache_dir: str | None,
    ) -> FasterWhisperBackend:
        return FasterWhisperBackend(profile, cache_dir=cache_dir)

    def _cached_profile_candidates(self) -> list[TranscriptionProfile]:
        profile_map = _profile_map()
        return [profile_map[name] for name in self._backend_cache if name in profile_map]

    def _prefer_cached_profiles(
        self,
        candidates: list[TranscriptionProfile],
    ) -> list[TranscriptionProfile]:
        if not self._backend_cache:
            return candidates
        cached_names = set(self._backend_cache)
        cached = [profile for profile in self._cached_profile_candidates() if profile in candidates]
        remainder = [profile for profile in candidates if profile.name not in cached_names]
        return cached + remainder

    def transcribe_path(
        self,
        input_path: Path,
        output_root: Path,
        *,
        language: str = "en",
        recursive: bool = False,
        glossary_path: Path | None = None,
    ) -> list[TranscriptionResult]:
        input_path = input_path.resolve()
        output_root = output_root.resolve()

        if input_path.is_dir():
            sources = list_video_sources(input_path, recursive=recursive)
        elif input_path.is_file():
            sources = [input_path]
        else:
            raise TranscriptionError(f"Input path not found: {input_path}")

        if not sources:
            raise TranscriptionError(f"No video files were found in {input_path}.")

        output_root.mkdir(parents=True, exist_ok=True)
        if glossary_path is not None:
            self._copy_glossary(glossary_path, output_root)

        results: list[TranscriptionResult] = []
        try:
            for source in sources:
                results.append(
                    self.transcribe_source(
                        source_path=source,
                        output_root=output_root,
                        input_root=input_path if input_path.is_dir() else None,
                        language=language,
                    )
                )

            self._write_batch_manifest(output_root, input_path, language, results, glossary_path)
            self._write_combined_transcript(output_root, results)
            return results
        finally:
            self.close()

    def transcribe_source(
        self,
        *,
        source_path: Path,
        output_root: Path,
        input_root: Path | None,
        language: str,
    ) -> TranscriptionResult:
        source_path = source_path.resolve()
        output_dir = self._source_output_dir(source_path, output_root, input_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_txt_path = output_dir / f"{source_path.stem}.raw.txt"
        raw_srt_path = output_dir / f"{source_path.stem}.raw.srt"
        segments_json_path = output_dir / f"{source_path.stem}.raw.segments.json"
        meta_json_path = output_dir / f"{source_path.stem}.raw.meta.json"

        if all(path.exists() for path in (raw_txt_path, raw_srt_path, segments_json_path, meta_json_path)):
            return self._load_existing_result(
                meta_json_path=meta_json_path,
                raw_txt_path=raw_txt_path,
            )

        snapshot = self.snapshot_provider()
        candidates, selection_note = ordered_profile_candidates(
            snapshot,
            self.requested_profile,
            self.low_memory_policy,
        )
        candidates = self._prefer_cached_profiles(candidates)

        temp_dir = Path(tempfile.mkdtemp(prefix=f"{safe_slug(source_path.stem)}-"))
        audio_path = temp_dir / f"{source_path.stem}.wav"
        self.ffmpeg.extract_audio(source_path=source_path, audio_path=audio_path)

        chosen_profile: TranscriptionProfile | None = None
        backend_meta: dict[str, Any] = {}
        cues: list[Cue] = []
        last_error: Exception | None = None
        try:
            for profile in candidates:
                backend = self._get_backend(profile)
                try:
                    cues, backend_meta = backend.transcribe(audio_path, language=language)
                    chosen_profile = profile
                    break
                except Exception as exc:
                    last_error = exc
                    if not _is_memory_failure(exc):
                        raise
                    self._discard_backend(profile)
                finally:
                    if chosen_profile is None and profile.name not in self._backend_cache:
                        backend.close()
            if chosen_profile is None:
                if self.low_memory_policy == "wait":
                    while chosen_profile is None:
                        time.sleep(10)
                        snapshot = self.snapshot_provider()
                        candidates, selection_note = ordered_profile_candidates(
                            snapshot,
                            self.requested_profile,
                            self.low_memory_policy,
                        )
                        candidates = self._prefer_cached_profiles(candidates)
                        for profile in candidates:
                            backend = self._get_backend(profile)
                            try:
                                cues, backend_meta = backend.transcribe(audio_path, language=language)
                                chosen_profile = profile
                                break
                            except Exception as exc:
                                last_error = exc
                                if not _is_memory_failure(exc):
                                    raise
                                self._discard_backend(profile)
                            finally:
                                if chosen_profile is None and profile.name not in self._backend_cache:
                                    backend.close()
                if chosen_profile is None:
                    raise TranscriptionError("All transcription profiles failed.") from last_error
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        transcript_text = self._render_transcript_text(cues)

        atomic_write_text(raw_txt_path, transcript_text)
        write_srt(raw_srt_path, cues)
        atomic_write_json(
            segments_json_path,
            [
                {
                    "index": cue.index,
                    "start": cue.start,
                    "end": cue.end,
                    "text": cue.text,
                }
                for cue in cues
            ],
        )

        duration = backend_meta.get("duration")
        if duration is not None:
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = None

        result = TranscriptionResult(
            source_path=str(source_path),
            relative_output_dir=str(
                output_dir.relative_to(input_root) if input_root is not None else Path(".")
            ),
            output_dir=str(output_dir),
            requested_profile=self.requested_profile,
            chosen_profile=chosen_profile.name,
            language=language,
            segment_count=len(cues),
            duration_seconds=duration,
            resource_snapshot=_resource_snapshot_dict(snapshot),
            profile=asdict(chosen_profile),
            backend={
                "engine": "faster-whisper",
                "model_id": chosen_profile.model_id,
                "device": chosen_profile.device,
                "compute_type": chosen_profile.compute_type,
                "batch_size": chosen_profile.batch_size,
                "beam_size": chosen_profile.beam_size,
            },
            outputs={
                "raw_txt": str(raw_txt_path),
                "raw_srt": str(raw_srt_path),
                "segments_json": str(segments_json_path),
                "meta_json": str(meta_json_path),
            },
            selection_note=selection_note,
            transcript_text=transcript_text,
        )
        atomic_write_json(meta_json_path, result.to_manifest_entry() | {"backend_info": backend_meta})
        return result

    def _get_backend(self, profile: TranscriptionProfile) -> FasterWhisperBackend:
        backend = self._backend_cache.get(profile.name)
        if backend is None:
            backend = self.backend_factory(profile, self.cache_dir)
            self._backend_cache[profile.name] = backend
        return backend

    def _discard_backend(self, profile: TranscriptionProfile) -> None:
        backend = self._backend_cache.pop(profile.name, None)
        if backend is not None:
            backend.close()

    def _load_existing_result(
        self,
        *,
        meta_json_path: Path,
        raw_txt_path: Path,
    ) -> TranscriptionResult:
        payload = json.loads(meta_json_path.read_text(encoding="utf-8"))
        payload.pop("backend_info", None)
        payload["transcript_text"] = raw_txt_path.read_text(encoding="utf-8")
        return TranscriptionResult(**payload)

    def _source_output_dir(
        self,
        source_path: Path,
        output_root: Path,
        input_root: Path | None,
    ) -> Path:
        if input_root is not None:
            return source_path.parent
        return source_path.parent

    def _render_transcript_text(self, cues: list[Cue]) -> str:
        lines = [
            f"[{format_srt_timestamp(cue.start)}] {cue.text.strip()}"
            for cue in cues
            if cue.text.strip()
        ]
        return "\n".join(lines).strip() + ("\n" if lines else "")

    def _copy_glossary(self, glossary_path: Path, output_root: Path) -> Path:
        source = glossary_path.resolve()
        if not source.exists():
            raise TranscriptionError(f"Glossary file not found: {source}")
        target = output_root / source.name
        atomic_write_text(target, source.read_text(encoding="utf-8"))
        return target

    def _write_batch_manifest(
        self,
        output_root: Path,
        input_path: Path,
        language: str,
        results: list[TranscriptionResult],
        glossary_path: Path | None,
    ) -> None:
        manifest = {
            "generated_at": now_iso(),
            "input_path": str(input_path),
            "output_path": str(output_root),
            "language": language,
            "requested_profile": self.requested_profile,
            "low_memory_policy": self.low_memory_policy,
            "glossary_path": str(glossary_path.resolve()) if glossary_path else None,
            "source_count": len(results),
            "entries": [result.to_manifest_entry() for result in results],
        }
        atomic_write_json(output_root / "course_manifest.json", manifest)

    def _write_combined_transcript(self, output_root: Path, results: list[TranscriptionResult]) -> None:
        parts: list[str] = []
        for result in results:
            title = Path(result.source_path).stem
            parts.append(f"# {title}\n\n{result.transcript_text.strip()}".strip())
        atomic_write_text(output_root / "course.raw.txt", "\n\n".join(parts).strip() + "\n")

    def close(self) -> None:
        for backend in self._backend_cache.values():
            backend.close()
        self._backend_cache.clear()
