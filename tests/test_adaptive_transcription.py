from __future__ import annotations

from pathlib import Path

import pytest

from local_subtitle_stack.adaptive_transcription import (
    CourseTranscriptionRunner,
    TranscriptionError,
    TranscriptionProfile,
    TranscriptionResult,
    TRANSCRIPTION_PROFILE_LADDER,
    ordered_profile_candidates,
    profile_fits,
)
from local_subtitle_stack.domain import Cue
from local_subtitle_stack.guards import ResourceSnapshot


class FakeFFmpeg:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def extract_audio(self, *, source_path: Path, audio_path: Path, progress_callback=None) -> None:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_text("audio", encoding="utf-8")
        self.calls.append((str(source_path), str(audio_path)))


class FakeBackend:
    def __init__(self, profile: TranscriptionProfile, cache_dir: str | None = None) -> None:
        self.profile = profile
        self.closed = False
        self.calls: list[tuple[str, str]] = []

    def transcribe(self, audio_path: Path, language: str):
        self.calls.append((str(audio_path), language))
        return (
            [
                Cue(index=1, start=0.0, end=1.0, text=f"{self.profile.name} one"),
                Cue(index=2, start=1.0, end=2.0, text=f"{self.profile.name} two"),
            ],
            {"duration": 2.0, "detected_language": language},
        )

    def close(self) -> None:
        self.closed = True


class MemoryFailingBackend(FakeBackend):
    def transcribe(self, audio_path: Path, language: str):
        raise RuntimeError("CUDA out of memory")


def test_profile_selection_scales_down_to_available_resources() -> None:
    high = ResourceSnapshot(free_ram_mb=15_000, process_rss_mb=500, gpu_free_mb=7_000, gpu_total_mb=8_192)
    balanced = ResourceSnapshot(free_ram_mb=11_000, process_rss_mb=500, gpu_free_mb=4_000, gpu_total_mb=8_192)
    low_gpu = ResourceSnapshot(free_ram_mb=8_000, process_rss_mb=500, gpu_free_mb=2_000, gpu_total_mb=8_192)
    cpu = ResourceSnapshot(free_ram_mb=5_000, process_rss_mb=500, gpu_free_mb=0, gpu_total_mb=0)

    assert profile_fits(high, TRANSCRIPTION_PROFILE_LADDER[0])
    assert ordered_profile_candidates(high, "auto", "downgrade")[0][0].name == "high"
    assert ordered_profile_candidates(balanced, "auto", "downgrade")[0][0].name == "balanced"
    assert ordered_profile_candidates(low_gpu, "auto", "downgrade")[0][0].name == "low_gpu"
    assert ordered_profile_candidates(cpu, "auto", "downgrade")[0][0].name == "cpu_fallback"


def test_all_profiles_keep_multilingual_model_for_japanese() -> None:
    assert all("distil" not in profile.model_id for profile in TRANSCRIPTION_PROFILE_LADDER)


def test_batch_runner_writes_outputs_and_manifest(monkeypatch, tmp_path: Path) -> None:
    ffmpeg = FakeFFmpeg()
    snapshot = ResourceSnapshot(free_ram_mb=15_000, process_rss_mb=500, gpu_free_mb=7_000, gpu_total_mb=8_192)
    chosen_profiles: list[str] = []

    def backend_factory(profile: TranscriptionProfile, cache_dir: str | None) -> FakeBackend:
        chosen_profiles.append(profile.name)
        return FakeBackend(profile, cache_dir)

    runner = CourseTranscriptionRunner(
        ffmpeg=ffmpeg,  # type: ignore[arg-type]
        cache_dir=None,
        requested_profile="auto",
        low_memory_policy="downgrade",
        snapshot_provider=lambda: snapshot,
        backend_factory=backend_factory,
    )
    input_dir = tmp_path / "course"
    input_dir.mkdir()
    (input_dir / "lesson one.mp4").write_text("video", encoding="utf-8")
    (input_dir / "lesson two.mp4").write_text("video", encoding="utf-8")
    output_dir = tmp_path / "out"

    results = runner.transcribe_path(input_dir, output_dir, language="en")

    assert len(results) == 2
    assert chosen_profiles == ["high"]
    assert (output_dir / "lesson one.raw.txt").exists()
    assert (output_dir / "lesson one.raw.srt").exists()
    assert (output_dir / "lesson one.raw.segments.json").exists()
    assert (output_dir / "lesson one.raw.meta.json").exists()
    assert (output_dir / "course.raw.txt").exists()
    assert (output_dir / "course_manifest.json").exists()
    assert "# lesson one" in (output_dir / "course.raw.txt").read_text(encoding="utf-8")


def test_batch_runner_resumes_existing_outputs_without_retranscribing(tmp_path: Path) -> None:
    ffmpeg = FakeFFmpeg()
    snapshot = ResourceSnapshot(free_ram_mb=15_000, process_rss_mb=500, gpu_free_mb=7_000, gpu_total_mb=8_192)
    backend_calls: list[str] = []

    def backend_factory(profile: TranscriptionProfile, cache_dir: str | None) -> FakeBackend:
        backend_calls.append(profile.name)
        return FakeBackend(profile, cache_dir)

    runner = CourseTranscriptionRunner(
        ffmpeg=ffmpeg,  # type: ignore[arg-type]
        cache_dir=None,
        requested_profile="auto",
        low_memory_policy="downgrade",
        snapshot_provider=lambda: snapshot,
        backend_factory=backend_factory,
    )
    input_dir = tmp_path / "course"
    input_dir.mkdir()
    source = input_dir / "lesson one.mp4"
    source.write_text("video", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    original = runner.transcribe_path(input_dir, output_dir, language="en")
    first_source_count = len(backend_calls)
    backend_calls.clear()

    resumed = runner.transcribe_path(input_dir, output_dir, language="en")

    assert len(original) == 1
    assert len(resumed) == 1
    assert backend_calls == []
    assert first_source_count == 1


def test_batch_runner_downgrades_when_requested_profile_fails(monkeypatch, tmp_path: Path) -> None:
    ffmpeg = FakeFFmpeg()
    snapshot = ResourceSnapshot(free_ram_mb=8_000, process_rss_mb=500, gpu_free_mb=2_000, gpu_total_mb=8_192)
    attempted: list[str] = []

    class FailingHighBackend(FakeBackend):
        def transcribe(self, audio_path: Path, language: str):
            if self.profile.name == "high":
                raise RuntimeError("CUDA out of memory")
            return super().transcribe(audio_path, language)

    def backend_factory(profile: TranscriptionProfile, cache_dir: str | None):
        attempted.append(profile.name)
        return FailingHighBackend(profile, cache_dir)

    runner = CourseTranscriptionRunner(
        ffmpeg=ffmpeg,  # type: ignore[arg-type]
        requested_profile="high",
        low_memory_policy="downgrade",
        snapshot_provider=lambda: snapshot,
        backend_factory=backend_factory,
    )
    input_dir = tmp_path / "course"
    input_dir.mkdir()
    (input_dir / "lesson.mp4").write_text("video", encoding="utf-8")
    output_dir = tmp_path / "out"

    results = runner.transcribe_path(input_dir, output_dir, language="en")

    assert results[0].chosen_profile in {"low_gpu", "cpu_fallback"}
    assert attempted[0] == "low_gpu" or attempted[0] == "cpu_fallback"


def test_wait_policy_times_out_instead_of_waiting_forever(tmp_path: Path) -> None:
    ffmpeg = FakeFFmpeg()
    snapshot = ResourceSnapshot(free_ram_mb=1, process_rss_mb=500, gpu_free_mb=0, gpu_total_mb=0)
    statuses: list[str] = []
    runner = CourseTranscriptionRunner(
        ffmpeg=ffmpeg,  # type: ignore[arg-type]
        requested_profile="high",
        low_memory_policy="wait",
        snapshot_provider=lambda: snapshot,
        backend_factory=lambda profile, cache_dir: MemoryFailingBackend(profile, cache_dir),
        wait_timeout_seconds=0.0,
        wait_poll_seconds=0.0,
        status_callback=statuses.append,
    )
    source = tmp_path / "lesson.mp4"
    source.write_text("video", encoding="utf-8")

    with pytest.raises(TranscriptionError, match="Timed out"):
        runner.transcribe_path(source, tmp_path / "out", language="en")

    assert (tmp_path / "out").exists()
