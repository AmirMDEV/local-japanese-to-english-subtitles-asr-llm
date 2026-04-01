from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_subtitle_stack.config import AppConfig, CachePaths, ModelConfig, ToolPaths, default_profiles
from local_subtitle_stack.domain import (
    ChunkPlan,
    Cue,
    JOB_STATUS_WORKING,
    SceneContextBlock,
    StageProgress,
    STAGE_ADAPTED,
    STAGE_EXTRACT,
    SOURCE_KIND_SUBTITLE,
    SOURCE_KIND_VIDEO,
    TRANSLATION_SOURCE_DIRECT_EN,
    TRANSLATION_SOURCE_JA,
)
from local_subtitle_stack.guards import ResourceSnapshot
from local_subtitle_stack.integrations import SubtitleEditClient
from local_subtitle_stack.queue import QueueError, QueueStore
from local_subtitle_stack.service import WorkerService
from local_subtitle_stack.utils import subtitle_output_dir


class FakeFFmpeg:
    def __init__(self) -> None:
        self.extract_calls: list[dict[str, float | str]] = []

    def create_chunk_plan(
        self,
        source_path: Path,
        chunks_dir: Path,
        chunk_seconds: int,
        overlap_seconds: int,
        progress_callback=None,
    ) -> list[ChunkPlan]:
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunks_dir / "chunk_0001.wav"
        if progress_callback is not None:
            progress_callback(
                {
                    "current_chunk": 1,
                    "total_chunks": 1,
                    "covered_seconds": 6.0,
                    "total_seconds": 12.0,
                }
            )
            progress_callback(
                {
                    "current_chunk": 1,
                    "total_chunks": 1,
                    "covered_seconds": 12.0,
                    "total_seconds": 12.0,
                }
            )
        return [ChunkPlan(index=1, start=0.0, end=12.0, path=str(chunk_path))]

    def extract_chunk(
        self,
        *,
        source_path: Path,
        chunk_path: Path,
        start: float,
        duration: float,
        progress_callback=None,
    ) -> None:
        self.extract_calls.append(
            {
                "source_path": str(source_path),
                "chunk_path": str(chunk_path),
                "start": start,
                "duration": duration,
            }
        )
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.write_text("chunk", encoding="utf-8")
        if progress_callback is not None:
            progress_callback(duration / 2)
            progress_callback(duration)


class FakeSubtitleEdit(SubtitleEditClient):
    def __init__(self) -> None:
        self.opened: list[Path] = []

    def open_files(self, paths: list[Path]) -> None:
        self.opened = paths


class FakeASR:
    def __init__(self, model_id: str, cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir

    def transcribe_chunk(self, chunk_path: Path, batch_size: int, device: str) -> list[Cue]:
        return [
            Cue(index=1, start=0.0, end=1.2, text="motto shite"),
            Cue(index=2, start=1.6, end=3.1, text="onegai"),
        ]

    def close(self) -> None:
        return None


class FailOnSecondChunkASR(FakeASR):
    def transcribe_chunk(self, chunk_path: Path, batch_size: int, device: str) -> list[Cue]:
        if "0002" in chunk_path.stem:
            raise RuntimeError("synthetic transcription failure")
        return [
            Cue(index=1, start=0.0, end=1.2, text="partial first chunk"),
        ]


class SuccessfulOllama:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate_json(self, model: str, prompt: str, temperature: float) -> dict[str, list[str]]:
        self.calls.append((model, prompt))
        count = prompt.count('"index":')
        prefix = "adapted" if '"literal_en"' in prompt else "literal"
        return {"translations": [f"{prefix} line {index + 1}" for index in range(count)]}


class AdaptedFallbackOllama(SuccessfulOllama):
    def generate_json(self, model: str, prompt: str, temperature: float) -> dict[str, list[str]]:
        if '"literal_en"' in prompt:
            self.calls.append((model, prompt))
            return {"translations": [""]}
        return super().generate_json(model, prompt, temperature)


class BadLiteralOllama(SuccessfulOllama):
    def generate_json(self, model: str, prompt: str, temperature: float) -> dict[str, list[str]]:
        if '"literal_en"' in prompt:
            return super().generate_json(model, prompt, temperature)
        return {"translations": [""]}


class SwitchableLiteralFailureOllama(SuccessfulOllama):
    def __init__(self) -> None:
        super().__init__()
        self.fail_literal = False

    def generate_json(self, model: str, prompt: str, temperature: float) -> dict[str, list[str]]:
        if self.fail_literal and '"literal_en"' not in prompt:
            return {"translations": [""]}
        return super().generate_json(model, prompt, temperature)


class SelectiveLiteralFailureOllama(SuccessfulOllama):
    def __init__(self, failing_filenames: set[str]) -> None:
        super().__init__()
        self.failing_filenames = failing_filenames

    def generate_json(self, model: str, prompt: str, temperature: float) -> dict[str, list[str]]:
        if '"literal_en"' not in prompt and any(f"filename={name}" in prompt for name in self.failing_filenames):
            return {"translations": [""]}
        return super().generate_json(model, prompt, temperature)


class RetryableJsonOllama(SuccessfulOllama):
    def __init__(self) -> None:
        super().__init__()
        self.fail_once = True

    def generate_json(self, model: str, prompt: str, temperature: float) -> dict[str, list[str]]:
        if self.fail_once:
            self.fail_once = False
            raise json.JSONDecodeError("bad json", "{", 1)
        return super().generate_json(model, prompt, temperature)


class SplitRecoveringOllama(SuccessfulOllama):
    def generate_json(self, model: str, prompt: str, temperature: float) -> dict[str, list[str]]:
        count = prompt.count('"index":')
        if count > 1:
            prefix = "adapted" if '"literal_en"' in prompt else "literal"
            return {"translations": [f"{prefix} split line {index + 1}" for index in range(count - 1)]}
        return super().generate_json(model, prompt, temperature)


class FailOnSecondLiteralGroupOllama(SuccessfulOllama):
    def __init__(self) -> None:
        super().__init__()
        self.literal_group_calls = 0

    def generate_json(self, model: str, prompt: str, temperature: float) -> dict[str, list[str]]:
        if '"literal_en"' not in prompt:
            self.literal_group_calls += 1
            if self.literal_group_calls == 2:
                raise RuntimeError("literal group 2 failed")
        return super().generate_json(model, prompt, temperature)


def build_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        config_path=str(tmp_path / "config.toml"),
        queue_root=str(tmp_path / "queue"),
        tools=ToolPaths(ffmpeg="ffmpeg", ffprobe="ffprobe", ollama="ollama", subtitle_edit="subtitle", python311="py311"),
        cache_paths=CachePaths(),
        models=ModelConfig(),
        profiles=default_profiles(),
    )


def patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_subtitle_stack.service.TransformersASRClient", FakeASR)
    monkeypatch.setattr("local_subtitle_stack.service.choose_device", lambda _min: "cpu")
    monkeypatch.setattr("local_subtitle_stack.service.ensure_safe_to_start_job", lambda *args, **kwargs: None)
    monkeypatch.setattr("local_subtitle_stack.service.ensure_safe_to_start_gpu_phase", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "local_subtitle_stack.service.capture_snapshot",
        lambda: ResourceSnapshot(free_ram_mb=12_000, process_rss_mb=512, gpu_free_mb=0, gpu_total_mb=0),
    )


def build_service(tmp_path: Path, ollama: SuccessfulOllama) -> WorkerService:
    config = build_config(tmp_path)
    store = QueueStore(config)
    return WorkerService(
        config=config,
        store=store,
        ffmpeg=FakeFFmpeg(),
        subtitle_edit=FakeSubtitleEdit(),
        ollama=ollama,
    )


class TwoChunkFFmpeg(FakeFFmpeg):
    def create_chunk_plan(
        self,
        source_path: Path,
        chunks_dir: Path,
        chunk_seconds: int,
        overlap_seconds: int,
        progress_callback=None,
    ) -> list[ChunkPlan]:
        chunks_dir.mkdir(parents=True, exist_ok=True)
        plans = [
            ChunkPlan(index=1, start=0.0, end=12.0, path=str(chunks_dir / "chunk_0001.wav")),
            ChunkPlan(index=2, start=12.0, end=24.0, path=str(chunks_dir / "chunk_0002.wav")),
        ]
        if progress_callback is not None:
            progress_callback(
                {
                    "current_chunk": 1,
                    "total_chunks": 2,
                    "covered_seconds": 12.0,
                    "total_seconds": 24.0,
                }
            )
            progress_callback(
                {
                    "current_chunk": 2,
                    "total_chunks": 2,
                    "covered_seconds": 24.0,
                    "total_seconds": 24.0,
                }
            )
        return plans


def write_srt_fixture(path: Path, entries: list[tuple[str, str, str]]) -> None:
    blocks: list[str] = []
    for index, (start, end, text) in enumerate(entries, start=1):
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def test_worker_creates_local_and_exported_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "scene.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")

    service.run_until_empty()

    job_dir, loaded = service.store.find_job(manifest.job_id)
    output_dir = Path(loaded.export_dir)
    assert job_dir.parent.name == "done"
    assert (job_dir / loaded.artifacts["ja_srt"]).exists()
    assert (job_dir / loaded.artifacts["literal_srt"]).exists()
    assert (job_dir / loaded.artifacts["adapted_srt"]).exists()
    assert (job_dir / loaded.artifacts["review"]).exists()
    assert output_dir == tmp_path / "scene.mp4 subtitles"
    assert (output_dir / loaded.artifacts["ja_srt"]).exists()
    assert (output_dir / loaded.artifacts["literal_srt"]).exists()
    assert (output_dir / loaded.artifacts["adapted_srt"]).exists()
    assert (output_dir / loaded.artifacts["review"]).exists()
    assert not (output_dir / "scene.en.literal.partial.srt").exists()
    assert not (output_dir / "scene.en.adapted.partial.srt").exists()


def test_enqueue_creates_source_side_output_folder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "queued-scene.mp4"
    source.write_text("video", encoding="utf-8")

    manifest = service.enqueue(source, profile="default")

    assert Path(manifest.export_dir).exists()


def test_transcribe_extracts_audio_chunks_on_demand_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "lazy-audio.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")

    service.run_until_empty()

    _job_dir, loaded = service.store.find_job(manifest.job_id)
    assert isinstance(service.ffmpeg, FakeFFmpeg)
    assert len(service.ffmpeg.extract_calls) == 1
    assert loaded.chunk_plan
    chunk_path = Path(loaded.chunk_plan[0].path)
    assert not chunk_path.exists()


def test_partial_japanese_srt_survives_when_transcription_fails_midway(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    monkeypatch.setattr("local_subtitle_stack.service.TransformersASRClient", FailOnSecondChunkASR)
    service = build_service(tmp_path, SuccessfulOllama())
    service.ffmpeg = TwoChunkFFmpeg()
    source = tmp_path / "partial-ja.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")

    service.run_until_empty()

    job_dir, loaded = service.store.find_job(manifest.job_id)
    export_dir = Path(loaded.export_dir)
    local_ja = job_dir / loaded.artifacts["ja_srt"]
    export_ja = export_dir / loaded.artifacts["ja_srt"]

    assert job_dir.parent.name == "failed"
    assert local_ja.exists()
    assert export_ja.exists()
    assert "partial first chunk" in local_ja.read_text(encoding="utf-8")
    assert "partial first chunk" in export_ja.read_text(encoding="utf-8")


def test_job_can_skip_easy_english_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "no-easy.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default", include_adapted_english=False)

    service.run_until_empty()

    job_dir, loaded = service.store.find_job(manifest.job_id)
    export_dir = Path(loaded.export_dir)
    preview = service.preview_rows(manifest.job_id)

    assert loaded.include_adapted_english is False
    assert loaded.checkpoint(STAGE_ADAPTED).status == "completed"
    assert loaded.checkpoint(STAGE_ADAPTED).details["mode"] == "skipped"
    assert (job_dir / loaded.artifacts["ja_srt"]).exists()
    assert (job_dir / loaded.artifacts["literal_srt"]).exists()
    assert not (job_dir / loaded.artifacts["adapted_srt"]).exists()
    assert (export_dir / loaded.artifacts["ja_srt"]).exists()
    assert (export_dir / loaded.artifacts["literal_srt"]).exists()
    assert not (export_dir / loaded.artifacts["adapted_srt"]).exists()
    assert all(row["adapted_english"] == "" for row in preview)
    assert service.open_review(manifest.job_id) == [
        export_dir / loaded.artifacts["ja_srt"],
        export_dir / loaded.artifacts["literal_srt"],
    ]


def test_open_review_prefers_exported_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "review.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")

    service.run_until_empty()

    paths = service.open_review(manifest.job_id)
    output_dir = tmp_path / "review.mp4 subtitles"
    assert paths == [
        output_dir / "review.ja.srt",
        output_dir / "review.en.literal.srt",
        output_dir / "review.en.adapted.srt",
    ]
    assert service.subtitle_edit.opened == paths


def test_finished_subtitles_are_exported_beside_source_even_if_finalize_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "finalize-boom.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")

    def boom(_job_dir: Path, failing_manifest) -> None:
        failing_manifest.current_stage = "finalize"
        failing_manifest.checkpoint("finalize").attempts += 1
        raise RuntimeError("finalize boom")

    monkeypatch.setattr(service, "_stage_finalize", boom)

    service.run_until_empty()

    job_dir, loaded = service.store.find_job(manifest.job_id)
    output_dir = Path(loaded.export_dir)
    assert job_dir.parent.name == "failed"
    assert (output_dir / loaded.artifacts["ja_srt"]).exists()
    assert (output_dir / loaded.artifacts["literal_srt"]).exists()
    assert (output_dir / loaded.artifacts["adapted_srt"]).exists()


def test_failed_translation_keeps_readable_partial_direct_english_beside_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    ollama = FailOnSecondLiteralGroupOllama()
    service = build_service(tmp_path, ollama)
    primary_srt = tmp_path / "partial-literal.ja.srt"
    write_srt_fixture(
        primary_srt,
        [
            ("00:00:00,000", "00:00:01,000", "line 1"),
            ("00:00:01,000", "00:00:02,000", "line 2"),
            ("00:00:02,000", "00:00:03,000", "line 3"),
            ("00:00:03,000", "00:00:04,000", "line 4"),
            ("00:00:04,000", "00:00:05,000", "line 5"),
            ("00:00:05,000", "00:00:06,000", "line 6"),
            ("00:00:06,000", "00:00:07,000", "line 7"),
        ],
    )

    manifest = service.import_existing(
        profile="conservative",
        primary_subtitle=primary_srt,
        japanese=primary_srt,
    )

    with pytest.raises(QueueError, match="literal group 2 failed"):
        service.rebuild_english(
            manifest.job_id,
            batch_label=None,
            overall_context=None,
            scene_contexts=[],
        )

    output_dir = subtitle_output_dir(primary_srt)
    partial_literal = output_dir / "partial-literal.ja.en.literal.partial.srt"
    final_literal = output_dir / "partial-literal.ja.en.literal.srt"
    assert partial_literal.exists()
    assert not final_literal.exists()
    assert "literal line 1" in partial_literal.read_text(encoding="utf-8")


def test_adapted_translation_uses_context_and_falls_back_to_literal_and_marks_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    ollama = AdaptedFallbackOllama()
    service = build_service(tmp_path, ollama)
    source = tmp_path / "fallback.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(
        source,
        profile="default",
        context="The scene compares appearance and family resemblance.",
        scene_contexts=[
            SceneContextBlock(start_seconds=0.0, end_seconds=10.0, notes="Travel conversation."),
        ],
    )

    service.run_until_empty()

    _job_dir, loaded = service.store.find_job(manifest.job_id)
    output_dir = Path(loaded.export_dir)
    literal = (output_dir / loaded.artifacts["literal_srt"]).read_text(encoding="utf-8")
    adapted = (output_dir / loaded.artifacts["adapted_srt"]).read_text(encoding="utf-8")
    review = (output_dir / loaded.artifacts["review"]).read_text(encoding="utf-8")
    assert literal == adapted
    assert "translation-fallback" in review
    assert any("appearance and family resemblance" in prompt for _model, prompt in ollama.calls if '"literal_en"' in prompt)
    assert any("Travel conversation." in prompt for _model, prompt in ollama.calls if '"literal_en"' in prompt)


def test_enqueue_folder_only_queues_video_files_and_skips_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source_dir = tmp_path / "folder"
    source_dir.mkdir()
    (source_dir / "one.mp4").write_text("video", encoding="utf-8")
    (source_dir / "two.mkv").write_text("video", encoding="utf-8")
    (source_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    nested = source_dir / "nested"
    nested.mkdir()
    (nested / "three.mp4").write_text("video", encoding="utf-8")

    manifests, skipped = service.enqueue_folder(source_dir, profile="default")
    assert [manifest.source_name for manifest in manifests] == ["one.mp4", "two.mkv"]
    assert skipped == []

    recursive_manifests, recursive_skipped = service.enqueue_folder(
        source_dir,
        profile="default",
        recursive=True,
    )
    assert [manifest.source_name for manifest in recursive_manifests] == ["three.mp4"]
    assert len(recursive_skipped) == 2

    manifests, skipped = service.enqueue_folder(source_dir, profile="default")
    assert manifests == []
    assert len(skipped) == 2


def test_open_output_folder_prefers_export_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "open-output.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")
    opened: list[list[str]] = []
    monkeypatch.setattr(
        "local_subtitle_stack.service.subprocess.Popen",
        lambda args: opened.append(args),
    )

    service.run_until_empty()

    output_dir = service.open_output_folder(manifest.job_id)
    assert output_dir == tmp_path / "open-output.mp4 subtitles"
    assert opened == [["explorer", str(output_dir)]]


def test_literal_failure_finishes_as_failed_without_crashing_the_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, BadLiteralOllama())
    source = tmp_path / "retry-literal.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")

    service.run_until_empty()

    job_dir, loaded = service.store.find_job(manifest.job_id)
    assert job_dir.parent.name == "failed"
    assert loaded.checkpoint("translate_literal").attempts == 2


def test_job_start_floor_switches_after_transcribe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "floor-switch.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="conservative")
    profile = service.config.profile("conservative")

    assert service._job_start_min_free_ram(manifest, profile) == profile.min_free_ram_mb

    manifest.checkpoint("transcribe").status = "completed"

    assert (
        service._job_start_min_free_ram(manifest, profile)
        == profile.min_free_ram_translation_mb
    )


def test_invalid_json_translation_retries_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_runtime(monkeypatch)
    ollama = RetryableJsonOllama()
    service = build_service(tmp_path, ollama)
    source = tmp_path / "retry-json.mp4"
    source.write_text("video", encoding="utf-8")

    service.enqueue(source, profile="default")
    service.run_until_empty()

    assert len(ollama.calls) >= 2


def test_preview_rows_and_rebuild_english_use_saved_notes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    ollama = SuccessfulOllama()
    service = build_service(tmp_path, ollama)
    source = tmp_path / "preview.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")
    service.run_until_empty()

    preview = service.preview_rows(manifest.job_id)
    assert len(preview) == 2
    assert preview[0]["japanese"] == "motto shite"
    assert preview[0]["literal_english"] == "literal line 1"
    assert preview[0]["adapted_english"] == "adapted line 1"

    ollama.calls.clear()
    service.rebuild_english(
        manifest.job_id,
        batch_label="Batch A",
        overall_context="Whole video is about appearance comparison and tone.",
        scene_contexts=[
            SceneContextBlock(start_seconds=0.0, end_seconds=10.0, notes="Travel talk about family resemblance."),
        ],
    )

    _job_dir, loaded = service.load_job(manifest.job_id)
    assert loaded.series == "Batch A"
    assert loaded.scene_contexts[0].notes == "Travel talk about family resemblance."
    assert any("Whole video is about appearance comparison and tone." in prompt for _model, prompt in ollama.calls)
    assert any("Travel talk about family resemblance." in prompt for _model, prompt in ollama.calls)


def test_update_subtitle_line_updates_local_and_exported_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "edit-line.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")
    service.run_until_empty()

    job_dir, loaded = service.store.find_job(manifest.job_id)
    export_dir = Path(loaded.export_dir)

    service.update_subtitle_line(
        manifest.job_id,
        cue_index=1,
        japanese_text="henshu shimashita",
        literal_english_text="edited direct line",
        adapted_english_text="edited easy line",
    )

    preview = service.preview_rows(manifest.job_id)
    assert preview[0]["japanese"] == "henshu shimashita"
    assert preview[0]["literal_english"] == "edited direct line"
    assert preview[0]["adapted_english"] == "edited easy line"
    assert "henshu shimashita" in (job_dir / loaded.artifacts["ja_srt"]).read_text(encoding="utf-8")
    assert "edited direct line" in (job_dir / loaded.artifacts["literal_srt"]).read_text(encoding="utf-8")
    assert "edited easy line" in (job_dir / loaded.artifacts["adapted_srt"]).read_text(encoding="utf-8")
    assert "henshu shimashita" in (export_dir / loaded.artifacts["ja_srt"]).read_text(encoding="utf-8")
    assert "edited direct line" in (export_dir / loaded.artifacts["literal_srt"]).read_text(encoding="utf-8")
    assert "edited easy line" in (export_dir / loaded.artifacts["adapted_srt"]).read_text(encoding="utf-8")


def test_update_subtitle_line_rejects_empty_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "edit-empty.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")
    service.run_until_empty()

    with pytest.raises(QueueError, match="Direct English text cannot be empty."):
        service.update_subtitle_line(
            manifest.job_id,
            cue_index=1,
            literal_english_text="   ",
        )


def test_worker_continues_to_later_jobs_after_one_job_exhausts_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SelectiveLiteralFailureOllama({"bad-batch"}))
    bad_source = tmp_path / "bad-batch.mp4"
    good_source = tmp_path / "good-batch.mp4"
    bad_source.write_text("video", encoding="utf-8")
    good_source.write_text("video", encoding="utf-8")
    bad_manifest = service.enqueue(bad_source, profile="default")
    good_manifest = service.enqueue(good_source, profile="default")

    service.run_until_empty()

    bad_job_dir, bad_loaded = service.store.find_job(bad_manifest.job_id)
    good_job_dir, good_loaded = service.store.find_job(good_manifest.job_id)
    assert bad_job_dir.parent.name == "failed"
    assert bad_loaded.checkpoint("translate_literal").attempts == 2
    assert good_job_dir.parent.name == "done"
    assert (Path(good_loaded.export_dir) / good_loaded.artifacts["adapted_srt"]).exists()


def test_invalid_profile_name_fails_at_enqueue_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "profile-check.mp4"
    source.write_text("video", encoding="utf-8")

    with pytest.raises(QueueError, match="Unknown profile 'turbo'"):
        service.enqueue(source, profile="turbo")


def test_rebuild_english_failure_keeps_previous_english_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    ollama = SwitchableLiteralFailureOllama()
    service = build_service(tmp_path, ollama)
    source = tmp_path / "rebuild-keep.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")
    service.run_until_empty()

    job_dir, loaded = service.store.find_job(manifest.job_id)
    export_dir = Path(loaded.export_dir)
    original_literal_local = (job_dir / loaded.artifacts["literal_srt"]).read_text(encoding="utf-8")
    original_adapted_local = (job_dir / loaded.artifacts["adapted_srt"]).read_text(encoding="utf-8")
    original_literal_export = (export_dir / loaded.artifacts["literal_srt"]).read_text(encoding="utf-8")
    original_adapted_export = (export_dir / loaded.artifacts["adapted_srt"]).read_text(encoding="utf-8")

    ollama.fail_literal = True

    with pytest.raises(QueueError):
        service.rebuild_english(
            manifest.job_id,
            batch_label="Batch B",
            overall_context="Updated context that should not replace good files on failure.",
            scene_contexts=[],
        )

    assert (job_dir / loaded.artifacts["literal_srt"]).read_text(encoding="utf-8") == original_literal_local
    assert (job_dir / loaded.artifacts["adapted_srt"]).read_text(encoding="utf-8") == original_adapted_local
    assert (export_dir / loaded.artifacts["literal_srt"]).read_text(encoding="utf-8") == original_literal_export
    assert (export_dir / loaded.artifacts["adapted_srt"]).read_text(encoding="utf-8") == original_adapted_export


def test_rebuild_without_easy_english_removes_previous_easy_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "toggle-easy.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")
    service.run_until_empty()

    job_dir, loaded = service.store.find_job(manifest.job_id)
    export_dir = Path(loaded.export_dir)
    assert (job_dir / loaded.artifacts["adapted_srt"]).exists()
    assert (export_dir / loaded.artifacts["adapted_srt"]).exists()

    service.save_job_notes(
        manifest.job_id,
        batch_label=None,
        overall_context=None,
        scene_contexts=[],
        include_adapted_english=False,
    )
    rebuilt = service.rebuild_english(
        manifest.job_id,
        batch_label=None,
        overall_context=None,
        scene_contexts=[],
        include_adapted_english=False,
    )

    _job_dir, reloaded = service.store.find_job(rebuilt.job_id)
    preview = service.preview_rows(rebuilt.job_id)

    assert reloaded.include_adapted_english is False
    assert reloaded.checkpoint(STAGE_ADAPTED).details["mode"] == "skipped"
    assert not (job_dir / reloaded.artifacts["adapted_srt"]).exists()
    assert not (export_dir / reloaded.artifacts["adapted_srt"]).exists()
    assert all(row["adapted_english"] == "" for row in preview)


def test_import_existing_video_linked_japanese_and_reference_tracks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    ollama = SuccessfulOllama()
    service = build_service(tmp_path, ollama)
    video = tmp_path / "imported-scene.mp4"
    video.write_text("video", encoding="utf-8")
    ja_srt = tmp_path / "imported-scene.ja.srt"
    reference_srt = tmp_path / "imported-scene.reference.srt"
    write_srt_fixture(
        ja_srt,
        [
            ("00:00:00,000", "00:00:01,200", "motto shite"),
            ("00:00:01,600", "00:00:03,100", "onegai"),
        ],
    )
    write_srt_fixture(
        reference_srt,
        [
            ("00:00:00,000", "00:00:01,000", "please keep going"),
        ],
    )

    manifest = service.import_existing(
        profile="default",
        video=video,
        japanese=ja_srt,
        reference=reference_srt,
        context="Bath scene.",
    )

    job_dir, loaded = service.store.find_job(manifest.job_id)
    preview = service.preview_rows(manifest.job_id)

    assert job_dir.parent.name == "done"
    assert loaded.source_kind == SOURCE_KIND_VIDEO
    assert loaded.translation_source_role == TRANSLATION_SOURCE_JA
    assert loaded.imported_tracks["ja"] == str(ja_srt.resolve())
    assert loaded.imported_tracks["reference"] == str(reference_srt.resolve())
    assert preview[0]["japanese"] == "motto shite"
    assert preview[0]["reference"] == "please keep going"
    assert (Path(loaded.export_dir) / loaded.artifacts["ja_srt"]).exists()


def test_import_existing_english_only_rebuilds_without_asr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    ollama = SuccessfulOllama()
    service = build_service(tmp_path, ollama)
    primary_srt = tmp_path / "english-only.srt"
    write_srt_fixture(
        primary_srt,
        [
            ("00:00:00,000", "00:00:01,200", "I match my mother's body type."),
            ("00:00:01,600", "00:00:03,100", "Look at my chest."),
        ],
    )

    manifest = service.import_existing(
        profile="default",
        primary_subtitle=primary_srt,
        context="Breast and body-type comparison scene.",
    )
    rebuilt = service.rebuild_english(
        manifest.job_id,
        batch_label="Imported batch",
        overall_context="Breast and body-type comparison scene.",
        scene_contexts=[],
    )

    job_dir, loaded = service.store.find_job(rebuilt.job_id)
    preview = service.preview_rows(rebuilt.job_id)

    assert loaded.source_kind == SOURCE_KIND_SUBTITLE
    assert loaded.translation_source_role == TRANSLATION_SOURCE_DIRECT_EN
    assert not (job_dir / loaded.artifacts["ja_cues"]).exists()
    assert preview[0]["literal_english"] == "literal line 1"
    assert preview[0]["adapted_english"] == "adapted line 1"
    assert any(
        "rewriting English subtitle lines into cleaner direct English" in prompt
        for _model, prompt in ollama.calls
    )
    assert any("for English dialogue" in prompt for _model, prompt in ollama.calls if '"literal_en"' in prompt)


def test_import_existing_reference_lines_are_injected_into_translation_prompts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    ollama = SuccessfulOllama()
    service = build_service(tmp_path, ollama)
    primary_srt = tmp_path / "scene-with-reference.srt"
    reference_srt = tmp_path / "scene-with-reference.reference.srt"
    write_srt_fixture(
        primary_srt,
        [
            ("00:00:00,000", "00:00:01,200", "mune"),
        ],
    )
    write_srt_fixture(
        reference_srt,
        [
            ("00:00:00,000", "00:00:01,200", "Talking about breasts here."),
        ],
    )

    manifest = service.import_existing(
        profile="default",
        primary_subtitle=primary_srt,
        japanese=primary_srt,
        reference=reference_srt,
    )
    service.rebuild_english(
        manifest.job_id,
        batch_label=None,
        overall_context=None,
        scene_contexts=[],
    )

    assert any("Talking about breasts here." in prompt for _model, prompt in ollama.calls)


def test_import_existing_reuses_matching_job_instead_of_duplicating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    primary_srt = tmp_path / "reuse-me.srt"
    write_srt_fixture(
        primary_srt,
        [
            ("00:00:00,000", "00:00:01,000", "hello there"),
        ],
    )

    first = service.import_existing(profile="default", primary_subtitle=primary_srt)
    second = service.import_existing(profile="default", primary_subtitle=primary_srt)
    rows = service.status_rows()

    assert first.job_id == second.job_id
    assert [row["job_id"] for row in rows] == [first.job_id]


def test_status_rows_include_stage_and_overall_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    source = tmp_path / "progress-scene.mp4"
    source.write_text("video", encoding="utf-8")
    manifest = service.enqueue(source, profile="default")
    job_dir, manifest = service.store.find_job(manifest.job_id)
    manifest.status = JOB_STATUS_WORKING
    manifest.current_stage = STAGE_EXTRACT
    manifest.current_progress = StageProgress(
        stage=STAGE_EXTRACT,
        current=600.0,
        total=1200.0,
        unit="seconds",
        percent=50.0,
        eta_seconds=600.0,
        done_seconds=600.0,
        total_seconds=1200.0,
        message="Audio chunk 2 of 4",
    )
    service.store.save_manifest(job_dir, manifest)

    row = service.status_rows()[0]

    assert row["step_text"].startswith("Getting the audio ready")
    assert row["stage_progress_percent"] == "50.00"
    assert row["overall_progress_percent"] == "10.00"
    assert row["stage_eta_seconds"] == "600.00"
    assert row["stage_progress_message"] == "Audio chunk 2 of 4"


def test_rebuild_english_splits_batches_when_model_returns_too_few_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SplitRecoveringOllama())
    primary_srt = tmp_path / "split-recovery.ja.srt"
    write_srt_fixture(
        primary_srt,
        [
            ("00:00:00,000", "00:00:01,000", "line 1"),
            ("00:00:01,000", "00:00:02,000", "line 2"),
            ("00:00:02,000", "00:00:03,000", "line 3"),
            ("00:00:03,000", "00:00:04,000", "line 4"),
            ("00:00:04,000", "00:00:05,000", "line 5"),
            ("00:00:05,000", "00:00:06,000", "line 6"),
        ],
    )

    manifest = service.import_existing(
        profile="conservative",
        primary_subtitle=primary_srt,
        japanese=primary_srt,
    )

    rebuilt = service.rebuild_english(
        manifest.job_id,
        batch_label=None,
        overall_context=None,
        scene_contexts=[],
    )

    _job_dir, loaded = service.store.find_job(rebuilt.job_id)
    preview = service.preview_rows(rebuilt.job_id)
    review_path = Path(loaded.export_dir) / loaded.artifacts["review"]
    review = review_path.read_text(encoding="utf-8")

    assert len(preview) == 6
    assert all(row["literal_english"] for row in preview)
    assert all(row["adapted_english"] for row in preview)
    assert "translation-batch-retry" in review


def test_import_existing_requires_a_real_source_track(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_runtime(monkeypatch)
    service = build_service(tmp_path, SuccessfulOllama())
    video = tmp_path / "no-source.mp4"
    video.write_text("video", encoding="utf-8")
    reference_srt = tmp_path / "no-source.reference.srt"
    write_srt_fixture(
        reference_srt,
        [
            ("00:00:00,000", "00:00:01,000", "reference only"),
        ],
    )

    with pytest.raises(QueueError, match="Japanese or Direct English subtitle source track"):
        service.import_existing(
            profile="default",
            video=video,
            reference=reference_srt,
        )
