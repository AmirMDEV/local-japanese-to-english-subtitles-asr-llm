from __future__ import annotations

import os
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .adaptive_transcription import (
    FasterWhisperBackend,
    TRANSCRIPTION_PROFILE_LADDER,
    TranscriptionProfile,
    _is_memory_failure,
    ordered_profile_candidates,
)
from .config import AppConfig
from .asr_models import QWEN3_ASR_1_7B_MODEL_ID, QWEN3_ASR_ENGINE, REAZON_K2_ENGINE, REAZON_K2_MODEL_ID
from .domain import (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PAUSED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_WORKING,
    STAGE_ADAPTED,
    STAGE_EXTRACT,
    STAGE_FINALIZE,
    STAGE_LITERAL,
    STAGE_TRANSCRIBE,
    Cue,
    JobEvent,
    JobManifest,
    ReviewFlag,
    SceneContextBlock,
    SOURCE_KIND_SUBTITLE,
    SOURCE_KIND_VIDEO,
    StageProgress,
    TRANSLATION_SOURCE_DIRECT_EN,
    TRANSLATION_SOURCE_JA,
)
from .guards import (
    capture_snapshot,
    choose_device,
    ensure_safe_to_start_gpu_phase,
    ensure_safe_to_start_job,
)
from .integrations import (
    FFmpegClient,
    OllamaClient,
    ReazonSpeechK2ASRClient,
    Qwen3ASRClient,
    SubtitleEditClient,
    TransformersASRClient,
    load_cues,
    save_cues,
)
from .pipeline import (
    apply_translations,
    build_adapted_prompt,
    build_coherence_pass_prompt,
    build_context_notes,
    build_direct_english_rewrite_prompt,
    build_literal_prompt_with_context,
    combine_chunk_cues,
    cue_groups,
    load_glossary,
    metadata_from_manifest,
    normalize_japanese_cues,
    parse_srt,
    strict_retry_prompt,
    subtitle_quality_flags,
    validate_translation_payload,
    write_review_flags,
    write_srt,
)
from .queue import QueueError, QueueStore
from .utils import (
    atomic_write_json,
    atomic_write_text,
    elapsed_seconds_since,
    format_duration_compact,
    format_timecode,
    list_video_sources,
    now_iso,
    parse_iso_datetime,
    parse_timecode,
    read_json,
    subtitle_output_dir,
)


class PauseRequested(RuntimeError):
    pass


STAGE_DISPLAY_LABELS = {
    STAGE_EXTRACT: "Getting the audio ready",
    STAGE_TRANSCRIBE: "Listening to the Japanese",
    STAGE_LITERAL: "Making direct English translation",
    STAGE_ADAPTED: "Making context-applied English",
    STAGE_FINALIZE: "Saving the subtitle files",
}
COHERENCE_REVIEW_FILENAME = "coherence-review.json"

ASR_ENGINE_FASTER_WHISPER = "faster-whisper"
ASR_ENGINE_KOTOBA = "kotoba"
ASR_ENGINE_QWEN3 = QWEN3_ASR_ENGINE
ASR_ENGINE_REAZON_K2 = REAZON_K2_ENGINE
QWEN3_ASR_CHUNK_SECONDS = 30
REAZON_K2_CHUNK_SECONDS = 6


class WorkerService:
    def __init__(
        self,
        config: AppConfig,
        store: QueueStore,
        ffmpeg: FFmpegClient,
        subtitle_edit: SubtitleEditClient,
        ollama: OllamaClient,
    ) -> None:
        self.config = config
        self.store = store
        self.ffmpeg = ffmpeg
        self.subtitle_edit = subtitle_edit
        self.ollama = ollama
        self._progress_save_cache: dict[str, tuple[float, float]] = {}
        self._cue_cache: dict[Path, tuple[tuple[int, int], list[Cue]]] = {}

    def enqueue(
        self,
        source: Path,
        profile: str,
        glossary: Path | None = None,
        series: str | None = None,
        context: str | None = None,
        scene_contexts: list[SceneContextBlock] | None = None,
        include_adapted_english: bool = True,
        prefer_fast_translation: bool = False,
    ) -> JobManifest:
        self._require_profile(profile)
        manifest = self.store.enqueue(
            source_path=source,
            profile=profile,
            glossary_path=glossary,
            series=series,
            job_context=context,
            scene_contexts=scene_contexts,
            include_adapted_english=include_adapted_english,
            prefer_fast_translation=prefer_fast_translation,
        )
        self._ensure_output_dir_for_manifest(manifest)
        job_dir, manifest = self.store.find_job(manifest.job_id)
        self._append_event(manifest, "info", f"Queued {manifest.source_name}.")
        self._save_manifest(job_dir, manifest)
        return manifest

    def enqueue_many(
        self,
        sources: list[Path],
        profile: str,
        glossary: Path | None = None,
        series: str | None = None,
        context: str | None = None,
        scene_contexts: list[SceneContextBlock] | None = None,
        include_adapted_english: bool = True,
        prefer_fast_translation: bool = False,
    ) -> tuple[list[JobManifest], list[Path]]:
        self._require_profile(profile)
        manifests: list[JobManifest] = []
        skipped: list[Path] = []
        existing = {
            self._source_key(Path(manifest.source_path))
            for _job_dir, manifest, _state in self.store.list_jobs()
        }
        seen: set[str] = set()
        for source in sources:
            resolved = source.resolve()
            key = self._source_key(resolved)
            if key in seen or key in existing:
                skipped.append(resolved)
                continue
            seen.add(key)
            existing.add(key)
            manifests.append(
                self.enqueue(
                    source=resolved,
                    profile=profile,
                    glossary=glossary,
                    series=series,
                    context=context,
                    scene_contexts=scene_contexts,
                    include_adapted_english=include_adapted_english,
                    prefer_fast_translation=prefer_fast_translation,
                )
            )
        return manifests, skipped

    def enqueue_folder(
        self,
        folder: Path,
        profile: str,
        glossary: Path | None = None,
        series: str | None = None,
        context: str | None = None,
        scene_contexts: list[SceneContextBlock] | None = None,
        recursive: bool = False,
        include_adapted_english: bool = True,
        prefer_fast_translation: bool = False,
    ) -> tuple[list[JobManifest], list[Path]]:
        if not folder.exists():
            raise QueueError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise QueueError(f"Folder path is not a directory: {folder}")
        self._require_profile(profile)
        return self.enqueue_many(
            list_video_sources(folder, recursive=recursive),
            profile=profile,
            glossary=glossary,
            series=series,
            context=context,
            scene_contexts=scene_contexts,
            include_adapted_english=include_adapted_english,
            prefer_fast_translation=prefer_fast_translation,
        )

    def detect_existing_subtitles(self, video: Path) -> dict[str, Path]:
        resolved = video.resolve()
        stem = resolved.stem
        output_dir = subtitle_output_dir(resolved)
        search_roots = [resolved.parent]
        if output_dir not in search_roots:
            search_roots.append(output_dir)
        patterns = {
            "ja": [
                f"{stem}.ja.srt",
                f"{stem}.jp.srt",
                f"{stem}.japanese.srt",
            ],
            "direct": [
                f"{stem}.en.literal.srt",
                f"{stem}.en.srt",
                f"{stem}.english.srt",
                f"{stem}.srt",
            ],
            "easy": [
                f"{stem}.en.adapted.srt",
                f"{stem}.adapted.srt",
            ],
            "reference": [
                f"{stem}.reference.srt",
                f"{stem}.ref.srt",
            ],
        }
        found: dict[str, Path] = {}
        for role, candidates in patterns.items():
            for root in search_roots:
                for filename in candidates:
                    candidate = root / filename
                    if candidate.exists():
                        found[role] = candidate.resolve()
                        break
                if role in found:
                    break
        return found

    def import_existing(
        self,
        *,
        profile: str,
        video: Path | None = None,
        primary_subtitle: Path | None = None,
        japanese: Path | None = None,
        direct: Path | None = None,
        easy: Path | None = None,
        reference: Path | None = None,
        series: str | None = None,
        context: str | None = None,
        scene_contexts: list[SceneContextBlock] | None = None,
        include_adapted_english: bool = True,
        prefer_fast_translation: bool = False,
    ) -> JobManifest:
        self._require_profile(profile)
        if bool(video) == bool(primary_subtitle):
            raise QueueError("Choose either a video or a primary subtitle file.")

        resolved_video = video.resolve() if video else None
        resolved_primary = primary_subtitle.resolve() if primary_subtitle else None
        auto_detected = self.detect_existing_subtitles(resolved_video) if resolved_video else {}

        tracks: dict[str, Path] = {}
        for role, provided in (
            ("ja", japanese),
            ("direct", direct),
            ("easy", easy),
            ("reference", reference),
        ):
            if provided:
                tracks[role] = provided.resolve()
            elif role in auto_detected:
                tracks[role] = auto_detected[role]

        if resolved_primary and "ja" not in tracks and "direct" not in tracks:
            tracks["direct"] = resolved_primary

        if not tracks:
            raise QueueError("No subtitle files were provided or detected for import.")
        if "ja" not in tracks and "direct" not in tracks:
            raise QueueError("Import needs a Japanese or Direct English translation subtitle source track.")

        for role, path in list(tracks.items()):
            if path.suffix.lower() != ".srt":
                raise QueueError(f"Only .srt files are supported right now: {path.name}")
            if not path.exists():
                raise QueueError(f"Subtitle file not found: {path}")
            tracks[role] = path.resolve()

        source_path = resolved_video or resolved_primary
        assert source_path is not None
        translation_source_role = (
            TRANSLATION_SOURCE_JA if "ja" in tracks else TRANSLATION_SOURCE_DIRECT_EN
        )

        existing = self._find_job_by_source_path(source_path)
        if existing is None:
            manifest = self.store.enqueue(
                source_path=source_path,
                profile=profile,
                series=series,
                job_context=context,
                scene_contexts=scene_contexts,
                source_kind=SOURCE_KIND_VIDEO if resolved_video else SOURCE_KIND_SUBTITLE,
                linked_video_path=resolved_video,
                translation_source_role=translation_source_role,
                imported_tracks={role: str(path) for role, path in tracks.items()},
                include_adapted_english=include_adapted_english,
                prefer_fast_translation=prefer_fast_translation,
            )
            job_dir, manifest = self.store.find_job(manifest.job_id)
        else:
            job_dir, manifest = existing
            manifest.profile = profile
            manifest.source_kind = SOURCE_KIND_VIDEO if resolved_video else SOURCE_KIND_SUBTITLE
            manifest.linked_video_path = str(resolved_video) if resolved_video else None
            manifest.translation_source_role = translation_source_role
            manifest.imported_tracks.update({role: str(path) for role, path in tracks.items()})
            manifest.include_adapted_english = include_adapted_english
            manifest.prefer_fast_translation = prefer_fast_translation
            if series is not None:
                manifest.series = series
            if context is not None:
                manifest.job_context = context
            if scene_contexts is not None:
                manifest.scene_contexts = list(scene_contexts)
        manifest.include_adapted_english = include_adapted_english
        manifest.prefer_fast_translation = prefer_fast_translation

        if series is not None:
            manifest.series = series
        if context is not None:
            manifest.job_context = context
        if scene_contexts is not None:
            manifest.scene_contexts = list(scene_contexts)

        self._ensure_output_dir_for_manifest(manifest)
        self._seed_imported_tracks(job_dir, manifest, tracks)
        self._mark_imported_source_checkpoints(manifest)
        self._append_event(manifest, "info", f"Imported existing subtitles for {manifest.source_name}.")
        manifest.status = JOB_STATUS_COMPLETED
        manifest.current_stage = STAGE_FINALIZE
        review_path = job_dir / manifest.artifacts["review"]
        self._write_review_file(review_path, manifest.review_flags)
        self._export_text_artifact(
            review_path,
            self._ensure_output_dir_for_manifest(manifest) / manifest.artifacts["review"],
        )
        self._save_manifest(job_dir, manifest)
        if job_dir.parent != self.store.done_dir:
            job_dir, manifest = self.store.mark_completed(job_dir, manifest)
            self._write_resume_state(job_dir, manifest)
        return manifest

    def status_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for job_dir, manifest, state in self.store.list_jobs():
            progress = manifest.current_progress
            progress_age_seconds = elapsed_seconds_since(progress.updated_at) if progress is not None else None
            stale_completed_progress = (
                manifest.status == JOB_STATUS_COMPLETED
                and progress is not None
                and progress_age_seconds is not None
                and progress_age_seconds > 300
            )
            visible_progress = None if stale_completed_progress else progress
            visible_progress_age_seconds = None if stale_completed_progress else progress_age_seconds
            latest_event = manifest.events[-1] if manifest.events else None
            rows.append(
                {
                    "job_id": manifest.job_id,
                    "state_dir": state,
                    "status": manifest.status,
                    "stage": manifest.current_stage,
                    "progress_stage": visible_progress.stage if visible_progress is not None else "",
                    "step_text": (
                        "Waiting to start. Press Start processing all jobs."
                        if manifest.status == JOB_STATUS_QUEUED
                        else "Saved subtitle files"
                        if stale_completed_progress and manifest.current_stage == STAGE_FINALIZE
                        else self._stage_display_text(manifest)
                    ),
                    "source": manifest.source_name,
                    "updated_at": manifest.updated_at,
                    "stage_progress_percent": f"{100.0 if stale_completed_progress else self._current_stage_percent(manifest):.2f}",
                    "overall_progress_percent": f"{100.0 if stale_completed_progress else self._overall_progress_percent(manifest):.2f}",
                    "stage_eta_seconds": (
                        f"{visible_progress.eta_seconds:.2f}"
                        if visible_progress is not None and visible_progress.eta_seconds is not None
                        else ""
                    ),
                    "stage_progress_message": visible_progress.message if visible_progress is not None and visible_progress.message else "",
                    "progress_updated_at": visible_progress.updated_at if visible_progress is not None else "",
                    "progress_age_seconds": f"{visible_progress_age_seconds:.0f}" if visible_progress_age_seconds is not None else "",
                    "progress_age_text": format_duration_compact(visible_progress_age_seconds) if visible_progress_age_seconds is not None else "",
                    "source_kind": manifest.source_kind,
                    "translation_source_role": manifest.translation_source_role,
                    "has_reference": "true" if manifest.imported_tracks.get("reference") else "false",
                    "include_adapted_english": "true" if manifest.include_adapted_english else "false",
                    "prefer_fast_translation": "true" if manifest.prefer_fast_translation else "false",
                    "stop_requested": "true" if self.store.job_stop_requested(job_dir) else "false",
                    "current_model": self._current_model_name(manifest),
                    "latest_event_message": latest_event.message if latest_event else "",
                    "latest_event_level": latest_event.level if latest_event else "",
                    "latest_event_at": latest_event.created_at if latest_event else "",
                    "latest_event_stage": latest_event.stage or "" if latest_event else "",
                }
            )
        return rows

    def load_job(self, job_id: str) -> tuple[Path, JobManifest]:
        return self.store.find_job(job_id)

    def preview_rows(self, job_id: str) -> list[dict[str, str | float | int]]:
        job_dir, manifest = self.store.find_job(job_id)
        ja_cues = self._load_optional_cues(job_dir, manifest, "ja_cues")
        literal_cues = self._load_optional_cues(job_dir, manifest, "literal_cues")
        adapted_cues = self._load_optional_cues(job_dir, manifest, "adapted_cues")
        reference_cues = self._load_optional_cues(job_dir, manifest, "reference_cues")

        ja_by_index = {cue.index: cue for cue in ja_cues}
        literal_by_index = {cue.index: cue for cue in literal_cues}
        adapted_by_index = {cue.index: cue for cue in adapted_cues}
        reference_by_index = {cue.index: cue for cue in reference_cues}
        indexes = sorted(set(ja_by_index) | set(literal_by_index) | set(adapted_by_index) | set(reference_by_index))

        rows: list[dict[str, str | float | int]] = []
        for cue_index in indexes:
            anchor = (
                ja_by_index.get(cue_index)
                or literal_by_index.get(cue_index)
                or adapted_by_index.get(cue_index)
                or reference_by_index[cue_index]
            )
            rows.append(
                {
                    "cue_index": cue_index,
                    "start": anchor.start,
                    "end": anchor.end,
                    "japanese": ja_by_index.get(cue_index).text if cue_index in ja_by_index else "",
                    "literal_english": literal_by_index.get(cue_index).text if cue_index in literal_by_index else "",
                    "adapted_english": adapted_by_index.get(cue_index).text if cue_index in adapted_by_index else "",
                    "reference": reference_by_index.get(cue_index).text if cue_index in reference_by_index else "",
                    "has_japanese": cue_index in ja_by_index,
                    "has_literal_english": cue_index in literal_by_index,
                    "has_adapted_english": cue_index in adapted_by_index,
                    "has_reference": cue_index in reference_by_index,
                }
            )
        return rows

    def update_subtitle_line(
        self,
        job_id: str,
        *,
        cue_index: int,
        japanese_text: str | None = None,
        literal_english_text: str | None = None,
        adapted_english_text: str | None = None,
        reference_text: str | None = None,
    ) -> JobManifest:
        job_dir, manifest = self.store.find_job(job_id)
        updates = (
            ("ja_cues", "ja_srt", japanese_text, "Japanese", True),
            ("literal_cues", "literal_srt", literal_english_text, "Direct English translation", True),
            ("adapted_cues", "adapted_srt", adapted_english_text, "Context-applied English", True),
            ("reference_cues", "reference_srt", reference_text, "Reference subtitle", False),
        )
        updated_any = False
        for cues_artifact, srt_artifact, text, label, should_export in updates:
            if text is None:
                continue
            normalized = text.strip()
            if not normalized:
                raise QueueError(f"{label} text cannot be empty.")
            cues_path = job_dir / manifest.artifacts[cues_artifact]
            if not cues_path.exists():
                raise QueueError(f"{label} subtitle lines are not ready yet for this job.")
            cues = self._load_cues_cached(cues_path)
            match = next((cue for cue in cues if cue.index == cue_index), None)
            if match is None:
                raise QueueError(f"Could not find subtitle line {cue_index} in {label}.")
            match.text = normalized
            self._save_cues_and_cache(cues_path, cues)
            local_srt_path = job_dir / manifest.artifacts[srt_artifact]
            write_srt(local_srt_path, cues)
            if should_export:
                export_dir = self._output_dir_for_manifest(manifest)
                self._export_text_artifact(local_srt_path, export_dir / manifest.artifacts[srt_artifact])
            updated_any = True

        if not updated_any:
            raise QueueError("No subtitle text changes were provided.")
        self._save_manifest(job_dir, manifest)
        return manifest

    def save_job_notes(
        self,
        job_id: str,
        *,
        batch_label: str | None,
        overall_context: str | None,
        scene_contexts: list[SceneContextBlock],
        include_adapted_english: bool | None = None,
        prefer_fast_translation: bool | None = None,
    ) -> JobManifest:
        job_dir, manifest = self.store.find_job(job_id)
        manifest.series = batch_label or None
        manifest.job_context = overall_context or None
        manifest.scene_contexts = list(scene_contexts)
        if include_adapted_english is not None:
            manifest.include_adapted_english = include_adapted_english
        if prefer_fast_translation is not None:
            manifest.prefer_fast_translation = prefer_fast_translation
        self._save_manifest(job_dir, manifest)
        return manifest

    def rebuild_english(
        self,
        job_id: str,
        *,
        batch_label: str | None,
        overall_context: str | None,
        scene_contexts: list[SceneContextBlock],
        include_adapted_english: bool | None = None,
        prefer_fast_translation: bool | None = None,
    ) -> JobManifest:
        job_dir, manifest = self.store.find_job(job_id)
        if not self._translation_source_path(job_dir, manifest).exists():
            raise QueueError("Source subtitle lines are not ready yet for this job.")

        self._require_profile(manifest.profile)
        manifest.series = batch_label or None
        manifest.job_context = overall_context or None
        manifest.scene_contexts = list(scene_contexts)
        if include_adapted_english is not None:
            manifest.include_adapted_english = include_adapted_english
        if prefer_fast_translation is not None:
            manifest.prefer_fast_translation = prefer_fast_translation
        self._save_manifest(job_dir, manifest)
        self._rebuild_english_transactional(job_dir, manifest)
        return manifest

    def rebuild_english_from_saved_notes(self, job_id: str) -> JobManifest:
        with self.store.acquire_worker_lock():
            job_dir, manifest = self.store.find_job(job_id)
            if not self._translation_source_path(job_dir, manifest).exists():
                raise QueueError("Source subtitle lines are not ready yet for this job.")
            self._require_profile(manifest.profile)
            self._rebuild_english_transactional(job_dir, manifest)
            return manifest

    def rebuild_english_range(
        self,
        job_id: str,
        *,
        batch_label: str | None,
        overall_context: str | None,
        scene_contexts: list[SceneContextBlock],
        start_seconds: float,
        end_seconds: float,
        include_adapted_english: bool | None = None,
        prefer_fast_translation: bool | None = None,
    ) -> JobManifest:
        job_dir, manifest = self.store.find_job(job_id)
        if end_seconds <= start_seconds:
            raise QueueError("The end time must be after the start time.")
        if not self._translation_source_path(job_dir, manifest).exists():
            raise QueueError("Source subtitle lines are not ready yet for this job.")
        manifest.series = batch_label or None
        manifest.job_context = overall_context or None
        manifest.scene_contexts = list(scene_contexts)
        if include_adapted_english is not None:
            manifest.include_adapted_english = include_adapted_english
        if prefer_fast_translation is not None:
            manifest.prefer_fast_translation = prefer_fast_translation
        self._save_manifest(job_dir, manifest)
        with self.store.acquire_worker_lock():
            self._rebuild_english_range_transactional(job_dir, manifest, start_seconds, end_seconds)
        return manifest

    def rebuild_english_range_from_saved_notes(
        self,
        job_id: str,
        *,
        start_timecode: str,
        end_timecode: str,
    ) -> JobManifest:
        with self.store.acquire_worker_lock():
            job_dir, manifest = self.store.find_job(job_id)
            if not self._translation_source_path(job_dir, manifest).exists():
                raise QueueError("Source subtitle lines are not ready yet for this job.")
            self._rebuild_english_range_transactional(
                job_dir,
                manifest,
                parse_timecode(start_timecode),
                parse_timecode(end_timecode),
            )
            return manifest

    def run_coherence_pass(
        self,
        job_id: str,
        *,
        batch_label: str | None,
        overall_context: str | None,
        scene_contexts: list[SceneContextBlock],
    ) -> JobManifest:
        job_dir, manifest = self.store.find_job(job_id)
        manifest.series = batch_label or None
        manifest.job_context = overall_context or None
        manifest.scene_contexts = list(scene_contexts)
        self._save_manifest(job_dir, manifest)
        with self.store.acquire_worker_lock():
            self._run_coherence_pass_transactional(job_dir, manifest)
        return manifest

    def run_coherence_pass_from_saved_notes(self, job_id: str) -> JobManifest:
        with self.store.acquire_worker_lock():
            job_dir, manifest = self.store.find_job(job_id)
            self._run_coherence_pass_transactional(job_dir, manifest)
            return manifest

    def coherence_review(self, job_id: str) -> list[dict[str, str | int | float]]:
        job_dir, _manifest = self.store.find_job(job_id)
        data = read_json(job_dir / COHERENCE_REVIEW_FILENAME, default=[]) or []
        return list(data) if isinstance(data, list) else []

    def attach_existing_subtitle(self, job_id: str, *, role: str, subtitle_path: Path) -> JobManifest:
        if role not in {"ja", "direct", "easy", "reference"}:
            raise QueueError(f"Unsupported subtitle role: {role}")
        if subtitle_path.suffix.lower() != ".srt":
            raise QueueError("Only .srt files can be attached right now.")
        if not subtitle_path.exists():
            raise QueueError(f"Subtitle file not found: {subtitle_path}")
        job_dir, manifest = self.store.find_job(job_id)
        self._seed_imported_tracks(job_dir, manifest, {role: subtitle_path.resolve()})
        manifest.imported_tracks[role] = str(subtitle_path.resolve())
        if role == "ja":
            manifest.translation_source_role = TRANSLATION_SOURCE_JA
            self._mark_imported_source_checkpoints(manifest)
        elif role == "direct" and not (job_dir / manifest.artifacts["ja_cues"]).exists():
            manifest.translation_source_role = TRANSLATION_SOURCE_DIRECT_EN
            self._mark_imported_source_checkpoints(manifest)
        self._append_event(
            manifest,
            "info",
            f"Attached {self._role_display_name(role)} subtitles from {subtitle_path.name}.",
        )
        self._save_manifest(job_dir, manifest)
        return manifest

    def run_until_empty(self) -> None:
        with self.store.acquire_worker_lock():
            while True:
                if self.store.pause_requested():
                    return
                claimed = self.store.claim_next_job()
                if not claimed:
                    return
                job_dir, manifest = claimed
                try:
                    self._run_job(job_dir, manifest)
                except QueueError:
                    continue

    def resume(self, job_id: str) -> JobManifest:
        job_dir, manifest = self.store.resume_job(job_id)
        self._append_event(manifest, "info", "The job was re-queued to continue from where it left off.")
        self._save_manifest(job_dir, manifest)
        return manifest

    def stop_job(self, job_id: str) -> JobManifest:
        job_dir, manifest = self.store.find_job(job_id)
        if manifest.status in {JOB_STATUS_COMPLETED, JOB_STATUS_FAILED}:
            raise QueueError("This job has already finished.")
        if manifest.status == JOB_STATUS_PAUSED:
            return manifest
        if manifest.status == JOB_STATUS_QUEUED:
            self._append_event(manifest, "info", "Job stopped before processing started.")
            paused_dir, paused_manifest = self.store.mark_paused(job_dir, manifest)
            self._write_resume_state(paused_dir, paused_manifest)
            return paused_manifest
        self.store.set_job_stop(job_dir, True)
        self._append_event(manifest, "info", "Job will stop after the next safe step.")
        self._save_manifest(job_dir, manifest)
        return manifest

    def open_review(self, job_id: str | None = None) -> list[Path]:
        job_dir, manifest = self._resolve_target_job(job_id)
        self._sync_existing_outputs_to_export(job_dir, manifest)
        outputs = self._review_output_paths(job_dir, manifest)
        if not outputs or not all(path.exists() for path in outputs):
            raise QueueError("Selected job does not have subtitle outputs yet.")
        self.subtitle_edit.open_files(outputs)
        return outputs

    def open_output_folder(self, job_id: str | None = None) -> Path:
        job_dir, manifest = self._resolve_target_job(job_id)
        self._sync_existing_outputs_to_export(job_dir, manifest)
        output_dir = self._ensure_output_dir_for_manifest(manifest)
        target = output_dir if output_dir.exists() else Path(manifest.source_path).parent
        subprocess.Popen(["explorer", str(target)])
        return target

    def subtitle_file_paths(self, job_id: str) -> dict[str, Path]:
        job_dir, manifest = self.store.find_job(job_id)
        self._sync_existing_outputs_to_export(job_dir, manifest)
        output_dir = self._ensure_output_dir_for_manifest(manifest)
        paths = {
            "ja": output_dir / manifest.artifacts["ja_srt"],
            "direct": output_dir / manifest.artifacts["literal_srt"],
            "easy": output_dir / manifest.artifacts["adapted_srt"],
            "review": output_dir / manifest.artifacts["review"],
        }
        for artifact_key, output_key in (("literal_srt", "direct-partial"), ("adapted_srt", "easy-partial")):
            _local_partial, export_partial = self._partial_srt_paths(job_dir, manifest, artifact_key)
            paths[output_key] = export_partial
        return paths

    def open_subtitle_file(self, job_id: str, kind: str) -> Path:
        paths = self.subtitle_file_paths(job_id)
        target = paths.get(kind)
        if target is None or not target.exists():
            raise QueueError(f"The selected {kind} subtitle file is not ready yet.")
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return target

    def health_check(self) -> dict[str, object]:
        checks: list[dict[str, str]] = []

        def add(status: str, name: str, detail: str) -> None:
            checks.append({"status": status, "name": name, "detail": detail})

        queue_root = self.config.queue_root_path
        add("ok" if queue_root.exists() else "error", "Queue folder", str(queue_root))

        for name, value, version_arg in (
            ("FFmpeg", self.config.tools.ffmpeg or "ffmpeg", "-version"),
            ("FFprobe", self.config.tools.ffprobe or "ffprobe", "-version"),
            ("Ollama", self.config.tools.ollama or "ollama", "--version"),
        ):
            try:
                subprocess.run(
                    [value, version_arg],
                    check=True,
                    capture_output=True,
                    text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception as exc:
                add("error", name, f"Not ready: {exc}")
            else:
                add("ok", name, f"Ready from {value}")

        subtitle_edit_path = self.config.tools.subtitle_edit
        if subtitle_edit_path and Path(subtitle_edit_path).exists():
            add("ok", "Subtitle Edit", subtitle_edit_path)
        else:
            add("warning", "Subtitle Edit", "Path is missing or not set.")

        models: list[str] = []
        try:
            self.ollama.ensure_available()
            models = self.ollama.list_models()
        except Exception as exc:
            add("error", "Ollama API", f"Could not connect: {exc}")
        else:
            add("ok", "Ollama API", f"Connected. {len(models)} model(s) found.")

        for name, model in (
            ("Direct English translation model", self.config.models.literal_translation),
            ("Context-applied English model", self.config.models.adapted_translation),
        ):
            if not models:
                add("warning", name, f"Could not verify {model} because Ollama was unavailable.")
            elif model in models:
                add("ok", name, model)
            else:
                add("warning", name, f"{model} is not installed in Ollama yet.")

        asr_model = self.config.models.asr
        add(
            "ok" if asr_model else "warning",
            "Japanese model",
            asr_model or "No Japanese model configured.",
        )

        snapshot = capture_snapshot()
        add("ok", "Free RAM", f"{snapshot.free_ram_mb} MB free")
        add("ok", "Free VRAM", f"{snapshot.gpu_free_mb or 0} MB free")

        worst = "ok"
        if any(item["status"] == "error" for item in checks):
            worst = "error"
        elif any(item["status"] == "warning" for item in checks):
            worst = "warning"
        summary = {
            "ok": "Everything looks ready for a subtitle run.",
            "warning": "The setup mostly works, but a few things need attention.",
            "error": "The setup is not fully ready yet.",
        }[worst]
        return {"status": worst, "summary": summary, "checks": checks}

    def _run_job(self, job_dir: Path, manifest: JobManifest) -> None:
        profile = self._require_profile(manifest.profile)
        try:
            ensure_safe_to_start_job(
                self._job_start_min_free_ram(manifest, profile),
                profile.max_rss_mb,
            )
            self._stage_extract(job_dir, manifest)
            self._stage_transcribe(job_dir, manifest)
            self._stage_translate_literal(job_dir, manifest)
            self._stage_translate_adapted(job_dir, manifest)
            self._stage_finalize(job_dir, manifest)
            completed_dir, completed_manifest = self.store.mark_completed(job_dir, manifest)
            self._write_resume_state(completed_dir, completed_manifest)
        except PauseRequested:
            return
        except Exception as exc:
            self._handle_stage_failure(job_dir, manifest, exc)

    def _job_start_min_free_ram(self, manifest: JobManifest, profile) -> int:
        if self._is_translation_resume(manifest):
            return profile.min_free_ram_translation_resume_mb
        if manifest.checkpoint(STAGE_TRANSCRIBE).status == "completed":
            return profile.min_free_ram_translation_mb
        return profile.min_free_ram_mb

    def _is_translation_resume(self, manifest: JobManifest) -> bool:
        if manifest.checkpoint(STAGE_TRANSCRIBE).status != "completed":
            return False
        for stage_name in (STAGE_LITERAL, STAGE_ADAPTED):
            checkpoint = manifest.checkpoint(stage_name)
            if checkpoint.attempts > 0:
                return True
            if int(checkpoint.details.get("completed_groups", 0) or 0) > 0:
                return True
        return False

    def _translation_stage_min_free_ram(self, manifest: JobManifest, profile, stage_name: str) -> int:
        checkpoint = manifest.checkpoint(stage_name)
        if checkpoint.attempts > 0 or int(checkpoint.details.get("completed_groups", 0) or 0) > 0:
            return profile.min_free_ram_translation_resume_mb
        return profile.min_free_ram_translation_mb

    def _effective_group_size(self, manifest: JobManifest, *, adapted: bool) -> int:
        profile = self.config.profile(manifest.profile)
        base = profile.adapted_group_size if adapted else profile.translation_group_size
        if not manifest.prefer_fast_translation:
            return base
        boosted = base + (2 if adapted else 4)
        return max(boosted, base)

    def _require_profile(self, profile_name: str):
        try:
            return self.config.profile(profile_name)
        except ValueError as exc:
            raise QueueError(str(exc)) from exc

    def _should_pause(self, job_dir: Path, manifest: JobManifest) -> None:
        if self.store.pause_requested() or self.store.job_stop_requested(job_dir):
            if self.store.job_stop_requested(job_dir):
                self._append_event(manifest, "info", "Job stopped after the current safe step.")
            paused_dir, paused_manifest = self.store.mark_paused(job_dir, manifest)
            self._write_resume_state(paused_dir, paused_manifest)
            raise PauseRequested()

    def _update_metrics(self, manifest: JobManifest) -> None:
        snapshot = capture_snapshot()
        manifest.metrics.peak_rss_mb = max(manifest.metrics.peak_rss_mb, snapshot.process_rss_mb)
        manifest.metrics.peak_gpu_used_mb = max(manifest.metrics.peak_gpu_used_mb, snapshot.gpu_used_mb)
        manifest.metrics.last_seen_ram_available_mb = snapshot.free_ram_mb
        manifest.metrics.last_seen_gpu_free_mb = snapshot.gpu_free_mb

    def _set_stage_progress(
        self,
        manifest: JobManifest,
        *,
        stage: str,
        current: float,
        total: float,
        unit: str,
        message: str | None = None,
        done_seconds: float | None = None,
        total_seconds: float | None = None,
    ) -> None:
        existing = manifest.current_progress
        started_at = (
            existing.started_at
            if existing is not None and existing.stage == stage
            else now_iso()
        )
        percent = 0.0
        if total > 0:
            percent = max(0.0, min((current / total) * 100.0, 100.0))

        eta_seconds: float | None = None
        elapsed = elapsed_seconds_since(started_at)
        rate_complete = current
        rate_total = total
        if (
            done_seconds is not None
            and total_seconds is not None
            and total_seconds > 0
            and done_seconds > 0
        ):
            rate_complete = done_seconds
            rate_total = total_seconds
        if elapsed is not None and elapsed > 0 and rate_complete > 0 and rate_total > rate_complete:
            rate = rate_complete / elapsed
            if rate > 0:
                eta_seconds = max((rate_total - rate_complete) / rate, 0.0)

        manifest.current_progress = StageProgress(
            stage=stage,
            current=current,
            total=total,
            unit=unit,
            percent=percent,
            started_at=started_at,
            updated_at=now_iso(),
            eta_seconds=eta_seconds,
            done_seconds=done_seconds,
            total_seconds=total_seconds,
            message=message,
        )

    def _clear_stage_progress(self, manifest: JobManifest) -> None:
        manifest.current_progress = None
        self._progress_save_cache.pop(manifest.job_id, None)

    def _save_progress(self, job_dir: Path, manifest: JobManifest, *, force: bool = False) -> None:
        progress = manifest.current_progress
        if progress is None:
            self._save_manifest(job_dir, manifest)
            return
        now_monotonic = time.monotonic()
        cached = self._progress_save_cache.get(manifest.job_id)
        should_save = force
        if cached is None:
            should_save = True
        else:
            last_percent, last_time = cached
            if abs(progress.percent - last_percent) >= 1.0 or now_monotonic - last_time >= 1.0:
                should_save = True
        if not should_save:
            return
        self._progress_save_cache[manifest.job_id] = (progress.percent, now_monotonic)
        self._save_manifest(job_dir, manifest)

    def _current_stage_percent(self, manifest: JobManifest) -> float:
        progress = manifest.current_progress
        if progress is not None:
            return progress.percent
        if manifest.status == JOB_STATUS_COMPLETED:
            return 100.0
        checkpoint = manifest.checkpoint(manifest.current_stage)
        if checkpoint.status == "completed":
            return 100.0
        return 0.0

    def _overall_progress_percent(self, manifest: JobManifest) -> float:
        progress = manifest.current_progress
        if progress is not None and manifest.status == JOB_STATUS_COMPLETED:
            return progress.percent
        if manifest.status == JOB_STATUS_COMPLETED:
            return 100.0
        stages = self._active_stages(manifest)
        completed = sum(
            1 for stage in stages if manifest.checkpoint(stage).status == "completed"
        )
        current_fraction = 0.0
        if manifest.current_stage in stages and manifest.checkpoint(manifest.current_stage).status != "completed":
            current_fraction = self._current_stage_percent(manifest) / 100.0
        return min(((completed + current_fraction) / len(stages)) * 100.0, 100.0)

    def _stage_display_text(self, manifest: JobManifest) -> str:
        progress = manifest.current_progress
        display_stage = progress.stage if progress is not None else manifest.current_stage
        base = STAGE_DISPLAY_LABELS.get(display_stage, display_stage)
        model_name = self._current_model_name(manifest)
        if progress is None:
            return f"{base} [{model_name}]" if model_name else base
        percent_text = f"{progress.percent:.0f}%"
        eta_text = (
            f" | {format_duration_compact(progress.eta_seconds)} left"
            if progress.eta_seconds is not None and progress.percent < 100.0
            else ""
        )
        message_text = f" | {progress.message}" if progress.message else ""
        model_text = f" | {model_name}" if model_name else ""
        return f"{base} ({percent_text}{eta_text}{message_text}{model_text})"

    def _current_model_name(self, manifest: JobManifest) -> str:
        if manifest.current_stage == STAGE_TRANSCRIBE:
            if self._asr_engine() == ASR_ENGINE_FASTER_WHISPER:
                return f"faster-whisper:{self._faster_whisper_profile_name()}"
            return self._asr_model_id_for_engine()
        if manifest.current_stage == STAGE_LITERAL:
            return self.config.models.literal_translation
        if manifest.current_stage == STAGE_ADAPTED:
            return self.config.models.adapted_translation
        return ""

    def _append_event(
        self,
        manifest: JobManifest,
        level: str,
        message: str,
        *,
        stage: str | None = None,
    ) -> None:
        manifest.events.append(JobEvent(level=level, message=message, stage=stage))
        if len(manifest.events) > 100:
            manifest.events = manifest.events[-100:]

    def _drain_ollama_events(self, manifest: JobManifest, *, stage: str | None = None) -> None:
        if not hasattr(self.ollama, "pop_recent_events"):
            return
        try:
            events = self.ollama.pop_recent_events()
        except Exception:
            return
        for item in events:
            self._append_event(
                manifest,
                str(item.get("level", "info")),
                str(item.get("message", "")),
                stage=stage,
            )

    def _save_manifest(self, job_dir: Path, manifest: JobManifest) -> None:
        self._update_metrics(manifest)
        self.store.save_manifest(job_dir, manifest)
        self._write_resume_state(job_dir, manifest)

    def _write_resume_state(self, job_dir: Path, manifest: JobManifest) -> None:
        try:
            output_dir = self._ensure_output_dir_for_manifest(manifest)
            path = output_dir / f"{Path(manifest.source_name).stem}.resume.json"
            atomic_write_json(
                path,
                {
                    "job_id": manifest.job_id,
                    "source_path": manifest.source_path,
                    "source_name": manifest.source_name,
                    "queue_job_dir": str(job_dir),
                    "status": manifest.status,
                    "current_stage": manifest.current_stage,
                    "updated_at": manifest.updated_at,
                    "current_progress": asdict(manifest.current_progress) if manifest.current_progress is not None else None,
                    "checkpoints": {
                        name: {
                            "status": checkpoint.status,
                            "attempts": checkpoint.attempts,
                            "updated_at": checkpoint.updated_at,
                            "details": checkpoint.details,
                        }
                        for name, checkpoint in manifest.checkpoints.items()
                    },
                    "chunk_plan": [
                        {
                            "index": chunk.index,
                            "start": chunk.start,
                            "end": chunk.end,
                            "path": chunk.path,
                        }
                        for chunk in manifest.chunk_plan
                    ],
                    "artifacts": manifest.artifacts,
                    "export_dir": manifest.export_dir,
                },
            )
        except OSError as exc:
            manifest.checkpoint(manifest.current_stage).details["resume_state_error"] = f"{type(exc).__name__}: {exc}"

    def _cue_signature(self, path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def _clone_cues(self, cues: list[Cue]) -> list[Cue]:
        return [Cue(index=cue.index, start=cue.start, end=cue.end, text=cue.text) for cue in cues]

    def _store_cues_cache(self, path: Path, cues: list[Cue]) -> None:
        try:
            signature = self._cue_signature(path)
        except FileNotFoundError:
            self._cue_cache.pop(path.resolve(), None)
            return
        self._cue_cache[path.resolve()] = (signature, self._clone_cues(cues))

    def _load_cues_cached(self, path: Path) -> list[Cue]:
        resolved = path.resolve()
        cached = self._cue_cache.get(resolved)
        if cached is not None:
            try:
                signature = self._cue_signature(path)
            except FileNotFoundError:
                self._cue_cache.pop(resolved, None)
            else:
                cached_signature, cached_cues = cached
                if cached_signature == signature:
                    return self._clone_cues(cached_cues)
                self._cue_cache.pop(resolved, None)
        cues = load_cues(path)
        self._store_cues_cache(path, cues)
        return self._clone_cues(cues)

    def _save_cues_and_cache(self, path: Path, cues: list[Cue]) -> None:
        save_cues(path, cues)
        self._store_cues_cache(path, cues)

    def _active_stages(self, manifest: JobManifest) -> list[str]:
        stages = [
            STAGE_EXTRACT,
            STAGE_TRANSCRIBE,
            STAGE_LITERAL,
        ]
        if manifest.include_adapted_english:
            stages.append(STAGE_ADAPTED)
        stages.append(STAGE_FINALIZE)
        return stages

    def _on_extract_progress(
        self,
        job_dir: Path,
        manifest: JobManifest,
        info: dict[str, float | int],
    ) -> None:
        covered_seconds = float(info.get("covered_seconds", 0.0))
        total_seconds = max(float(info.get("total_seconds", 0.0)), 0.0)
        current_chunk = int(info.get("current_chunk", 0))
        total_chunks = max(int(info.get("total_chunks", 0)), 1)
        self._set_stage_progress(
            manifest,
            stage=STAGE_EXTRACT,
            current=covered_seconds,
            total=total_seconds or 1.0,
            unit="seconds",
            message=f"Audio chunk {current_chunk} of {total_chunks}",
            done_seconds=covered_seconds,
            total_seconds=total_seconds or None,
        )
        self._save_progress(job_dir, manifest)

    def _stage_extract(self, job_dir: Path, manifest: JobManifest) -> None:
        checkpoint = manifest.checkpoint(STAGE_EXTRACT)
        if checkpoint.status == "completed":
            return
        if self._translation_source_path(job_dir, manifest).exists():
            manifest.current_stage = STAGE_EXTRACT
            checkpoint.attempts += 1
            checkpoint.status = "completed"
            checkpoint.details = {
                "mode": "imported-subtitles",
                "source_role": manifest.translation_source_role,
            }
            self._clear_stage_progress(manifest)
            self._save_manifest(job_dir, manifest)
            return
        if self._asr_engine() == ASR_ENGINE_FASTER_WHISPER:
            manifest.current_stage = STAGE_EXTRACT
            checkpoint.attempts += 1
            duration = self.ffmpeg.probe_duration(Path(manifest.source_path))
            checkpoint.status = "completed"
            checkpoint.details = {
                "mode": "single-audio-pass",
                "engine": ASR_ENGINE_FASTER_WHISPER,
                "total_seconds": duration,
            }
            self._clear_stage_progress(manifest)
            self._save_manifest(job_dir, manifest)
            return
        manifest.current_stage = STAGE_EXTRACT
        checkpoint.attempts += 1
        chunks_dir = job_dir / "chunks"
        profile = self.config.profile(manifest.profile)
        self._set_stage_progress(
            manifest,
            stage=STAGE_EXTRACT,
            current=0.0,
            total=1.0,
            unit="seconds",
            message="Checking the video length",
        )
        self._save_progress(job_dir, manifest, force=True)
        chunk_seconds = profile.chunk_seconds
        if self._asr_engine() == ASR_ENGINE_REAZON_K2:
            chunk_seconds = min(chunk_seconds, REAZON_K2_CHUNK_SECONDS)
        elif self._asr_engine() == ASR_ENGINE_QWEN3:
            chunk_seconds = min(chunk_seconds, QWEN3_ASR_CHUNK_SECONDS)
        manifest.chunk_plan = self.ffmpeg.create_chunk_plan(
            source_path=Path(manifest.source_path),
            chunks_dir=chunks_dir,
            chunk_seconds=chunk_seconds,
            overlap_seconds=profile.chunk_overlap_seconds,
            progress_callback=lambda info: self._on_extract_progress(job_dir, manifest, info),
        )
        checkpoint.status = "completed"
        total_seconds = max((chunk.end for chunk in manifest.chunk_plan), default=0.0)
        checkpoint.details = {
            "chunk_count": len(manifest.chunk_plan),
            "total_seconds": total_seconds,
            "mode": "lazy-chunk-extraction",
            "engine": self._asr_engine(),
            "chunk_seconds": chunk_seconds,
        }
        self._clear_stage_progress(manifest)
        self._save_manifest(job_dir, manifest)

    def _on_transcribe_extract_progress(
        self,
        job_dir: Path,
        manifest: JobManifest,
        *,
        chunk_index: int,
        total_chunks: int,
        chunk_start: float,
        chunk_end: float,
        local_seconds: float,
        total_seconds: float,
    ) -> None:
        covered_seconds = min(chunk_start + local_seconds, chunk_end)
        self._set_stage_progress(
            manifest,
            stage=STAGE_TRANSCRIBE,
            current=covered_seconds,
            total=total_seconds or 1.0,
            unit="seconds",
            message=f"Getting audio for chunk {chunk_index} of {total_chunks}",
            done_seconds=covered_seconds,
            total_seconds=total_seconds or None,
        )
        self._save_progress(job_dir, manifest)

    def _asr_engine(self) -> str:
        engine = getattr(self.config.models, "asr_engine", ASR_ENGINE_FASTER_WHISPER)
        normalized = str(engine or ASR_ENGINE_FASTER_WHISPER).strip().lower()
        if normalized in {"faster_whisper", "faster whisper", "whisper"}:
            return ASR_ENGINE_FASTER_WHISPER
        if normalized in {"kotoba", "transformers", "kotoba-whisper"}:
            return ASR_ENGINE_KOTOBA
        if normalized in {"qwen3-asr", "qwen3 asr", "qwen-asr", "qwen asr", "qwen"}:
            return ASR_ENGINE_QWEN3
        if normalized in {"reazonspeech-k2", "reazonspeech k2", "reazon-k2", "reazon", "reazonspeech-k2-experimental"}:
            return ASR_ENGINE_REAZON_K2
        return ASR_ENGINE_FASTER_WHISPER

    def _asr_model_id_for_engine(self) -> str:
        configured = str(getattr(self.config.models, "asr", "") or "").strip()
        if self._asr_engine() == ASR_ENGINE_REAZON_K2 and "reazonspeech" not in configured.lower():
            return REAZON_K2_MODEL_ID
        if self._asr_engine() == ASR_ENGINE_QWEN3 and "qwen/qwen3-asr" not in configured.lower():
            return QWEN3_ASR_1_7B_MODEL_ID
        return configured

    def _faster_whisper_profile_name(self) -> str:
        profile = getattr(self.config.models, "faster_whisper_profile", "auto")
        return str(profile or "auto").strip() or "auto"

    def _on_fast_asr_audio_progress(
        self,
        job_dir: Path,
        manifest: JobManifest,
        seconds: float,
        total_seconds: float,
    ) -> None:
        done = min(seconds, total_seconds) if total_seconds > 0 else seconds
        self._set_stage_progress(
            manifest,
            stage=STAGE_TRANSCRIBE,
            current=done,
            total=total_seconds or 1.0,
            unit="seconds",
            message="Preparing one clean audio track",
            done_seconds=done if total_seconds > 0 else None,
            total_seconds=total_seconds or None,
        )
        self._save_progress(job_dir, manifest)

    def _run_faster_whisper_transcription(self, job_dir: Path, manifest: JobManifest) -> list[Cue]:
        checkpoint = manifest.checkpoint(STAGE_TRANSCRIBE)
        source_path = Path(manifest.source_path)
        audio_path = job_dir / manifest.artifacts.get("audio", "source.wav")
        total_seconds = float(manifest.checkpoint(STAGE_EXTRACT).details.get("total_seconds") or 0.0)
        if total_seconds <= 0:
            total_seconds = self.ffmpeg.probe_duration(source_path)

        if not audio_path.exists():
            self._set_stage_progress(
                manifest,
                stage=STAGE_TRANSCRIBE,
                current=0.0,
                total=total_seconds or 1.0,
                unit="seconds",
                message="Preparing one clean audio track",
                done_seconds=0.0,
                total_seconds=total_seconds or None,
            )
            self._save_progress(job_dir, manifest, force=True)
            self.ffmpeg.extract_audio(
                source_path=source_path,
                audio_path=audio_path,
                progress_callback=lambda seconds: self._on_fast_asr_audio_progress(
                    job_dir,
                    manifest,
                    seconds,
                    total_seconds,
                ),
            )

        snapshot = capture_snapshot()
        candidates, selection_note = ordered_profile_candidates(
            snapshot,
            self._faster_whisper_profile_name(),
            "downgrade",
        )
        last_error: Exception | None = None
        for profile in candidates:
            backend = FasterWhisperBackend(
                profile,
                cache_dir=self.config.cache_paths.hf_hub_cache or None,
            )
            try:
                self._set_stage_progress(
                    manifest,
                    stage=STAGE_TRANSCRIBE,
                    current=0.0,
                    total=total_seconds or 1.0,
                    unit="seconds",
                    message=f"Listening with {profile.name}",
                    done_seconds=0.0,
                    total_seconds=total_seconds or None,
                )
                self._save_progress(job_dir, manifest, force=True)
                cues, metadata = backend.transcribe(audio_path, language="ja")
                checkpoint.details.update(
                    {
                        "mode": "single-audio-pass",
                        "engine": ASR_ENGINE_FASTER_WHISPER,
                        "requested_profile": self._faster_whisper_profile_name(),
                        "chosen_profile": profile.name,
                        "selection_note": selection_note,
                        "model_id": profile.model_id,
                        "device": profile.device,
                        "compute_type": profile.compute_type,
                        "batch_size": profile.batch_size,
                        "beam_size": profile.beam_size,
                        "detected_language": metadata.get("detected_language"),
                        "language_probability": metadata.get("language_probability"),
                        "duration_seconds": metadata.get("duration") or total_seconds,
                    }
                )
                self._set_stage_progress(
                    manifest,
                    stage=STAGE_TRANSCRIBE,
                    current=total_seconds or 1.0,
                    total=total_seconds or 1.0,
                    unit="seconds",
                    message=f"Finished with {profile.name}",
                    done_seconds=total_seconds or None,
                    total_seconds=total_seconds or None,
                )
                self._save_progress(job_dir, manifest)
                return cues
            except Exception as exc:
                last_error = exc
                if not _is_memory_failure(exc):
                    raise
            finally:
                backend.close()
        raise RuntimeError("All faster-whisper profiles failed.") from last_error

    def _stage_transcribe(self, job_dir: Path, manifest: JobManifest) -> None:
        checkpoint = manifest.checkpoint(STAGE_TRANSCRIBE)
        if checkpoint.status == "completed":
            return
        if manifest.source_kind == SOURCE_KIND_SUBTITLE and self._translation_source_path(job_dir, manifest).exists():
            manifest.current_stage = STAGE_TRANSCRIBE
            checkpoint.attempts += 1
            checkpoint.status = "completed"
            checkpoint.details = {
                "mode": "imported-subtitles",
                "source_role": manifest.translation_source_role,
            }
            self._append_event(
                manifest,
                "info",
                "Skipped Japanese listening because subtitle text was already imported.",
                stage=STAGE_TRANSCRIBE,
            )
            self._clear_stage_progress(manifest)
            self._save_manifest(job_dir, manifest)
            return
        self._should_pause(job_dir, manifest)

        manifest.current_stage = STAGE_TRANSCRIBE
        checkpoint.attempts += 1
        if self._asr_engine() == ASR_ENGINE_FASTER_WHISPER:
            cues = self._run_faster_whisper_transcription(job_dir, manifest)
            normalized = normalize_japanese_cues(cues)
            self._persist_partial_japanese_outputs(job_dir, manifest, normalized)
            self._detect_tail_gap(manifest, normalized)
            checkpoint.status = "completed"
            self._clear_stage_progress(manifest)
            self._save_manifest(job_dir, manifest)
            return
        profile = self.config.profile(manifest.profile)
        engine = self._asr_engine()
        if engine == ASR_ENGINE_REAZON_K2:
            device = "cpu"
            ensure_safe_to_start_job(profile.min_free_ram_mb, profile.max_rss_mb)
        else:
            device = choose_device(profile.min_free_vram_mb)
        if device == "cuda":
            ensure_safe_to_start_gpu_phase(profile.min_free_ram_mb, profile.min_free_vram_mb, profile.max_rss_mb)
        elif engine != ASR_ENGINE_REAZON_K2:
            ensure_safe_to_start_job(profile.min_free_ram_mb, profile.max_rss_mb)

        transcript_dir = job_dir / "chunk-transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        model_id = self._asr_model_id_for_engine()
        if engine == ASR_ENGINE_REAZON_K2:
            asr = ReazonSpeechK2ASRClient(
                model_id,
                cache_dir=self.config.cache_paths.hf_hub_cache or None,
            )
            checkpoint.details.update(
                {
                    "engine": ASR_ENGINE_REAZON_K2,
                    "model_id": model_id,
                    "mode": "chunked-reazonspeech-k2",
                    "benchmark_note": "Reported CER: JSUT 6.45, Common Voice v8 Japanese 7.85, TEDxJP-10K 9.09.",
                }
            )
        elif engine == ASR_ENGINE_QWEN3:
            asr = Qwen3ASRClient(
                model_id,
                cache_dir=self.config.cache_paths.hf_hub_cache or None,
            )
            checkpoint.details.update(
                {
                    "engine": ASR_ENGINE_QWEN3,
                    "model_id": model_id,
                    "mode": "chunked-qwen3-asr-forced-aligner",
                    "aligner_model_id": Qwen3ASRClient.ALIGNER_MODEL_ID,
                    "speaker_separation": "not-enabled",
                }
            )
        else:
            asr = TransformersASRClient(
                model_id,
                cache_dir=self.config.cache_paths.hf_hub_cache or None,
            )
            checkpoint.details.update(
                {
                    "engine": ASR_ENGINE_KOTOBA,
                    "model_id": model_id,
                    "mode": "chunked-transformers",
                }
            )
        all_chunk_cues: list[tuple[float, list[Cue]]] = []
        batch_size = profile.asr_batch_size
        try:
            total_chunks = max(len(manifest.chunk_plan), 1)
            total_seconds = max((chunk.end for chunk in manifest.chunk_plan), default=1.0)
            self._set_stage_progress(
                manifest,
                stage=STAGE_TRANSCRIBE,
                current=0.0,
                total=total_seconds,
                unit="seconds",
                message=f"Listening to chunk 0 of {total_chunks}",
                done_seconds=0.0,
                total_seconds=total_seconds,
            )
            self._save_progress(job_dir, manifest, force=True)
            for chunk in manifest.chunk_plan:
                self._should_pause(job_dir, manifest)
                transcript_path = transcript_dir / f"chunk_{chunk.index:04d}.json"
                if transcript_path.exists():
                    local_cues = self._load_cues_cached(transcript_path)
                else:
                    chunk_path = Path(chunk.path)
                    if not chunk_path.exists():
                        self.ffmpeg.extract_chunk(
                            source_path=Path(manifest.source_path),
                            chunk_path=chunk_path,
                            start=chunk.start,
                            duration=chunk.end - chunk.start,
                            progress_callback=lambda local_seconds, *, current_chunk=chunk.index, total_chunk_count=total_chunks, start_seconds=chunk.start, end_seconds=chunk.end, source_seconds=total_seconds: self._on_transcribe_extract_progress(
                                job_dir,
                                manifest,
                                chunk_index=current_chunk,
                                total_chunks=total_chunk_count,
                                chunk_start=start_seconds,
                                chunk_end=end_seconds,
                                local_seconds=local_seconds,
                                total_seconds=source_seconds,
                            ),
                        )
                    try:
                        local_cues = asr.transcribe_chunk(chunk_path, batch_size=batch_size, device=device)
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower() and batch_size > 1:
                            batch_size = 1
                            local_cues = asr.transcribe_chunk(chunk_path, batch_size=batch_size, device=device)
                        else:
                            raise
                    self._save_cues_and_cache(transcript_path, local_cues)
                    chunk_path.unlink(missing_ok=True)
                all_chunk_cues.append((chunk.start, local_cues))
                partial_merged = combine_chunk_cues(all_chunk_cues)
                partial_normalized = normalize_japanese_cues(partial_merged)
                self._persist_partial_japanese_outputs(job_dir, manifest, partial_normalized)
                checkpoint.details["completed_chunks"] = chunk.index
                checkpoint.details["completed_seconds"] = chunk.end
                self._set_stage_progress(
                    manifest,
                    stage=STAGE_TRANSCRIBE,
                    current=chunk.end,
                    total=total_seconds,
                    unit="seconds",
                    message=f"Listening to chunk {chunk.index} of {total_chunks}",
                    done_seconds=chunk.end,
                    total_seconds=total_seconds,
                )
                self._save_progress(job_dir, manifest)
        finally:
            asr.close()

        merged = combine_chunk_cues(all_chunk_cues)
        normalized = normalize_japanese_cues(merged)
        self._persist_partial_japanese_outputs(job_dir, manifest, normalized)
        self._detect_tail_gap(manifest, normalized)
        checkpoint.status = "completed"
        self._clear_stage_progress(manifest)
        self._save_manifest(job_dir, manifest)

    def _persist_partial_japanese_outputs(
        self,
        job_dir: Path,
        manifest: JobManifest,
        cues: list[Cue],
    ) -> None:
        ja_cues_path = job_dir / manifest.artifacts["ja_cues"]
        ja_srt_path = job_dir / manifest.artifacts["ja_srt"]
        self._save_cues_and_cache(ja_cues_path, cues)
        write_srt(ja_srt_path, cues)
        checkpoint = manifest.checkpoint(STAGE_TRANSCRIBE)
        try:
            export_dir = self._ensure_output_dir_for_manifest(manifest)
            self._export_text_artifact(ja_srt_path, export_dir / manifest.artifacts["ja_srt"])
            checkpoint.details.pop("partial_export_error", None)
        except OSError as exc:
            checkpoint.details["partial_export_error"] = f"{type(exc).__name__}: {exc}"

    def _run_translation_prompt(self, model_name: str, prompt: str, expected_count: int, adapted: bool) -> list[str]:
        try:
            payload = self.ollama.generate_json(
                model=model_name,
                prompt=prompt,
                temperature=0.1 if not adapted else 0.3,
            )
            return validate_translation_payload(payload, expected_count=expected_count)
        except ValueError:
            retry_payload = self.ollama.generate_json(
                model=model_name,
                prompt=strict_retry_prompt(prompt),
                temperature=0.0 if not adapted else 0.2,
            )
            return validate_translation_payload(retry_payload, expected_count=expected_count)

    def _build_translation_prompt(
        self,
        *,
        manifest: JobManifest,
        group: list[Cue],
        group_start_index: int,
        source_cues: list[Cue],
        context_source_cues: list[Cue],
        literal_cues: list[Cue],
        glossary: list[dict[str, str]],
        metadata: str,
        reference_cues: list[Cue],
        adapted: bool,
    ) -> str:
        context_notes = build_context_notes(
            group=group,
            global_context=manifest.job_context,
            scene_contexts=manifest.scene_contexts,
            reference_cues=reference_cues,
            surrounding_cues=self._surrounding_context_cues(group, context_source_cues),
        )
        if adapted:
            literal_group = literal_cues[group_start_index : group_start_index + len(group)]
            prev_context, next_context = self._previous_next_context_cues(group, context_source_cues, count=2)
            return build_adapted_prompt(
                group=group,
                literal_group=literal_group,
                prev_context=prev_context,
                next_context=next_context,
                glossary=glossary,
                metadata=metadata,
                context_notes=context_notes,
                source_language=self._translation_source_language(manifest),
            )
        if manifest.translation_source_role == TRANSLATION_SOURCE_DIRECT_EN:
            return build_direct_english_rewrite_prompt(
                group=group,
                glossary=glossary,
                metadata=metadata,
                context_notes=context_notes,
            )
        return build_literal_prompt_with_context(
            group=group,
            glossary=glossary,
            metadata=metadata,
            context_notes=context_notes,
        )

    def _translate_group_with_backoff(
        self,
        *,
        manifest: JobManifest,
        stage_name: str,
        model_name: str,
        group: list[Cue],
        group_start_index: int,
        source_cues: list[Cue],
        context_source_cues: list[Cue],
        literal_cues: list[Cue],
        glossary: list[dict[str, str]],
        metadata: str,
        reference_cues: list[Cue],
        adapted: bool,
    ) -> tuple[list[str], str | None]:
        prompt = self._build_translation_prompt(
            manifest=manifest,
            group=group,
            group_start_index=group_start_index,
            source_cues=source_cues,
            context_source_cues=context_source_cues,
            literal_cues=literal_cues,
            glossary=glossary,
            metadata=metadata,
            reference_cues=reference_cues,
            adapted=adapted,
        )
        try:
            result = self._run_translation_prompt(
                model_name,
                prompt,
                len(group),
                adapted=adapted,
            )
            self._drain_ollama_events(manifest, stage=stage_name)
            return (
                result,
                None,
            )
        except Exception as exc:
            self._drain_ollama_events(manifest, stage=stage_name)
            if len(group) <= 1:
                raise
            midpoint = max(len(group) // 2, 1)
            left_texts, _left_note = self._translate_group_with_backoff(
                manifest=manifest,
                stage_name=stage_name,
                model_name=model_name,
                group=group[:midpoint],
                group_start_index=group_start_index,
                source_cues=source_cues,
                context_source_cues=context_source_cues,
                literal_cues=literal_cues,
                glossary=glossary,
                metadata=metadata,
                reference_cues=reference_cues,
                adapted=adapted,
            )
            right_texts, _right_note = self._translate_group_with_backoff(
                manifest=manifest,
                stage_name=stage_name,
                model_name=model_name,
                group=group[midpoint:],
                group_start_index=group_start_index + midpoint,
                source_cues=source_cues,
                context_source_cues=context_source_cues,
                literal_cues=literal_cues,
                glossary=glossary,
                metadata=metadata,
                reference_cues=reference_cues,
                adapted=adapted,
            )
            return (
                left_texts + right_texts,
                f"Recovered by splitting a {len(group)}-cue batch after {type(exc).__name__}: {exc}",
            )

    def _translate_stage(
        self,
        *,
        job_dir: Path,
        manifest: JobManifest,
        stage_name: str,
        model_name: str,
        output_artifact: str,
        output_srt_artifact: str,
        group_size: int,
        adapted: bool,
        source_cues_path: Path | None = None,
        literal_input_path: Path | None = None,
        output_cues_path: Path | None = None,
        output_srt_path: Path | None = None,
        partial_path: Path | None = None,
        prompt_context_cues_path: Path | None = None,
    ) -> None:
        checkpoint = manifest.checkpoint(stage_name)
        if checkpoint.status == "completed":
            return
        self._should_pause(job_dir, manifest)

        manifest.current_stage = stage_name
        checkpoint.attempts += 1
        profile = self.config.profile(manifest.profile)
        ensure_safe_to_start_job(
            self._translation_stage_min_free_ram(manifest, profile, stage_name),
            profile.max_rss_mb,
        )
        if checkpoint.attempts == 1:
            speed_label = "faster batches" if manifest.prefer_fast_translation else "safe batches"
            self._append_event(
                manifest,
                "info",
                f"Started {STAGE_DISPLAY_LABELS.get(stage_name, stage_name).lower()} with {model_name} using {speed_label}.",
                stage=stage_name,
            )

        source_path = source_cues_path or self._translation_source_path(job_dir, manifest)
        literal_source_path = literal_input_path or (job_dir / manifest.artifacts["literal_cues"])
        final_cues_path = output_cues_path or (job_dir / manifest.artifacts[output_artifact])
        final_srt_path = output_srt_path or (job_dir / manifest.artifacts[output_srt_artifact])
        partial_output_path = partial_path or (job_dir / f"{output_artifact}.partial.json")

        source_cues = self._load_cues_cached(source_path)
        context_source_cues = self._load_cues_cached(prompt_context_cues_path) if prompt_context_cues_path else source_cues
        literal_cues = self._load_cues_cached(literal_source_path) if adapted else []
        reference_cues = self._reference_cues_for_job(job_dir, manifest)
        glossary = load_glossary(manifest.glossary_path)
        metadata = metadata_from_manifest(manifest.source_name, manifest.series)
        groups = cue_groups(source_cues, group_size)
        partial_rows = read_json(partial_output_path, default=[]) or []
        translated_cues = [Cue.from_dict(item) for item in partial_rows]
        start_group = int(checkpoint.details.get("completed_groups", 0))
        total_groups = max(len(groups), 1)

        self._set_stage_progress(
            manifest,
            stage=stage_name,
            current=float(start_group),
            total=float(total_groups),
            unit="groups",
            message=f"Subtitle group {start_group} of {total_groups}" if start_group else f"Subtitle group 0 of {total_groups}",
        )
        self._save_progress(job_dir, manifest, force=True)

        for group_index in range(start_group, len(groups)):
            self._should_pause(job_dir, manifest)
            group = groups[group_index]
            try:
                group_start_index = group_index * group_size
                self._set_stage_progress(
                    manifest,
                    stage=stage_name,
                    current=float(group_index) + 0.1,
                    total=float(total_groups),
                    unit="groups",
                    message=f"Working on subtitle group {group_index + 1} of {total_groups}",
                )
                self._save_progress(job_dir, manifest, force=True)
                translations, recovery_note = self._translate_group_with_backoff(
                    manifest=manifest,
                    stage_name=stage_name,
                    model_name=model_name,
                    group=group,
                    group_start_index=group_start_index,
                    source_cues=source_cues,
                    context_source_cues=context_source_cues,
                    literal_cues=literal_cues,
                    glossary=glossary,
                    metadata=metadata,
                    reference_cues=reference_cues,
                    adapted=adapted,
                )
            except Exception as exc:
                if adapted:
                    fallback = literal_cues[
                        group_index * group_size : group_index * group_size + len(group)
                    ]
                    translated_cues.extend(fallback)
                    manifest.review_flags.append(
                        ReviewFlag(
                            stage=stage_name,
                            group_index=group_index,
                            reason="translation-fallback",
                            detail=str(exc),
                        )
                    )
                else:
                    raise
            else:
                if recovery_note:
                    manifest.review_flags.append(
                        ReviewFlag(
                            stage=stage_name,
                            group_index=group_index,
                            reason="translation-batch-retry",
                            detail=recovery_note,
                        )
                    )
                    self._append_event(
                        manifest,
                        "warning",
                        recovery_note,
                        stage=stage_name,
                    )
                new_cues = apply_translations(group, translations)
                previous_text = translated_cues[-1].text if translated_cues else None
                translated_cues.extend(new_cues)
                for cue_index, reason, detail in subtitle_quality_flags(
                    group,
                    new_cues,
                    glossary,
                    previous_text=previous_text,
                ):
                    manifest.review_flags.append(
                        ReviewFlag(
                            stage=stage_name,
                            group_index=group_index,
                            reason=reason,
                            detail=detail,
                        )
                    )

            checkpoint.details["completed_groups"] = group_index + 1
            atomic_write_json(
                partial_output_path,
                [
                    {"index": cue.index, "start": cue.start, "end": cue.end, "text": cue.text}
                    for cue in translated_cues
                ],
            )
            self._write_partial_translation_srt(
                job_dir,
                manifest,
                output_srt_artifact,
                translated_cues,
            )
            self._set_stage_progress(
                manifest,
                stage=stage_name,
                current=float(group_index + 1),
                total=float(total_groups),
                unit="groups",
                message=f"Subtitle group {group_index + 1} of {total_groups}",
            )
            self._save_progress(job_dir, manifest)

        self._save_cues_and_cache(final_cues_path, translated_cues)
        write_srt(final_srt_path, translated_cues)
        self._export_text_artifact(
            final_srt_path,
            self._ensure_output_dir_for_manifest(manifest) / manifest.artifacts[output_srt_artifact],
        )
        self._remove_partial_translation_srt(job_dir, manifest, output_srt_artifact)
        checkpoint.status = "completed"
        self._append_event(
            manifest,
            "info",
            f"Finished {STAGE_DISPLAY_LABELS.get(stage_name, stage_name).lower()}.",
            stage=stage_name,
        )
        self._clear_stage_progress(manifest)
        self._save_manifest(job_dir, manifest)
        partial_output_path.unlink(missing_ok=True)

    def _stage_translate_literal(
        self,
        job_dir: Path,
        manifest: JobManifest,
        *,
        source_cues_path: Path | None = None,
        output_cues_path: Path | None = None,
        output_srt_path: Path | None = None,
        partial_path: Path | None = None,
        prompt_context_cues_path: Path | None = None,
    ) -> None:
        self._translate_stage(
            job_dir=job_dir,
            manifest=manifest,
            stage_name=STAGE_LITERAL,
            model_name=self.config.models.literal_translation,
            output_artifact="literal_cues",
            output_srt_artifact="literal_srt",
            group_size=self._effective_group_size(manifest, adapted=False),
            adapted=False,
            source_cues_path=source_cues_path,
            output_cues_path=output_cues_path,
            output_srt_path=output_srt_path,
            partial_path=partial_path,
            prompt_context_cues_path=prompt_context_cues_path,
        )

    def _mark_adapted_stage_skipped(self, manifest: JobManifest) -> None:
        checkpoint = manifest.checkpoint(STAGE_ADAPTED)
        checkpoint.status = "completed"
        checkpoint.details = {"mode": "skipped", "reason": "easy-english-disabled"}

    def _stage_translate_adapted(
        self,
        job_dir: Path,
        manifest: JobManifest,
        *,
        source_cues_path: Path | None = None,
        literal_input_path: Path | None = None,
        output_cues_path: Path | None = None,
        output_srt_path: Path | None = None,
        partial_path: Path | None = None,
        prompt_context_cues_path: Path | None = None,
    ) -> None:
        if not manifest.include_adapted_english:
            self._mark_adapted_stage_skipped(manifest)
            self._append_event(
                manifest,
                "info",
                "Skipped context-applied English because this job is set to direct English translation only.",
                stage=STAGE_ADAPTED,
            )
            self._clear_stage_progress(manifest)
            self._save_manifest(job_dir, manifest)
            return
        self._translate_stage(
            job_dir=job_dir,
            manifest=manifest,
            stage_name=STAGE_ADAPTED,
            model_name=self.config.models.adapted_translation,
            output_artifact="adapted_cues",
            output_srt_artifact="adapted_srt",
            group_size=self._effective_group_size(manifest, adapted=True),
            adapted=True,
            source_cues_path=source_cues_path,
            literal_input_path=literal_input_path,
            output_cues_path=output_cues_path,
            output_srt_path=output_srt_path,
            partial_path=partial_path,
            prompt_context_cues_path=prompt_context_cues_path,
        )

    def _stage_finalize(self, job_dir: Path, manifest: JobManifest) -> None:
        checkpoint = manifest.checkpoint(STAGE_FINALIZE)
        if checkpoint.status == "completed":
            return
        manifest.current_stage = STAGE_FINALIZE
        checkpoint.attempts += 1
        self._set_stage_progress(
            manifest,
            stage=STAGE_FINALIZE,
            current=0.0,
            total=1.0,
            unit="steps",
            message="Writing the final subtitle files",
        )
        self._save_progress(job_dir, manifest, force=True)
        review_path = job_dir / manifest.artifacts["review"]
        write_review_flags(
            review_path,
            [
                {
                    "stage": flag.stage,
                    "group_index": flag.group_index,
                    "reason": flag.reason,
                    "detail": flag.detail,
                    "created_at": flag.created_at,
                }
                for flag in manifest.review_flags
            ],
        )
        output_dir = self._export_final_outputs(job_dir, manifest)
        checkpoint.status = "completed"
        checkpoint.details = {"export_dir": str(output_dir)}
        self._clear_stage_progress(manifest)
        self._save_manifest(job_dir, manifest)

    def _handle_stage_failure(self, job_dir: Path, manifest: JobManifest, exc: Exception) -> None:
        checkpoint = manifest.checkpoint(manifest.current_stage)
        detail = f"{type(exc).__name__}: {exc}"
        self._append_event(
            manifest,
            "error",
            detail,
            stage=manifest.current_stage,
        )
        if checkpoint.attempts >= 2:
            failed_dir, failed_manifest = self.store.mark_failed(job_dir, manifest, detail)
            self._write_resume_state(failed_dir, failed_manifest)
            raise QueueError(detail)
        self._clear_stage_progress(manifest)
        queued_dir, queued_manifest = self.store.requeue_working(job_dir, manifest, detail)
        self._write_resume_state(queued_dir, queued_manifest)
        raise QueueError(detail)

    def _source_key(self, path: Path) -> str:
        return str(path.resolve()).casefold()

    def _find_job_by_source_path(self, source_path: Path) -> tuple[Path, JobManifest] | None:
        target = self._source_key(source_path)
        for job_dir, manifest, _state in self.store.list_jobs():
            if self._source_key(Path(manifest.source_path)) == target:
                return job_dir, manifest
        return None

    def _translation_source_artifact(self, manifest: JobManifest) -> str:
        if manifest.translation_source_role == TRANSLATION_SOURCE_DIRECT_EN:
            return "literal_cues"
        return "ja_cues"

    def _translation_source_language(self, manifest: JobManifest) -> str:
        if manifest.translation_source_role == TRANSLATION_SOURCE_DIRECT_EN:
            return "English"
        return "Japanese"

    def _translation_source_path(self, job_dir: Path, manifest: JobManifest) -> Path:
        return job_dir / manifest.artifacts[self._translation_source_artifact(manifest)]

    def _reference_cues_for_job(self, job_dir: Path, manifest: JobManifest) -> list[Cue]:
        return self._load_optional_cues(job_dir, manifest, "reference_cues")

    def _seed_imported_tracks(
        self,
        job_dir: Path,
        manifest: JobManifest,
        tracks: dict[str, Path],
    ) -> None:
        for role, path in tracks.items():
            cues = parse_srt(path)
            if role == "ja":
                cues = normalize_japanese_cues(cues)
            artifact_key, srt_key = self._import_role_artifacts(role)
            local_cues_path = job_dir / manifest.artifacts[artifact_key]
            local_srt_path = job_dir / manifest.artifacts[srt_key]
            self._save_cues_and_cache(local_cues_path, cues)
            write_srt(local_srt_path, cues)
            if role in {"ja", "direct", "easy"}:
                export_dir = self._ensure_output_dir_for_manifest(manifest)
                self._export_text_artifact(local_srt_path, export_dir / manifest.artifacts[srt_key])

    def _import_role_artifacts(self, role: str) -> tuple[str, str]:
        mapping = {
            "ja": ("ja_cues", "ja_srt"),
            "direct": ("literal_cues", "literal_srt"),
            "easy": ("adapted_cues", "adapted_srt"),
            "reference": ("reference_cues", "reference_srt"),
        }
        return mapping[role]

    def _mark_imported_source_checkpoints(self, manifest: JobManifest) -> None:
        detail = {
            "mode": "imported-subtitles",
            "source_role": manifest.translation_source_role,
        }
        extract_checkpoint = manifest.checkpoint(STAGE_EXTRACT)
        extract_checkpoint.status = "completed"
        extract_checkpoint.details = dict(detail)
        transcribe_checkpoint = manifest.checkpoint(STAGE_TRANSCRIBE)
        transcribe_checkpoint.status = "completed"
        transcribe_checkpoint.details = dict(detail)

    def _resolve_target_job(self, job_id: str | None) -> tuple[Path, JobManifest]:
        if job_id:
            return self.store.find_job(job_id)
        completed = [
            (job_dir, manifest)
            for job_dir, manifest, state in self.store.list_jobs()
            if state == "done" and manifest.status == JOB_STATUS_COMPLETED
        ]
        if not completed:
            raise QueueError("No completed job found.")
        return completed[-1]

    def _output_dir_for_manifest(self, manifest: JobManifest) -> Path:
        if manifest.export_dir:
            return Path(manifest.export_dir)
        return subtitle_output_dir(Path(manifest.source_path))

    def _ensure_output_dir_for_manifest(self, manifest: JobManifest) -> Path:
        output_dir = self._output_dir_for_manifest(manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _sync_existing_outputs_to_export(self, job_dir: Path, manifest: JobManifest) -> Path:
        output_dir = self._ensure_output_dir_for_manifest(manifest)
        for artifact in ("ja_srt", "literal_srt", "adapted_srt", "review"):
            local_path = job_dir / manifest.artifacts[artifact]
            if local_path.exists():
                self._export_text_artifact(local_path, output_dir / manifest.artifacts[artifact])
        for artifact in ("literal_srt", "adapted_srt"):
            local_partial_srt, export_partial_srt = self._partial_srt_paths(job_dir, manifest, artifact)
            if local_partial_srt.exists():
                self._export_text_artifact(local_partial_srt, export_partial_srt)
        return output_dir

    def _review_output_paths(self, job_dir: Path, manifest: JobManifest) -> list[Path]:
        export_dir = self._output_dir_for_manifest(manifest)
        exported = [
            export_dir / manifest.artifacts[artifact]
            for artifact in ("ja_srt", "literal_srt", "adapted_srt")
            if (export_dir / manifest.artifacts[artifact]).exists()
        ]
        if exported:
            return exported
        return [
            job_dir / manifest.artifacts[artifact]
            for artifact in ("ja_srt", "literal_srt", "adapted_srt")
            if (job_dir / manifest.artifacts[artifact]).exists()
        ]

    def _export_text_artifact(self, source_path: Path, target_path: Path) -> None:
        atomic_write_text(target_path, source_path.read_text(encoding="utf-8"))

    def _rebuild_english_transactional(self, job_dir: Path, manifest: JobManifest) -> None:
        profile = self._require_profile(manifest.profile)
        ensure_safe_to_start_job(profile.min_free_ram_translation_mb, profile.max_rss_mb)
        self._append_event(
            manifest,
            "info",
            "Started redo English for the whole job.",
            stage=STAGE_LITERAL,
        )

        temp_paths = self._rebuild_temp_paths(job_dir, manifest)
        original_manifest = JobManifest.from_dict(manifest.to_dict())
        working_manifest = JobManifest.from_dict(manifest.to_dict())
        working_manifest.error = None
        working_manifest.review_flags = [
            flag for flag in working_manifest.review_flags if flag.stage not in {STAGE_LITERAL, STAGE_ADAPTED}
        ]
        for stage_name in (STAGE_LITERAL, STAGE_ADAPTED, STAGE_FINALIZE):
            checkpoint = working_manifest.checkpoint(stage_name)
            checkpoint.status = "pending"
            checkpoint.attempts = 0
            checkpoint.details = {}

        self._cleanup_paths(temp_paths.values())
        try:
            self._save_manifest(job_dir, working_manifest)
            self._stage_translate_literal(
                job_dir,
                working_manifest,
                output_cues_path=temp_paths["literal_cues"],
                output_srt_path=temp_paths["literal_srt"],
                partial_path=temp_paths["literal_partial"],
            )
            if working_manifest.include_adapted_english:
                self._stage_translate_adapted(
                    job_dir,
                    working_manifest,
                    literal_input_path=temp_paths["literal_cues"],
                    output_cues_path=temp_paths["adapted_cues"],
                    output_srt_path=temp_paths["adapted_srt"],
                    partial_path=temp_paths["adapted_partial"],
                )
            else:
                self._mark_adapted_stage_skipped(working_manifest)

            working_manifest.current_stage = STAGE_FINALIZE
            finalize_checkpoint = working_manifest.checkpoint(STAGE_FINALIZE)
            finalize_checkpoint.attempts += 1
            self._write_review_file(temp_paths["review"], working_manifest.review_flags)
            self._save_manifest(job_dir, working_manifest)

            self._promote_rebuild_outputs(job_dir, manifest, temp_paths)
            manifest.review_flags = list(working_manifest.review_flags)
            manifest.error = None
            manifest.current_stage = STAGE_FINALIZE
            for stage_name in (STAGE_LITERAL, STAGE_ADAPTED, STAGE_FINALIZE):
                manifest.checkpoints[stage_name] = working_manifest.checkpoint(stage_name)
            output_dir = self._export_final_outputs(job_dir, manifest)
            finalize_checkpoint = manifest.checkpoint(STAGE_FINALIZE)
            finalize_checkpoint.status = "completed"
            finalize_checkpoint.details = {"export_dir": str(output_dir)}
            self._append_event(
                manifest,
                "info",
                "Finished redo English for the whole job.",
                stage=STAGE_FINALIZE,
            )
            self._save_manifest(job_dir, manifest)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            original_manifest.error = detail
            self._append_event(original_manifest, "error", detail, stage=original_manifest.current_stage)
            self._save_manifest(job_dir, original_manifest)
            raise QueueError(detail) from exc
        finally:
            self._cleanup_paths(temp_paths.values())

    def _rebuild_temp_paths(self, job_dir: Path, manifest: JobManifest) -> dict[str, Path]:
        literal_cues = job_dir / manifest.artifacts["literal_cues"]
        adapted_cues = job_dir / manifest.artifacts["adapted_cues"]
        literal_srt = job_dir / manifest.artifacts["literal_srt"]
        adapted_srt = job_dir / manifest.artifacts["adapted_srt"]
        review = job_dir / manifest.artifacts["review"]
        return {
            "literal_cues": literal_cues.with_name(f"{literal_cues.stem}.rebuild{literal_cues.suffix}"),
            "adapted_cues": adapted_cues.with_name(f"{adapted_cues.stem}.rebuild{adapted_cues.suffix}"),
            "literal_srt": literal_srt.with_name(f"{literal_srt.stem}.rebuild{literal_srt.suffix}"),
            "adapted_srt": adapted_srt.with_name(f"{adapted_srt.stem}.rebuild{adapted_srt.suffix}"),
            "review": review.with_name(f"{review.stem}.rebuild{review.suffix}"),
            "literal_partial": job_dir / "literal_cues.rebuild.partial.json",
            "adapted_partial": job_dir / "adapted_cues.rebuild.partial.json",
        }

    def _write_review_file(self, path: Path, review_flags: list[ReviewFlag]) -> None:
        write_review_flags(
            path,
            [
                {
                    "stage": flag.stage,
                    "group_index": flag.group_index,
                    "reason": flag.reason,
                    "detail": flag.detail,
                    "created_at": flag.created_at,
                }
                for flag in review_flags
            ],
        )

    def _promote_rebuild_outputs(self, job_dir: Path, manifest: JobManifest, temp_paths: dict[str, Path]) -> None:
        for artifact_key in ("literal_cues", "literal_srt", "review"):
            self._export_text_artifact(temp_paths[artifact_key], job_dir / manifest.artifacts[artifact_key])
        if manifest.include_adapted_english:
            for artifact_key in ("adapted_cues", "adapted_srt"):
                if temp_paths[artifact_key].exists():
                    self._export_text_artifact(temp_paths[artifact_key], job_dir / manifest.artifacts[artifact_key])
        else:
            self._remove_adapted_outputs(job_dir, manifest)

    def _cleanup_paths(self, paths) -> None:
        for path in paths:
            Path(path).unlink(missing_ok=True)

    def _export_final_outputs(self, job_dir: Path, manifest: JobManifest) -> Path:
        output_dir = self._ensure_output_dir_for_manifest(manifest)
        for artifact in ("ja_srt", "literal_srt", "adapted_srt", "review"):
            source_path = job_dir / manifest.artifacts[artifact]
            if not source_path.exists():
                continue
            target_path = output_dir / manifest.artifacts[artifact]
            self._export_text_artifact(source_path, target_path)
        return output_dir

    def _partial_srt_name_for_artifact(self, manifest: JobManifest, output_srt_artifact: str) -> str:
        filename = manifest.artifacts[output_srt_artifact]
        path = Path(filename)
        return f"{path.stem}.partial{path.suffix}"

    def _partial_srt_paths(
        self,
        job_dir: Path,
        manifest: JobManifest,
        output_srt_artifact: str,
    ) -> tuple[Path, Path]:
        partial_name = self._partial_srt_name_for_artifact(manifest, output_srt_artifact)
        return (
            job_dir / partial_name,
            self._ensure_output_dir_for_manifest(manifest) / partial_name,
        )

    def _write_partial_translation_srt(
        self,
        job_dir: Path,
        manifest: JobManifest,
        output_srt_artifact: str,
        cues: list[Cue],
    ) -> None:
        local_partial_srt, export_partial_srt = self._partial_srt_paths(
            job_dir,
            manifest,
            output_srt_artifact,
        )
        write_srt(local_partial_srt, cues)
        self._export_text_artifact(local_partial_srt, export_partial_srt)

    def _remove_partial_translation_srt(
        self,
        job_dir: Path,
        manifest: JobManifest,
        output_srt_artifact: str,
    ) -> None:
        local_partial_srt, export_partial_srt = self._partial_srt_paths(
            job_dir,
            manifest,
            output_srt_artifact,
        )
        local_partial_srt.unlink(missing_ok=True)
        export_partial_srt.unlink(missing_ok=True)

    def _load_optional_cues(self, job_dir: Path, manifest: JobManifest, artifact_key: str) -> list[Cue]:
        artifact = manifest.artifacts.get(artifact_key)
        if not artifact:
            return []
        path = job_dir / artifact
        if not path.exists():
            return []
        return self._load_cues_cached(path)

    def _clear_translation_outputs(self, job_dir: Path, manifest: JobManifest) -> None:
        for filename in (
            "literal_cues.partial.json",
            "adapted_cues.partial.json",
        ):
            (job_dir / filename).unlink(missing_ok=True)
        self._remove_partial_translation_srt(job_dir, manifest, "literal_srt")
        self._remove_partial_translation_srt(job_dir, manifest, "adapted_srt")

        for artifact in ("literal_cues", "adapted_cues", "literal_srt", "adapted_srt", "review"):
            local_path = job_dir / manifest.artifacts[artifact]
            local_path.unlink(missing_ok=True)
            export_path = self._output_dir_for_manifest(manifest) / manifest.artifacts[artifact]
            export_path.unlink(missing_ok=True)

    def _remove_adapted_outputs(self, job_dir: Path, manifest: JobManifest) -> None:
        for filename in ("adapted_cues.partial.json", "adapted_cues.rebuild.partial.json"):
            (job_dir / filename).unlink(missing_ok=True)
        self._remove_partial_translation_srt(job_dir, manifest, "adapted_srt")
        for artifact in ("adapted_cues", "adapted_srt"):
            local_path = job_dir / manifest.artifacts[artifact]
            local_path.unlink(missing_ok=True)
            self._cue_cache.pop(local_path.resolve(), None)
            export_path = self._output_dir_for_manifest(manifest) / manifest.artifacts[artifact]
            export_path.unlink(missing_ok=True)

    def _role_display_name(self, role: str) -> str:
        return {
            "ja": "Japanese",
            "direct": "Direct English translation",
            "easy": "Context-applied English",
            "reference": "Reference",
        }.get(role, role)

    def _detect_tail_gap(self, manifest: JobManifest, cues: list[Cue], *, threshold_seconds: float = 15.0) -> None:
        if not manifest.chunk_plan or not cues:
            return
        total_seconds = max((chunk.end for chunk in manifest.chunk_plan), default=0.0)
        last_end = max((cue.end for cue in cues), default=0.0)
        gap = max(total_seconds - last_end, 0.0)
        if gap < threshold_seconds:
            return
        detail = (
            f"The Japanese subtitles end around {format_timecode(last_end)}, but the video reaches about "
            f"{format_timecode(total_seconds)}. Check the missing tail section."
        )
        if not any(flag.reason == "tail-gap" for flag in manifest.review_flags):
            manifest.review_flags.append(
                ReviewFlag(
                    stage=STAGE_TRANSCRIBE,
                    group_index=0,
                    reason="tail-gap",
                    detail=detail,
                )
            )
        self._append_event(manifest, "warning", detail, stage=STAGE_TRANSCRIBE)

    def _surrounding_context_cues(
        self,
        group: list[Cue],
        source_cues: list[Cue],
        *,
        before: int = 3,
        after: int = 3,
    ) -> list[Cue]:
        prev_context, next_context = self._previous_next_context_cues(group, source_cues, count=max(before, after))
        return prev_context[-before:] + next_context[:after]

    def _previous_next_context_cues(
        self,
        group: list[Cue],
        source_cues: list[Cue],
        *,
        count: int,
    ) -> tuple[list[Cue], list[Cue]]:
        if not group or not source_cues:
            return [], []
        source_by_index = {cue.index: position for position, cue in enumerate(source_cues)}
        positions = [source_by_index[cue.index] for cue in group if cue.index in source_by_index]
        if not positions:
            return [], []
        start = min(positions)
        end = max(positions) + 1
        return source_cues[max(0, start - count) : start], source_cues[end : end + count]

    def _selected_range_indexes(self, source_cues: list[Cue], start_seconds: float, end_seconds: float) -> set[int]:
        return {
            cue.index
            for cue in source_cues
            if cue.end >= start_seconds and cue.start <= end_seconds
        }

    def _merge_cue_updates(
        self,
        base_cues: list[Cue],
        updates: list[Cue],
        *,
        allowed_indexes: set[int],
    ) -> list[Cue]:
        update_map = {cue.index: cue for cue in updates}
        merged: list[Cue] = []
        for cue in base_cues:
            if cue.index in allowed_indexes and cue.index in update_map:
                replacement = update_map[cue.index]
                merged.append(
                    Cue(
                        index=cue.index,
                        start=cue.start,
                        end=cue.end,
                        text=replacement.text,
                    )
                )
            else:
                merged.append(Cue(index=cue.index, start=cue.start, end=cue.end, text=cue.text))
        return merged

    def _copy_existing_translation_or_raise(
        self,
        job_dir: Path,
        manifest: JobManifest,
        artifact_key: str,
        *,
        label: str,
    ) -> list[Cue]:
        path = job_dir / manifest.artifacts[artifact_key]
        if not path.exists():
            raise QueueError(f"{label} is not ready yet, so only a whole-job English rebuild is possible.")
        return self._load_cues_cached(path)

    def _run_coherence_pass_transactional(self, job_dir: Path, manifest: JobManifest) -> None:
        profile = self._require_profile(manifest.profile)
        ensure_safe_to_start_job(
            self._translation_stage_min_free_ram(manifest, profile, STAGE_ADAPTED),
            profile.max_rss_mb,
        )
        source_path = self._translation_source_path(job_dir, manifest)
        literal_path = job_dir / manifest.artifacts["literal_cues"]
        adapted_path = job_dir / manifest.artifacts["adapted_cues"]
        if not source_path.exists():
            raise QueueError("Source subtitle lines are not ready yet for this job.")
        if not literal_path.exists():
            raise QueueError("Direct English translation is not ready yet for this job.")
        if not adapted_path.exists():
            raise QueueError("Context-applied English is not ready yet for this job.")

        source_by_index = {cue.index: cue for cue in self._load_cues_cached(source_path)}
        literal_by_index = {cue.index: cue for cue in self._load_cues_cached(literal_path)}
        original_adapted = self._load_cues_cached(adapted_path)
        rewritten: list[Cue] = []
        review_rows: list[dict[str, str | int | float]] = []
        groups = cue_groups(original_adapted, self._effective_group_size(manifest, adapted=True))
        metadata = metadata_from_manifest(manifest.source_name, manifest.series)
        self._append_event(manifest, "info", "Started second-pass subtitle coherence review.", stage=STAGE_ADAPTED)
        self._set_stage_progress(
            manifest,
            stage=STAGE_ADAPTED,
            current=0.0,
            total=float(max(len(groups), 1)),
            unit="groups",
            message=f"Second-pass group 0 of {len(groups)}",
        )
        self._save_progress(job_dir, manifest, force=True)
        for group_index, group in enumerate(groups):
            self._should_pause(job_dir, manifest)
            source_group = [source_by_index.get(cue.index, cue) for cue in group]
            literal_group = [literal_by_index.get(cue.index, cue) for cue in group]
            _prev, next_context = self._previous_next_context_cues(group, original_adapted, count=3)
            context_notes = build_context_notes(
                source_group,
                manifest.job_context,
                manifest.scene_contexts,
                reference_cues=self._reference_cues_for_job(job_dir, manifest),
                surrounding_cues=rewritten[-3:] + next_context[:3],
            )
            prompt = build_coherence_pass_prompt(
                group=group,
                source_group=source_group,
                literal_group=literal_group,
                current_group=group,
                previous_final=rewritten[-12:],
                next_context=next_context[:6],
                context_notes=context_notes,
                metadata=metadata,
            )
            translations = self._run_translation_prompt(
                self.config.models.adapted_translation,
                prompt,
                len(group),
                adapted=True,
            )
            new_cues = apply_translations(group, translations)
            for before, after in zip(group, new_cues, strict=True):
                rewritten.append(after)
                if before.text.strip() != after.text.strip():
                    review_rows.append(
                        {
                            "cue_index": before.index,
                            "start": before.start,
                            "end": before.end,
                            "before": before.text,
                            "after": after.text,
                        }
                    )
            self._set_stage_progress(
                manifest,
                stage=STAGE_ADAPTED,
                current=float(group_index + 1),
                total=float(max(len(groups), 1)),
                unit="groups",
                message=f"Second-pass group {group_index + 1} of {len(groups)}",
            )
            self._save_progress(job_dir, manifest)

        self._save_cues_and_cache(adapted_path, rewritten)
        adapted_srt = job_dir / manifest.artifacts["adapted_srt"]
        write_srt(adapted_srt, rewritten)
        self._export_text_artifact(
            adapted_srt,
            self._ensure_output_dir_for_manifest(manifest) / manifest.artifacts["adapted_srt"],
        )
        atomic_write_json(job_dir / COHERENCE_REVIEW_FILENAME, review_rows)
        self._append_event(
            manifest,
            "info",
            f"Finished second-pass coherence review with {len(review_rows)} changed lines.",
            stage=STAGE_ADAPTED,
        )
        self._clear_stage_progress(manifest)
        self._save_manifest(job_dir, manifest)

    def _rebuild_english_range_transactional(
        self,
        job_dir: Path,
        manifest: JobManifest,
        start_seconds: float,
        end_seconds: float,
    ) -> None:
        profile = self._require_profile(manifest.profile)
        ensure_safe_to_start_job(
            self._translation_stage_min_free_ram(manifest, profile, STAGE_LITERAL),
            profile.max_rss_mb,
        )
        source_path = self._translation_source_path(job_dir, manifest)
        source_cues = self._load_cues_cached(source_path)
        selected_indexes = self._selected_range_indexes(source_cues, start_seconds, end_seconds)
        if not selected_indexes:
            raise QueueError("No subtitle lines overlap that selected time range.")

        subset_source = [cue for cue in source_cues if cue.index in selected_indexes]
        temp_dir = job_dir / "range-rebuild"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_source = temp_dir / "source.cues.json"
        self._save_cues_and_cache(temp_source, subset_source)

        temp_paths = {
            "literal_cues": temp_dir / "literal.cues.json",
            "literal_srt": temp_dir / "literal.srt",
            "literal_partial": temp_dir / "literal.partial.json",
            "adapted_cues": temp_dir / "adapted.cues.json",
            "adapted_srt": temp_dir / "adapted.srt",
            "adapted_partial": temp_dir / "adapted.partial.json",
            "review": temp_dir / "review.json",
        }
        self._cleanup_paths(temp_paths.values())
        self._append_event(
            manifest,
            "info",
            f"Redoing English only for {format_timecode(start_seconds)} to {format_timecode(end_seconds)}.",
            stage=STAGE_LITERAL,
        )
        saved_stage = manifest.current_stage
        saved_progress = deepcopy(manifest.current_progress)
        saved_checkpoints = {
            stage: deepcopy(manifest.checkpoint(stage))
            for stage in (STAGE_LITERAL, STAGE_ADAPTED)
        }
        for stage in (STAGE_LITERAL, STAGE_ADAPTED):
            checkpoint = manifest.checkpoint(stage)
            checkpoint.status = "pending"
            checkpoint.attempts = 0
            checkpoint.details = {}
        try:
            self._stage_translate_literal(
                job_dir,
                manifest,
                output_cues_path=temp_paths["literal_cues"],
                output_srt_path=temp_paths["literal_srt"],
                partial_path=temp_paths["literal_partial"],
                source_cues_path=temp_source,
                prompt_context_cues_path=source_path,
            )
            existing_literal = self._copy_existing_translation_or_raise(
                job_dir,
                manifest,
                "literal_cues",
                label="Direct English translation",
            )
            merged_literal = self._merge_cue_updates(
                existing_literal,
                self._load_cues_cached(temp_paths["literal_cues"]),
                allowed_indexes=selected_indexes,
            )
            literal_cues_path = job_dir / manifest.artifacts["literal_cues"]
            literal_srt_path = job_dir / manifest.artifacts["literal_srt"]
            self._save_cues_and_cache(literal_cues_path, merged_literal)
            write_srt(literal_srt_path, merged_literal)
            self._export_text_artifact(
                literal_srt_path,
                self._ensure_output_dir_for_manifest(manifest) / manifest.artifacts["literal_srt"],
            )

            if manifest.include_adapted_english:
                temp_literal_merged = temp_dir / "literal-merged.cues.json"
                subset_literal = [cue for cue in merged_literal if cue.index in selected_indexes]
                self._save_cues_and_cache(temp_literal_merged, subset_literal)
                self._stage_translate_adapted(
                    job_dir,
                    manifest,
                    literal_input_path=temp_literal_merged,
                    output_cues_path=temp_paths["adapted_cues"],
                    output_srt_path=temp_paths["adapted_srt"],
                    partial_path=temp_paths["adapted_partial"],
                    source_cues_path=temp_source,
                    prompt_context_cues_path=source_path,
                )
                existing_adapted = self._copy_existing_translation_or_raise(
                    job_dir,
                    manifest,
                    "adapted_cues",
                    label="Context-applied English",
                )
                merged_adapted = self._merge_cue_updates(
                    existing_adapted,
                    self._load_cues_cached(temp_paths["adapted_cues"]),
                    allowed_indexes=selected_indexes,
                )
                adapted_cues_path = job_dir / manifest.artifacts["adapted_cues"]
                adapted_srt_path = job_dir / manifest.artifacts["adapted_srt"]
                self._save_cues_and_cache(adapted_cues_path, merged_adapted)
                write_srt(adapted_srt_path, merged_adapted)
                self._export_text_artifact(
                    adapted_srt_path,
                    self._ensure_output_dir_for_manifest(manifest) / manifest.artifacts["adapted_srt"],
                )

            self._append_event(
                manifest,
                "info",
                f"Finished redo English for {format_timecode(start_seconds)} to {format_timecode(end_seconds)}.",
                stage=STAGE_FINALIZE,
            )
            self._save_manifest(job_dir, manifest)
        finally:
            manifest.current_stage = saved_stage
            manifest.current_progress = saved_progress
            for stage, checkpoint in saved_checkpoints.items():
                manifest.checkpoints[stage] = checkpoint
            self._save_manifest(job_dir, manifest)
            self._cleanup_paths(temp_paths.values())
            for extra_path in (temp_source, temp_dir / "literal-merged.cues.json"):
                extra_path.unlink(missing_ok=True)
            temp_dir.rmdir() if temp_dir.exists() and not any(temp_dir.iterdir()) else None
