from __future__ import annotations

import subprocess
from pathlib import Path

from .config import AppConfig
from .domain import (
    JOB_STATUS_COMPLETED,
    STAGE_ADAPTED,
    STAGE_EXTRACT,
    STAGE_FINALIZE,
    STAGE_LITERAL,
    STAGE_TRANSCRIBE,
    Cue,
    JobManifest,
    ReviewFlag,
    SceneContextBlock,
    SOURCE_KIND_SUBTITLE,
    SOURCE_KIND_VIDEO,
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
    SubtitleEditClient,
    TransformersASRClient,
    load_cues,
    save_cues,
)
from .pipeline import (
    apply_translations,
    build_adapted_prompt,
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
    validate_translation_payload,
    write_review_flags,
    write_srt,
)
from .queue import QueueError, QueueStore
from .utils import atomic_write_json, atomic_write_text, list_video_sources, read_json, subtitle_output_dir


class PauseRequested(RuntimeError):
    pass


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

    def enqueue(
        self,
        source: Path,
        profile: str,
        glossary: Path | None = None,
        series: str | None = None,
        context: str | None = None,
        scene_contexts: list[SceneContextBlock] | None = None,
    ) -> JobManifest:
        self._require_profile(profile)
        return self.store.enqueue(
            source_path=source,
            profile=profile,
            glossary_path=glossary,
            series=series,
            job_context=context,
            scene_contexts=scene_contexts,
        )

    def enqueue_many(
        self,
        sources: list[Path],
        profile: str,
        glossary: Path | None = None,
        series: str | None = None,
        context: str | None = None,
        scene_contexts: list[SceneContextBlock] | None = None,
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
            raise QueueError("Import needs a Japanese or Direct English subtitle source track.")

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
            )
            job_dir, manifest = self.store.find_job(manifest.job_id)
        else:
            job_dir, manifest = existing
            manifest.profile = profile
            manifest.source_kind = SOURCE_KIND_VIDEO if resolved_video else SOURCE_KIND_SUBTITLE
            manifest.linked_video_path = str(resolved_video) if resolved_video else None
            manifest.translation_source_role = translation_source_role
            manifest.imported_tracks.update({role: str(path) for role, path in tracks.items()})
            if series is not None:
                manifest.series = series
            if context is not None:
                manifest.job_context = context
            if scene_contexts is not None:
                manifest.scene_contexts = list(scene_contexts)

        if series is not None:
            manifest.series = series
        if context is not None:
            manifest.job_context = context
        if scene_contexts is not None:
            manifest.scene_contexts = list(scene_contexts)

        self._seed_imported_tracks(job_dir, manifest, tracks)
        self._mark_imported_source_checkpoints(manifest)
        manifest.status = JOB_STATUS_COMPLETED
        manifest.current_stage = STAGE_FINALIZE
        review_path = job_dir / manifest.artifacts["review"]
        self._write_review_file(review_path, manifest.review_flags)
        self._export_text_artifact(
            review_path,
            self._output_dir_for_manifest(manifest) / manifest.artifacts["review"],
        )
        self._save_manifest(job_dir, manifest)
        if job_dir.parent != self.store.done_dir:
            job_dir, manifest = self.store.mark_completed(job_dir, manifest)
        return manifest

    def status_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for _job_dir, manifest, state in self.store.list_jobs():
            rows.append(
                {
                    "job_id": manifest.job_id,
                    "state_dir": state,
                    "status": manifest.status,
                    "stage": manifest.current_stage,
                    "source": manifest.source_name,
                    "updated_at": manifest.updated_at,
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
            ("literal_cues", "literal_srt", literal_english_text, "Direct English", True),
            ("adapted_cues", "adapted_srt", adapted_english_text, "Easy English", True),
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
            cues = load_cues(cues_path)
            match = next((cue for cue in cues if cue.index == cue_index), None)
            if match is None:
                raise QueueError(f"Could not find subtitle line {cue_index} in {label}.")
            match.text = normalized
            save_cues(cues_path, cues)
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
    ) -> JobManifest:
        job_dir, manifest = self.store.find_job(job_id)
        manifest.series = batch_label or None
        manifest.job_context = overall_context or None
        manifest.scene_contexts = list(scene_contexts)
        self._save_manifest(job_dir, manifest)
        return manifest

    def rebuild_english(
        self,
        job_id: str,
        *,
        batch_label: str | None,
        overall_context: str | None,
        scene_contexts: list[SceneContextBlock],
    ) -> JobManifest:
        job_dir, manifest = self.store.find_job(job_id)
        if not self._translation_source_path(job_dir, manifest).exists():
            raise QueueError("Source subtitle lines are not ready yet for this job.")

        self._require_profile(manifest.profile)
        manifest.series = batch_label or None
        manifest.job_context = overall_context or None
        manifest.scene_contexts = list(scene_contexts)
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
        _job_dir, manifest = self.store.resume_job(job_id)
        return manifest

    def open_review(self, job_id: str | None = None) -> list[Path]:
        job_dir, manifest = self._resolve_target_job(job_id)
        outputs = self._review_output_paths(job_dir, manifest)
        if not outputs or not all(path.exists() for path in outputs):
            raise QueueError("Selected job does not have subtitle outputs yet.")
        self.subtitle_edit.open_files(outputs)
        return outputs

    def open_output_folder(self, job_id: str | None = None) -> Path:
        _job_dir, manifest = self._resolve_target_job(job_id)
        output_dir = self._output_dir_for_manifest(manifest)
        target = output_dir if output_dir.exists() else Path(manifest.source_path).parent
        subprocess.Popen(["explorer", str(target)])
        return target

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
            self.store.mark_completed(job_dir, manifest)
        except PauseRequested:
            return
        except Exception as exc:
            self._handle_stage_failure(job_dir, manifest, exc)

    def _job_start_min_free_ram(self, manifest: JobManifest, profile) -> int:
        if manifest.checkpoint(STAGE_TRANSCRIBE).status == "completed":
            return profile.min_free_ram_translation_mb
        return profile.min_free_ram_mb

    def _require_profile(self, profile_name: str):
        try:
            return self.config.profile(profile_name)
        except ValueError as exc:
            raise QueueError(str(exc)) from exc

    def _should_pause(self, job_dir: Path, manifest: JobManifest) -> None:
        if self.store.pause_requested():
            self.store.mark_paused(job_dir, manifest)
            raise PauseRequested()

    def _update_metrics(self, manifest: JobManifest) -> None:
        snapshot = capture_snapshot()
        manifest.metrics.peak_rss_mb = max(manifest.metrics.peak_rss_mb, snapshot.process_rss_mb)
        manifest.metrics.peak_gpu_used_mb = max(manifest.metrics.peak_gpu_used_mb, snapshot.gpu_used_mb)
        manifest.metrics.last_seen_ram_available_mb = snapshot.free_ram_mb
        manifest.metrics.last_seen_gpu_free_mb = snapshot.gpu_free_mb

    def _save_manifest(self, job_dir: Path, manifest: JobManifest) -> None:
        self._update_metrics(manifest)
        self.store.save_manifest(job_dir, manifest)

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
            self._save_manifest(job_dir, manifest)
            return
        manifest.current_stage = STAGE_EXTRACT
        checkpoint.attempts += 1
        chunks_dir = job_dir / "chunks"
        profile = self.config.profile(manifest.profile)
        manifest.chunk_plan = self.ffmpeg.create_chunk_plan(
            source_path=Path(manifest.source_path),
            chunks_dir=chunks_dir,
            chunk_seconds=profile.chunk_seconds,
            overlap_seconds=profile.chunk_overlap_seconds,
        )
        checkpoint.status = "completed"
        checkpoint.details = {"chunk_count": len(manifest.chunk_plan)}
        self._save_manifest(job_dir, manifest)

    def _stage_transcribe(self, job_dir: Path, manifest: JobManifest) -> None:
        checkpoint = manifest.checkpoint(STAGE_TRANSCRIBE)
        if checkpoint.status == "completed":
            return
        if self._translation_source_path(job_dir, manifest).exists():
            manifest.current_stage = STAGE_TRANSCRIBE
            checkpoint.attempts += 1
            checkpoint.status = "completed"
            checkpoint.details = {
                "mode": "imported-subtitles",
                "source_role": manifest.translation_source_role,
            }
            self._save_manifest(job_dir, manifest)
            return
        self._should_pause(job_dir, manifest)

        manifest.current_stage = STAGE_TRANSCRIBE
        checkpoint.attempts += 1
        profile = self.config.profile(manifest.profile)
        device = choose_device(profile.min_free_vram_mb)
        if device == "cuda":
            ensure_safe_to_start_gpu_phase(profile.min_free_ram_mb, profile.min_free_vram_mb, profile.max_rss_mb)
        else:
            ensure_safe_to_start_job(profile.min_free_ram_mb, profile.max_rss_mb)

        transcript_dir = job_dir / "chunk-transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        asr = TransformersASRClient(
            self.config.models.asr,
            cache_dir=self.config.cache_paths.hf_hub_cache or None,
        )
        all_chunk_cues: list[tuple[float, list[Cue]]] = []
        batch_size = profile.asr_batch_size
        try:
            for chunk in manifest.chunk_plan:
                self._should_pause(job_dir, manifest)
                transcript_path = transcript_dir / f"chunk_{chunk.index:04d}.json"
                if transcript_path.exists():
                    local_cues = load_cues(transcript_path)
                else:
                    try:
                        local_cues = asr.transcribe_chunk(Path(chunk.path), batch_size=batch_size, device=device)
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower() and batch_size > 1:
                            batch_size = 1
                            local_cues = asr.transcribe_chunk(Path(chunk.path), batch_size=batch_size, device=device)
                        else:
                            raise
                    save_cues(transcript_path, local_cues)
                all_chunk_cues.append((chunk.start, local_cues))
                checkpoint.details["completed_chunks"] = chunk.index
                self._save_manifest(job_dir, manifest)
        finally:
            asr.close()

        merged = combine_chunk_cues(all_chunk_cues)
        normalized = normalize_japanese_cues(merged)
        ja_cues_path = job_dir / manifest.artifacts["ja_cues"]
        save_cues(ja_cues_path, normalized)
        write_srt(job_dir / manifest.artifacts["ja_srt"], normalized)
        checkpoint.status = "completed"
        self._save_manifest(job_dir, manifest)

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
    ) -> None:
        checkpoint = manifest.checkpoint(stage_name)
        if checkpoint.status == "completed":
            return
        self._should_pause(job_dir, manifest)

        manifest.current_stage = stage_name
        checkpoint.attempts += 1
        profile = self.config.profile(manifest.profile)
        ensure_safe_to_start_job(profile.min_free_ram_translation_mb, profile.max_rss_mb)

        source_path = source_cues_path or self._translation_source_path(job_dir, manifest)
        literal_source_path = literal_input_path or (job_dir / manifest.artifacts["literal_cues"])
        final_cues_path = output_cues_path or (job_dir / manifest.artifacts[output_artifact])
        final_srt_path = output_srt_path or (job_dir / manifest.artifacts[output_srt_artifact])
        partial_output_path = partial_path or (job_dir / f"{output_artifact}.partial.json")

        source_cues = load_cues(source_path)
        literal_cues = load_cues(literal_source_path) if adapted else []
        reference_cues = self._reference_cues_for_job(job_dir, manifest)
        glossary = load_glossary(manifest.glossary_path)
        metadata = metadata_from_manifest(manifest.source_name, manifest.series)
        groups = cue_groups(source_cues, group_size)
        partial_rows = read_json(partial_output_path, default=[]) or []
        translated_cues = [Cue.from_dict(item) for item in partial_rows]
        start_group = int(checkpoint.details.get("completed_groups", 0))

        for group_index in range(start_group, len(groups)):
            self._should_pause(job_dir, manifest)
            group = groups[group_index]
            if adapted:
                literal_group = literal_cues[group_index * group_size : group_index * group_size + len(group)]
                prev_context = source_cues[max(0, group_index * group_size - 2) : group_index * group_size]
                next_context = source_cues[
                    group_index * group_size + len(group) : group_index * group_size + len(group) + 2
                ]
                prompt = build_adapted_prompt(
                    group=group,
                    literal_group=literal_group,
                    prev_context=prev_context,
                    next_context=next_context,
                    glossary=glossary,
                    metadata=metadata,
                    context_notes=build_context_notes(
                        group=group,
                        global_context=manifest.job_context,
                        scene_contexts=manifest.scene_contexts,
                        reference_cues=reference_cues,
                    ),
                    source_language=self._translation_source_language(manifest),
                )
            else:
                context_notes = build_context_notes(
                    group=group,
                    global_context=manifest.job_context,
                    scene_contexts=manifest.scene_contexts,
                    reference_cues=reference_cues,
                )
                if manifest.translation_source_role == TRANSLATION_SOURCE_DIRECT_EN:
                    prompt = build_direct_english_rewrite_prompt(
                        group=group,
                        glossary=glossary,
                        metadata=metadata,
                        context_notes=context_notes,
                    )
                else:
                    prompt = build_literal_prompt_with_context(
                        group=group,
                        glossary=glossary,
                        metadata=metadata,
                        context_notes=context_notes,
                    )

            try:
                translations = self._run_translation_prompt(model_name, prompt, len(group), adapted=adapted)
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
                translated_cues.extend(apply_translations(group, translations))

            checkpoint.details["completed_groups"] = group_index + 1
            atomic_write_json(
                partial_output_path,
                [
                    {"index": cue.index, "start": cue.start, "end": cue.end, "text": cue.text}
                    for cue in translated_cues
                ],
            )
            self._save_manifest(job_dir, manifest)

        save_cues(final_cues_path, translated_cues)
        write_srt(final_srt_path, translated_cues)
        checkpoint.status = "completed"
        self._save_manifest(job_dir, manifest)
        partial_output_path.unlink(missing_ok=True)

    def _stage_translate_literal(
        self,
        job_dir: Path,
        manifest: JobManifest,
        *,
        output_cues_path: Path | None = None,
        output_srt_path: Path | None = None,
        partial_path: Path | None = None,
    ) -> None:
        self._translate_stage(
            job_dir=job_dir,
            manifest=manifest,
            stage_name=STAGE_LITERAL,
            model_name=self.config.models.literal_translation,
            output_artifact="literal_cues",
            output_srt_artifact="literal_srt",
            group_size=self.config.profile(manifest.profile).translation_group_size,
            adapted=False,
            output_cues_path=output_cues_path,
            output_srt_path=output_srt_path,
            partial_path=partial_path,
        )

    def _stage_translate_adapted(
        self,
        job_dir: Path,
        manifest: JobManifest,
        *,
        literal_input_path: Path | None = None,
        output_cues_path: Path | None = None,
        output_srt_path: Path | None = None,
        partial_path: Path | None = None,
    ) -> None:
        self._translate_stage(
            job_dir=job_dir,
            manifest=manifest,
            stage_name=STAGE_ADAPTED,
            model_name=self.config.models.adapted_translation,
            output_artifact="adapted_cues",
            output_srt_artifact="adapted_srt",
            group_size=self.config.profile(manifest.profile).adapted_group_size,
            adapted=True,
            literal_input_path=literal_input_path,
            output_cues_path=output_cues_path,
            output_srt_path=output_srt_path,
            partial_path=partial_path,
        )

    def _stage_finalize(self, job_dir: Path, manifest: JobManifest) -> None:
        checkpoint = manifest.checkpoint(STAGE_FINALIZE)
        if checkpoint.status == "completed":
            return
        manifest.current_stage = STAGE_FINALIZE
        checkpoint.attempts += 1
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
        self._save_manifest(job_dir, manifest)

    def _handle_stage_failure(self, job_dir: Path, manifest: JobManifest, exc: Exception) -> None:
        checkpoint = manifest.checkpoint(manifest.current_stage)
        detail = f"{type(exc).__name__}: {exc}"
        if checkpoint.attempts >= 2:
            self.store.mark_failed(job_dir, manifest, detail)
            raise QueueError(detail)
        self.store.requeue_working(job_dir, manifest, detail)
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
            save_cues(local_cues_path, cues)
            write_srt(local_srt_path, cues)
            if role in {"ja", "direct", "easy"}:
                export_dir = self._output_dir_for_manifest(manifest)
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
            self._stage_translate_adapted(
                job_dir,
                working_manifest,
                literal_input_path=temp_paths["literal_cues"],
                output_cues_path=temp_paths["adapted_cues"],
                output_srt_path=temp_paths["adapted_srt"],
                partial_path=temp_paths["adapted_partial"],
            )

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
            self._save_manifest(job_dir, manifest)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            original_manifest.error = detail
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
        for artifact_key in ("literal_cues", "adapted_cues", "literal_srt", "adapted_srt", "review"):
            self._export_text_artifact(temp_paths[artifact_key], job_dir / manifest.artifacts[artifact_key])

    def _cleanup_paths(self, paths) -> None:
        for path in paths:
            Path(path).unlink(missing_ok=True)

    def _export_final_outputs(self, job_dir: Path, manifest: JobManifest) -> Path:
        output_dir = self._output_dir_for_manifest(manifest)
        for artifact in ("ja_srt", "literal_srt", "adapted_srt", "review"):
            source_path = job_dir / manifest.artifacts[artifact]
            if not source_path.exists():
                continue
            target_path = output_dir / manifest.artifacts[artifact]
            self._export_text_artifact(source_path, target_path)
        return output_dir

    def _load_optional_cues(self, job_dir: Path, manifest: JobManifest, artifact_key: str) -> list[Cue]:
        path = job_dir / manifest.artifacts[artifact_key]
        if not path.exists():
            return []
        return load_cues(path)

    def _clear_translation_outputs(self, job_dir: Path, manifest: JobManifest) -> None:
        for filename in (
            "literal_cues.partial.json",
            "adapted_cues.partial.json",
        ):
            (job_dir / filename).unlink(missing_ok=True)

        for artifact in ("literal_cues", "adapted_cues", "literal_srt", "adapted_srt", "review"):
            local_path = job_dir / manifest.artifacts[artifact]
            local_path.unlink(missing_ok=True)
            export_path = self._output_dir_for_manifest(manifest) / manifest.artifacts[artifact]
            export_path.unlink(missing_ok=True)
