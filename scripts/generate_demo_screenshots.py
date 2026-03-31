from __future__ import annotations

from pathlib import Path

from PIL import ImageGrab

import local_subtitle_stack.ui as ui_module
from local_subtitle_stack.config import AppConfig
from local_subtitle_stack.domain import JobManifest, SceneContextBlock
from local_subtitle_stack.ui import ImportExistingDialog, SubtitleStackApp


class FakeStore:
    def __init__(self) -> None:
        self.paused = False

    def pause_requested(self) -> bool:
        return self.paused

    def set_pause(self, paused: bool) -> None:
        self.paused = paused


class FakeService:
    def __init__(self) -> None:
        self.config = AppConfig(config_path="config.toml", queue_root="queue")
        self.store = FakeStore()
        self.job_dirs: dict[str, Path] = {}
        self.manifests: dict[str, JobManifest] = {}
        self.preview_by_job: dict[str, list[dict[str, str | float | int | bool]]] = {}
        self._build_demo_jobs()

    def _build_demo_jobs(self) -> None:
        jobs = [
            ("job-one", "sample-video.mp4", "completed", "finalize"),
            ("job-two", "meeting-example.mp4", "queued", "transcribe"),
        ]
        for index, (job_id, filename, status, stage) in enumerate(jobs, start=1):
            manifest = JobManifest(
                job_id=job_id,
                source_path=str(Path(f"D:/Videos/{filename}")),
                source_name=filename,
                profile="conservative",
                status=status,
                current_stage=stage,
                export_dir=str(Path(f"D:/Videos/{filename} subtitles")),
                source_kind="video",
                translation_source_role="ja",
            )
            manifest.artifacts = {
                "job": f"{Path(filename).stem}.job.json",
                "review": f"{Path(filename).stem}.review.json",
                "ja_srt": f"{Path(filename).stem}.ja.srt",
                "literal_srt": f"{Path(filename).stem}.en.literal.srt",
                "adapted_srt": f"{Path(filename).stem}.en.adapted.srt",
                "audio": "source.wav",
                "ja_cues": "ja.cues.json",
                "literal_cues": "literal.cues.json",
                "adapted_cues": "adapted.cues.json",
                "reference_srt": f"{Path(filename).stem}.reference.srt",
                "reference_cues": "reference.cues.json",
            }
            manifest.updated_at = f"2026-04-01T09:0{index}:00+00:00"
            self.manifests[job_id] = manifest
            self.job_dirs[job_id] = Path(f"D:/Queue/{job_id}")

        self.preview_by_job["job-one"] = [
            {
                "cue_index": 1,
                "start": 0.0,
                "end": 2.0,
                "japanese": "今日は少し予定が変わった。",
                "literal_english": "Today's schedule changed a little.",
                "adapted_english": "The schedule changed a bit today.",
                "reference": "This short section is about a small schedule change.",
                "has_japanese": True,
                "has_literal_english": True,
                "has_adapted_english": True,
                "has_reference": True,
            },
            {
                "cue_index": 2,
                "start": 2.2,
                "end": 5.4,
                "japanese": "会議が少し遅れるから、先に温泉へ行こう。",
                "literal_english": "The meeting will be a little late, so let's go to the hot spring first.",
                "adapted_english": "The meeting is running late, so let's head to the hot spring first.",
                "reference": "",
                "has_japanese": True,
                "has_literal_english": True,
                "has_adapted_english": True,
                "has_reference": False,
            },
            {
                "cue_index": 3,
                "start": 5.8,
                "end": 8.5,
                "japanese": "この場面は待ち合わせと移動の話。",
                "literal_english": "This scene is about meeting up and moving to the next place.",
                "adapted_english": "This part is about meeting up and heading somewhere else.",
                "reference": "",
                "has_japanese": True,
                "has_literal_english": True,
                "has_adapted_english": True,
                "has_reference": False,
            },
        ]
        self.preview_by_job["job-two"] = [
            {
                "cue_index": 1,
                "start": 0.0,
                "end": 2.0,
                "japanese": "出発前に荷物を確認しよう。",
                "literal_english": "Let's check the bags before we leave.",
                "adapted_english": "",
                "reference": "",
                "has_japanese": True,
                "has_literal_english": True,
                "has_adapted_english": False,
                "has_reference": False,
            }
        ]

    def status_rows(self) -> list[dict[str, str]]:
        return [
            {
                "job_id": job_id,
                "state_dir": "done" if manifest.status == "completed" else "incoming",
                "status": manifest.status,
                "stage": manifest.current_stage,
                "source": manifest.source_name,
                "updated_at": manifest.updated_at,
            }
            for job_id, manifest in self.manifests.items()
        ]

    def load_job(self, job_id: str) -> tuple[Path, JobManifest]:
        return self.job_dirs[job_id], self.manifests[job_id]

    def preview_rows(self, job_id: str) -> list[dict[str, str | float | int]]:
        return list(self.preview_by_job[job_id])

    def save_job_notes(
        self,
        job_id: str,
        *,
        batch_label: str | None,
        overall_context: str | None,
        scene_contexts: list[SceneContextBlock],
    ) -> JobManifest:
        manifest = self.manifests[job_id]
        manifest.series = batch_label or None
        manifest.job_context = overall_context or None
        manifest.scene_contexts = list(scene_contexts)
        return manifest

    def resume(self, job_id: str) -> JobManifest:
        return self.manifests[job_id]

    def open_review(self, job_id: str) -> list[Path]:
        return []

    def open_output_folder(self, job_id: str) -> Path:
        return Path(self.manifests[job_id].export_dir or "")

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
        row = next(item for item in self.preview_by_job[job_id] if int(item["cue_index"]) == cue_index)
        if japanese_text is not None:
            row["japanese"] = japanese_text
        if literal_english_text is not None:
            row["literal_english"] = literal_english_text
        if adapted_english_text is not None:
            row["adapted_english"] = adapted_english_text
        if reference_text is not None:
            row["reference"] = reference_text
            row["has_reference"] = True
        return self.manifests[job_id]

    def detect_existing_subtitles(self, _video: Path) -> dict[str, Path]:
        return {
            "ja": Path(r"D:\Subtitles\sample-video.ja.srt"),
            "direct": Path(r"D:\Subtitles\sample-video.en.literal.srt"),
            "easy": Path(r"D:\Subtitles\sample-video.en.adapted.srt"),
            "reference": Path(r"D:\Subtitles\sample-video.reference.srt"),
        }


def capture_window(widget, target: Path) -> None:
    widget.update_idletasks()
    widget.update()
    bbox = (
        widget.winfo_rootx(),
        widget.winfo_rooty(),
        widget.winfo_rootx() + widget.winfo_width(),
        widget.winfo_rooty() + widget.winfo_height(),
    )
    ImageGrab.grab(bbox=bbox).save(target)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    images_dir = repo_root / "docs" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    service = FakeService()
    original_build_service = ui_module.build_service
    original_start_snapshot = SubtitleStackApp._start_snapshot_thread
    original_schedule_refresh = SubtitleStackApp._schedule_refresh
    ui_module.build_service = lambda: service
    SubtitleStackApp._start_snapshot_thread = lambda self: None
    SubtitleStackApp._schedule_refresh = lambda self: None

    app = None
    dialog = None
    try:
        app = SubtitleStackApp()
        app.geometry("1540x1120+40+40")
        app.attributes("-topmost", True)
        app.update()
        app.job_tree.selection_set("job-one")
        app._on_job_selected()
        app.preview_tree.selection_set(("cue-1",))
        app.preview_tree.focus("cue-1")
        app._on_preview_lines_selected()
        app.batch_label_var.set("Example Batch")
        app.context_text.insert(
            "1.0",
            "Whole-video note: this example is about a small schedule change, a delayed meeting, and moving to a hot spring.",
        )
        app.note_start_var.set("00:00:00")
        app.note_end_var.set("00:00:05")
        app.range_notes_text.insert(
            "1.0",
            "Short conversation about the meeting being late and going to the hot spring first.",
        )
        app.scene_contexts = [
            SceneContextBlock(
                start_seconds=0.0,
                end_seconds=5.0,
                notes="This section is about a delayed meeting and a plan change.",
            )
        ]
        app._render_scene_blocks()
        app.update()
        capture_window(app, images_dir / "app-overview.png")

        app.preview_tree.selection_set(("cue-1", "cue-2"))
        app.preview_tree.focus("cue-2")
        app._on_preview_lines_selected()
        app.use_selected_lines_for_note_range()
        app.range_notes_text.delete("1.0", "end")
        app.range_notes_text.insert(
            "1.0",
            "Use this helper note when the lines talk about schedule changes, meeting delays, and going somewhere first.",
        )
        app.scroll_canvas.yview_moveto(0.48)
        app.update()
        capture_window(app, images_dir / "app-context-notes.png")

        dialog = ImportExistingDialog(app, detect_callback=service.detect_existing_subtitles)
        dialog.mode_var.set("video")
        dialog.video_var.set(r"D:\Videos\sample-video.mp4")
        dialog.ja_var.set(r"D:\Subtitles\sample-video.ja.srt")
        dialog.direct_var.set(r"D:\Subtitles\sample-video.en.literal.srt")
        dialog.easy_var.set(r"D:\Subtitles\sample-video.en.adapted.srt")
        dialog.reference_var.set(r"D:\Subtitles\sample-video.reference.srt")
        dialog._sync_mode()
        dialog.update()
        capture_window(dialog, images_dir / "app-import-existing.png")
    finally:
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()
        if app is not None and app.winfo_exists():
            app.destroy()
        ui_module.build_service = original_build_service
        SubtitleStackApp._start_snapshot_thread = original_start_snapshot
        SubtitleStackApp._schedule_refresh = original_schedule_refresh


if __name__ == "__main__":
    main()
