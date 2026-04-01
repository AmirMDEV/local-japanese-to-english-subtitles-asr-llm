from __future__ import annotations

from pathlib import Path

import pytest

from local_subtitle_stack.config import AppConfig
from local_subtitle_stack.domain import JobManifest, SceneContextBlock
from local_subtitle_stack.ui import DONATE_URL, SubtitleStackApp, ordered_preview_range, wrap_preview_text


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
        self.reset()

    def reset(self) -> None:
        self.saved_notes_calls = []
        self.updated_line_calls = []
        self.job_dirs = {}
        self.manifests = {}
        self.preview_by_job = {}
        for suffix, filename in (("one", "scene-one.mp4"), ("two", "scene-two.mp4")):
            job_id = f"job-{suffix}"
            manifest = JobManifest(
                job_id=job_id,
                source_path=str(Path(f"C:/videos/{filename}")),
                source_name=filename,
                profile="conservative",
                status="completed",
                current_stage="finalize",
                export_dir=str(Path(f"C:/videos/{filename} subtitles")),
                include_adapted_english=(suffix == "one"),
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
            self.job_dirs[job_id] = Path(f"C:/queue/{job_id}")
            self.manifests[job_id] = manifest
            self.preview_by_job[job_id] = [
                {
                    "cue_index": 1,
                    "start": 0.0,
                    "end": 1.2,
                    "japanese": f"jp {suffix} 1",
                    "literal_english": f"literal {suffix} 1",
                    "adapted_english": f"adapted {suffix} 1" if suffix == "one" else "",
                    "reference": f"reference {suffix} 1" if suffix == "one" else "",
                    "has_japanese": True,
                    "has_literal_english": True,
                    "has_adapted_english": suffix == "one",
                    "has_reference": suffix == "one",
                },
                {
                    "cue_index": 2,
                    "start": 1.5,
                    "end": 2.8,
                    "japanese": f"jp {suffix} 2",
                    "literal_english": f"literal {suffix} 2",
                    "adapted_english": f"adapted {suffix} 2" if suffix == "one" else "",
                    "reference": "",
                    "has_japanese": True,
                    "has_literal_english": True,
                    "has_adapted_english": suffix == "one",
                    "has_reference": False,
                },
                {
                    "cue_index": 3,
                    "start": 3.0,
                    "end": 4.0,
                    "japanese": f"jp {suffix} 3",
                    "literal_english": f"literal {suffix} 3",
                    "adapted_english": f"adapted {suffix} 3" if suffix == "one" else "",
                    "reference": "",
                    "has_japanese": True,
                    "has_literal_english": True,
                    "has_adapted_english": suffix == "one",
                    "has_reference": False,
                },
            ]

    def status_rows(self) -> list[dict[str, str]]:
        return [
            {
                "job_id": job_id,
                "state_dir": "done",
                "status": "working" if job_id == "job-one" else manifest.status,
                "stage": "transcribe" if job_id == "job-one" else manifest.current_stage,
                "step_text": "Listening to the Japanese | 50% done | about 5m left | Audio chunk 1 of 2" if job_id == "job-one" else "Saving the subtitle files",
                "source": manifest.source_name,
                "updated_at": manifest.updated_at,
                "stage_progress_percent": "50.00" if job_id == "job-one" else "100.00",
                "overall_progress_percent": "72.00" if job_id == "job-one" else "100.00",
                "stage_eta_seconds": "300.00" if job_id == "job-one" else "",
                "stage_progress_message": "Audio chunk 1 of 2" if job_id == "job-one" else "",
                "source_kind": "video",
                "translation_source_role": "ja",
                "has_reference": "true" if job_id == "job-one" else "false",
                "include_adapted_english": "true" if manifest.include_adapted_english else "false",
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
        include_adapted_english: bool | None = None,
    ) -> JobManifest:
        manifest = self.manifests[job_id]
        manifest.series = batch_label or None
        manifest.job_context = overall_context or None
        manifest.scene_contexts = list(scene_contexts)
        if include_adapted_english is not None:
            manifest.include_adapted_english = include_adapted_english
        self.saved_notes_calls.append(
            {
                "job_id": job_id,
                "batch_label": batch_label,
                "overall_context": overall_context,
                "scene_contexts": list(scene_contexts),
                "include_adapted_english": include_adapted_english,
            }
        )
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
        rows = self.preview_by_job[job_id]
        row = next(item for item in rows if int(item["cue_index"]) == cue_index)
        if japanese_text is not None:
            row["japanese"] = japanese_text
        if literal_english_text is not None:
            row["literal_english"] = literal_english_text
        if adapted_english_text is not None:
            row["adapted_english"] = adapted_english_text
        if reference_text is not None:
            row["reference"] = reference_text
            row["has_reference"] = True
        self.updated_line_calls.append(
            {
                "job_id": job_id,
                "cue_index": cue_index,
                "japanese_text": japanese_text,
                "literal_english_text": literal_english_text,
                "adapted_english_text": adapted_english_text,
                "reference_text": reference_text,
            }
        )
        return self.manifests[job_id]


@pytest.fixture(scope="module")
def app_context() -> tuple[SubtitleStackApp, FakeService]:
    service = FakeService()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("local_subtitle_stack.ui.build_service", lambda: service)
    monkeypatch.setattr(SubtitleStackApp, "_start_snapshot_thread", lambda self: None)
    monkeypatch.setattr(SubtitleStackApp, "_schedule_refresh", lambda self: None)
    monkeypatch.setattr("local_subtitle_stack.ui.messagebox.showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr("local_subtitle_stack.ui.messagebox.showerror", lambda *args, **kwargs: None)
    try:
        window = SubtitleStackApp()
        window.withdraw()
        yield window, service
    finally:
        if "window" in locals() and window.winfo_exists():
            window.destroy()
        monkeypatch.undo()


@pytest.fixture
def app(app_context: tuple[SubtitleStackApp, FakeService]) -> SubtitleStackApp:
    window, service = app_context
    service.reset()
    window.current_job_id = None
    window.loaded_job_id = None
    window.editor_drafts.clear()
    window.scene_contexts.clear()
    window.preview_selected_cue_indexes = []
    window.preview_row_data = {}
    window.preview_mark_start_item = None
    window.preview_mark_end_item = None
    window.line_editor_cue_index = None
    window.batch_label_var.set("")
    window.include_adapted_english_var.set(True)
    window.note_start_var.set("")
    window.note_end_var.set("")
    window.status_var.set("Ready")
    window.selected_file_var.set("Pick or click a job on the left.")
    window.selected_job_state_var.set("Nothing is selected yet.")
    window.selected_stage_progress_var.set(0.0)
    window.selected_overall_progress_var.set(0.0)
    window.selected_stage_progress_text_var.set("This step: waiting")
    window.selected_overall_progress_text_var.set("Whole job: 0% done")
    window.marked_range_var.set("Marked range: none")
    window.preview_hint_var.set(
        "When you click a job, its Japanese lines and English lines show up here."
    )
    window.context_text.delete("1.0", "end")
    window.range_notes_text.delete("1.0", "end")
    window.line_editor_time_var.set("")
    window.line_editor_status_var.set("Click one subtitle line to edit it here.")
    for widget in (
        window.line_editor_japanese_text,
        window.line_editor_literal_text,
        window.line_editor_adapted_text,
        window.line_editor_reference_text,
    ):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.configure(state="disabled")
    for tree in (window.preview_tree, window.note_tree):
        for item_id in tree.get_children():
            tree.delete(item_id)
    if window.job_tree.get_children():
        window.job_tree.selection_remove(window.job_tree.selection())
    window.refresh()
    return window


def select_job(app: SubtitleStackApp, job_id: str) -> None:
    app.job_tree.selection_set(job_id)
    app._on_job_selected()


def test_ordered_preview_range_includes_all_items_between_markers() -> None:
    item_ids = ["cue-1", "cue-2", "cue-3", "cue-4"]

    assert ordered_preview_range(item_ids, "cue-2", "cue-4") == ["cue-2", "cue-3", "cue-4"]
    assert ordered_preview_range(item_ids, "cue-4", "cue-2") == ["cue-2", "cue-3", "cue-4"]


def test_ordered_preview_range_returns_empty_when_marker_is_missing() -> None:
    item_ids = ["cue-1", "cue-2", "cue-3"]

    assert ordered_preview_range(item_ids, "cue-1", "cue-9") == []


def test_wrap_preview_text_wraps_space_separated_text() -> None:
    wrapped = wrap_preview_text("this line should wrap into something easier to read", 12)

    assert "\n" in wrapped
    assert len(wrapped.splitlines()) <= 3


def test_wrap_preview_text_wraps_japanese_without_spaces() -> None:
    wrapped = wrap_preview_text("これはとても長い日本語の字幕行で折り返しが必要です", 8)

    assert "\n" in wrapped
    assert all(len(line) <= 8 for line in wrapped.splitlines())


def test_refresh_keeps_preview_selection_for_current_job(app: SubtitleStackApp) -> None:
    select_job(app, "job-one")
    app.preview_tree.selection_set(("cue-2",))
    app.preview_tree.focus("cue-2")

    app.refresh()
    app.refresh()

    assert app.current_job_id == "job-one"
    assert app.loaded_job_id == "job-one"
    assert app.preview_tree.selection() == ("cue-2",)


def test_refresh_updates_selected_progress_bars_and_text(app: SubtitleStackApp) -> None:
    select_job(app, "job-one")

    app.refresh()

    assert app.selected_stage_progress_var.get() == pytest.approx(50.0)
    assert app.selected_overall_progress_var.get() == pytest.approx(72.0)
    assert "Listening to the Japanese" in app.selected_stage_progress_text_var.get()
    assert "about 5m left" in app.selected_stage_progress_text_var.get()
    assert "Whole job: 72% done" in app.selected_overall_progress_text_var.get()


def test_marked_range_survives_refresh_cycles(app: SubtitleStackApp) -> None:
    select_job(app, "job-one")
    app.preview_tree.selection_set(("cue-1",))
    app.preview_tree.focus("cue-1")
    app.mark_preview_start_line()
    app.preview_tree.selection_set(("cue-2",))
    app.preview_tree.focus("cue-2")
    app.mark_preview_end_line()

    app.refresh()

    assert app.preview_mark_start_item == "cue-1"
    assert app.preview_mark_end_item == "cue-2"
    assert app.note_start_var.get() == "00:00:00"
    assert app.note_end_var.get() == "00:00:02"
    assert "00:00:00 to 00:00:02" in app.marked_range_var.get()


def test_switching_jobs_restores_unsaved_drafts(app: SubtitleStackApp) -> None:
    select_job(app, "job-one")
    app.batch_label_var.set("Batch A")
    app.context_text.insert("1.0", "Whole scene is set at the bath house.")
    app.note_start_var.set("00:00")
    app.note_end_var.set("00:03")
    app.range_notes_text.insert("1.0", "Talking about body type and proportions.")
    app.scene_contexts = [
        SceneContextBlock(start_seconds=0.0, end_seconds=3.0, notes="Bath scene."),
    ]
    app._render_scene_blocks()
    app.preview_selected_cue_indexes = [2]
    app.preview_mark_start_item = "cue-1"
    app.preview_mark_end_item = "cue-2"
    app._update_marked_range_status()

    select_job(app, "job-two")
    app.context_text.insert("1.0", "Home scene.")

    select_job(app, "job-one")

    assert app.batch_label_var.get() == "Batch A"
    assert app.context_text.get("1.0", "end").strip() == "Whole scene is set at the bath house."
    assert app.note_start_var.get() == "00:00"
    assert app.note_end_var.get() == "00:03"
    assert app.range_notes_text.get("1.0", "end").strip() == "Talking about body type and proportions."
    assert app.scene_contexts[0].notes == "Bath scene."


def test_reload_lines_keeps_unsaved_draft_and_selection(app: SubtitleStackApp) -> None:
    select_job(app, "job-one")
    app.context_text.insert("1.0", "Keep this draft.")
    app.range_notes_text.insert("1.0", "Keep this note too.")
    app.note_start_var.set("00:01")
    app.note_end_var.set("00:04")
    app.preview_selected_cue_indexes = [1, 2]
    app.preview_mark_start_item = "cue-1"
    app.preview_mark_end_item = "cue-2"

    app.reload_selected_preview()

    assert app.context_text.get("1.0", "end").strip() == "Keep this draft."
    assert app.range_notes_text.get("1.0", "end").strip() == "Keep this note too."
    assert app.note_start_var.get() == "00:01"
    assert app.note_end_var.get() == "00:04"


def test_selecting_one_preview_line_loads_quick_editor(app: SubtitleStackApp) -> None:
    select_job(app, "job-one")

    app.preview_tree.selection_set(("cue-2",))
    app.preview_tree.focus("cue-2")
    app._on_preview_lines_selected()

    assert app.line_editor_cue_index == 2
    assert app.line_editor_time_var.get() == "00:00:01 - 00:00:02"
    assert app.line_editor_japanese_text.get("1.0", "end").strip() == "jp one 2"
    assert app.line_editor_literal_text.get("1.0", "end").strip() == "literal one 2"
    assert app.line_editor_adapted_text.get("1.0", "end").strip() == "adapted one 2"
    assert app.line_editor_reference_text.get("1.0", "end").strip() == ""
    assert not app.save_line_button.instate(("disabled",))


def test_reference_column_appears_only_when_present(app: SubtitleStackApp) -> None:
    select_job(app, "job-one")

    assert "reference" in [role for role, _label in app.preview_visible_columns]
    assert "reference" in app.preview_tree["displaycolumns"]

    select_job(app, "job-two")

    assert "reference" not in [role for role, _label in app.preview_visible_columns]
    assert "reference" not in app.preview_tree["displaycolumns"]


def test_easy_english_column_and_checkbox_follow_selected_job(app: SubtitleStackApp) -> None:
    select_job(app, "job-one")

    assert "adapted_english" in [role for role, _label in app.preview_visible_columns]
    assert "adapted" in app.preview_tree["displaycolumns"]
    assert app.include_adapted_english_var.get() is True

    select_job(app, "job-two")

    assert "adapted_english" not in [role for role, _label in app.preview_visible_columns]
    assert "adapted" not in app.preview_tree["displaycolumns"]
    assert app.include_adapted_english_var.get() is False


def test_double_click_preview_focuses_matching_editor(app_context: tuple[SubtitleStackApp, FakeService]) -> None:
    window, service = app_context
    service.reset()
    window.refresh()
    select_job(window, "job-one")
    focus_calls: list[str] = []
    original_focus_set = window.line_editor_literal_text.focus_set

    def record_focus() -> None:
        focus_calls.append("literal")
        original_focus_set()

    window.line_editor_literal_text.focus_set = record_focus  # type: ignore[method-assign]

    class Event:
        x = 260
        y = 42

    window.preview_tree.selection_set(("cue-2",))
    window.preview_tree.focus("cue-2")
    window.preview_tree.identify_row = lambda _y: "cue-2"  # type: ignore[method-assign]
    window.preview_tree.identify_column = lambda _x: "#3"  # type: ignore[method-assign]

    result = window._on_preview_tree_double_click(Event())

    assert result == "break"
    assert window.line_editor_cue_index == 2
    assert focus_calls == ["literal"]


def test_global_mousewheel_routes_to_preview_when_pointer_is_over_preview(
    app: SubtitleStackApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_job(app, "job-one")
    preview_scrolls: list[tuple[int, str]] = []
    outer_scrolls: list[tuple[int, str]] = []
    monkeypatch.setattr(app.preview_tree, "yview_scroll", lambda units, what: preview_scrolls.append((units, what)))
    monkeypatch.setattr(app.scroll_canvas, "yview_scroll", lambda units, what: outer_scrolls.append((units, what)))
    monkeypatch.setattr(app, "winfo_pointerx", lambda: 10)
    monkeypatch.setattr(app, "winfo_pointery", lambda: 10)
    monkeypatch.setattr(app, "winfo_containing", lambda *_args: app.preview_tree)

    class Event:
        delta = -120
        num = None

    app._on_global_mousewheel(Event())

    assert preview_scrolls == [(1, "units")]
    assert outer_scrolls == []


def test_shift_mousewheel_routes_horizontally_to_outer_canvas(
    app: SubtitleStackApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer_x_scrolls: list[tuple[int, str]] = []
    outer_y_scrolls: list[tuple[int, str]] = []
    monkeypatch.setattr(app.scroll_canvas, "xview_scroll", lambda units, what: outer_x_scrolls.append((units, what)))
    monkeypatch.setattr(app.scroll_canvas, "yview_scroll", lambda units, what: outer_y_scrolls.append((units, what)))
    monkeypatch.setattr(app, "winfo_pointerx", lambda: 10)
    monkeypatch.setattr(app, "winfo_pointery", lambda: 10)
    monkeypatch.setattr(app, "winfo_containing", lambda *_args: app.scroll_root)

    class Event:
        delta = -120
        num = None
        state = 0x0001

    app._on_global_mousewheel(Event())

    assert outer_x_scrolls == [(1, "units")]
    assert outer_y_scrolls == []


def test_canvas_configure_keeps_requested_width_when_content_is_wider(
    app: SubtitleStackApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[tuple[object, int]] = []
    monkeypatch.setattr(app.scroll_root, "winfo_reqwidth", lambda: 1600)
    monkeypatch.setattr(app.scroll_canvas, "itemconfigure", lambda item, **kwargs: configured.append((item, kwargs["width"])))

    class Event:
        width = 900

    app._on_canvas_configure(Event())

    assert configured == [(app.scroll_window, 1600)]


def test_saving_selected_line_updates_preview_and_service(app_context: tuple[SubtitleStackApp, FakeService]) -> None:
    window, service = app_context
    service.reset()
    window.current_job_id = None
    window.loaded_job_id = None
    window.editor_drafts.clear()
    window.scene_contexts.clear()
    window.preview_selected_cue_indexes = []
    window.preview_row_data = {}
    window.preview_mark_start_item = None
    window.preview_mark_end_item = None
    window.line_editor_cue_index = None
    window.batch_label_var.set("")
    window.include_adapted_english_var.set(True)
    window.note_start_var.set("")
    window.note_end_var.set("")
    window.status_var.set("Ready")
    window.context_text.delete("1.0", "end")
    window.range_notes_text.delete("1.0", "end")
    for widget in (
        window.line_editor_japanese_text,
        window.line_editor_literal_text,
        window.line_editor_adapted_text,
        window.line_editor_reference_text,
    ):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.configure(state="disabled")
    for tree in (window.preview_tree, window.note_tree):
        for item_id in tree.get_children():
            tree.delete(item_id)
    if window.job_tree.get_children():
        window.job_tree.selection_remove(window.job_tree.selection())
    window.refresh()

    select_job(window, "job-one")
    window.preview_tree.selection_set(("cue-2",))
    window.preview_tree.focus("cue-2")
    window._on_preview_lines_selected()
    window.line_editor_japanese_text.delete("1.0", "end")
    window.line_editor_japanese_text.insert("1.0", "edited jp")
    window.line_editor_literal_text.delete("1.0", "end")
    window.line_editor_literal_text.insert("1.0", "edited literal")
    window.line_editor_adapted_text.delete("1.0", "end")
    window.line_editor_adapted_text.insert("1.0", "edited adapted")

    window.save_selected_line_edit()

    assert service.updated_line_calls[-1] == {
        "job_id": "job-one",
        "cue_index": 2,
        "japanese_text": "edited jp",
        "literal_english_text": "edited literal",
        "adapted_english_text": "edited adapted",
        "reference_text": None,
    }
    assert window.preview_row_data[2]["japanese"] == "edited jp"
    assert window.preview_row_data[2]["literal_english"] == "edited literal"
    assert window.preview_row_data[2]["adapted_english"] == "edited adapted"
    assert window.status_var.get() == "Saved changes for subtitle line 2"


def test_redo_english_launches_background_process_and_disables_conflicting_buttons(
    app: SubtitleStackApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_job(app, "job-one")
    app.context_text.insert("1.0", "Use this saved note.")

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def communicate(self) -> tuple[str, str]:
            return ("Rebuilt English for job-one", "")

    fake_process = FakeProcess()
    scheduled: list[tuple[int, object]] = []
    monkeypatch.setattr(app, "after", lambda delay, callback: scheduled.append((delay, callback)) or "after-1")
    monkeypatch.setattr(
        "local_subtitle_stack.ui.subprocess.Popen",
        lambda *args, **kwargs: fake_process,
    )

    app.redo_english_selected()

    assert app.rebuild_process is fake_process
    assert app.rebuild_job_id == "job-one"
    assert app.service.saved_notes_calls[-1]["overall_context"] == "Use this saved note."
    assert app.service.saved_notes_calls[-1]["include_adapted_english"] is True
    assert app.start_processing_button.instate(("disabled",))
    assert app.retry_selected_button.instate(("disabled",))
    assert app.save_notes_button.instate(("disabled",))
    assert app.redo_english_button.instate(("disabled",))
    assert scheduled

    fake_process.returncode = 0
    app._poll_rebuild_process()

    assert app.rebuild_process is None
    assert app.rebuild_job_id is None
    assert not app.start_processing_button.instate(("disabled",))
    assert not app.redo_english_button.instate(("disabled",))
    assert app.status_var.get() == "English subtitles were rebuilt for the selected job"


def test_open_donate_page_opens_browser_and_updates_status(
    app: SubtitleStackApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr("local_subtitle_stack.ui.webbrowser.open_new_tab", lambda url: opened.append(url))

    app.open_donate_page()

    assert opened == [DONATE_URL]
    assert app.status_var.get() == "Opened the donate page in your browser"
