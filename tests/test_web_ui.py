from pathlib import Path

from local_subtitle_stack.web_ui import HTML, close_other_web_ui_processes, count_completed_outputs, count_sources, format_bytes


def test_web_ui_counts_sources_and_completed_outputs(tmp_path: Path) -> None:
    folder = tmp_path / "course"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    video = folder / "one.mp4"
    nested_video = nested / "two.mov"
    video.write_text("video", encoding="utf-8")
    nested_video.write_text("video", encoding="utf-8")
    (folder / "one.raw.meta.json").write_text("{}", encoding="utf-8")

    assert count_sources([folder], recursive=False) == 1
    assert count_sources([folder], recursive=True) == 2
    assert count_sources([video], recursive=True) == 1
    assert count_completed_outputs([folder], recursive=True) == 1


def test_web_ui_has_responsive_layout_shell() -> None:
    assert "Fast Multilanguage Transcriber" in HTML
    assert "className:\"layout\"" in HTML
    assert "@media (max-width: 980px)" in HTML
    assert "target-list" in HTML
    assert "max-height: min(34vh, 340px)" in HTML
    assert "/api/models" in HTML
    assert "Ollama storage" in HTML
    assert "Preview and line editor" in HTML
    assert "Transcription and translation models" in HTML
    assert "Load existing subtitles" in HTML
    assert "Check setup" in HTML
    assert "Start processing all jobs" in HTML
    assert "What to do now" in HTML
    assert "Open direct English translation in Subtitle Edit" in HTML
    assert "Context-applied English" in HTML
    assert "Open review bundle in Subtitle Edit" in HTML
    assert "Add files and start processing" in HTML
    assert "added to every English translation prompt after the Japanese audio has been transcribed" in HTML
    assert "Overall video context" in HTML
    assert "Time-range context" in HTML
    assert "Selected subtitle lines fill these times automatically" in HTML
    assert "Add time-range context" in HTML
    assert "Retranslate selected time range" in HTML
    assert "Shift-click selects a continuous range" in HTML
    assert "Pick context-applied" in HTML
    assert "Pick natural" not in HTML
    assert "Series or project name" in HTML
    assert "Drop an .srt file here to edit existing subtitles" in HTML
    assert "/api/upload-subtitle" in HTML
    assert "Saved subtitle files" in HTML
    assert "Delete job from list" in HTML
    assert "/api/job/delete" in HTML
    assert "selectNewJob" in HTML


def test_web_ui_formats_model_sizes() -> None:
    assert format_bytes(4_435_931_324) == "4.1 GB"


def test_web_ui_closes_old_web_processes_only(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, pid: int, command: list[str]) -> None:
            self.pid = pid
            self.info = {"pid": pid, "cmdline": command}
            self.terminated = False
            self.killed = False

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    current = FakeProcess(10, ["python", "-m", "local_subtitle_stack", "web-ui"])
    parent = FakeProcess(9, ["python", "-m", "local_subtitle_stack", "web-ui"])
    old_web = FakeProcess(11, ["python", "-m", "local_subtitle_stack.cli", "web-ui", "--port", "8767"])
    worker = FakeProcess(12, ["python", "-m", "local_subtitle_stack.cli", "worker"])
    other = FakeProcess(13, ["python", "-m", "other_app", "web-ui"])

    monkeypatch.setattr("local_subtitle_stack.web_ui.os.getpid", lambda: 10)
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.Process", lambda _pid: type("Current", (), {"parents": lambda _self: [parent]})())
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.process_iter", lambda _attrs: [parent, current, old_web, worker, other])
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.wait_procs", lambda processes, timeout: (processes, []))

    close_other_web_ui_processes()

    assert old_web.terminated is True
    assert worker.terminated is False
    assert other.terminated is False
    assert current.terminated is False
    assert parent.terminated is False
