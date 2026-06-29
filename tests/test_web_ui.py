from pathlib import Path

from local_subtitle_stack.web_ui import HTML, count_completed_outputs, count_sources, format_bytes


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
    assert "Open direct English subtitles" in HTML
    assert "Add files and start processing" in HTML


def test_web_ui_formats_model_sizes() -> None:
    assert format_bytes(4_435_931_324) == "4.1 GB"
