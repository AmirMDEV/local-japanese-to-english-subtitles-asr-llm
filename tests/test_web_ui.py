from pathlib import Path

from local_subtitle_stack.web_ui import count_completed_outputs, count_sources


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
