from __future__ import annotations

from pathlib import Path

import pytest

from local_subtitle_stack.domain import Cue
from local_subtitle_stack.pipeline import (
    build_context_notes,
    parse_srt,
    validate_translation_payload,
    write_srt,
)


def test_refusal_boilerplate_is_rejected() -> None:
    with pytest.raises(ValueError, match="refusal or sanitization boilerplate"):
        validate_translation_payload({"translations": ["I cannot comply with that."]}, expected_count=1)


def test_normal_dialogue_with_cant_still_passes() -> None:
    assert validate_translation_payload(
        {"translations": ["I can't help it if we have the same body type."]},
        expected_count=1,
    ) == ["I can't help it if we have the same body type."]


def test_parse_srt_supports_bom_multiline_and_numbering_gaps(tmp_path: Path) -> None:
    srt_path = tmp_path / "sample.srt"
    srt_path.write_text(
        "\ufeff1\n00:00:00,000 --> 00:00:01,250\nfirst line\nsecond line\n\n5\n00:00:02.000 --> 00:00:03.500\nthird line\n",
        encoding="utf-8",
    )

    cues = parse_srt(srt_path)

    assert [(cue.index, cue.start, cue.end, cue.text) for cue in cues] == [
        (1, 0.0, 1.25, "first line\nsecond line"),
        (2, 2.0, 3.5, "third line"),
    ]


def test_parse_srt_rejects_malformed_timestamp_block(tmp_path: Path) -> None:
    srt_path = tmp_path / "broken.srt"
    srt_path.write_text(
        "1\n00:00:00,000 - 00:00:01,000\nbroken\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Malformed SRT timestamp"):
        parse_srt(srt_path)


def test_write_srt_clamps_tiny_asr_overlaps(tmp_path: Path) -> None:
    srt_path = tmp_path / "overlap.srt"

    write_srt(
        srt_path,
        [
            Cue(index=1, start=1.0, end=2.05, text="first"),
            Cue(index=2, start=2.0, end=3.0, text="second"),
        ],
    )

    cues = parse_srt(srt_path)
    assert cues[0].end == 1.999
    assert cues[1].start == 2.0


def test_write_srt_moves_out_of_order_starts_forward(tmp_path: Path) -> None:
    srt_path = tmp_path / "out-of-order.srt"

    write_srt(
        srt_path,
        [
            Cue(index=1, start=10.0, end=10.001, text="first"),
            Cue(index=2, start=9.98, end=11.0, text="second"),
        ],
    )

    cues = parse_srt(srt_path)
    assert cues[0].end == 10.5
    assert cues[1].start == 10.5


def test_build_context_notes_includes_overlapping_reference_lines() -> None:
    notes = build_context_notes(
        group=[Cue(index=1, start=5.0, end=7.0, text="target")],
        global_context="Whole scene note.",
        scene_contexts=[],
        reference_cues=[
            Cue(index=1, start=4.5, end=5.5, text="reference one"),
            Cue(index=2, start=9.0, end=10.0, text="out of range"),
        ],
    )

    assert notes is not None
    assert "Whole scene note." in notes
    assert "reference one" in notes
    assert "out of range" not in notes
