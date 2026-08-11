"""VTT caption parsing for the short-video agent.

These cover a defect found by running the agent end to end against a real
15-minute talk: the parser passed YouTube's per-word timing markup straight
through to the model, and kept every rolling-caption repeat. The transcript came
out at 98,217 characters where the actual speech is ~23,000 — four times the
tokens, all of it noise. On a long podcast that difference is the context window.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_agents.agents.short_video_generator.tools.get_short_clips_tool import (
    _parse_vtt_dir,
    _strip_overlap,
)


# --------------------------------------------------------------------------
# _strip_overlap
# --------------------------------------------------------------------------

def test_exact_repeat_collapses_to_nothing():
    assert _strip_overlap("this program is brought to you by",
                          "this program is brought to you by") == ""


def test_rolling_window_keeps_only_the_new_words():
    assert _strip_overlap(
        "this program is brought to you by",
        "this program is brought to you by Stanford University",
    ) == "Stanford University"


def test_partial_tail_overlap_is_trimmed():
    assert _strip_overlap("a b c d", "c d e f") == "e f"


def test_unrelated_text_passes_through_whole():
    assert _strip_overlap("a b c", "x y z") == "x y z"


def test_no_previous_text_passes_through():
    assert _strip_overlap("", "first line") == "first line"


def test_empty_current_yields_empty():
    assert _strip_overlap("a b", "") == ""


# --------------------------------------------------------------------------
# _parse_vtt_dir
# --------------------------------------------------------------------------

# Shaped exactly like a real YouTube auto-caption file: per-word <timestamp> and
# <c> tags, and each cue restating the tail of the one before it.
AUTO_VTT = """WEBVTT
Kind: captions
Language: en

00:00:07.520 --> 00:00:08.950
this<00:00:07.720><c> program</c><00:00:08.160><c> is</c>

00:00:08.960 --> 00:00:11.870
this program is
brought<00:00:09.200><c> to</c><00:00:09.500><c> you</c><00:00:09.800><c> by</c>

00:00:11.880 --> 00:00:20.509
brought to you by
Stanford<00:00:12.100><c> University</c>
"""


def test_parser_strips_markup_and_overlap(tmp_path: Path):
    (tmp_path / "captions.en.vtt").write_text(AUTO_VTT)
    transcript = _parse_vtt_dir(tmp_path)

    spoken = " ".join(
        line for line in transcript.splitlines()
        if line and not line.startswith("[")
    )

    assert "<c>" not in transcript and "</c>" not in transcript
    assert "00:00:07.720" not in spoken, "per-word timing markup leaked into the text"
    assert spoken.split() == [
        "this", "program", "is", "brought", "to", "you", "by",
        "Stanford", "University",
    ], f"expected each word once, got: {spoken}"


def test_parser_keeps_the_cue_timestamps(tmp_path: Path):
    (tmp_path / "captions.en.vtt").write_text(AUTO_VTT)
    transcript = _parse_vtt_dir(tmp_path)
    # Timestamps are what the clip selector cuts on, so they must survive intact.
    assert "[00:00:07,520 --> 00:00:08,950]" in transcript


def test_parser_raises_when_there_are_no_captions(tmp_path: Path):
    import pytest

    with pytest.raises(Exception, match="No captions available"):
        _parse_vtt_dir(tmp_path)
