"""CPU-only tests for the ROSVOT post-processing helpers."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rh_lyrics import (  # noqa: E402
    clean_metadata,
    merge_held_note_sps,
    merge_invalid_repeated_chars,
    parse_metadata,
)


def _track(texts, durs=None, pitches=None, types=None, f0="0.0 0.0 0.0"):
    n = len(texts)
    durs = durs or [0.25] * n
    pitches = pitches or [60] * n
    types = types or [2] * n
    return {
        "index": "seg",
        "language": "Mandarin",
        "time": [0, int(sum(durs) * 1000)],
        "duration": " ".join(f"{d:.2f}" for d in durs),
        "text": " ".join(texts),
        "phoneme": " ".join("p" for _ in texts),
        "note_pitch": " ".join(map(str, pitches)),
        "note_type": " ".join(map(str, types)),
        "f0": f0,
    }


def _meta(*tracks):
    return json.dumps(list(tracks), ensure_ascii=False)


def test_merge_held_note_sps_removes_pause_inside_word():
    track = parse_metadata(_meta(_track(["雾", "<SP>", "雾"])))[0]
    merged = merge_held_note_sps(track)
    assert [t.text for t in merged.tokens] == ["雾", "雾"]
    assert merged.tokens[1].note_type == 3
    # Duration is preserved (0.25 SP folded into the first note).
    assert sum(t.duration for t in merged.tokens) == pytest.approx(0.75)


def test_merge_repeated_chars_drops_melisma_duplicates():
    track = parse_metadata(_meta(_track(["雾", "散", "落", "河", "河"])))[0]
    merged = merge_invalid_repeated_chars(track)
    assert [t.text for t in merged.tokens] == ["雾", "散", "落", "河"]
    assert sum(t.duration for t in merged.tokens) == pytest.approx(1.25)


def test_valid_reduplication_is_kept():
    """哥哥 is a real AA word and must survive."""
    track = parse_metadata(_meta(_track(["哥", "哥"])))[0]
    merged = merge_invalid_repeated_chars(track)
    assert [t.text for t in merged.tokens] == ["哥", "哥"]


def test_reference_lyrics_decides_run_length():
    """Reference 滔滔 keeps 2; reference 河 keeps 1."""
    track = parse_metadata(_meta(_track(["滔", "滔", "滔"])))[0]
    assert len(merge_invalid_repeated_chars(track, "滔滔").tokens) == 2
    assert len(merge_invalid_repeated_chars(track, "河").tokens) == 1


def test_clean_metadata_keeps_f0_and_time():
    meta = _meta(_track(["雾", "<SP>", "雾", "散"]))
    out = clean_metadata(meta)
    before, after = parse_metadata(meta)[0], parse_metadata(out)[0]
    assert after.f0 == before.f0
    assert after.meta["time"] == before.meta["time"]
    # [雾, <SP>, 雾, 散] -> held-merge -> [雾, 雾, 散] -> dedup -> [雾, 散]
    assert len(after.tokens) == 2
    assert sum(t.duration for t in after.tokens) == pytest.approx(
        sum(t.duration for t in before.tokens)
    )


def test_switches_are_respected():
    meta = _meta(_track(["雾", "<SP>", "雾"]))
    assert len(parse_metadata(clean_metadata(meta, merge_held_notes=False))[0].tokens) == 3
