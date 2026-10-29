"""CPU-only tests for the lyrics editor (no GPU, no models, no ComfyUI).

Run with::

    pytest tests/test_lyrics_editor.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rh_lyrics import (  # noqa: E402
    align_lyrics,
    parse_metadata,
    preview_lyrics,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "vocal_sample.json")


@pytest.fixture()
def metadata() -> str:
    with open(FIXTURE, encoding="utf-8") as fh:
        return fh.read()


def _tokens(tracks):
    return [t for track in tracks for t in track.tokens]


def test_parse_serialize_roundtrip(metadata):
    tracks = parse_metadata(metadata)
    assert len(tracks) == 1
    before = _tokens(tracks)
    after = _tokens(parse_metadata(
        json.dumps([t for t in json.loads(metadata)], ensure_ascii=False)
    ))
    assert [t.text for t in before] == [t.text for t in after]
    assert [t.duration for t in before] == [t.duration for t in after]
    assert [t.note_pitch for t in before] == [t.note_pitch for t in after]


def test_alignment_preserves_performance(metadata):
    tracks = parse_metadata(metadata)
    out_json, preview, warnings = align_lyrics(metadata, "云飘过海山岗")

    result = parse_metadata(out_json)
    before, after = _tokens(tracks), _tokens(result)

    # Structure, timing and pitch are untouched ...
    assert len(after) == len(before)
    assert [t.duration for t in after] == pytest.approx([t.duration for t in before])
    assert [t.note_pitch for t in after] == [t.note_pitch for t in before]
    assert result[0].f0 == tracks[0].f0
    assert result[0].meta["time"] == tracks[0].meta["time"]
    # note_type encodes word/section boundaries derived from the *text*, so it
    # is expected to change with the lyrics — but only to valid values.
    assert set(t.note_type for t in after) <= {1, 2, 3}

    # ... only the words change.
    new_text = "".join(t.text for t in after if not t.is_sp)
    assert new_text == "云飘过海山岗"
    assert new_text != "".join(t.text for t in before if not t.is_sp)

    # Preview exposes the phrase structure so the user can check the phrasing.
    assert "云飘过海" in preview
    assert "山岗" in preview
    assert warnings == ""


def test_too_many_chars_warns_instead_of_silently_dropping(metadata):
    """More characters than notes must be reported, not silently merged."""
    out_json, preview, warnings = align_lyrics(metadata, "云飘过大海之畔\n微光在眼底流转")
    assert "多于可用音符" in warnings
    # The alignment still succeeds (best effort) and keeps the note structure.
    assert len(_tokens(parse_metadata(out_json))) == 7



def test_same_lyrics_is_idempotent(metadata):
    original = "".join(
        t.text for t in _tokens(parse_metadata(metadata)) if not t.is_sp
    )
    first, _, _ = align_lyrics(metadata, original)
    second, _, _ = align_lyrics(first, original)
    # Re-running with the same lyrics must keep notes and timing stable.
    a, b = _tokens(parse_metadata(first)), _tokens(parse_metadata(second))
    assert [t.text for t in a] == [t.text for t in b]
    assert [t.duration for t in a] == pytest.approx([t.duration for t in b])


def test_empty_lyrics_raises(metadata):
    with pytest.raises(ValueError):
        align_lyrics(metadata, "   ")


def test_invalid_metadata_raises():
    with pytest.raises(ValueError):
        align_lyrics("{not json", "新的歌词")


def test_preview_lists_phrase_lengths(metadata):
    preview = preview_lyrics(parse_metadata(metadata))
    assert preview.startswith("[0] (4/2)")
    assert "4 个音符" not in preview  # 5 non-SP notes in the fixture


def test_speed_changes_duration_only(metadata):
    tracks = parse_metadata(metadata)
    before = sum(t.duration for t in _tokens(tracks))
    out, _, _ = align_lyrics(metadata, "云飘过大海之畔", speed=1.2)
    after = sum(t.duration for t in _tokens(parse_metadata(out)))
    assert after == pytest.approx(before / 1.2, rel=0.02)
