from __future__ import annotations
"""Metadata helpers for the lyrics-editing nodes.

The RunningHub nodes exchange SoulX-Singer metadata as a JSON **string** under
the custom ``SoulXSinger_Audio_Metadata`` type.  ``rh_lyrics`` works on parsed
:class:`~rh_lyrics.midi_format.Track` objects, so this module is the single
place that converts between the two, plus the two user-facing helpers:

* :func:`distribute_lyrics` — split one lyric block across the metadata's
  segments (a 60 s ``max_merge_duration`` produces several).
* :func:`preview_lyrics` — render the aligned result as one line per segment,
  with the character count of every phrase, so the user can *see* how the new
  lyrics were broken up without reading JSON.
"""


import json

from .midi_format import Track, parse_tracks, serialize_tracks

__all__ = [
    "parse_metadata",
    "serialize_metadata",
    "distribute_lyrics",
    "preview_lyrics",
]


def parse_metadata(metadata) -> list[Track]:
    """Normalise any accepted metadata representation into ``list[Track]``.

    Accepts the JSON string produced by ``RunningHub SoulX-Singer
    Preprocessor`` (the normal case), an already-parsed list of segment dicts,
    or a single segment dict.
    """
    if metadata is None:
        raise ValueError("metadata is empty — connect a Preprocessor node first")

    if isinstance(metadata, str):
        text = metadata.strip()
        if not text:
            raise ValueError("metadata is an empty string — run a Preprocessor first")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata is not valid JSON: {exc}") from exc
    else:
        data = metadata

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"unsupported metadata type: {type(data).__name__}")

    return parse_tracks(json.dumps(list(data), ensure_ascii=False))


def serialize_metadata(tracks: list[Track]) -> str:
    """Serialise tracks back to the JSON string the SVS node expects."""
    return serialize_tracks(tracks)


def distribute_lyrics(lyrics: str, tracks: list[Track]) -> list[str]:
    """Split a full lyric block across tracks by duration proportion.

    Ported from ``ComfyUI-MIDI-Edit/core/edit_algorithm.py``.

    A single lyric string fed into every segment independently causes
    catastrophic splits in short segments, so partition along line boundaries
    with a budget proportional to each track's non-SP duration (duration, not
    token count, tracks real acoustic capacity).
    """
    if len(tracks) <= 1:
        return [lyrics]

    capacities = [
        sum(t.duration for t in track.tokens if not t.is_sp) for track in tracks
    ]
    total_cap = sum(capacities)
    if total_cap <= 0:
        return [lyrics] * len(tracks)

    lines = [ln for ln in lyrics.split("\n") if ln.strip()]
    if not lines:
        lines = [lyrics]

    total_chars = sum(len(line) for line in lines)
    result: list[str] = []
    line_idx = 0

    for i in range(len(tracks)):
        if i == len(tracks) - 1:
            result.append("\n".join(lines[line_idx:]))
            break

        target_chars = int(total_chars * capacities[i] / total_cap)

        accumulated = 0
        end_idx = line_idx
        while end_idx < len(lines) and accumulated < target_chars:
            accumulated += len(lines[end_idx])
            end_idx += 1

        result.append("\n".join(lines[line_idx:end_idx]))
        line_idx = end_idx

    while len(result) < len(tracks):
        result.append("")
    return result


def preview_lyrics(tracks: list[Track]) -> str:
    """Render the alignment as ``[段] (每句字数) 逐句文字``.

    Example::

        [0] (14/3/1/4/2) 雾散落银河之间星如指尖般盘旋 / 任由那 / 光 / 定格在这 / 初见
        总计 24 字 · 5 句 · 音高 36 个音符

    Phrase boundaries are the original ``<SP>`` (rest) tokens, i.e. exactly the
    segmentation ``preserve_sp`` keeps.
    """
    lines: list[str] = []
    for idx, track in enumerate(tracks):
        phrases: list[str] = []
        current: list[str] = []
        for token in track.tokens:
            if token.is_sp:
                if current:
                    phrases.append("".join(current))
                    current = []
            else:
                current.append(token.text)
        if current:
            phrases.append("".join(current))

        if not phrases:
            lines.append(f"[{idx}] (空)")
            continue

        counts = "/".join(str(len(p)) for p in phrases)
        body = " / ".join(phrases)
        total = sum(len(p) for p in phrases)
        notes = sum(1 for t in track.tokens if not t.is_sp)
        lines.append(
            f"[{idx}] ({counts}) {body}\n    总计 {total} 字 · {len(phrases)} 句 · {notes} 个音符"
        )

    return "\n".join(lines)
