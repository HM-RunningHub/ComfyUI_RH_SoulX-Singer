from __future__ import annotations
# Ported from ComfyUI-MIDI-Edit (MIT License).
# Source: https://github.com/ahkimkoo/ComfyUI-MIDI-Edit  (core/midi_format.py)
# See NOTICE in this directory for the full attribution.
# core/midi_format.py
"""MIDI token/track data structures and JSON parse/serialize.

Houses the ``Token`` / ``Track`` dataclasses and the conversion layer between
the MIDI JSON wire format and the internal representation. Frame rate (FPS) is
fixed at 50 (SoulX-Singer: sample_rate=24000, hop_size=480).
"""

import json
from dataclasses import dataclass, field


# SoulX-Singer data_processor.py: sample_rate=24000, hop_size=480 -> 50fps.
FPS = 50


@dataclass(frozen=True)
class Token:
    """A single MIDI token (internal representation of one JSON entry)."""

    text: str            # "<SP>" or an actual char/word
    phoneme: str         # original phoneme
    duration: float      # seconds
    note_pitch: int      # MIDI note number (0 = rest)
    note_type: int       # 1 = section tail / 2 = normal·word head / 3 = word-internal continuation
    index: int           # original index within the track

    @property
    def is_sp(self) -> bool:
        return self.text == "<SP>"


@dataclass
class Track:
    """Internal representation of one MIDI track (preserves non-token fields)."""

    tokens: list[Token]
    meta: dict = field(default_factory=dict)  # index/language/time etc. from the original JSON
    f0: str = ""          # frame-level f0, preserved verbatim


def parse_tracks(midi_json_str: str) -> list[Track]:
    """Parse a MIDI JSON string into a list of ``Track`` objects.

    Raises:
        ValueError: JSON parse failure or a missing required field.
    """
    if not midi_json_str or not midi_json_str.strip():
        raise ValueError("midi_json is empty")
    try:
        data = json.loads(midi_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError("midi_json must be a list of track objects")
    return [_parse_track(t, i) for i, t in enumerate(data)]


def _parse_track(track_data: dict, track_idx: int) -> Track:
    """Parse a single track dict."""
    required = ["text", "phoneme", "duration", "note_pitch", "note_type"]
    for field_name in required:
        if field_name not in track_data:
            raise ValueError(f"track {track_idx} missing field: {field_name}")

    texts = track_data["text"].split()
    phonemes = track_data["phoneme"].split()
    durations = track_data["duration"].split()
    pitches = track_data["note_pitch"].split()
    note_types = track_data["note_type"].split()

    n = len(texts)
    if not (len(phonemes) == len(durations) == len(pitches) == len(note_types) == n):
        raise ValueError(
            f"track {track_idx}: field length mismatch "
            f"(text={n}, phoneme={len(phonemes)}, duration={len(durations)}, "
            f"pitch={len(pitches)}, type={len(note_types)})"
        )

    tokens = [
        Token(
            text=texts[i],
            phoneme=phonemes[i],
            duration=float(durations[i]),
            note_pitch=int(pitches[i]),
            note_type=int(note_types[i]),
            index=i,
        )
        for i in range(n)
    ]

    meta = {k: v for k, v in track_data.items()
            if k not in [*required, "f0"]}
    f0 = track_data.get("f0", "")
    return Track(tokens=tokens, meta=meta, f0=f0)


def serialize_track(track: Track) -> dict:
    """Serialize a ``Track`` back into a track dict (wire-format compatible)."""
    tokens = track.tokens
    result = dict(track.meta)
    result["text"] = " ".join(t.text for t in tokens)
    result["phoneme"] = " ".join(t.phoneme for t in tokens)
    result["duration"] = " ".join(f"{t.duration:.2f}" for t in tokens)
    result["note_pitch"] = " ".join(str(t.note_pitch) for t in tokens)
    result["note_type"] = " ".join(str(t.note_type) for t in tokens)
    if track.f0:
        result["f0"] = track.f0
    return result


def serialize_tracks(tracks: list[Track]) -> str:
    """Serialize a list of ``Track`` objects into a JSON string."""
    return json.dumps([serialize_track(t) for t in tracks],
                      ensure_ascii=False, indent=2)
