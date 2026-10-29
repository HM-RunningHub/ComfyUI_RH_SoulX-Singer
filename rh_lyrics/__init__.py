from __future__ import annotations
"""Lyrics editing for SoulX-Singer metadata (ported from ComfyUI-MIDI-Edit).

Public surface used by ``nodes.py``:

* :func:`align_lyrics` — replace a metadata's lyrics while keeping notes,
  durations, pitch and f0 of the original recording.
* :func:`parse_metadata` / :func:`serialize_metadata` — metadata I/O.
* :func:`preview_lyrics` — human-readable per-phrase alignment preview.
* :func:`wrap_transcriber` — force alignment against a reference lyric sheet.
* :data:`CostWeights` — alignment cost weights (defaults from MIDI-Edit).

See ``NOTICE`` for attribution (MIT, ahkimkoo/ComfyUI-MIDI-Edit).
"""


from .align_algorithm import CostWeights, align_track
from .ct_transformer import restore_punctuation
from .force_align import ReferenceLyricsTranscriber, wrap_transcriber
from .metadata import (
    distribute_lyrics,
    parse_metadata,
    preview_lyrics,
    serialize_metadata,
)
from .midi_format import FPS, Track, parse_tracks, serialize_tracks
from .postprocess import (
    clean_tracks,
    merge_held_note_sps,
    merge_invalid_repeated_chars,
)
from .speed import apply_speed_change

__all__ = [
    "align_lyrics",
    "clean_metadata",
    "parse_metadata",
    "serialize_metadata",
    "preview_lyrics",
    "distribute_lyrics",
    "wrap_transcriber",
    "ReferenceLyricsTranscriber",
    "CostWeights",
    "align_track",
    "restore_punctuation",
    "apply_speed_change",
    "parse_tracks",
    "serialize_tracks",
    "merge_held_note_sps",
    "merge_invalid_repeated_chars",
    "Track",
    "FPS",
]


def clean_metadata(
    metadata,
    *,
    merge_held_notes: bool = True,
    merge_repeated_chars: bool = True,
    reference_lyrics: str = "",
) -> str:
    """Repair ROSVOT artefacts in freshly transcribed metadata.

    Returns the cleaned metadata as a JSON string, ready for the SVS node.
    """
    tracks = parse_metadata(metadata)
    cleaned = clean_tracks(
        tracks,
        merge_held_notes=merge_held_notes,
        merge_repeated_chars=merge_repeated_chars,
        reference_lyrics=reference_lyrics,
    )
    return serialize_metadata(cleaned)


def align_lyrics(
    metadata,
    new_lyrics: str,
    *,
    preserve_sp: bool = True,
    normalize_digits: bool = True,
    force_tone4: bool = False,
    speed: float = 1.0,
    high_pitch_threshold: int = 79,
) -> tuple[str, str, str]:
    """Replace lyrics in ``metadata`` and keep the original performance.

    Returns ``(metadata_json, lyrics_preview, warnings)``.
    """
    new_lyrics = "" if new_lyrics is None else str(new_lyrics)
    if not new_lyrics.strip():
        raise ValueError("new_lyrics is empty — type the new lyrics first")

    tracks = parse_metadata(metadata)
    if not tracks:
        raise ValueError("metadata contains no segments")

    weights = CostWeights()
    lyrics_per_track = distribute_lyrics(new_lyrics, tracks)

    result_tracks = []
    warnings: list[str] = []
    for idx, track in enumerate(tracks):
        track_lyrics = lyrics_per_track[idx] if idx < len(lyrics_per_track) else ""
        if not track_lyrics or not track_lyrics.strip():
            result_tracks.append(track)
            continue

        # A section has exactly one token per note: asking for more characters
        # than there are notes silently merges the surplus into single notes.
        # Surface it — this is the #1 cause of "some words are missing" audio.
        slots = sum(1 for t in track.tokens if not t.is_sp)
        chars = sum(1 for ch in track_lyrics if not ch.isspace())
        if chars > slots:
            warnings.append(
                f"[{idx}] 新词 {chars} 字多于可用音符 {slots} 个，多出的字会被合并/丢失；"
                f"请按 {slots} 字写这一句，或把句子拆到下一段"
            )

        try:
            new_track, warns = align_track(
                track,
                track_lyrics,
                weights,
                normalize_digits,
                force_tone4,
                punctuate_fn=restore_punctuation,
                preserve_sp=preserve_sp,
            )
        except ValueError as exc:
            raise ValueError(f"lyrics alignment failed for segment {idx}: {exc}") from exc

        warnings.extend(f"[{idx}] {w}" for w in warns)
        result_tracks.append(new_track)

    if speed and float(speed) != 1.0:
        result_tracks = apply_speed_change(result_tracks, float(speed))

    return (
        serialize_metadata(result_tracks),
        preview_lyrics(result_tracks),
        "; ".join(warnings),
    )
