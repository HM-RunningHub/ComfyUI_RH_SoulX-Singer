"""Post-processing for ROSVOT note output.

Ported from ``ComfyUI-MIDI-Edit/core/soulsx_singer.py``
(``_merge_rosvot_held_note_sps`` / ``_merge_invalid_repeated_chars``), MIT
License — see ``NOTICE``.

Two artefacts of ROSVOT note transcription show up as duplicated syllables in
the synthesised audio:

1. **Held notes split by an ``<SP>``** — the singer sustains one syllable, but
   ROSVOT assigns the middle note to the ``<SP>`` word index, producing
   ``[X, <SP>, X]``.  Synthesis then inserts an audible pause inside the word.
2. **Melisma duplication** — a sustained syllable spread over several notes
   gets its character repeated (``雾散落银河河之间间``).  Only *valid* AA-form
   reduplications (哥哥, 滔滔, 纷纷 …) should keep two characters.

Both are fixed here, on the final ``Track`` objects, before any lyrics editing:
durations are merged into the surviving note, ``f0`` (frame level) is never
touched, so the total length of the segment is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import replace

from .midi_format import Token, Track

__all__ = [
    "merge_held_note_sps",
    "merge_invalid_repeated_chars",
    "clean_tracks",
    "load_reduplication_wordlist",
]

_WORDLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "reduplication-verbs.txt")
_wordlist_cache: frozenset[str] | None = None


def load_reduplication_wordlist() -> frozenset[str]:
    """Valid AA-form words (哥哥, 滔滔 …); empty set when the data file is absent."""
    global _wordlist_cache
    if _wordlist_cache is None:
        try:
            with open(_WORDLIST_PATH, encoding="utf-8") as fh:
                _wordlist_cache = frozenset(
                    line.strip() for line in fh if line.strip()
                )
        except OSError:
            _wordlist_cache = frozenset()
    return _wordlist_cache


def merge_held_note_sps(track: Track) -> Track:
    """Turn ``[X, <SP>, X]`` into ``[X, X]`` (second X = continuation, type 3)."""
    tokens = track.tokens
    if len(tokens) < 3:
        return track

    out: list[Token] = []
    changed = 0
    for i, token in enumerate(tokens):
        if (out
                and token.is_sp
                and not out[-1].is_sp
                and i + 1 < len(tokens)
                and tokens[i + 1].text == out[-1].text):
            prev = out[-1]
            out[-1] = replace(prev, duration=prev.duration + token.duration)
            changed += 1
            continue
        out.append(token)

    # The lookahead above consumed the SP but the following X still needs to be
    # marked as a continuation of the previous note.
    result: list[Token] = []
    for token in out:
        if (result
                and not token.is_sp
                and token.text == result[-1].text
                and token.note_type != 3):
            token = replace(token, note_type=3, note_pitch=result[-1].note_pitch)
        result.append(token)

    if not changed:
        return track
    return Track(tokens=result, meta=track.meta, f0=track.f0)


def _ref_max_run(reference: str) -> dict[str, int]:
    """Longest consecutive run per character in the reference lyrics."""
    runs: dict[str, int] = {}
    i = 0
    while i < len(reference):
        ch = reference[i]
        run = 1
        while i + run < len(reference) and reference[i + run] == ch:
            run += 1
        runs[ch] = max(runs.get(ch, 0), run)
        i += run
    return runs


def merge_invalid_repeated_chars(
    track: Track,
    reference_lyrics: str = "",
    wordlist: frozenset[str] | None = None,
) -> Track:
    """Collapse melisma duplicates, keeping valid AA-form reduplications.

    With ``reference_lyrics`` the number of characters kept per run follows the
    reference (``滔滔`` keeps 2, ``谁负谁`` keeps 1).  Without it, a run keeps 2
    characters only when the pair is a known AA word, otherwise 1.
    """
    tokens = track.tokens
    if len(tokens) < 2:
        return track

    wordlist = load_reduplication_wordlist() if wordlist is None else wordlist
    ref_runs = _ref_max_run(reference_lyrics) if reference_lyrics else {}

    out: list[Token] = []
    i = 0
    changed = 0
    while i < len(tokens):
        token = tokens[i]
        if token.is_sp:
            out.append(token)
            i += 1
            continue

        run_end = i + 1
        while run_end < len(tokens) and tokens[run_end].text == token.text:
            run_end += 1
        run_len = run_end - i

        if run_len < 2:
            out.append(token)
            i = run_end
            continue

        if ref_runs:
            keep = ref_runs.get(token.text, 1)
        else:
            keep = 2 if (token.text + token.text) in wordlist else 1
        keep = max(1, min(keep, run_len))

        # Keep the first `keep` notes of the run; fold the rest into the last
        # kept note so the segment's total duration never changes.
        for k in range(i, i + keep):
            out.append(tokens[k])
        for k in range(i + keep, run_end):
            out[-1] = replace(out[-1], duration=out[-1].duration + tokens[k].duration)

        changed += run_len - keep
        i = run_end

    if not changed:
        return track
    return Track(tokens=out, meta=track.meta, f0=track.f0)


def clean_tracks(
    tracks: list[Track],
    *,
    merge_held_notes: bool = True,
    merge_repeated_chars: bool = True,
    reference_lyrics: str = "",
) -> list[Track]:
    """Apply the enabled fixes to every track."""
    cleaned = []
    for track in tracks:
        if merge_held_notes:
            track = merge_held_note_sps(track)
        if merge_repeated_chars:
            track = merge_invalid_repeated_chars(track, reference_lyrics)
        cleaned.append(track)
    return cleaned
