from __future__ import annotations
# Ported from ComfyUI-MIDI-Edit (MIT License).
# Source: https://github.com/ahkimkoo/ComfyUI-MIDI-Edit  (core/speed.py)
# See NOTICE in this directory for the full attribution.
# core/speed.py
"""Speed adjustment: duration scaling and f0 resampling.

Provides both the dict-level ``apply_speed`` (operates on the raw MIDI JSON
track dicts) and the ``apply_speed_change`` adapter (operates on parsed
``Track`` objects). Formatting helpers round durations to 2 decimals while
keeping the rounded total equal to the true total.
"""


def apply_speed(midi_data: list, speed: float) -> list:
    """Apply speed change to all tracks: scale durations and resample f0.

    Args:
        midi_data: List of track dicts (already processed by replace_lyrics).
        speed: Speed multiplier (e.g. 1.5 = 150% speed = faster = shorter duration,
               0.5 = 50% speed = slower = longer duration).
               duration_new = duration_orig / speed

    Returns:
        Modified midi_data with scaled durations and resampled f0.
    """
    import numpy as _np

    ratio = 1.0 / speed  # duration scale factor (speed up → ratio < 1 → shorter)

    for track in midi_data:
        # Scale durations
        if "duration" in track:
            dur_vals = [float(x) * ratio for x in track["duration"].split(" ")]
            track["duration"] = " ".join(format_durations(dur_vals))

        # Scale time range (used by downstream to preallocate audio buffer)
        if "time" in track and isinstance(track["time"], list) and len(track["time"]) == 2:
            track["time"] = [round(track["time"][0] * ratio), round(track["time"][1] * ratio)]

        # Resample f0 (frame-level data at ~50fps)
        if "f0" in track and track["f0"].strip():
            f0_vals = [float(x) for x in track["f0"].split(" ")]
            orig_len = len(f0_vals)
            new_len = max(1, round(orig_len * ratio))

            if new_len == orig_len:
                # No change needed
                track["f0"] = " ".join(format_f0(v) for v in f0_vals)
            elif new_len > orig_len:
                # Stretch: linear interpolation
                old_indices = _np.linspace(0, orig_len - 1, orig_len)
                new_indices = _np.linspace(0, orig_len - 1, new_len)
                resampled = _np.interp(new_indices, old_indices, f0_vals)
                track["f0"] = " ".join(format_f0(v) for v in resampled)
            else:
                # Shrink: linear interpolation then take fewer samples
                old_indices = _np.linspace(0, orig_len - 1, orig_len)
                new_indices = _np.linspace(0, orig_len - 1, new_len)
                resampled = _np.interp(new_indices, old_indices, f0_vals)
                track["f0"] = " ".join(format_f0(v) for v in resampled)

    return midi_data


def _fmt_dur(v: float) -> str:
    """Format a single duration value, cleaning up float artifacts. Keeps 2 decimal places."""
    return f"{v:.2f}"


def format_durations(durations: list[float]) -> list[str]:
    """Format a list of durations to 2 decimal places, adjusting the last
    element so the rounded total matches the true total."""
    if not durations:
        return []
    true_total = sum(durations)
    rounded = [round(d, 2) for d in durations]
    rounded_total = sum(rounded)
    diff = round(true_total - rounded_total, 2)
    if diff != 0 and rounded:
        rounded[-1] = round(rounded[-1] + diff, 2)
    return [f"{d:.2f}" for d in rounded]


def format_f0(v) -> str:
    """Format an f0 value, cleaning up float artifacts."""
    f = float(v)
    if f == 0.0:
        return "0.0"
    s = f"{f:.1f}".rstrip("0").rstrip(".")
    return s if s else "0"


def apply_speed_change(tracks: list, speed: float) -> list:
    """Apply speed change to parsed ``Track`` objects (speed ≠ 1).

    Serializes each track to its dict form, runs :func:`apply_speed`, then
    re-parses the result back into ``Track`` objects (avoiding a redundant
    JSON round-trip).
    """
    from .midi_format import serialize_track, _parse_track

    if speed == 1.0:
        return tracks
    track_dicts = [serialize_track(t) for t in tracks]
    result_dicts = apply_speed(track_dicts, speed)
    return [_parse_track(d, i) for i, d in enumerate(result_dicts)]
