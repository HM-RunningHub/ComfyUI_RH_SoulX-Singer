from __future__ import annotations
"""Force-align transcription to a user-supplied lyric sheet.

Ported from ``ComfyUI-MIDI-Edit/core/soulsx_singer.py``
(``_ForceAlignLyricTranscriber``), MIT License — see ``NOTICE``.

Why this exists
---------------
The stock ``Preprocessor`` runs ASR and takes whatever it hears.  For a cover
workflow the user already *knows* the original lyrics, and ASR mistakes (and
`<SP>` misplacement) propagate straight into note segmentation, which is what
makes a re-sung line sound strained.  Biasing the ASR with the reference
lyrics pins the text to what was actually sung, while all timing / pitch / f0
data still comes from the audio.

Strategy (per segment, only when reference lyrics are supplied):

1. First-pass ASR (no hotword) → rough character count.
2. Slice the reference by that count → phrase chunks for the hotword.
3. Second-pass ASR with ``hotword=<phrase chunks>`` (Paraformer biasing).
4. Re-slice the reference using the second pass's count (more accurate).
5. Character-level DTW between the reference slice and the ASR output: the
   text always comes from the reference, the timestamps always come from the
   ASR.

The wrapper keeps the ``process(wav_fn, language)`` signature so
``PreprocessPipeline`` can use it transparently.
"""


import os
import re

import numpy as np

__all__ = ["ReferenceLyricsTranscriber", "wrap_transcriber", "strip_reference_lyrics"]


def _transcription_utils():
    """Lazily import the pack's transcription helpers.

    Lazy on purpose: importing this module (e.g. from a CPU-only unit test)
    must not require the parent ComfyUI package to be importable.
    """
    from ..preprocess.tools.lyric_transcription import (
        _build_words_with_gaps,
        _word_dur_post_process,
    )

    return _build_words_with_gaps, _word_dur_post_process

# Keep only CJK chars + ASCII letters: punctuation/whitespace/digits are not
# useful as ASR hotwords and would shift the character budget.
_REF_LYRICS_STRIP_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z]")


def strip_reference_lyrics(lyrics: str | None) -> str:
    """Normalise a lyric sheet to the character sequence used for alignment."""
    return _REF_LYRICS_STRIP_RE.sub("", lyrics or "")


class ReferenceLyricsTranscriber:
    """Wrap a ``LyricTranscriber`` and force its text output to the reference.

    When no reference is given, every call delegates to the inner transcriber
    unchanged (fully backwards compatible).
    """

    # Paraformer hotword phrases longer than this are less effective;
    # 5 chars per phrase works well for Mandarin lyrics.
    _HOTWORD_PHRASE_LEN = 5

    def __init__(self, inner, reference_lyrics: str | None = None):
        self.inner = inner
        self.reference_chars = strip_reference_lyrics(reference_lyrics)
        self._ref_pos = 0
        self.diagnostics: list[dict] = []
        self.captured_segment_words: list[list[str]] = []

    # -- passthroughs used by PreprocessPipeline / logging -------------------
    @property
    def verbose(self):
        return getattr(self.inner, "verbose", False)

    @property
    def device(self):
        return getattr(self.inner, "device", "cuda")

    @property
    def reference_remaining(self) -> int:
        return max(0, len(self.reference_chars) - self._ref_pos)

    def process(self, wav_fn: str, language: str | None = "Mandarin"):
        lang = (language or "").lower()
        is_zh_path = lang in {"mandarin", "cantonese", "zh", "中文", ""}
        if (not self.reference_chars
                or self._ref_pos >= len(self.reference_chars)
                or not is_zh_path):
            words, word_durs = self.inner.process(wav_fn, language)
            self.captured_segment_words.append(list(words))
            return words, word_durs

        # ---- Step 1: first-pass ASR → rough char count --------------------
        out_first = self.inner.zh_model.model.generate(wav_fn, output_timestamp=True)[0]
        first_words = out_first["text"].replace("@", "").split(" ")

        # ---- Step 2: slice for the hotword only (rough) -------------------
        seg_ref_for_hotword = self.reference_chars[
            self._ref_pos:self._ref_pos + len(first_words)
        ]
        if not seg_ref_for_hotword:
            words, word_durs = self.inner.process(wav_fn, language)
            self.captured_segment_words.append(list(words))
            return words, word_durs

        # ---- Step 3: second-pass ASR biased with hotword ------------------
        hotword = " ".join(self._split_phrases(seg_ref_for_hotword))
        try:
            out = self.inner.zh_model.model.generate(
                wav_fn, output_timestamp=True, hotword=hotword,
            )[0]
        except Exception as exc:  # hotword unsupported / model error
            print(f"[SoulX-RH] force-align: hotword ASR failed ({exc}); "
                  f"falling back to first pass")
            out = out_first

        asr_words = out["text"].replace("@", "").split(" ")
        asr_ts = [[t[0] / 1000, t[1] / 1000] for t in out["timestamp"]]

        # ---- Step 4: re-slice with the second-pass count ------------------
        seg_ref = self.reference_chars[
            self._ref_pos:self._ref_pos + len(asr_words)
        ]
        self._ref_pos += len(seg_ref)

        # ---- Step 5: DTW ---------------------------------------------------
        final_words = self._dtw_align(list(seg_ref), asr_words)
        self.diagnostics.append({
            "wav_fn": wav_fn,
            "hotword": hotword,
            "asr_words": asr_words,
            "final_words": final_words,
            "ref_slice": seg_ref,
        })

        # ---- Step 6: rebuild (words, word_durs) exactly like the ASR path --
        build_words_with_gaps, word_dur_post_process = _transcription_utils()
        words, word_durs = build_words_with_gaps(final_words, asr_ts, wav_fn)
        f0_path = os.path.splitext(wav_fn)[0] + "_f0.npy"
        if os.path.exists(f0_path):
            words, word_durs = word_dur_post_process(words, word_durs, np.load(f0_path))

        self.captured_segment_words.append(list(words))
        return words, word_durs

    @classmethod
    def _split_phrases(cls, text: str) -> list[str]:
        n = cls._HOTWORD_PHRASE_LEN
        return [text[i:i + n] for i in range(0, len(text), n)]

    @staticmethod
    def _dtw_align(ref_chars: list[str], asr_words: list[str]) -> list[str]:
        """Char-level DTW: output length always equals ``len(asr_words)``.

        Keeps ASR's structure (including melisma duplications that ROSVOT will
        later expand) while replacing each word's identity with the reference
        character that best explains it.
        """
        M, N = len(ref_chars), len(asr_words)
        if N == 0:
            return []
        if M == 0:
            return list(asr_words)

        INF = float("inf")
        dp = [[INF] * (N + 1) for _ in range(M + 1)]
        back = [[None] * (N + 1) for _ in range(M + 1)]
        dp[0][0] = 0

        for i in range(M + 1):
            for j in range(N + 1):
                if i == 0 and j == 0:
                    continue
                if i > 0 and j > 0:
                    cost = 0 if ref_chars[i - 1] == asr_words[j - 1] else 1
                    if dp[i - 1][j - 1] + cost < dp[i][j]:
                        dp[i][j] = dp[i - 1][j - 1] + cost
                        back[i][j] = ("sub", i - 1, j - 1)
                if i > 0:
                    if dp[i - 1][j] + 1 < dp[i][j]:
                        dp[i][j] = dp[i - 1][j] + 1
                        back[i][j] = ("ins", i - 1, j)
                if j > 0:
                    if dp[i][j - 1] + 1 < dp[i][j]:
                        dp[i][j] = dp[i][j - 1] + 1
                        back[i][j] = ("del", i, j - 1)

        ops = []
        i, j = M, N
        while i > 0 or j > 0:
            op, pi, pj = back[i][j]
            ops.append((op, pi, pj))
            i, j = pi, pj
        ops.reverse()

        new_words: list[str | None] = [None] * N
        for op, pi, pj in ops:
            if op == "sub":
                new_words[pj] = ref_chars[pi]
            elif op == "del":
                # Extra ASR position: give it the nearest reference char so
                # every detected note still carries a meaningful syllable.
                new_words[pj] = ref_chars[min(pi, M - 1)]
        for k in range(N):
            if new_words[k] is None:
                new_words[k] = asr_words[k]
        return new_words  # type: ignore[return-value]


def wrap_transcriber(transcriber, reference_lyrics: str | None):
    """Return ``transcriber`` wrapped for force alignment, or unwrapped.

    Always derive from the *inner* transcriber so a shared pipeline (several
    Preprocessor nodes use the same pipeline object) can never inherit another
    node's reference lyrics: an empty reference explicitly unwraps.
    """
    inner = (
        transcriber.inner
        if isinstance(transcriber, ReferenceLyricsTranscriber)
        else transcriber
    )
    reference = strip_reference_lyrics(reference_lyrics)
    if not reference:
        return inner
    return ReferenceLyricsTranscriber(inner, reference)
