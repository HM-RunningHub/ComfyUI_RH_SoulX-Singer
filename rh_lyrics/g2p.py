from __future__ import annotations
# Ported from ComfyUI-MIDI-Edit (MIT License).
# Source: https://github.com/ahkimkoo/ComfyUI-MIDI-Edit  (core/g2p.py)
# See NOTICE in this directory for the full attribution.
# core/g2p.py
"""G2P model loading and phoneme inference (g2pM + g2p_en).

Single source of truth for character/word → phoneme conversion and the
related digit/charset helpers. Both G2P models are imported lazily inside
``_get_g2p_zh`` / ``_get_g2p_en`` so that importing this module (and hence
ComfyUI startup) is not blocked by the heavy g2pM / g2p_en dependencies.
"""

import os
import re
import warnings

# Suppress numpy runtime warnings from g2pM model inference.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- NLTK data directory ---
# Download to ComfyUI's models/nltk/ on first run, not bundled in this repo.
# Resolve: find ComfyUI root by looking for "main.py" in parent directories.
def _find_comfyui_models_dir():
    """Find ComfyUI's models/ directory. Falls back to ~/.cache/nltk_data."""
    try:
        import folder_paths  # ComfyUI runtime

        return os.path.join(folder_paths.models_dir, "nltk")
    except Exception:
        pass
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _ in range(5):
        if os.path.isfile(os.path.join(p, "main.py")):
            return os.path.join(p, "models", "nltk")
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    # Not found (e.g., running tests outside ComfyUI) → use default cache
    import tempfile
    return os.path.join(os.path.expanduser("~"), ".cache", "nltk_data")

_NLTK_DATA_DIR = _find_comfyui_models_dir()
os.makedirs(_NLTK_DATA_DIR, exist_ok=True)
os.environ.setdefault("NLTK_DATA", _NLTK_DATA_DIR)

# --- Constants ---

_EN_WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)*$")
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

ZH_FLAG = "zh_"
EN_FLAG = "en_"

# --- G2P singletons (lazy) ---

_g2p_zh = None
_g2p_en = None

_NLTK_PACKAGES = [
    "taggers/averaged_perceptron_tagger",
    "taggers/averaged_perceptron_tagger_eng",
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
]


def ensure_nltk_data():
    """Ensure NLTK data path is registered and required packages are downloaded.

    ComfyUI-MIDI-Edit stores NLTK data under ``ComfyUI/models/nltk``
    (``_NLTK_DATA_DIR``). This canonical entry point performs two steps:

    1. Inserts ``_NLTK_DATA_DIR`` at the front of ``nltk.data.path``. This is
       required when ``nltk`` was imported *before* this module's ``NLTK_DATA``
       env-var ``setdefault`` executed — in that case the env var no longer
       affects ``nltk.data.path``, so ``nltk.data.find`` / models (e.g.
       ``nltk.pos_tag`` used by ``g2p_en``) cannot discover the directory. This
       is exactly what breaks the SoulX-Singer transcription path: the preprocess
       chain imports ``g2p_en`` (which pulls in nltk) independently of
       :func:`_get_g2p_en`, so by the time we reach the tagger the env var has
       had no effect.
    2. Downloads any of :data:`_NLTK_PACKAGES` still missing into
       ``_NLTK_DATA_DIR``.

    Safe to call multiple times (idempotent). Single source of truth shared by
    the MIDIEdit lyrics nodes (via :func:`_get_g2p_en`) and the SoulX-Singer
    preprocess pipeline
    (:func:`core.soulsx_singer._get_preprocess_pipeline`).
    """
    import nltk

    if _NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, _NLTK_DATA_DIR)

    for pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(pkg)
        except LookupError:
            print(f"[SoulX-RH] Downloading NLTK data: {pkg}")
            nltk.download(pkg, download_dir=_NLTK_DATA_DIR, quiet=True)


def _ensure_nltk_data():
    """Backward-compatible alias for :func:`ensure_nltk_data`."""
    ensure_nltk_data()


def _get_g2p_zh():
    """Lazy-initialise the Chinese G2P model (downloads on first use)."""
    global _g2p_zh
    if _g2p_zh is None:
        # Lazy import: g2pM is heavy and must not block module import.
        from g2pM import G2pM

        _g2p_zh = G2pM()
    return _g2p_zh


def _get_g2p_en():
    """Lazy-initialise the English G2P model."""
    global _g2p_en
    if _g2p_en is None:
        # Lazy import: g2p_en pulls in nltk + model data on construction.
        from g2p_en import G2p as G2pE

        ensure_nltk_data()
        _g2p_en = G2pE()
    return _g2p_en


# --- Helpers ---


def is_chinese_char(char: str) -> bool:
    return len(char) == 1 and bool(_ZH_CHAR_RE.fullmatch(char))


def is_english_word(word: str) -> bool:
    return bool(word) and bool(_EN_WORD_RE.fullmatch(word))


# Arabic digits → Chinese number chars (g2pM can pronounce these)
_DIGIT_TO_ZH = str.maketrans("0123456789", "零一二三四五六七八九")


def normalize_digits(text: str) -> str:
    """Convert Arabic digits to Chinese number chars so they get proper phonemes."""
    if not text:
        return text
    return text.translate(_DIGIT_TO_ZH)


# Single English letter → single ARPAbet phoneme.
# Letter-name pronunciation (e.g., g2p_en("s") → "EH1-S") produces multi-phoneme
# tokens that don't fit in short MIDI notes (< 0.1s). Instead, map each letter to
# its most common consonant/vowel sound so each note has exactly 1 phoneme.
_EN_LETTER_TO_PHONEME: dict[str, str] = {
    # Vowels — most common sound (schwa/short vowel)
    "a": "AH0", "e": "EH0", "i": "IH0", "o": "OW0", "u": "AH0",
    # Consonants — direct ARPAbet mapping
    "b": "B", "c": "K", "d": "D", "f": "F", "g": "G",
    "h": "HH", "j": "JH", "k": "K", "l": "L", "m": "M",
    "n": "N", "p": "P", "q": "K", "r": "R", "s": "S",
    "t": "T", "v": "V", "w": "W", "x": "K", "y": "Y", "z": "Z",
}


def char_to_phoneme(char: str, lang: str = "Mandarin") -> str:
    """Convert a single character to its phoneme representation.

    Chinese chars   -> zh_<pinyin_with_tone>
    English letters -> en_<single ARPAbet phoneme>  (consonant/vowel, not letter name)
    English words   -> en_<ARPAbet_phonemes_joined_by_dash>  (via g2p_en)
    Unknown / SP    -> <SP>
    """
    if char == "<SP>":
        return "<SP>"
    if is_chinese_char(char):
        g2p = _get_g2p_zh()
        result = g2p(char, tone=True, char_split=False)
        return ZH_FLAG + result[0]
    if len(char) == 1 and char.isascii() and char.isalpha():
        # Single English letter — use single ARPAbet phoneme (not letter name)
        return EN_FLAG + _EN_LETTER_TO_PHONEME.get(char.lower(), "AH0")
    if is_english_word(char):
        g2p = _get_g2p_en()
        result = g2p(char.lower())
        return EN_FLAG + "-".join(result)
    return "<SP>"


def word_to_phoneme(word: str) -> str:
    """Convert an English word to its ARPAbet phoneme string (en_X-Y-Z)."""
    g2p = _get_g2p_en()
    result = g2p(word.lower())
    return EN_FLAG + "-".join(result)
