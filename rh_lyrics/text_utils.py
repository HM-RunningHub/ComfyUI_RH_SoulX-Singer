from __future__ import annotations
# Ported from ComfyUI-MIDI-Edit (MIT License).
# Source: https://github.com/ahkimkoo/ComfyUI-MIDI-Edit  (core/text_utils.py)
# See NOTICE in this directory for the full attribution.
# core/text_utils.py
"""Text utilities: digit normalization, reduplication detection, sentence splitting.

Re-exports ``normalize_digits`` from :mod:`core.g2p` so callers have a single
text-processing surface, and provides the MIDIEditLyrics sentence splitter plus
the Chinese reduplication (叠词) table used by both alignment algorithms.
"""

import re

from .g2p import normalize_digits  # noqa: F401  (re-exported public API)


def clean_lyrics(text: str) -> str:
    """Remove punctuation and newlines.

    Keeps Chinese chars, English letters, and spaces. Arabic digits are
    converted to Chinese number chars.
    """
    if not text:
        return ""
    text = normalize_digits(text)
    return re.sub(r"[^\u4e00-\u9fffA-Za-z ]", "", text)


# --- 叠词判断 ---
# 叠词（reduplication）是中文里两字相同但构成独立词汇的词，
# 如"哥哥""慢慢"。它们的每个字都应独立演唱（type=2），
# 不应标 type=3（延续音，SoulX-Singer 不独立唱第二个字）。
#
# 与之相对的是"节奏重复"——如"完完""了了"，第二个字是第一个的
# 延续（type=3），原 MIDI 数据里这类重复的第二个字确实是 type=3。
_REDUP_WORDS = frozenset({
    # 家人/称呼
    "爸爸", "妈妈", "爷爷", "奶奶", "哥哥", "弟弟", "姐姐", "妹妹",
    "叔叔", "伯伯", "舅舅", "姑姑", "婆婆", "公公", "外婆", "外公",
    "阿姨", "嫂嫂", "婶婶", "太太", "爹爹", "宝宝", "娃娃",
    # 常用叠词
    "星星", "慢慢", "轻轻", "悄悄", "渐渐", "常常", "刚刚",
    "处处", "人人", "天天", "年年", "步步", "声声", "点点",
    "阵阵", "朵朵", "条条", "闪闪", "洋洋", "匆匆", "纷纷",
    "茫茫", "圆圆", "方方", "长长", "短短", "高高", "低低",
    "大大", "小小", "多多", "少少", "好好", "坏坏", "快快",
    "暖暖", "凉凉", "红红", "白白", "黑黑", "黄黄", "绿绿",
    "甜甜", "苦苦", "酸酸", "辣辣", "香香", "臭臭",
    "嘻嘻", "哈哈", "呵呵", "嗯嗯", "哎哎",
})


def is_reduplication(char: str, prev_char: str) -> bool:
    """判断两个字是否构成叠词（独立词汇，不是节奏重复）。

    Returns True 如果 prev_char+char 在叠词表中（如"哥哥"），
    表示两个字都应独立演唱（type=2），不应标 type=3。
    """
    if len(char) != 1 or len(prev_char) != 1 or char != prev_char:
        return False
    return (prev_char + char) in _REDUP_WORDS


# --- MIDIEditLyrics sentence splitting ---

_SENTENCE_DELIM_RE = re.compile(r"[\n，。！？；：、,.\!\?;:]+")


def split_lyrics_to_sentences(new_lyrics: str) -> list[str]:
    """Split user lyrics into sentences by newlines and punctuation.

    Returns a list of cleaned (non-empty) sentence strings.
    Preserves spaces within sentences (needed for English word boundaries).
    """
    if not new_lyrics:
        return []
    raw = _SENTENCE_DELIM_RE.split(new_lyrics)
    result = []
    for s in raw:
        # Strip leading/trailing whitespace and collapse multiple spaces
        cleaned = " ".join(s.split())
        if cleaned:
            result.append(cleaned)
    return result
