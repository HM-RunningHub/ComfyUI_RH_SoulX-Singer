from __future__ import annotations
# Ported from ComfyUI-MIDI-Edit (MIT License).
# Source: https://github.com/ahkimkoo/ComfyUI-MIDI-Edit  (core/ct_transformer.py)
# See NOTICE in this directory for the full attribution.
# core/ct_transformer.py
"""CT-Transformer punctuation model loading and inference.

Loads the ModelScope ONNX punctuation model lazily and restores punctuation on
raw lyrics text, used by the smart sentence splitting in both editing and
alignment algorithms.
"""

import json
import os

_MODELSCOPE_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx"
_PUNC_MODEL_FILES = {"model_quant.onnx", "tokens.json"}
_PUNC_LIST = ["<unk>", "_", ",", "。", "?", "、"]
_PUNC_LIST_NORMALIZED = ["<unk>", "_", "，", "。", "？", "、"]
_SPLIT_SIZE = 20

_ct_transformer = None  # module-level singleton


def _get_models_base_dir():
    """Return the base models directory (ComfyUI models_dir or local fallback)."""
    try:
        import folder_paths

        return folder_paths.models_dir
    except ImportError:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )


def _ensure_punc_model():
    """Download CT-Transformer model from ModelScope if not present.

    Stores model files under {models_base}/ct-transformer-punc/.
    """
    model_dir = os.path.join(_get_models_base_dir(), "ct-transformer-punc")
    if all(os.path.exists(os.path.join(model_dir, f)) for f in _PUNC_MODEL_FILES):
        return model_dir

    os.makedirs(model_dir, exist_ok=True)
    print("[SoulX-RH] Downloading CT-Transformer punctuation model from ModelScope...")

    from modelscope import snapshot_download

    downloaded = snapshot_download(
        _MODELSCOPE_MODEL_ID,
        cache_dir=model_dir,
    )

    # snapshot_download may create a subdirectory; copy needed files to model_dir
    if downloaded != model_dir:
        for fname in _PUNC_MODEL_FILES:
            src = os.path.join(downloaded, fname)
            dst = os.path.join(model_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil

                shutil.copy2(src, dst)

    # Verify files exist after download
    missing = [f for f in _PUNC_MODEL_FILES
               if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"[SoulX-RH] Failed to download CT-Transformer model. "
            f"Missing files: {missing}"
        )

    print(f"[SoulX-RH] CT-Transformer punctuation model ready at {model_dir}")
    return model_dir


def _load_token_list(tokens_path: str) -> list[str]:
    """Load the token list from a JSON file."""
    with open(tokens_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Copied utility functions from cttPunctuator (adapted, no typeguard dep) ---


def _code_mix_split_words(text: str) -> list[str]:
    """Split text into words: each Chinese char is a word, consecutive ASCII is one word."""
    words = []
    segs = text.split()
    for seg in segs:
        current_word = ""
        for c in seg:
            if len(c.encode()) == 1:
                current_word += c
            else:
                if len(current_word) > 0:
                    words.append(current_word)
                    current_word = ""
                words.append(c)
        if len(current_word) > 0:
            words.append(current_word)
    return words


def _split_to_mini_sentence(words: list, word_limit: int = _SPLIT_SIZE) -> list[list]:
    """Split word list into chunks of at most word_limit."""
    if len(words) <= word_limit:
        return [words]
    sentences = []
    length = len(words)
    sentence_len = length // word_limit
    for i in range(sentence_len):
        sentences.append(words[i * word_limit: (i + 1) * word_limit])
    if length % word_limit > 0:
        sentences.append(words[sentence_len * word_limit:])
    return sentences


class _TokenIDConverter:
    """Bidirectional converter between token strings and integer IDs."""

    def __init__(self, token_list: list[str]):
        self.token_list = token_list
        self.unk_symbol = token_list[-1]
        self.token2id = {v: i for i, v in enumerate(self.token_list)}
        self.unk_id = self.token2id[self.unk_symbol]

    def tokens2ids(self, tokens: list[str]) -> list[int]:
        return [self.token2id.get(t, self.unk_id) for t in tokens]


class _CTTransformerPunc:
    """Lightweight CT-Transformer punctuation restorer.

    Directly loads the ModelScope ONNX model and runs inference,
    without the lovemefan/cttPunctuator dependency.
    """

    def __init__(self, model_dir: str):
        import numpy as np
        import onnxruntime

        tokens_path = os.path.join(model_dir, "tokens.json")
        model_path = os.path.join(model_dir, "model_quant.onnx")

        token_list = _load_token_list(tokens_path)
        self.converter = _TokenIDConverter(token_list)

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.log_severity_level = 3  # suppress warnings
        self.session = onnxruntime.InferenceSession(model_path, sess_options=opts)

        # Determine period index in punc list
        self.punc_list = list(_PUNC_LIST_NORMALIZED)
        self.period_idx = _PUNC_LIST_NORMALIZED.index("。")

    def add_punctuation(self, text: str) -> str:
        """Run CT-Transformer inference to add punctuation to text.

        Returns the text with punctuation inserted.
        """
        import numpy as np

        split_text = _code_mix_split_words(text)
        if not split_text:
            return text

        split_text_id = self.converter.tokens2ids(split_text)
        mini_sentences = _split_to_mini_sentence(split_text)
        mini_sentences_id = _split_to_mini_sentence(split_text_id)

        cache_sent = []
        cache_sent_id = []
        result_parts = []

        for mini_i in range(len(mini_sentences)):
            mini_sentence = cache_sent + mini_sentences[mini_i]
            mini_sentence_id = cache_sent_id + mini_sentences_id[mini_i]
            # ModelScope quantized model uses int32 input named "inputs"
            mini_sentence_id = np.array(mini_sentence_id, dtype="int32")
            text_lengths = np.array([len(mini_sentence)], dtype="int32")

            mapped_feed = {
                "inputs": mini_sentence_id[None, :],
                "text_lengths": text_lengths,
            }

            outputs = self.session.run(None, mapped_feed)
            punctuations = np.argmax(outputs[0], axis=-1)[0]

            # Search for last period/question mark as cache boundary
            if mini_i < len(mini_sentences) - 1:
                sentence_end = -1
                last_comma_idx = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    p = self.punc_list[punctuations[i]]
                    if p == "。" or p == "？":
                        sentence_end = i
                        break
                    if last_comma_idx < 0 and p == "，":
                        last_comma_idx = i

                if sentence_end < 0 and len(mini_sentence) > 200 and last_comma_idx >= 0:
                    sentence_end = last_comma_idx
                    punctuations[sentence_end] = self.period_idx

                cache_sent = mini_sentence[sentence_end + 1:]
                cache_sent_id = mini_sentence_id[sentence_end + 1:].tolist()
                mini_sentence = mini_sentence[:sentence_end + 1]
                punctuations = punctuations[:sentence_end + 1]

            # Build words with punctuation
            for i in range(len(mini_sentence)):
                if i > 0:
                    # Add space between two consecutive ASCII words
                    if (len(mini_sentence[i][0].encode()) == 1
                            and len(mini_sentence[i - 1][0].encode()) == 1):
                        result_parts.append(" ")
                result_parts.append(mini_sentence[i])
                p = self.punc_list[punctuations[i]]
                if p != "_":
                    result_parts.append(p)

        # Force sentence to end with period
        result = "".join(result_parts)
        if result and result[-1] in ("，", "、"):
            result = result[:-1] + "。"
        elif result and result[-1] not in ("。", "？"):
            result = result + "。"

        return result


def get_ct_transformer() -> _CTTransformerPunc:
    """Lazy-initialise the CT-Transformer punctuation model (downloads on first use)."""
    global _ct_transformer
    if _ct_transformer is None:
        model_dir = _ensure_punc_model()
        _ct_transformer = _CTTransformerPunc(model_dir)
    return _ct_transformer


def restore_punctuation(text: str) -> str:
    """Run CT-Transformer to add punctuation to a sentence."""
    model = get_ct_transformer()
    return model.add_punctuation(text)
