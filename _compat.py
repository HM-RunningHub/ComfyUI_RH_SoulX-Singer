"""Compatibility shims so this pack runs on a newer dependency stack
than the pinned ``preprocess/requirements.txt``.

This machine runs ComfyUI 0.41 on AMD ROCm with:

* ``transformers`` 5.14  -> ``LlamaConfig(0, 256, 1024, 1, 1)`` (positional
  construction inside ``soulxsinger/models/modules/llama.py``) raises
  ``TypeError`` at *import* time, and ``LlamaAttention.forward`` expects
  ``position_embeddings``.
* ``librosa`` 1.0        -> ``librosa.get_duration(filename=...)`` was removed.
* ``torchaudio`` 2.11    -> ``torchaudio.load`` requires TorchCodec, which has
  no wheel for this ROCm build.

Everything here is a no-op on the dependency versions the pack pins, so the
shims are safe to keep enabled unconditionally.
"""

from __future__ import annotations

import importlib.metadata as _md


def _version(pkg: str) -> tuple[int, int]:
    try:
        parts = _md.version(pkg).split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------

_LEGACY_LLAMA_POS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
)


def _patch_llama_config() -> None:
    """Allow legacy positional construction of ``LlamaConfig``."""
    try:
        from transformers import LlamaConfig
    except Exception:
        return

    if getattr(LlamaConfig, "_soulx_patched", False):
        return

    orig_init = LlamaConfig.__init__

    def init(self, *args, **kwargs):
        if args:
            for name, value in zip(_LEGACY_LLAMA_POS, args):
                kwargs.setdefault(name, value)
            args = ()
        orig_init(self, *args, **kwargs)
        if getattr(self, "_attn_implementation", None) is None:
            try:
                self._attn_implementation = "eager"
            except Exception:
                pass

    LlamaConfig.__init__ = init
    LlamaConfig._soulx_patched = True


def _patch_llama_attention() -> None:
    """Supply ``position_embeddings`` and keep the 3-tuple return contract."""
    if _version("transformers") < (4, 53):
        return

    try:
        import torch
        from transformers.models.llama.modeling_llama import LlamaAttention
    except Exception:
        return

    if getattr(LlamaAttention, "_soulx_patched", False):
        return

    orig_forward = LlamaAttention.forward

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if position_embeddings is None:
            rotary = getattr(self, "rotary_emb", None)
            if rotary is None:
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

                head_dim = getattr(self, "head_dim", None) or (
                    self.config.hidden_size // self.config.num_attention_heads
                )
                rotary = LlamaRotaryEmbedding(
                    config=self.config, head_dim=head_dim
                ).to(hidden_states.device, hidden_states.dtype)
                self.rotary_emb = rotary
            if position_ids is None:
                position_ids = torch.arange(
                    hidden_states.shape[1], device=hidden_states.device
                ).unsqueeze(0)
            position_embeddings = rotary(hidden_states, position_ids)

        out = orig_forward(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(out, tuple) and len(out) == 2:
            return (out[0], out[1], None)
        return out

    LlamaAttention.forward = forward
    LlamaAttention._soulx_patched = True


# ---------------------------------------------------------------------------
# librosa
# ---------------------------------------------------------------------------


def _patch_librosa() -> None:
    """``librosa.get_duration(filename=...)`` -> ``path=``."""
    if _version("librosa") < (1, 0):
        return

    import librosa

    if getattr(librosa.get_duration, "_soulx_patched", False):
        return

    orig = librosa.get_duration

    def get_duration(*args, **kwargs):
        if "filename" in kwargs:
            kwargs.setdefault("path", kwargs.pop("filename"))
        return orig(*args, **kwargs)

    get_duration._soulx_patched = True
    librosa.get_duration = get_duration


# ---------------------------------------------------------------------------
# torchaudio
# ---------------------------------------------------------------------------


def _patch_torchaudio() -> None:
    """soundfile-backed ``torchaudio.load`` when TorchCodec is unavailable."""
    try:
        import torchaudio
    except Exception:
        return

    if getattr(torchaudio.load, "_soulx_patched", False):
        return

    try:
        import torchcodec  # noqa: F401

        return
    except Exception:
        pass

    import numpy as np
    import soundfile as sf
    import torch

    def load(path, *args, **kwargs):
        target_sr = kwargs.get("sample_rate", kwargs.get("sr"))
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wave = torch.from_numpy(np.ascontiguousarray(data.T))
        if target_sr and int(target_sr) != int(sr):
            import torchaudio.functional as F

            wave = F.resample(wave, int(sr), int(target_sr))
            sr = int(target_sr)
        return wave, sr

    load._soulx_patched = True
    torchaudio.load = load

    # torchaudio.save() has the same TorchCodec dependency.
    def save(uri, src, sample_rate, *args, **kwargs):
        data = src.detach().cpu().float().numpy()
        if data.ndim == 2 and data.shape[0] <= 8:  # (channels, frames) -> (frames, channels)
            data = data.T
        sf.write(str(uri), data, int(sample_rate))

    save._soulx_patched = True
    torchaudio.save = save


def apply() -> None:
    _patch_llama_config()
    _patch_llama_attention()
    _patch_librosa()
    _patch_torchaudio()


apply()
