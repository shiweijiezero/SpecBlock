"""LLaMA SpecBlock-Shift draft model for SGLang inference.

Phase 0 wrapper: delegates the actual forward to the proven HF
benchmarks_hf SpecBlockShiftInferenceModel implementation (tree build /
KV cache / shift logic already validated by the HF eval pipeline that
gave 2.27x avg speedup on Qwen3-8B no-think).

Phase 1+ will refactor module-by-module to use SGLang's attention backend
and paged KV cache. For now this file's role is:
  - register the model class so SGLang can find it via EntryClass
  - load training-ckpt safetensors via load_weights
  - expose set_embed_and_head / get_hot_token_id hooks
"""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

# Phase 1.1: self-contained — inference model code lives next to this file
# in _specblock_shift_inference.py (and _specblock_inference.py for the base).
# These were copied from benchmarks_hf/algorithms/* and only the relative
# import inside _specblock_shift_inference.py was rewritten.
from ._specblock_shift_inference import SpecBlockShiftInferenceModel


class LlamaForCausalLMSpecBlockShift(nn.Module):
    """SGLang-compatible wrapper around SpecBlockShiftInferenceModel."""

    config_class = LlamaConfig

    def __init__(
        self,
        config: LlamaConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        # pp_group is fetched lazily in Phase 1 when SGLang server initializes
        # the distributed group; Phase 0 standalone spike doesn't need it.
        self.pp_group = None

        self.inner = SpecBlockShiftInferenceModel(config)

        self.hot_token_id = None
        self.capture_aux_hidden_states = True

    @property
    def K(self):
        return self.inner.K

    @property
    def num_layers(self):
        return self.inner.num_layers

    def forward_with_cache(self, *args, **kwargs):
        return self.inner.forward_with_cache(*args, **kwargs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load weights from training checkpoint.

        Training ckpt's parameter names match SpecBlockShiftInferenceModel's
        structure exactly (input_layer.*, layers.N.*, norm.*, lm_head.*,
        rank_head.*, shift_proj.N.*, t2d, d2t). We strip any leading
        ``model.`` / ``inner.`` / ``module.`` prefix and copy_ tensors in.
        """
        params_dict = dict(self.inner.named_parameters())
        bufs_dict = dict(self.inner.named_buffers())
        loaded, skipped = 0, []

        for name, loaded_weight in weights:
            for pfx in ("model.", "inner.", "module."):
                if name.startswith(pfx):
                    name = name[len(pfx):]

            if name in bufs_dict:
                bufs_dict[name].copy_(loaded_weight)
                if name == "d2t":
                    self.hot_token_id = (
                        loaded_weight
                        + torch.arange(
                            loaded_weight.shape[0],
                            device=loaded_weight.device,
                            dtype=loaded_weight.dtype,
                        )
                    )
                continue

            param = params_dict.get(name)
            if param is None:
                skipped.append(name)
                continue
            with torch.no_grad():
                param.copy_(loaded_weight)
            loaded += 1

        if skipped:
            print(
                f"[LlamaForCausalLMSpecBlockShift] loaded={loaded} "
                f"skipped={len(skipped)} (first 5: {skipped[:5]})"
            )

    def set_embed(self, embed):
        del self.inner.input_layer.embed_tokens.weight
        self.inner.input_layer.embed_tokens.weight = embed

    def set_embed_and_head(self, embed, head):
        """Set embed from target. Draft has its own lm_head (draft_vocab), don't overwrite."""
        del self.inner.input_layer.embed_tokens.weight
        self.inner.input_layer.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_hot_token_id(self):
        return self.hot_token_id


EntryClass = [LlamaForCausalLMSpecBlockShift]
