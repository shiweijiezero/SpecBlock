"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Inference-only LLaMA-DiffSpec model compatible with training weights.

Supports two modes based on config.diffspec_use_condition_as_token:
1. use_condition_as_token=True (default): H-dimensional input attention
   - QKV input dimension: H (standard LlamaAttention)
2. use_condition_as_token=False: 2*H concat mode (EAGLE3-style)
   - Concatenate embeds and hidden_states
   - QKV input dimension: 2*H

Architecture:
- Multiple decoder layers (num_hidden_layers, default 2)
- 3*H -> H projection for hidden states from target model
- Independent lm_head with draft_vocab_size
"""

import copy
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP
from sglang.srt.utils import add_prefix


class LlamaDiffSpecDecoderLayer(LlamaDecoderLayer):
    """
    DiffSpec Decoder Layer with configurable input mode.

    For use_condition_as_token=True:
        - Input: embeds (batch_size, H) - standard attention
        - QKV input dimension: H
    For use_condition_as_token=False (EAGLE3-style):
        - Input: concat([embeds, condition], dim=-1) = 2*H
        - QKV input dimension: 2*H
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_condition_as_token: bool = True,
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)
        self.use_condition_as_token = use_condition_as_token

        if not use_condition_as_token:
            # Override qkv to accept 2*H input (EAGLE3-style)
            self.self_attn.qkv_proj = QKVParallelLinear(
                2 * self.hidden_size,
                self.self_attn.head_dim,
                self.self_attn.total_num_heads,
                self.self_attn.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("qkv_proj", prefix),
            )

        # MLP with correct intermediate size
        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = LlamaMLP(
            config.hidden_size, inter_size, config.hidden_act, quant_config, prefix
        )

        # Separate norm for condition hidden states
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass following EAGLE3's pattern with fused kernels.

        This is equivalent to training model's logic:
        - Training: residual add + norm + mlp + residual add (manual)
        - Inference: fused (residual add + norm) after attention, then mlp,
                    then fused (residual add + norm) in model.norm()
        """
        # Normalize separately (like EAGLE3)
        embeds_normalized = self.input_layernorm(embeds)
        hidden_states_normalized = self.hidden_norm(hidden_states)

        if self.use_condition_as_token:
            # H-dimensional input mode: just use normalized embeds
            # Residual should be embeds (what we're actually processing)
            residual = embeds
            attn_input = embeds_normalized
        else:
            # 2*H concat mode (EAGLE3-style)
            # Residual should be hidden_states (the condition)
            residual = hidden_states
            attn_input = torch.cat([embeds_normalized, hidden_states_normalized], dim=-1)

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=attn_input,
            forward_batch=forward_batch,
        )

        # Fused: residual add + norm after attention (like EAGLE3)
        # This does: hidden_states = hidden_states + residual, then norm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP (no residual add here, done in model.norm())
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaDiffSpecModel(nn.Module):
    """
    DiffSpec Model backbone with multi-layer support.
    """

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        # Check for use_condition_as_token mode
        self.use_condition_as_token = getattr(config, 'diffspec_use_condition_as_token', True)

        self.is_mrope_enabled = (
            hasattr(config, "rope_scaling")
            and config.rope_scaling is not None
            and "mrope_section" in config.rope_scaling
        )
        if self.is_mrope_enabled:
            config.rope_scaling["rope_type"] = "default"

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Hidden size for condition projection
        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        # Condition projector: 3*H -> H
        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        # Multiple decoder layers
        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList([
            LlamaDiffSpecDecoderLayer(
                config, i, quant_config, prefix,
                use_condition_as_token=self.use_condition_as_token
            )
            for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        """
        Forward pass following EAGLE3's pattern.
        """
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        # Get condition hidden states from spec_info
        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            # Project 3*H -> H
            hidden_states = self.fc(hidden_states)

        # Handle empty batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        # Forward through all decoder layers (like EAGLE3)
        # embeds stays the same, hidden_states is updated each layer
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                embeds,        # embeds stays the same across layers
                hidden_states, # hidden_states is updated (output of previous layer)
                forward_batch,
                residual,      # residual from previous layer
            )

        # Final norm with fused residual add (like EAGLE3)
        # This does the second residual add + norm
        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # Return format: (hidden_states_for_logits, [aux_hidden_states])
        return hidden_states_to_logits, [hidden_states_to_aux]


class LlamaForCausalLMDiffSpec(LlamaForCausalLM):
    """
    DiffSpec Model for speculative decoding inference.
    """

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = LlamaDiffSpecModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # LM head setup
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            draft_vocab_size = getattr(config, 'draft_vocab_size', None)
            if draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
                draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        config_.vocab_size = getattr(config, 'draft_vocab_size', config.vocab_size)
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """
        Load weights from training checkpoint.

        Handles mapping from training model's separate q/k/v projections
        to inference model's combined qkv_proj.
        """
        params_dict = dict(self.named_parameters())

        # Mapping for stacked parameters
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            # Skip vocab mapping buffers
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(
                    loaded_weight.shape[0],
                    device=loaded_weight.device,
                    dtype=loaded_weight.dtype
                )
                continue

            if "t2d" in name or "target_to_draft" in name:
                continue

            # Map old single-layer "midlayer" to "layers.0" for backwards compatibility
            if "midlayer" in name:
                name = name.replace("midlayer", "layers.0")

            # Handle stacked parameters (qkv_proj, gate_up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                # Try with and without model. prefix
                if name in params_dict:
                    full_param_name = name
                elif f"model.{name}" in params_dict:
                    full_param_name = f"model.{name}"
                else:
                    break  # Parameter not found

                param = params_dict[full_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Regular parameters (not stacked)
                if name in params_dict:
                    full_param_name = name
                elif f"model.{name}" in params_dict:
                    full_param_name = f"model.{name}"
                else:
                    continue  # Parameter not found

                param = params_dict[full_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def get_hot_token_id(self):
        return self.hot_token_id

    def set_embed(self, embed):
        """Set embedding weight from target model."""
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed

    def set_embed_and_head(self, embed, head):
        """Set embedding and lm_head weights from target model."""
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        if hasattr(self.lm_head, 'weight'):
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = [LlamaForCausalLMDiffSpec]
