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

# Adapted from Medusa (https://github.com/FasterDecoding/Medusa)
"""Inference-only LLaMA-Medusa model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.utils import add_prefix


class ResBlock(nn.Module):
    """
    A Residual Block module for Medusa heads.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class LlamaModel(nn.Module):
    """
    Medusa model for Llama architecture.

    Original Medusa design:
    - Only uses last layer hidden states from target model
    - NO fc projection layer
    - NO embedding addition
    - Multiple parallel heads (ResBlock(s) + Linear) directly process hidden states

    Key components:
    - embed_tokens: Embedding layer (kept for compatibility, but NOT used in original design)
    - medusa_head: Multiple parallel heads (ResBlock(s) + Linear)
    """

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Original Medusa: NO fc projection layer!
        # Hidden states from target model's last layer are used directly
        if hasattr(config, "target_hidden_size"):
            self.target_hidden_size = config.target_hidden_size
        else:
            self.target_hidden_size = config.hidden_size

        # Medusa-specific parameters
        self.medusa_num_heads = getattr(config, "medusa_num_heads", 4)
        self.medusa_num_layers = getattr(config, "medusa_num_layers", 1)

        # Note: Medusa has NO Transformer layers (unlike EAGLE3's midlayer)
        # Instead, it uses multiple parallel heads

        # We don't create medusa_head here because we need draft_vocab_size
        # which might not be in config yet. It will be created in LlamaForCausalLMMedusa

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass for Medusa model.

        Original Medusa design:
        - Directly uses hidden states from target model's last layer
        - NO projection, NO embedding addition

        Args:
            input_ids: Input token IDs (NOT used in original Medusa)
            positions: Position IDs (unused in Medusa)
            forward_batch: Batch information containing spec_info.hidden_states
            input_embeds: Optional pre-computed embeddings (NOT used in original Medusa)
            pp_proxy_tensors: Pipeline parallel proxy tensors (unused)

        Returns:
            hidden_states: Hidden states ready for Medusa heads
            aux_states: Auxiliary states for further processing
        """
        # Original Medusa: Directly use hidden states from target model's last layer
        # NO projection, NO embedding addition!
        hidden_states = forward_batch.spec_info.hidden_states

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        # Return hidden states directly
        # Medusa heads will process this
        return hidden_states, [hidden_states]


class LlamaForCausalLMMedusa(LlamaForCausalLM):
    """
    LLaMA model with Medusa heads for speculative decoding.

    This model extends LlamaForCausalLM to support Medusa-style
    multi-head speculative decoding.

    Original Medusa design:
    - No Transformer layers (num_hidden_layers = 0)
    - Multiple parallel medusa_heads (ResBlock(s) only, NO independent Linear)
    - Shared lm_head with target model
    - Each head predicts a different future token position
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

        # Medusa requires num_hidden_layers = 0 (no Transformer layers)
        if self.config.num_hidden_layers != 0:
            raise ValueError(
                f"Medusa requires num_hidden_layers = 0, but got {self.config.num_hidden_layers}. "
                "Please update your config.json."
            )

        self.model = LlamaModel(config, quant_config, prefix)

        # Medusa uses vocab_size (shared with target model in original design)
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)

        # Create multiple Medusa heads
        medusa_num_heads = getattr(config, "medusa_num_heads", 4)
        medusa_num_layers = getattr(config, "medusa_num_layers", 1)

        # Use target_hidden_size for Medusa heads (original Medusa uses target model's hidden_size)
        if hasattr(config, "target_hidden_size"):
            head_hidden_size = config.target_hidden_size
        else:
            head_hidden_size = config.hidden_size

        # Original Medusa design: Each head only has ResBlock(s), NO independent Linear layer!
        # The lm_head is shared with target model
        self.medusa_heads = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(head_hidden_size)] * medusa_num_layers),
                )
                for _ in range(medusa_num_heads)
            ]
        )

        # Shared lm_head (will be set to target model's lm_head)
        # Create a placeholder that will be replaced by set_embed_and_head()
        self.lm_head = ParallelLMHead(
            self.draft_vocab_size,
            head_hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )

        # Logits processor for vocabulary mapping (not used in draft generation)
        self.logits_processor = LogitsProcessor(config, self.draft_vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """
        Forward pass returning logits from all Medusa heads.

        Original Medusa design:
        1. Each head processes hidden_states through ResBlock(s)
        2. Then shared lm_head produces logits

        Args:
            input_ids: Input token IDs
            positions: Position IDs
            forward_batch: Batch information
            input_embeds: Optional pre-computed embeddings
            pp_proxy_tensors: Pipeline parallel proxy tensors

        Returns:
            logits: Logits from Medusa heads (stacked)
        """
        # Get combined hidden states from model
        hidden_states, _ = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

        # Apply all Medusa heads (original Medusa design with shared lm_head)
        # Each head independently processes the hidden states through ResBlock(s)
        # Then shared lm_head produces logits
        medusa_logits = []
        for head in self.medusa_heads:
            # Process through ResBlock(s)
            mhidden_states = head(hidden_states)
            # Apply shared lm_head using F.linear (SGLang's ParallelLMHead doesn't support direct call)
            logits = F.linear(mhidden_states, self.lm_head.weight)
            medusa_logits.append(logits)

        # Stack logits from all heads
        # Shape: (num_heads, batch_size, seq_len, vocab_size)
        return torch.stack(medusa_logits, dim=0)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from all Medusa heads given hidden states.

        This is a simplified interface for draft generation that bypasses
        the full forward() method and directly computes logits from hidden states.

        Args:
            hidden_states: Hidden states from target model, shape (bs, hidden_size)

        Returns:
            medusa_logits: Stacked logits from all heads, shape (num_heads, bs, vocab_size)
        """
        medusa_logits = []
        for head in self.medusa_heads:
            # Process through ResBlock(s)
            mhidden_states = head(hidden_states)
            # Apply shared lm_head
            logits = F.linear(mhidden_states, self.lm_head.weight)
            medusa_logits.append(logits)

        # Stack: (num_heads, bs, vocab_size)
        return torch.stack(medusa_logits, dim=0)

    def get_embed_and_head(self):
        """Get embedding and head weights for sharing."""
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """Set embedding and lm_head weights to share with target model.

        Original Medusa design: Share both embedding and lm_head with target model.

        Args:
            embed: Embedding weight tensor (not the nn.Embedding module)
            head: lm_head weight tensor (IMPORTANT: used for shared lm_head in original Medusa)
        """
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed

        # Original Medusa design: Share lm_head with target model
        if head is not None:
            del self.lm_head.weight
            self.lm_head.weight = head

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Load model weights.

        Original Medusa weights structure (from training):
        - model.embed_tokens.weight (optional, usually shared with target)
        - lm_head.weight (shared with target model, NOT loaded from checkpoint)
        - medusa_heads.0.0.linear.weight (Head 0, ResBlock linear)
        - medusa_heads.1.0.linear.weight (Head 1, ResBlock linear)
        - etc.

        Note: Original Medusa design does NOT have independent Linear layers per head.
        The lm_head is shared with target model and set via set_embed_and_head().
        """
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name in params_dict:
                param = params_dict[name]
            elif name.replace("model.", "") in params_dict:
                param = params_dict[name.replace("model.", "")]
            else:
                # Skip weights that don't match (e.g., from base model)
                continue

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


# Register the model
EntryClass = [LlamaForCausalLMMedusa]
