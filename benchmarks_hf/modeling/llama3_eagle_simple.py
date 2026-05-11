"""
Simplified EAGLE3 inference wrapper for benchmarks_hf.
Compatible with SpecForge-trained models.

Based on EAGLE-main's cnets.py architecture with SpecForge weight loading.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ModelComponents(nn.Module):
    """Container for model components to match SpecForge structure."""
    def __init__(self, config):
        super().__init__()
        from specforge.modeling.draft.llama3_eagle import LlamaDecoderLayer, LlamaRMSNorm
        from benchmarks_hf.modeling.inference_attention import InferenceAttentionWithCache

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        # Create decoder layer with training weights
        temp_layer = LlamaDecoderLayer(config, attention_backend="sdpa")
        self.midlayer = temp_layer

        # Replace attention with inference-optimized version that supports KV caching
        # Create fresh rotary_emb instead of reusing from training (to avoid cached buffer issues)
        from specforge.modeling.draft.llama3_eagle import LlamaRotaryEmbedding
        rotary_emb = LlamaRotaryEmbedding(
            temp_layer.self_attn.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, 'rope_theta', 10000)
        )

        self.inference_attn = InferenceAttentionWithCache(
            q_proj=temp_layer.self_attn.q_proj,
            k_proj=temp_layer.self_attn.k_proj,
            v_proj=temp_layer.self_attn.v_proj,
            o_proj=temp_layer.self_attn.o_proj,
            rotary_emb=rotary_emb,
            config=config
        )

        # fc layer: projects 3*hidden_size -> hidden_size
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(config.target_hidden_size * 3, config.hidden_size, bias=False)
        else:
            self.fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LlamaForCausalLMEagle3Simple(nn.Module):
    """
    Simplified EAGLE3 inference model.

    Wraps SpecForge training model components for inference use.
    Supports both:
    - Phase 1: Target hidden states (3*hidden_size) -> apply fc
    - Phase 2: Draft hidden states (hidden_size) -> skip fc
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize model components in 'model' namespace to match training structure
        self.model = ModelComponents(config)

        # lm_head at root level
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        # Vocab mapping buffers (will be loaded from checkpoint)
        t2d = torch.zeros(config.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(config.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input token IDs.

        input_ids should be target vocab IDs, matching training behavior.
        """
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass for inference.

        Args:
            input_embeds: Token embeddings (batch, seq_len, hidden_size)
            hidden_states: Either target hidden states (batch, seq_len, hidden_size*3)
                          or draft hidden states (batch, seq_len, hidden_size)
            position_ids: Position IDs (batch, seq_len)
            attention_mask: Optional attention mask
            past_key_values: Optional KV cache tuple (key, value)
            use_cache: Whether to return updated KV cache
            output_attentions: Whether to return attention weights (unused)

        Returns:
            (logits, past_key_values, hidden_states_out)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        expected_hidden_size = self.config.hidden_size

        # Phase 1: Target hidden states (3*hidden_size) -> apply fc
        # Phase 2: Draft hidden states (hidden_size) -> skip fc
        if hidden_dim == expected_hidden_size * 3:
            # Phase 1: Apply fc to project 3*hidden -> hidden
            hidden_states = self.model.fc(hidden_states)
        elif hidden_dim == expected_hidden_size:
            # Phase 2: Already hidden_size, skip fc
            pass
        else:
            raise ValueError(
                f"Invalid hidden_states size: {hidden_dim}. "
                f"Expected {expected_hidden_size} (draft) or {expected_hidden_size * 3} (target)"
            )

        # Custom decoder forward with inference-optimized attention + KV caching
        # Based on EAGLE's LlamaDecoderLayer structure but using InferenceAttentionWithCache

        # Handle hidden_states length mismatch (following EAGLE-main cnets.py:657-661)
        # When hidden_states is shorter (e.g., single token), repeat to match input_embeds
        if hidden_states.shape[1] != input_embeds.shape[1]:
            if hidden_states.shape[1] == 1:
                hidden_states = hidden_states.repeat(1, input_embeds.shape[1], 1)
            else:
                # Take last hidden state and repeat
                hidden_states = hidden_states[:, -1:].repeat(1, input_embeds.shape[1], 1)

        residual = hidden_states

        # Norms (EAGLE-specific: normalize both branches)
        hidden_states = self.model.midlayer.hidden_norm(hidden_states)
        input_embeds = self.model.midlayer.input_layernorm(input_embeds)

        # Concatenate (EAGLE-specific: concat embeds + hidden_states)
        hidden_states = torch.cat((input_embeds, hidden_states), dim=-1)

        # Self-attention with KV caching
        attn_output, new_past_kv = self.model.inference_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,  # ✅ Pass attention_mask (tree_mask support)
        )

        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.model.midlayer.post_attention_layernorm(hidden_states)
        hidden_states = self.model.midlayer.mlp(hidden_states)
        hidden_states_out = residual + hidden_states

        # Compute logits
        norm_hidden = self.model.norm(hidden_states_out)
        logits = self.lm_head(norm_hidden)

        return logits, new_past_kv, hidden_states_out

    def __call__(self, *args, **kwargs):
        """Direct call to forward without PreTrainedModel wrapper."""
        return self.forward(*args, **kwargs)

    def init_tree(self, top_k: int):
        """Initialize tree generation state (EAGLE-main aligned).

        Sets up:
        - stable_kv: KV cache for draft model
        - tree_mask_init: Initial tree mask for Phase 2 (identity matrix)
        - tree_mask: Current tree mask (set during Phase 2)
        - tree_position_ids: Position IDs for tree nodes (all 0 initially)

        Args:
            top_k: Top-k candidates per step (beam width)
        """
        # Get device from model parameters
        device = next(self.parameters()).device

        self.stable_kv = None
        self.tree_mask = None
        # tree_mask_init: [1, 1, topk, topk] identity matrix (each token only attends to itself in new batch)
        self.tree_mask_init = torch.eye(top_k, dtype=torch.float32, device=device)[None, None, :, :]
        # tree_position_ids: [topk] all zeros (same position for parallel candidates)
        self.tree_position_ids = torch.zeros(top_k, dtype=torch.long, device=device)

    def reset_kv(self):
        """Reset KV cache for new generation."""
        self.stable_kv = None
        self.tree_mask = None
