"""KV Cache Manager for HuggingFace Speculative Decoding.

This module provides utilities for managing KV cache in speculative decoding algorithms.
Key features:
- Uses HuggingFace's DynamicCache for compatibility
- Supports tree-based verification (clone cache for branches)
- Handles cache updates after token acceptance/rejection
"""

import torch
from transformers.cache_utils import DynamicCache
from typing import Optional, List, Tuple


class KVCacheManager:
    """Manager for KV cache in speculative decoding.

    Handles:
    - Cache initialization and reuse across generation
    - Only passing new tokens (not full sequences)
    - Tree verification cache handling (clone for branches)
    - Cache cleanup after acceptance/rejection
    """

    def __init__(self):
        """Initialize the KV cache manager."""
        self.cache: Optional[DynamicCache] = None
        self.original_cache: Optional[DynamicCache] = None

    def reset(self):
        """Reset cache for new generation sequence."""
        self.cache = None
        self.original_cache = None

    def initialize_if_needed(self) -> DynamicCache:
        """Initialize cache if not exists, otherwise return existing.

        Returns:
            DynamicCache object
        """
        if self.cache is None:
            self.cache = DynamicCache()
        return self.cache

    def clone_for_tree_verification(self) -> DynamicCache:
        """Clone current cache for tree verification.

        Tree verification may accept/reject branches, so we need to preserve
        the original cache state before verification.

        Returns:
            Cloned cache for verification
        """
        if self.cache is None:
            return DynamicCache()

        # Save original cache
        self.original_cache = self._deep_clone_cache(self.cache)

        # Return cloned cache for verification
        return self._deep_clone_cache(self.cache)

    def rollback_to_original(self):
        """Rollback to original cache if verification failed.

        Used when tree verification rejects all candidates.
        """
        if self.original_cache is not None:
            self.cache = self.original_cache
            self.original_cache = None

    def commit_verified_cache(self, verified_cache: DynamicCache):
        """Commit verified cache after acceptance.

        Args:
            verified_cache: Cache after tree verification
        """
        self.cache = verified_cache
        self.original_cache = None

    def truncate_to_length(self, length: int):
        """Truncate cache to specified sequence length.

        Used when rejecting speculative tokens.

        Args:
            length: Target sequence length
        """
        if self.cache is None:
            return

        # DynamicCache stores as list of tuples (key, value) per layer
        for layer_idx in range(len(self.cache.key_cache)):
            # key/value shape: [batch_size, num_heads, seq_len, head_dim]
            key = self.cache.key_cache[layer_idx]
            value = self.cache.value_cache[layer_idx]

            if key.shape[2] > length:
                self.cache.key_cache[layer_idx] = key[:, :, :length, :]
                self.cache.value_cache[layer_idx] = value[:, :, :length, :]

    @staticmethod
    def _deep_clone_cache(cache: DynamicCache) -> DynamicCache:
        """Deep clone a DynamicCache object.

        Args:
            cache: Original cache

        Returns:
            Deep cloned cache
        """
        cloned = DynamicCache()

        for key, value in zip(cache.key_cache, cache.value_cache):
            cloned.update(
                key.clone(),
                value.clone(),
                layer_idx=len(cloned.key_cache)
            )

        return cloned

    def get_cache(self) -> Optional[DynamicCache]:
        """Get current cache.

        Returns:
            Current DynamicCache or None
        """
        return self.cache

    def get_cache_length(self) -> int:
        """Get current cache sequence length.

        Returns:
            Sequence length in cache (0 if no cache)
        """
        if self.cache is None or len(self.cache.key_cache) == 0:
            return 0

        # Return sequence length from first layer
        return self.cache.key_cache[0].shape[2]


def create_position_ids(
    start_pos: int,
    length: int,
    batch_size: int = 1,
    device: str = "cuda"
) -> torch.Tensor:
    """Create position IDs for new tokens.

    Args:
        start_pos: Starting position (usually cache length)
        length: Number of new tokens
        batch_size: Batch size
        device: Device for tensor

    Returns:
        Position IDs tensor of shape [batch_size, length]
    """
    position_ids = torch.arange(
        start_pos,
        start_pos + length,
        dtype=torch.long,
        device=device
    )
    return position_ids.unsqueeze(0).expand(batch_size, -1)


def extract_last_hidden_state(
    hidden_states: torch.Tensor,
    num_new_tokens: int
) -> torch.Tensor:
    """Extract the last token's hidden state from model output.

    Args:
        hidden_states: Full hidden states from model output
                      Shape: [batch_size, num_new_tokens, hidden_dim]
        num_new_tokens: Number of tokens that were forwarded

    Returns:
        Last token's hidden state: [batch_size, hidden_dim]
    """
    # Return the last token's hidden state
    return hidden_states[:, -1, :]


def forward_with_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Optional[DynamicCache],
    output_hidden_states: bool = False,
    use_cache: bool = True,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], DynamicCache]:
    """Forward pass with KV cache.

    Args:
        model: HuggingFace model
        input_ids: Input token IDs (only NEW tokens, not full sequence!)
        past_key_values: Previous KV cache
        output_hidden_states: Whether to return hidden states
        use_cache: Whether to use cache
        **kwargs: Additional model arguments

    Returns:
        Tuple of:
        - logits: Model output logits
        - hidden_states: Last layer hidden states (if requested)
        - past_key_values: Updated cache
    """
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        **kwargs
    )

    logits = outputs.logits
    hidden_states = outputs.hidden_states[-1] if output_hidden_states else None
    updated_cache = outputs.past_key_values

    return logits, hidden_states, updated_cache
