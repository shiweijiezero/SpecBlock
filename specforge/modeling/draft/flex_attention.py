import torch
import torch._dynamo as dynamo
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    or_masks,
)
from transformers.utils import is_torchdynamo_compiling

dynamo.config.recompile_limit = 64


# Reference Implementation https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/flex_attention.py
class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        """
        Initialize or update the singleton instance.
        """
        if not self._is_flex_compiled:
            # Enable dynamic shapes to handle different input sizes
            self._compiled_flex_attention = torch.compile(
                flex_attention,
                # mode="max-autotune-no-cudagraphs",
            )
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    # Do not use compiled version if already compiling forward (it raises issues)
    flex_attention_compiled = (
        WrappedFlexAttention()() if not is_torchdynamo_compiling() else flex_attention
    )
    return flex_attention_compiled(
        query,
        key,
        value,
        **kwargs,
    )


class WrappedCreateBlockMask:
    _instance = None
    _is_create_block_mask_compiled = False
    _compiled_create_block_mask = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        if not self._is_create_block_mask_compiled:
            self._compiled_create_block_mask = torch.compile(create_block_mask)
            self._is_create_block_mask_compiled = True

    def __call__(self):
        return self._compiled_create_block_mask


def compile_friendly_create_block_mask(
    mask_mod,
    B,
    H,
    Q_LEN,
    KV_LEN,
    device,
):
    create_block_mask_compiled = (
        WrappedCreateBlockMask()()
        if not is_torchdynamo_compiling()
        else create_block_mask
    )
    return create_block_mask_compiled(
        mask_mod,
        B,
        H,
        Q_LEN,
        KV_LEN,
        device,
    )


def generate_eagle3_mask(
    seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int, lck: int = 0
):

    def causal_mask(b, h, q_idx, kv_idx):
        # Causal will keep shrinking by 1 diagnol due to appended suffix
        # Shirnk the causal by diagnol
        causal_mask = q_idx >= kv_idx
        padding_mask = (kv_idx < seq_lengths[b]) & (q_idx < seq_lengths[b])
        return causal_mask & padding_mask

    def suffix_mask(b, h, q_idx, kv_idx):
        suffix_mask = kv_idx >= Q_LEN
        padding_mask = kv_idx % Q_LEN < seq_lengths[b]
        diagnol_mask = (kv_idx - q_idx) % Q_LEN == 0
        return suffix_mask & padding_mask & diagnol_mask

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_lck_{lck}"
    return mask_mod


def generate_diffspec_parallel_mask(
    seq_lengths: torch.Tensor,  # [batch] actual sequence lengths
    seq_len: int,  # original sequence length
    K: int,  # number of draft positions
):
    """
    Generate attention mask for block-level parallel decoding.

    The expanded sequence has length seq_len * K.
    Position encoding: expanded_idx = orig_pos * K + draft_pos

    Attention rules for position (i, k) querying (j, m):
    - Can attend if: (j < i) OR (j == i AND m <= k)
    - Plus padding mask based on seq_lengths

    This creates a block-wise causal structure:
    - Full attention to all draft positions of previous original positions
    - Causal attention within the same original position
    """

    def mask_mod(b, h, q_idx, kv_idx):
        # Decode expanded positions
        q_orig = q_idx // K
        q_draft = q_idx % K
        kv_orig = kv_idx // K
        kv_draft = kv_idx % K

        # Causal: can attend to all earlier original positions
        earlier_pos_mask = kv_orig < q_orig

        # Same original position: can only attend to earlier or equal draft positions
        same_pos_mask = (kv_orig == q_orig) & (kv_draft <= q_draft)

        # Combine attention masks
        attend_mask = earlier_pos_mask | same_pos_mask

        # Padding mask
        q_valid = q_orig < seq_lengths[b]
        kv_valid = kv_orig < seq_lengths[b]

        return attend_mask & q_valid & kv_valid

    mask_mod.__name__ = f"diffspec_parallel_mask_seq_{seq_len}_K_{K}"
    return mask_mod
