"""
SpecBlock: Block Causal Speculative Decoding

Architecture: 增加 Rank Head 用于推理时动态决定:
  - Block 长度 M（第一个 rank > 0 的位置停下）
  - 分支因子（class 1→factor=4, class 2→factor=10）
  - 是否继续迭代（class 3 = give up 则终止该分支）

Training uses standard causal attention (each position predicts the next token); 
Block 结构是推理时的概念。

架构：
1. 输入层：3 路融合 (Condition, Token, Query) → H 
2. Decoder：跨位置因果 Attention + MLP 
3. 输出层：
   - LM Head → draft 分布 [B, S, K, V_draft]
   - Rank Head → rank 预测 [B, S, K, 4]（detached normed hidden + top-10 logit values + entropy, 4-class bucket）
"""

import math
from functools import lru_cache
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

_compiled_flex_cache = {}

def get_compiled_flex_attention(kv_len: int):
    """Return a compiled flex_attention function keyed by KV length.

    Each unique KV length (Block 1/2/3) gets its own compiled function,
    avoiding recompilation when block_mask objects differ between blocks.
    """
    if kv_len not in _compiled_flex_cache:
        _compiled_flex_cache[kv_len] = torch.compile(flex_attention)
    return _compiled_flex_cache[kv_len]

from transformers import LlamaConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache

from .base import Eagle3DraftModel


# ============================================================
#                    Cached Block Mask
# ============================================================

@lru_cache(maxsize=8)
def get_cached_causal_block_mask(K: int, batch_size: int, device_str: str):
    device = torch.device(device_str)
    return create_block_mask(
        mask_mod=generate_causal_mask_for_k(K),
        B=batch_size, H=1,
        Q_LEN=K, KV_LEN=K,
        device=device,
    )


@lru_cache(maxsize=8)
def get_cached_cross_position_causal_mask(S: int, K: int, batch_size: int, device_str: str):
    device = torch.device(device_str)
    total_len = S * K
    return create_block_mask(
        mask_mod=generate_cross_position_causal_mask(K),
        B=batch_size, H=1,
        Q_LEN=total_len, KV_LEN=total_len,
        device=device,
    )


@lru_cache(maxsize=16)
def get_cached_cross_position_causal_mask_with_kv(
    S: int, K: int, C: int, batch_size: int, device_str: str
):
    """Get block mask for attention with cached slot_0 KV from previous blocks.

    Args:
        S: sequence length (number of positions)
        K: draft tokens per position
        C: total cached slot_0 count (num_prev_blocks * S)
        batch_size: batch size
        device_str: device string
    """
    device = torch.device(device_str)
    q_len = S * K
    kv_len = C + S * K
    return create_block_mask(
        mask_mod=generate_cross_position_causal_mask_with_cache(K, C, S),
        B=batch_size, H=1,
        Q_LEN=q_len, KV_LEN=kv_len,
        device=device,
    )


def build_ttt_block_mask(
    S: int, K: int, C: int, ttt_valid_counts: torch.Tensor,
    batch_size: int, device: torch.device,
):
    """Build block mask for attention with TTT all-K-slots cache + variable valid mask.

    NOT cached because ttt_valid_counts is data-dependent (varies per batch).

    Args:
        S: sequence length (number of positions)
        K: draft tokens per position
        C: total cached entries = num_prev_blocks * S * K
        ttt_valid_counts: [num_prev_blocks, S] valid slot count per block per position
        batch_size: batch size
        device: torch device

    KV layout: [ttt_cache (C entries)] + [current_block S*K entries]
    ttt_cache layout per block: S positions × K slots = S*K entries
    """
    num_prev_blocks = C // (S * K) if C > 0 else 0
    block_SK = S * K
    stride = K

    # Move to device for mask_mod
    _ttt_valid = ttt_valid_counts.to(device)

    # Clamp bounds for safe indexing (mask_mod evaluates all indices, no short-circuit)
    max_block_idx = max(num_prev_blocks - 1, 0)
    max_cache_pos = S - 1

    def mask_mod(b, h, q_idx, kv_idx):
        q_pos = q_idx // stride

        is_cached = kv_idx < C

        # --- TTT cache region: same position, valid slot only ---
        # Clamp to avoid OOB when kv_idx >= C (result masked out by is_cached)
        block_idx = torch.clamp(kv_idx // block_SK, max=max_block_idx)
        within_block = kv_idx % block_SK
        cache_pos = torch.clamp(within_block // stride, max=max_cache_pos)
        cache_slot = within_block % stride
        ttt_mask = is_cached & (cache_pos == q_pos) & (cache_slot < _ttt_valid[block_idx, cache_pos])

        # --- Current block region: standard cross-position causal ---
        current_kv_idx = kv_idx - C
        kv_pos = current_kv_idx // stride
        kv_slot = current_kv_idx % stride

        cross_ok = (~is_cached) & (kv_pos < q_pos)
        within_ok = (~is_cached) & (kv_pos == q_pos) & (kv_slot <= (q_idx % stride))

        return ttt_mask | cross_ok | within_ok

    mask_mod.__name__ = f"ttt_allslot_K{K}_C{C}_S{S}"

    q_len = S * K
    kv_len = C + S * K
    return create_block_mask(
        mask_mod=mask_mod,
        B=batch_size, H=1,
        Q_LEN=q_len, KV_LEN=kv_len,
        device=device,
    )


def build_block2plus_mask(
    S: int, K: int, C_ttt: int, ttt_valid_counts: torch.Tensor,
    batch_size: int, device: torch.device,
):
    """Block 2+ attention mask: cross from Block 1's all-K-slots + TTT + within-position.

    KV layout: [Block1_allslots (S*K entries)] + [ttt_cache (C_ttt entries)] + [current_block (S*K entries)]
    Q layout:  [current_block (S*K entries)]

    Cross region uses strict < (not <=) because same-position Block 1 KV is already
    in ttt_cache, so <= would double-count.

    Args:
        S: sequence length
        K: draft tokens per position
        C_ttt: total ttt cache entries = num_prev_blocks * S * K
        ttt_valid_counts: [num_prev_blocks, S] valid slot count per block per position.
        batch_size: batch size
        device: torch device
    """
    C_cross = S * K  # Block 1's all K slots, K per position
    C_total = C_cross + C_ttt
    stride = K
    num_prev_blocks = C_ttt // (S * K) if C_ttt > 0 else 0
    block_SK = S * K

    _ttt_valid = ttt_valid_counts.to(device) if ttt_valid_counts is not None else None
    max_block_idx = max(num_prev_blocks - 1, 0)
    max_cache_pos = S - 1

    def mask_mod(b, h, q_idx, kv_idx):
        q_pos = q_idx // stride

        is_cross = kv_idx < C_cross
        is_ttt = (kv_idx >= C_cross) & (kv_idx < C_total)
        is_current = kv_idx >= C_total

        # Region 1: Block 1's all-K-slots → cross-position causal (strictly < q_pos)
        # Layout: [pos0_slot0, pos0_slot1, ..., pos0_slotK-1, pos1_slot0, ...]
        cross_pos = torch.clamp(kv_idx // stride, max=max_cache_pos)
        cross_ok = is_cross & (cross_pos < q_pos)

        # Region 2: TTT cache → same position, valid slots only
        ttt_local = torch.clamp(kv_idx - C_cross, min=0)
        block_idx = torch.clamp(ttt_local // block_SK, min=0, max=max_block_idx)
        within_block = ttt_local % block_SK
        cache_pos = torch.clamp(within_block // stride, min=0, max=max_cache_pos)
        cache_slot = within_block % stride
        ttt_ok = is_ttt & (cache_pos == q_pos) & (cache_slot < _ttt_valid[block_idx, cache_pos])

        # Region 3: Current block → cross-position all-slot + within-position causal
        current_kv_idx = kv_idx - C_total
        kv_pos = current_kv_idx // stride
        kv_slot = current_kv_idx % stride
        cross_current_ok = is_current & (kv_pos < q_pos)
        within_ok = is_current & (kv_pos == q_pos) & (kv_slot <= (q_idx % stride))

        return cross_ok | ttt_ok | cross_current_ok | within_ok

    mask_mod.__name__ = f"block2plus_allslot_K{K}_Cttt{C_ttt}_S{S}"

    q_len = S * K
    kv_len = C_total + S * K
    return create_block_mask(
        mask_mod=mask_mod,
        B=batch_size, H=1,
        Q_LEN=q_len, KV_LEN=kv_len,
        device=device,
    )


# ============================================================
#                      基础组件
# ============================================================

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=8192,
        base=10000,
        device=None,
        scaling_factor=None,
        low_freq_factor=None,
        high_freq_factor=None,
        orig_max_position=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )

        if all(v is not None for v in [scaling_factor, low_freq_factor, high_freq_factor, orig_max_position]):
            low_freq_wavelen = orig_max_position / low_freq_factor
            high_freq_wavelen = orig_max_position / high_freq_factor
            wave_len = 2 * math.pi / inv_freq

            if low_freq_factor != high_freq_factor:
                smooth = (orig_max_position / wave_len - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
            else:
                smooth = 0

            new_freqs = torch.where(
                wave_len < high_freq_wavelen,
                inv_freq,
                torch.where(
                    wave_len > low_freq_wavelen,
                    inv_freq / scaling_factor,
                    (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
                ),
            )
            inv_freq = new_freqs

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings + 20, device, torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(dynamic=True)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================
#                  输入层：三路融合
# ============================================================

class SpecBlockInputLayer(nn.Module):
    """
    输入层：3 路融合到 H

    路 1 - Condition: target hidden (3H → H)
    路 2 - Token: embed(input_ids) (H)
    路 3 - Query: learnable embedding (H)

    融合：3 路 concat (3H) → Linear → H
    """

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        K = getattr(config, 'diffspec_draft_token_num', 7)
        target_H = getattr(config, 'target_hidden_size', H)

        self.K = K
        self.hidden_size = H

        self.condition_proj = nn.Linear(target_H * 3, H, bias=False)
        self.condition_norm = LlamaRMSNorm(H, eps=getattr(config, 'rms_norm_eps', 1e-5))
        self.embed_tokens = nn.Embedding(config.vocab_size, H)
        self.token_norm = LlamaRMSNorm(H, eps=getattr(config, 'rms_norm_eps', 1e-5))
        self.position_queries = nn.Embedding(K, H)
        nn.init.normal_(self.position_queries.weight, std=0.02)
        self.query_norm = LlamaRMSNorm(H, eps=getattr(config, 'rms_norm_eps', 1e-5))
        self.fuse = nn.Linear(H * 3, H, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_draft_condition: bool = False,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        K = self.K
        dtype = hidden_states.dtype

        if use_draft_condition:
            condition = hidden_states
        else:
            # Target hidden is 3H, project to H
            condition = self.condition_proj(hidden_states)
        condition = self.condition_norm(condition)
        condition = condition.unsqueeze(2).expand(-1, -1, K, -1)

        token = self.embed_tokens(input_ids).to(dtype)
        token = self.token_norm(token)
        token = token.unsqueeze(2).expand(-1, -1, K, -1)

        query = self.position_queries.weight.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        query = self.query_norm(query)

        W = self.fuse.weight  # [H, 3H]
        H = W.shape[0]
        output = F.linear(condition, W[:, :H]) + F.linear(token, W[:, H:2*H]) + F.linear(query, W[:, 2*H:])

        return output


# ============================================================
#                Attention（跨位置因果）
# ============================================================

class SpecBlockAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self._init_rope(config)

    def _init_rope(self, config):
        rope_scaling = getattr(config, 'rope_scaling', None)

        if rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=getattr(config, 'max_position_embeddings', 8192),
                base=getattr(config, 'rope_theta', 10000),
            )
        else:
            def rope_get(key, default=None):
                if isinstance(rope_scaling, dict):
                    return rope_scaling.get(key, default)
                return getattr(rope_scaling, key, default)

            scaling_type = rope_get("rope_type", rope_get("type"))

            if scaling_type == "llama3":
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=getattr(config, 'max_position_embeddings', 8192),
                    base=getattr(config, 'rope_theta', 10000),
                    scaling_factor=rope_get("factor", 1.0),
                    low_freq_factor=rope_get("low_freq_factor"),
                    high_freq_factor=rope_get("high_freq_factor"),
                    orig_max_position=rope_get("original_max_position_embeddings"),
                )
            else:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=getattr(config, 'max_position_embeddings', 8192),
                    base=getattr(config, 'rope_theta', 10000),
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_mask,
        position_ids: torch.Tensor,
        max_position: Optional[int] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        slot0_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            kv_cache: cached (K, V) from previous blocks, each [B, num_kv_heads, C, head_dim]
            slot0_indices: unused (kept for API compatibility)

        Returns:
            attn_output: [B, L, H]
            all_kv: (all_k, all_v) each [B, num_kv_heads, S*K, head_dim] (RoPE-rotated)
        """
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if max_position is None:
            max_pos = position_ids.max().item() + 1
        else:
            max_pos = max_position
        cos, sin = self.rotary_emb(q, seq_len=max_pos)
        cos, sin = cos.to(q.device), sin.to(q.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Save all KV for both cross-position cache and TTT cache
        all_k = k  # [B, num_kv_heads, S*K, head_dim]
        all_v = v  # [B, num_kv_heads, S*K, head_dim]

        # If we have cached KV from previous blocks, prepend to current K/V
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)  # [B, num_kv_heads, C+L, head_dim]
            v = torch.cat([kv_cache[1], v], dim=2)  # [B, num_kv_heads, C+L, head_dim]

        attn_output = get_compiled_flex_attention(k.shape[2])(
            query=q,
            key=k.contiguous(),
            value=v.contiguous(),
            block_mask=block_mask,
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(attn_output), (all_k, all_v)


# ============================================================
#                         MLP
# ============================================================

class SpecBlockMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        intermediate = config.intermediate_size

        self.gate_proj = nn.Linear(H, intermediate, bias=False)
        self.up_proj = nn.Linear(H, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, H, bias=False)
        self.act_fn = ACT2FN[getattr(config, 'hidden_act', 'silu')]

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ============================================================
#                     Decoder Layer
# ============================================================

class SpecBlockDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        H = config.hidden_size

        self.input_layernorm = LlamaRMSNorm(H, eps=getattr(config, 'rms_norm_eps', 1e-5))
        self.post_attention_layernorm = LlamaRMSNorm(H, eps=getattr(config, 'rms_norm_eps', 1e-5))
        self.self_attn = SpecBlockAttention(config)
        self.mlp = SpecBlockMLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        block_mask,
        position_ids: torch.Tensor,
        max_position: Optional[int] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        slot0_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            hidden: [B, L, H]
            all_kv: (all_k, all_v) for both cross-position and TTT caching (all K slots)
        """
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden, all_kv = self.self_attn(
            hidden, block_mask, position_ids, max_position,
            kv_cache=kv_cache, slot0_indices=slot0_indices,
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden

        return hidden, all_kv


# ============================================================
#                    Attention Mask
# ============================================================

def generate_causal_mask_for_k(K: int):
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    mask_mod.__name__ = f"causal_mask_K{K}"
    return mask_mod


def generate_cross_position_causal_mask(K: int):
    stride = K

    def mask_mod(b, h, q_idx, kv_idx):
        q_pos = q_idx // stride
        q_slot = q_idx % stride
        kv_pos = kv_idx // stride
        kv_slot = kv_idx % stride

        cross_position_mask = (kv_pos < q_pos)
        within_position_mask = (kv_pos == q_pos) & (kv_slot <= q_slot)

        return cross_position_mask | within_position_mask

    mask_mod.__name__ = f"cross_position_causal_allslot_K{K}"
    return mask_mod


def generate_cross_position_causal_mask_with_cache(K: int, C: int, S: int):
    """
    Mask for current block attending to cached all-K-slot KV + own S*K tokens.

    KV layout: [cached_allslots (C tokens)] + [current_block S*K tokens]
    Q layout:  [current_block S*K tokens]

    C = num_prev_blocks * S * K (total cached all-slot count across all previous blocks).
    Cached tokens at index j correspond to position (j // K) % S from block j // (S*K).

    Rules:
    - Current token at (pos_q, slot_k) attends to:
      1. Cached all-K-slots at position p if p <= pos_q (causal constraint)
      2. Current block: cross-position all-slot + within-position causal
    """
    stride = K

    def mask_mod(b, h, q_idx, kv_idx):
        q_pos = q_idx // stride
        q_slot = q_idx % stride

        is_cached = kv_idx < C

        # Cached: effective position = (kv_idx // K) % S (wraps for multi-block cache)
        cached_pos = (kv_idx // stride) % S
        cached_mask = is_cached & (cached_pos <= q_pos)

        # Current block tokens (offset by C in KV dimension)
        current_kv_idx = kv_idx - C
        kv_pos = current_kv_idx // stride
        kv_slot = current_kv_idx % stride

        cross_position_mask = (~is_cached) & (kv_pos < q_pos)
        within_position_mask = (~is_cached) & (kv_pos == q_pos) & (kv_slot <= q_slot)

        return cached_mask | cross_position_mask | within_position_mask

    mask_mod.__name__ = f"cross_pos_causal_cached_allslot_K{K}_C{C}_S{S}"
    return mask_mod


# ============================================================
#                    完整模型
# ============================================================

class _LlamaForCausalLMSpecBlockBase(Eagle3DraftModel):
    """
    SpecBlock: block-level draft with Rank Head

    增加 rank_head，用 detached normed hidden + top-10 logit features + entropy
    预测 target token 在 draft 分布中的排名（4 分类 bucket）。

    Rank 分类含义：
      - class 0: rank=1, top-1 正确，继续
      - class 1: rank 2-4, 小分叉 (factor=4)
      - class 2: rank 5-10, 大分叉 (factor=10)
      - class 3: rank>10, 放弃，终止该分支
    """

    config_class = LlamaConfig

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.K = getattr(config, 'diffspec_draft_token_num', 7)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = getattr(config, 'draft_vocab_size', 32000)
        self.rank_classes = getattr(config, 'rank_classes', 4)

        # 输入层
        self.input_layer = SpecBlockInputLayer(config)

        # Decoder layers（可配置层数）
        num_layers = getattr(config, 'num_hidden_layers', 1)
        self.layers = nn.ModuleList([
            SpecBlockDecoderLayer(config)
            for _ in range(num_layers)
        ])

        # 输出层：LM Head（draft 分布）
        self.norm = LlamaRMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-5))
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)

        # Rank Head（4 分类 bucket）
        # 输入: detached normed hidden (H) + top10_lp(10) + gaps(3) + cumprob_top1(1) + entropy(1) = H+15
        self.rank_feature_dim = 10 + 3 + 1 + 1  # top-10 log_probs + 3 gaps + cumprob_top1 + entropy
        self.rank_head = nn.Sequential(
            nn.Linear(config.hidden_size + self.rank_feature_dim, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, self.rank_classes, bias=False),
        )

        # Vocab mapping buffers
        self.register_buffer("t2d", torch.ones(self.vocab_size, dtype=torch.bool))
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))

    def forward_parallel(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_draft_condition: bool = False,
        prev_kv_cache: Optional[list] = None,
        ttt_valid_counts: Optional[torch.Tensor] = None,
        cross_kv_cache: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, list]:
        """
        并行前向传播

        Args:
            hidden_states: [B, S, 3H] (target) or [B, S, H] (draft, when use_draft_condition=True)
            input_ids: [B, S]
            attention_mask: [B, S]
            use_draft_condition: if True, hidden_states is draft hidden (H), skip condition_proj
            prev_kv_cache: per-layer TTT cache with ALL K slots from previous blocks.
                List of (cached_k, cached_v) each [B, num_kv_heads, C, head_dim]
                where C = num_prev_blocks * S * K.
                None for the first block.
            ttt_valid_counts: [num_prev_blocks, S] valid slot count per block per position.
                Required when prev_kv_cache is not None. Used to mask ttt attention
                so only valid slots are attended to.
            cross_kv_cache: Block 1's all-K-slot KV for cross-position attention (Block 2+ only).
                Per-layer list of (all_k, all_v) each [B, num_kv_heads, S*K, head_dim].
                None for Block 1.

        Returns:
            logits: [B, S, K, draft_vocab_size] - draft token 分布
            rank_logits: [B, S, K, rank_classes] - rank 预测（detached）
            draft_hidden: [B, S, K, H] - decoder hidden（norm 前），用于 TTT 链式传递
            new_cross_kv: per-layer all-K-slots (K, V) for cross-position caching.
                List of (all_k, all_v) each [B, num_kv_heads, S*K, head_dim].
            new_ttt_kv: same as new_cross_kv (all K slots used for both).
        """
        B, S, _ = hidden_states.shape
        K = self.K
        total_len = S * K

        # ===== 输入层 =====
        inputs = self.input_layer(
            hidden_states, input_ids, attention_mask,
            use_draft_condition=use_draft_condition,
        )

        # Reshape for decoder: [B, S, K, H] → [B, S*K, H]
        hidden = inputs.view(B, total_len, self.hidden_size)

        # ===== Position IDs =====
        # Sequential: position i 的 slot k 使用 position_id = i + 1 + k
        base_pos = torch.arange(1, S + 1, device=hidden.device)
        slot_offsets = torch.arange(K, device=hidden.device)
        position_ids = (base_pos.unsqueeze(1) + slot_offsets.unsqueeze(0)).reshape(-1)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)

        # ===== Pre-compute max position =====
        max_position = S + K

        # ===== Slot_0 indices (precompute once) =====
        slot0_indices = torch.arange(0, total_len, K, device=hidden.device)  # [0, K, 2K, ...]

        # ===== Attention Mask & Combined Cache =====
        if cross_kv_cache is not None:
            # Block 2+: use new mask with Block 1's all-K-slot as cross-position KV
            C_ttt = prev_kv_cache[0][0].shape[2] if prev_kv_cache is not None else 0
            block_mask = build_block2plus_mask(
                S, K, C_ttt, ttt_valid_counts, B, hidden_states.device,
            )
            # Merge cross_kv (all K slots) + ttt_kv into combined cache for attention
            combined_cache = []
            for l in range(len(self.layers)):
                cross_k, cross_v = cross_kv_cache[l]  # [B, n_kv_heads, S*K, head_dim]
                if prev_kv_cache is not None:
                    ttt_k, ttt_v = prev_kv_cache[l]    # [B, n_kv_heads, C_ttt, head_dim]
                    combined_cache.append((
                        torch.cat([cross_k, ttt_k], dim=2),
                        torch.cat([cross_v, ttt_v], dim=2),
                    ))
                else:
                    combined_cache.append((cross_k, cross_v))
        elif prev_kv_cache is not None:
            # Fallback: TTT cache without cross_kv (should not happen in normal flow)
            C = prev_kv_cache[0][0].shape[2]
            block_mask = build_ttt_block_mask(
                S, K, C, ttt_valid_counts, B, hidden_states.device,
            )
            combined_cache = prev_kv_cache
        else:
            # Block 1: standard cross-position causal
            block_mask = get_cached_cross_position_causal_mask(S, K, B, str(hidden_states.device))
            combined_cache = None

        # ===== Decoder Layers =====
        new_all_kv = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = combined_cache[layer_idx] if combined_cache is not None else None
            hidden, all_kv = layer(
                hidden, block_mask, position_ids, max_position,
                kv_cache=layer_cache, slot0_indices=slot0_indices,
            )
            new_all_kv.append(all_kv)

        # ===== 输出层 =====
        # [B, S*K, H] → [B, S, K, H]
        hidden = hidden.view(B, S, K, self.hidden_size)

        # Save pre-norm hidden for TTT chaining
        draft_hidden = hidden

        # LM Head: normal gradient flow through norm
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)  # [B, S, K, draft_vocab_size]

        # Rank Head: top-20 fast path (lse over top-20 ≈ full-vocab lse).
        # Must stay aligned with SpecBlockInferenceModel._rank_forward.
        with torch.no_grad():
            top20_logits, _ = logits.detach().topk(20, dim=-1)
            top20_logits = top20_logits.float()
            top20_lse = torch.logsumexp(top20_logits, dim=-1, keepdim=True)

            top10_logits = top20_logits[..., :10]
            top10_lp = top10_logits - top20_lse
            top10_p = top10_lp.exp()
            gap_12 = top10_logits[..., 0:1] - top10_logits[..., 1:2]
            gap_13 = top10_logits[..., 0:1] - top10_logits[..., 2:3]
            gap_15 = top10_logits[..., 0:1] - top10_logits[..., 4:5]
            cumprob_top1 = top10_p[..., 0:1]

            top20_lp = top20_logits - top20_lse
            top20_p = top20_lp.exp()
            entropy = -(top20_p * top20_lp).nan_to_num(0.0).sum(-1, keepdim=True)

            rank_features = torch.cat(
                [top10_lp, gap_12, gap_13, gap_15, cumprob_top1, entropy], dim=-1,
            ).to(hidden.dtype)
        rank_input = torch.cat([hidden.detach(), rank_features], dim=-1)
        rank_logits = self.rank_head(rank_input)

        # cross_kv and ttt_kv are now the same (all K slots)
        return logits, rank_logits, draft_hidden, new_all_kv, new_all_kv

    # ========== 兼容接口 ==========

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.input_layer.embed_tokens(input_ids)

    @property
    def embed_tokens(self):
        return self.input_layer.embed_tokens

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.input_layer.condition_proj(hidden_states)

    @property
    def fc(self):
        return self.input_layer.condition_proj

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "SpecBlock uses forward_parallel() instead of backbone(). "
            "Call forward_parallel() for parallel multi-token prediction."
        )

    def load_embedding(self, target_model_path: str, embedding_key: str = "model.embed_tokens.weight"):
        import os
        from safetensors import safe_open

        if not os.path.exists(target_model_path):
            from huggingface_hub import snapshot_download
            print(f"Downloading model from HuggingFace: {target_model_path}")
            target_model_path = snapshot_download(repo_id=target_model_path)

        if os.path.isdir(target_model_path):
            safetensor_files = [f for f in os.listdir(target_model_path) if f.endswith('.safetensors')]
        else:
            safetensor_files = [os.path.basename(target_model_path)]
            target_model_path = os.path.dirname(target_model_path)

        for sf in safetensor_files:
            sf_path = os.path.join(target_model_path, sf)
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                if embedding_key in f.keys():
                    embed_weight = f.get_tensor(embedding_key)
                    self.input_layer.embed_tokens.weight.data.copy_(embed_weight)
                    print(f"Loaded embedding from {sf_path}")
                    return

        print(f"Warning: Could not find {embedding_key} in {target_model_path}")

    def freeze_embedding(self):
        self.input_layer.embed_tokens.weight.requires_grad = False

    def load_vocab_mapping(self, vocab_mapping_path: str):
        vocab_mapping = torch.load(vocab_mapping_path, weights_only=False)
        self.t2d.copy_(vocab_mapping["t2d"])
        self.d2t.copy_(vocab_mapping["d2t"])
        self.vocab_mapping_loaded = True
        print(f"Loaded vocab mapping: {self.t2d.sum().item()} mapped tokens")

    def save_pretrained(self, save_directory: str, state_dict=None):
        import os
        import json
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)

        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        if state_dict is None:
            state_dict = {k: v for k, v in self.state_dict().items() if 'embed' not in k.lower()}

        save_file(state_dict, os.path.join(save_directory, 'model.safetensors'))
