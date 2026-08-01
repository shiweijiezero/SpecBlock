"""
SpecBlock Inference Model with KV Cache

推理专用模型，基于 DiffSpec Inference Model 扩展：
1. 增加 rank_head 输出
2. 增加 use_draft_condition 支持多 block TTT
3. 增加 ttt_cache 支持同一 position 跨 block 的 all-K-slots KV 缓存 + variable mask

与训练模型 (specforge/modeling/draft/llama3_specblock.py) 权重兼容。

两层 cache 语义：
- cross_cache: 所有已解码 position 的 all-K-slot KV（跨 position 注意力），整个生成过程维护
- ttt_cache: 同一 position 前面 block 的 all-K-slots KV，单次 iteration 内累积，结束丢弃
  - 物理上每个 block 存 K 个 slot 的 KV（固定大小）
  - 通过 ttt_mask: [B, C_ttt] bool 控制哪些 slot 有效
  - 语义: "ABC12→M" 而非 "ABC123→M"（只看到有效预测，不看截断后的 slot）
"""

import math
import os
from collections import defaultdict
from functools import lru_cache
from typing import List, Optional, Tuple

# Profile-deep state: accumulates (label, start_event, end_event) tuples for
# in-call attention/MLP/norm timing breakdown. Drained by static_draft.build_tree.
_PROFILE_EVENTS: list = []

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

# Optional sgl_kernel fused kernels for faster draft forward.
# Falls back to native PyTorch when unavailable (e.g., non-CUDA).
try:
    from sgl_kernel import rmsnorm as _sgl_rmsnorm
    from sgl_kernel import silu_and_mul as _sgl_silu_and_mul
    from sgl_kernel import apply_rope_with_cos_sin_cache_inplace as _sgl_apply_rope
    from sgl_kernel import fused_add_rmsnorm as _sgl_fused_add_rmsnorm
    _SGL_KERNEL_AVAILABLE = True
except ImportError:
    _SGL_KERNEL_AVAILABLE = False
    _sgl_apply_rope = None
    _sgl_fused_add_rmsnorm = None

# SpecBlock paged KV pool (worker-local).  Imported lazily to avoid
# pulling sglang.srt.speculative into model code at import time.
try:
    from sglang.srt.speculative.spec_kv_pool import SpecBlockKVPool
    _SPEC_KV_POOL_AVAILABLE = True
except Exception:
    SpecBlockKVPool = None  # type: ignore
    _SPEC_KV_POOL_AVAILABLE = False


def _is_paged_cache(cross_cache) -> bool:
    """True when cross_cache uses paged layout.

    Paged layout::

        [kv_pool, layer_id, count, cross_loc, new_cross_loc, valid_new_slots?]

    Dense layout (legacy)::

        [k_buf, v_buf, count, max_len]
    """
    if not _SPEC_KV_POOL_AVAILABLE or SpecBlockKVPool is None:
        return False
    return len(cross_cache) >= 5 and isinstance(cross_cache[0], SpecBlockKVPool)


def _read_cross_kv_paged(cross_cache):
    """Gather paged cross K/V into [B, n_heads, count, head_dim] views.

    cross_cache layout: ``[kv_pool, layer_id, count, cross_loc, new_cross_loc]``.
    cross_loc shape:
      * 1D ``[count]``: single-req gather, returns [1, n_heads, count, D].
      * 2D ``[B, max_count]``: batched padded gather, returns
        [B, n_heads, max_count, D].  Pad indices should be 0 (the pool
        sentinel slot, always zero -- mask out via full_kv_mask).
    """
    kv_pool = cross_cache[0]
    layer_id = cross_cache[1]
    cross_loc = cross_cache[3]
    if cross_loc is None or cross_loc.numel() == 0:
        # Empty cross — return a single sentinel slot view.  Caller's
        # cross_count guard prevents this from being read.
        sentinel_idx = torch.zeros(1, dtype=torch.int64, device=kv_pool.device)
        k_paged = kv_pool.get_k(layer_id)[sentinel_idx]  # [1, n_heads, D]
        v_paged = kv_pool.get_v(layer_id)[sentinel_idx]
        cross_k = k_paged.permute(1, 0, 2).unsqueeze(0).contiguous()
        cross_v = v_paged.permute(1, 0, 2).unsqueeze(0).contiguous()
        return cross_k, cross_v

    if cross_loc.dim() == 1:
        k_paged = kv_pool.get_k(layer_id)[cross_loc]  # [count, n_heads, D]
        v_paged = kv_pool.get_v(layer_id)[cross_loc]
        cross_k = k_paged.permute(1, 0, 2).unsqueeze(0).contiguous()
        cross_v = v_paged.permute(1, 0, 2).unsqueeze(0).contiguous()
    else:
        # 2D [B, max_count]
        k_paged = kv_pool.get_k(layer_id)[cross_loc]  # [B, max_count, n_heads, D]
        v_paged = kv_pool.get_v(layer_id)[cross_loc]
        cross_k = k_paged.permute(0, 2, 1, 3).contiguous()  # [B, n_heads, max_count, D]
        cross_v = v_paged.permute(0, 2, 1, 3).contiguous()
    return cross_k, cross_v


def _write_cross_kv_paged(cross_cache, new_k, new_v):
    """Write GQA-expanded ``new_k``/``new_v`` to the paged pool at
    ``cross_cache[4]`` (= new_cross_loc).

    new_k / new_v shape:
      * [B, n_heads, K_new, head_dim] (single-pos forward, K_new = K).
      * [B, n_heads, NK_new, head_dim] (forward_batch, NK_new = N*K).

    new_cross_loc shape (must match the K_new flattened dim):
      * [K_new] for single-req single-pos.
      * [N*K_new] flattened for forward_batch.
      * [B, K_new] for batched single-pos (caller handles flatten).
    """
    kv_pool = cross_cache[0]
    layer_id = cross_cache[1]
    new_cross_loc = cross_cache[4] if len(cross_cache) > 4 else None
    if new_cross_loc is None:
        return
    if not torch.is_tensor(new_cross_loc) or new_cross_loc.numel() == 0:
        return

    B, n_heads, K_new, D = new_k.shape
    valid_new_slots = cross_cache[5] if len(cross_cache) > 5 else None
    if new_cross_loc.dim() == 2 and valid_new_slots is not None:
        kv_pool.set_kv_padded(
            layer_id,
            new_cross_loc,
            new_k,
            new_v,
            valid_new_slots,
        )
        return
    if new_cross_loc.dim() == 1:
        # Flatten [B, n_heads, K_new, D] -> [B*K_new, n_heads, D].
        # The caller MUST guarantee new_cross_loc.numel() == B*K_new.
        k_flat = new_k.permute(0, 2, 1, 3).reshape(B * K_new, n_heads, D)
        v_flat = new_v.permute(0, 2, 1, 3).reshape(B * K_new, n_heads, D)
    else:
        # 2D [B, K_new] -> reshape, but K_new must match.
        assert new_cross_loc.shape == (B, K_new), (
            f"new_cross_loc shape {tuple(new_cross_loc.shape)} != (B, K_new) "
            f"= ({B}, {K_new})"
        )
        k_flat = new_k.permute(0, 2, 1, 3).reshape(B * K_new, n_heads, D)
        v_flat = new_v.permute(0, 2, 1, 3).reshape(B * K_new, n_heads, D)
        new_cross_loc = new_cross_loc.reshape(B * K_new)
    kv_pool.set_kv(layer_id, new_cross_loc, k_flat, v_flat)
    # Update count tracker (kept in cross_cache[2] for caller convenience).
    cross_cache[2] = int(cross_cache[2]) + new_cross_loc.numel()


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


# ============================================================
#              Cross-position Causal Mask (cached)
# ============================================================

@lru_cache(maxsize=16)
def _build_cross_position_mask(N: int, K: int, device_str: str) -> torch.Tensor:
    """Build cross-position causal mask for N positions with K slots each."""
    device = torch.device(device_str)
    NK = N * K
    q_idx = torch.arange(NK, device=device)
    q_pos = q_idx // K
    q_slot = q_idx % K

    kv_idx = torch.arange(NK, device=device)
    kv_pos = kv_idx // K
    kv_slot = kv_idx % K

    cross_ok = (kv_pos[None, :] < q_pos[:, None])
    within_ok = (kv_pos[None, :] == q_pos[:, None]) & (kv_slot[None, :] <= q_slot[:, None])

    mask = torch.where(cross_ok | within_ok, 0.0, float('-inf'))
    return mask


# Stream cache mask helper (StreamingLLM-style cache compression for draft forward).
# Variants:
#   "off": disabled
#   "B": near positions keep all K slots; far keep only slot_0 (+ sink)
#   "C": progressive — near32 all K; mid (32..near_win) slot_0+1; far slot_0 only
@lru_cache(maxsize=64)
def _stream_mask_1d(
    cross_count: int, K: int, variant: str,
    near_win: int, sink_pos: int, mid_win: int,
    device_str: str,
) -> torch.Tensor:
    """Build [cross_count] bool mask — True = keep, False = drop in attention."""
    device = torch.device(device_str)
    if variant == "off" or cross_count == 0:
        return torch.ones(cross_count, dtype=torch.bool, device=device)
    idx = torch.arange(cross_count, device=device)
    pos_idx = idx // K
    slot_idx = idx % K
    n_pos = cross_count // K
    near_start = max(0, n_pos - near_win)
    if variant == "B":
        keep = (pos_idx < sink_pos) | (pos_idx >= near_start) | (slot_idx == 0)
    elif variant == "C":
        near32_start = max(0, n_pos - 32)
        mid_start = max(0, n_pos - mid_win)
        keep = (
            (pos_idx < sink_pos) | (pos_idx >= near32_start)
            | ((pos_idx >= mid_start) & (pos_idx < near32_start) & (slot_idx < 2))
            | (slot_idx == 0)
        )
    else:
        keep = torch.ones(cross_count, dtype=torch.bool, device=device)
    return keep


def _stream_mask(
    cross_count: int, K: int, B: int, device,
    variant: str = None, near_win: int = None,
    sink_pos: int = None, mid_win: int = None,
) -> Optional[torch.Tensor]:
    """Return [B, cross_count] mask or None if stream cache disabled."""
    variant = variant if variant is not None else os.environ.get('STREAM_CACHE_VARIANT', 'off')
    if variant == 'off' or cross_count == 0:
        return None
    near_win = near_win if near_win is not None else int(os.environ.get('STREAM_NEAR_WIN', '128'))
    sink_pos = sink_pos if sink_pos is not None else int(os.environ.get('STREAM_SINK_POS', '8'))
    mid_win = mid_win if mid_win is not None else int(os.environ.get('STREAM_MID_WIN', '128'))
    base = _stream_mask_1d(cross_count, K, variant, near_win, sink_pos, mid_win, str(device))
    return base.unsqueeze(0).expand(B, -1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Fused sgl_kernel rmsnorm (CUDA bf16/fp16 fast path).
        # Shape: sgl_kernel.rmsnorm expects [M, H] 2D input — flatten leading dims.
        if (
            _SGL_KERNEL_AVAILABLE
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.is_contiguous()
        ):
            orig_shape = x.shape
            x_flat = x.reshape(-1, orig_shape[-1])
            out = _sgl_rmsnorm(x_flat, self.weight.data, self.eps)
            return out.reshape(orig_shape)
        return F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_pos=8192, base=10000, scaling_factor=None,
                 low_freq_factor=None, high_freq_factor=None, orig_max_position=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        if all(v is not None for v in [scaling_factor, low_freq_factor, high_freq_factor, orig_max_position]):
            low_wavelen = orig_max_position / low_freq_factor
            high_wavelen = orig_max_position / high_freq_factor
            wavelen = 2 * math.pi / inv_freq
            smooth = (orig_max_position / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor + 1e-8)
            smooth = smooth.clamp(0, 1)
            inv_freq = torch.where(wavelen < high_wavelen, inv_freq,
                       torch.where(wavelen > low_wavelen, inv_freq / scaling_factor,
                                   (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # _cos_bf/_sin_bf are lazy bf16 caches; built on first forward when dtype is known.
        self._cos_bf16 = None
        self._sin_bf16 = None
        self._build_cache(max_pos)

    def _build_cache(self, max_pos):
        t = torch.arange(max_pos, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)
        # Invalidate bf16 cache (rebuilt lazily on first forward call).
        self._cos_bf16 = None
        self._sin_bf16 = None
        # Invalidate sgl_kernel cos_sin_cache (rebuilt lazily on first fused-rope call).
        self._sgl_cos_sin_cache = None
        self.max_pos = max_pos

    def get_sgl_cos_sin_cache(self, seq_len, device):
        """Build/cached cos_sin_cache for sgl_kernel.apply_rope_with_cos_sin_cache_inplace.

        sgl_kernel expects shape (max_seq_len, rotary_dim) and dtype float32.
        First rotary_dim/2 columns are cos, last rotary_dim/2 are sin.
        Our self.cos/sin are [1,1,max_pos,head_dim] with both halves identical
        (Neox style); extract the first-half cos/sin and concat.
        """
        cache = self._sgl_cos_sin_cache
        if (cache is None or cache.device != device or cache.shape[0] < seq_len):
            half = self.cos.shape[-1] // 2  # rotary_dim/2
            cos_half = self.cos[0, 0, :, :half].float()   # [max_pos, head_dim/2]
            sin_half = self.sin[0, 0, :, :half].float()
            self._sgl_cos_sin_cache = torch.cat([cos_half, sin_half], dim=-1).to(
                device=device, dtype=torch.float32,
            ).contiguous()
        return self._sgl_cos_sin_cache

    def forward(self, seq_len, dtype=None, device=None):
        if seq_len > self.max_pos:
            self._build_cache(seq_len + 100)
        if dtype is None:
            return self.cos[:, :, :seq_len], self.sin[:, :, :seq_len]
        # Cached dtype-cast copies (one per dtype on current device).
        if self._cos_bf16 is None or self._cos_bf16.dtype != dtype or \
           (device is not None and self._cos_bf16.device != device):
            dev = device if device is not None else self.cos.device
            self._cos_bf16 = self.cos.to(device=dev, dtype=dtype)
            self._sin_bf16 = self.sin.to(device=dev, dtype=dtype)
        return self._cos_bf16[:, :, :seq_len], self._sin_bf16[:, :, :seq_len]


def apply_rope(q, k, cos, sin, pos_ids):
    cos = cos[0, 0]
    sin = sin[0, 0]
    cos = cos[pos_ids].unsqueeze(1)
    sin = sin[pos_ids].unsqueeze(1)

    def rotate(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


# ============================================================
#              SpecBlock Input Layer
# ============================================================

class SpecBlockInputLayer(nn.Module):
    """3-path input fusion with use_draft_condition support.

    Block 1: condition_proj(3H) + token_embed + position_query → fuse → H
    Block 2+: draft_hidden(H) + token_embed + position_query → fuse → H
    """

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        K = getattr(config, 'diffspec_draft_token_num', 7)
        target_H = getattr(config, 'target_hidden_size', H)

        self.K = K
        self.H = H
        self.condition_proj = nn.Linear(target_H * 3, H, bias=False)
        self.condition_norm = RMSNorm(H, getattr(config, 'rms_norm_eps', 1e-5))
        self.embed_tokens = nn.Embedding(config.vocab_size, H)
        self.token_norm = RMSNorm(H, getattr(config, 'rms_norm_eps', 1e-5))
        self.position_queries = nn.Embedding(K, H)
        self.query_norm = RMSNorm(H, getattr(config, 'rms_norm_eps', 1e-5))
        self.fuse = nn.Linear(H * 3, H, bias=False)

    def forward(self, hidden, input_ids, use_draft_condition=False):
        """Single position: [B, 1, 3H|H] → [B, K, H]

        Fused fuse path: cat([cond, tok, query], dim=-1) then single
        F.linear with fuse.weight [H, 3H], replacing 3 separate matmul +
        2 adds. Saves ~150us per call (production profile: input_layer
        was 882us in b2_fwd; matmul count 3 → 1 dominates).
        """
        B, K, H = hidden.shape[0], self.K, self.H
        dtype = hidden.dtype

        if use_draft_condition:
            cond = hidden
        else:
            cond = self.condition_proj(hidden)
        cond = self.condition_norm(cond).expand(-1, K, -1)
        tok = self.embed_tokens(input_ids).to(dtype)
        tok = self.token_norm(tok).expand(-1, K, -1)
        query = self.position_queries.weight[None].expand(B, -1, -1)
        query = self.query_norm(query)

        # Single fused matmul instead of 3 separate F.linear + 2 adds.
        combined = torch.cat([cond, tok, query], dim=-1)  # [B, K, 3H]
        return self.fuse(combined)

    def forward_batch(self, hidden_3h, input_ids):
        """Batch N positions: [B, N, 3H] → [B, N*K, H] (always target condition)"""
        B, N, _ = hidden_3h.shape
        K, H = self.K, self.H
        dtype = hidden_3h.dtype

        cond = self.condition_proj(hidden_3h)
        cond = self.condition_norm(cond).unsqueeze(2).expand(-1, -1, K, -1)
        tok = self.embed_tokens(input_ids).to(dtype)
        tok = self.token_norm(tok).unsqueeze(2).expand(-1, -1, K, -1)
        query = self.position_queries.weight[None, None].expand(B, N, -1, -1)
        query = self.query_norm(query)

        # Single fused matmul instead of 3 separate F.linear + 2 adds.
        combined = torch.cat([cond, tok, query], dim=-1)  # [B, N, K, 3H]
        out = self.fuse(combined)
        return out.reshape(B, N * K, H)


# ============================================================
#              SpecBlock Attention with Cache
# ============================================================

class SpecBlockAttentionWithCache(nn.Module):
    """Attention with cross-position cache and TTT block cache."""

    def __init__(self, config):
        super().__init__()
        self.H = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = self.H // self.n_heads
        self.n_kv_heads = getattr(config, 'num_key_value_heads', self.n_heads)
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.K = getattr(config, 'diffspec_draft_token_num', 7)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # Column boundaries in the fused QKV output.
        self._q_out = self.n_heads * self.head_dim
        self._k_out = self.n_kv_heads * self.head_dim
        self._v_out = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.H, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.H, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.H, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.H, bias=False)
        # Lazy-cached fused QKV weight (built on first fast-path call once all
        # three proj weights are loaded). Replaces 3 matmul launches with 1.
        self._qkv_weight = None

        rope_cfg = getattr(config, 'rope_scaling', None) or {}
        self.rope = RotaryEmbedding(
            self.head_dim,
            max_pos=getattr(config, 'max_position_embeddings', 8192),
            base=getattr(config, 'rope_theta', 10000),
            scaling_factor=rope_cfg.get('factor'),
            low_freq_factor=rope_cfg.get('low_freq_factor'),
            high_freq_factor=rope_cfg.get('high_freq_factor'),
            orig_max_position=rope_cfg.get('original_max_position_embeddings'),
        )

        self._causal_mask = None

    def _qkv_proj(self, x):
        """Fused Q/K/V projection (1 matmul instead of 3).

        Lazily builds a fused weight on first fast-path call. Falls back to
        separate projections if bf16/fp16 fused weight can't be used.

        Dual-GPU fix (2026-04-24 Day 24): rebuild if device mismatch.
        `.to(device)` on nn.Module moves params + buffers but NOT arbitrary
        cached attributes like `self._qkv_weight`. When draft_train is
        .to(cuda:1) for dual-GPU training, this cache stays on cuda:0 →
        F.linear device mismatch. Solution: detect x.device vs cached
        device, rebuild if different.
        """
        # Skip fused-cat path when weights are torchao quantized
        # (AffineQuantizedTensor doesn't support aten.cat).
        q_w = self.q_proj.weight
        is_quantized = (
            type(q_w).__name__ in ("AffineQuantizedTensor",)
            or not isinstance(q_w, torch.Tensor)
        )
        if is_quantized:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            return q, k, v
        if self._qkv_weight is None or self._qkv_weight.device != x.device:
            self._qkv_weight = torch.cat([
                self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
            ], dim=0).contiguous()
        qkv = F.linear(x, self._qkv_weight)
        q_out = self._q_out
        k_out = self._k_out
        q = qkv[..., :q_out]
        k = qkv[..., q_out:q_out + k_out]
        v = qkv[..., q_out + k_out:]
        return q, k, v

    def _get_causal_mask(self, device):
        if self._causal_mask is None or self._causal_mask.device != device:
            self._causal_mask = torch.triu(
                torch.full((self.K, self.K), float('-inf'), device=device), diagonal=1
            )
        return self._causal_mask

    @staticmethod
    def _ensure_cache_buf(cache, needed, template):
        """Lazily init or grow pre-allocated cache buffer."""
        if cache[0] is None:
            max_len = cache[3] if len(cache) > 3 else 2048
            max_len = max(max_len, needed)
            cache[0] = torch.zeros(
                template.shape[0], template.shape[1], max_len, template.shape[3],
                device=template.device, dtype=template.dtype,
            )
            cache[1] = torch.zeros_like(cache[0])
        elif needed > cache[0].shape[2]:
            old_count = cache[2]
            new_max = max(cache[0].shape[2] * 2, needed)
            new_k = torch.zeros(
                *cache[0].shape[:2], new_max, cache[0].shape[3],
                device=cache[0].device, dtype=cache[0].dtype,
            )
            new_v = torch.zeros_like(new_k)
            if old_count > 0:
                new_k[:, :, :old_count, :] = cache[0][:, :, :old_count, :]
                new_v[:, :, :old_count, :] = cache[1][:, :, :old_count, :]
            cache[0], cache[1] = new_k, new_v
            if len(cache) > 3:
                cache[3] = new_max

    def forward(self, x, cross_cache, pos_ids, max_position: int = None,
                ttt_cache=None, ttt_mask=None, update_cross_cache=True,
                full_kv_mask=None):
        """Single position forward with optional TTT cache.

        Args:
            x: [B, K, H]
            cross_cache: [k_buf, v_buf, count, max_len] - pre-allocated all-K-slot cache
            pos_ids: [B, K] position ids
            max_position: for RoPE
            ttt_cache: (ttt_k, ttt_v) each [B, n_kv_heads, C_ttt, head_dim]
            ttt_mask: [B, C_ttt] bool mask, True for valid entries
            update_cross_cache: if True, append all K slots to cross_cache
            full_kv_mask: optional pre-built bool mask of shape [B, cross_count + C_ttt].
                When provided, skips the internal torch.cat([ones, ttt_mask]).
                Safe to share across layers (identical mask applied at every layer).

        Returns:
            output: [B, K, H]
            all_kv: (all_k, all_v) each [B, n_kv_heads, K, head_dim]
        """
        B, K, _ = x.shape

        _deep = os.environ.get("STATIC_DRAFT_PROFILE_DEEP", "0") == "1"
        # 3-way path: skip cat, pass cross/ttt/curr separately to attention kernel.
        # Default ON (TREE_ATTN_3WAY='1'). Falls back to cat path when:
        #   - TREE_ATTN_3WAY=0 explicitly set
        #   - sdpa fallback explicitly requested (TREE_ATTN_SDPA=1)
        #   - tree_K=0 (block-1 single-position calls; cat is cheap there)
        _use_3way = (
            os.environ.get("TREE_ATTN_3WAY", "1") == "1"
            and self.head_dim == 128
            and os.environ.get("TREE_ATTN_SDPA", "0") != "1"
        )

        if _deep:
            _ev_qkv_s = torch.cuda.Event(enable_timing=True); _ev_qkv_s.record()

        q_flat, k_flat, v_flat = self._qkv_proj(x)

        if _deep:
            _ev_qkv_e = torch.cuda.Event(enable_timing=True); _ev_qkv_e.record()
            _PROFILE_EVENTS.append(("qkv_proj", _ev_qkv_s, _ev_qkv_e))
            _ev_rope_s = torch.cuda.Event(enable_timing=True); _ev_rope_s.record()

        # Avoid pos_ids.max().item() host sync in hot path. All production callers
        # pass max_position explicitly; fallback to RoPE cache ceiling when absent.
        rope_len = (max_position + 1) if max_position is not None else self.rope.max_pos

        # Fused RoPE via sgl_kernel.apply_rope_with_cos_sin_cache_inplace when
        # available (single kernel vs ~10 PyTorch ops). Opt out via
        # ROPE_FUSED=0; falls back to apply_rope() PyTorch path.
        _use_fused_rope = (
            _SGL_KERNEL_AVAILABLE
            and _sgl_apply_rope is not None
            and q_flat.is_cuda
            and q_flat.dtype in (torch.bfloat16, torch.float16)
            and os.environ.get("ROPE_FUSED", "1") == "1"
        )
        if _use_fused_rope:
            # q_flat shape [B, K, n_heads*head_dim], k_flat [B, K, n_kv_heads*head_dim]
            # sgl_kernel expects (nnz, num_heads * head_size) flat.
            cos_sin_cache = self.rope.get_sgl_cos_sin_cache(rope_len, q_flat.device)
            # In-place modify q_flat / k_flat.
            _sgl_apply_rope(
                positions=pos_ids.reshape(-1),
                query=q_flat.view(B * K, self.n_heads * self.head_dim),
                key=k_flat.view(B * K, self.n_kv_heads * self.head_dim),
                head_size=self.head_dim,
                cos_sin_cache=cos_sin_cache,
                is_neox=True,
            )
            q = q_flat.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
            k = k_flat.view(B, K, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v_flat.view(B, K, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            q = q_flat.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
            k = k_flat.view(B, K, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v_flat.view(B, K, self.n_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.rope(rope_len, dtype=q.dtype, device=q.device)
            q, k = apply_rope(q, k, cos, sin, pos_ids)

        # Save for TTT cache (before GQA repeat)
        all_k = k  # [B, n_kv_heads, K, head_dim] (post-RoPE, non-expanded)
        all_v = v

        # Defer repeat_kv: 3WAY path passes non-expanded all_k/v directly to kernel
        # (kernel does pid_h // n_kv_groups for curr, same as ttt). Legacy/SDPA cat
        # path repeats below.

        if _deep:
            _ev_rope_e = torch.cuda.Event(enable_timing=True); _ev_rope_e.record()
            _PROFILE_EVENTS.append(("rope_repeat", _ev_rope_s, _ev_rope_e))

        cross_count = cross_cache[2]
        has_ttt = ttt_cache is not None
        is_paged = _is_paged_cache(cross_cache)

        # ---- 3-way path: no cat, attention kernel reads from 3 buffers ----
        if _use_3way:
            from ._tree_attention_triton import (
                three_part_attention,
                three_part_attention_paged,
            )
            if _deep:
                _ev_attn_s = torch.cuda.Event(enable_timing=True); _ev_attn_s.record()
                # Track kv_cat as 0 (skipped) to keep profile output structure stable.
                _ev_cat_dummy = torch.cuda.Event(enable_timing=True); _ev_cat_dummy.record()
                _PROFILE_EVENTS.append(("kv_cat", _ev_cat_dummy, _ev_cat_dummy))
            ttt_count_eff = ttt_cache[0].shape[2] if has_ttt else 0
            cross_count_int = (
                int(cross_count) if not torch.is_tensor(cross_count) else cross_count.item()
            )
            if is_paged:
                kv_pool = cross_cache[0]
                layer_id = cross_cache[1]
                cross_loc = cross_cache[3]
                # 1D loc (single-req): unsqueeze to [1, count] for the paged
                # wrapper which expects [B, count].  2D already shaped right.
                if cross_loc.dim() == 1:
                    cross_loc_2d = cross_loc.unsqueeze(0)
                else:
                    cross_loc_2d = cross_loc

                # ---- Stage D: flashinfer paged cross + Triton ttt+curr merge ----
                # Path is taken when (a) the worker initialised a
                # SpecBlockFlashInferCross instance (kv_pool's
                # `flashinfer_cross` attr is non-None, gated by
                # SPECBLOCK_FLASHINFER=1), and (b) the cache carries
                # per-row cross_counts (cross_cache[5]) so we can strip
                # padding before passing to flashinfer.
                flashinfer_cross = getattr(kv_pool, "flashinfer_cross", None)
                per_row_counts = (
                    cross_cache[5]
                    if (len(cross_cache) >= 6 and cross_cache[5] is not None)
                    else None
                )
                use_flashinfer = (
                    flashinfer_cross is not None
                    and per_row_counts is not None
                    and os.environ.get("SPECBLOCK_FLASHINFER", "0") == "1"
                )

                if use_flashinfer:
                    from ._tree_attention_triton import (
                        two_part_attention,
                        merge_attn_outputs,
                    )
                    # Cross stage via flashinfer.  The plan is
                    # pre-built once per decode step in
                    # _batched_block2_forward (cross_loc / counts are
                    # layer-invariant); here we only run().
                    # q layout: [B, n_heads, K, D] -> [B, K, n_heads, D]
                    # -> [B*K, n_heads, D] (flashinfer's token-major).
                    B_q, n_heads_q, K_q, D_q = q.shape
                    q_flat = (
                        q.permute(0, 2, 1, 3).reshape(B_q * K_q, n_heads_q, D_q)
                    )
                    pool_k_layer_3d = kv_pool.get_k(layer_id)
                    pool_v_layer_3d = kv_pool.get_v(layer_id)
                    out_cross_flat, lse_cross_flat = flashinfer_cross.run_layer(
                        q_flat, pool_k_layer_3d, pool_v_layer_3d,
                    )
                    out_cross = (
                        out_cross_flat.reshape(B_q, K_q, n_heads_q, D_q)
                        .permute(0, 2, 1, 3)
                        .contiguous()
                    )
                    lse_cross = (
                        lse_cross_flat.reshape(B_q, K_q, n_heads_q)
                        .permute(0, 2, 1)
                        .contiguous()
                    )
                    out_other, lse_other = two_part_attention(
                        q,
                        ttt_cache[0] if has_ttt else None,
                        ttt_cache[1] if has_ttt else None,
                        all_k, all_v,
                        ttt_count=ttt_count_eff,
                        ttt_mask=ttt_mask if has_ttt else None,
                        k_slots=K, n_kv_groups=self.n_kv_groups, scale=self.scale,
                    )
                    out = merge_attn_outputs(
                        out_cross, lse_cross, out_other, lse_other,
                    )
                else:
                    # Paged kernel reads cross K/V directly from the pool via
                    # cross_loc indices -- no fancy-index gather, so cuda graph
                    # private pool stays in KB-class instead of GB-class.
                    if full_kv_mask is not None and cross_count_int > 0:
                        cross_mask_view = full_kv_mask[..., :cross_count_int]
                    else:
                        cross_mask_view = None
                    pool_k_layer = kv_pool.get_k(layer_id)
                    pool_v_layer = kv_pool.get_v(layer_id)
                    if cross_count_int == 0:
                        # Provide a length-1 sentinel slice so the kernel's
                        # n_in_bounds mask trivially zero-fills the cross loop.
                        cross_loc_2d = torch.zeros(
                            cross_loc_2d.shape[0], 1, dtype=torch.int64, device=q.device,
                        )
                        cross_count_int = 1
                        cross_mask_view = torch.zeros(
                            cross_loc_2d.shape[0], 1, dtype=torch.bool, device=q.device,
                        )
                    request_batch = int(cross_loc_2d.shape[0])
                    if request_batch != B:
                        # Captured block-2 expands a fixed pending-leaf bucket
                        # per request.  Group those leaves in one attention
                        # launch so each prefix K/V tile is shared across
                        # multiple leaves instead of reloaded B_pending times.
                        from ._tree_attention_triton import (
                            grouped_three_part_attention_paged,
                        )

                        if B % request_batch != 0:
                            raise RuntimeError(
                                "Grouped SpecBlock attention has inconsistent "
                                f"batch geometry: flat_batch={B}, "
                                f"request_batch={request_batch}."
                            )
                        leaves_per_request = B // request_batch
                        if not has_ttt or ttt_mask is None:
                            raise RuntimeError(
                                "Grouped SpecBlock block-2 attention requires "
                                "request-level TTT KV and leaf masks."
                            )
                        out = grouped_three_part_attention_paged(
                            q,
                            pool_k_layer, pool_v_layer,
                            cross_loc_2d, cross_mask_view,
                            ttt_cache[0], ttt_cache[1],
                            all_k, all_v,
                            ttt_mask,
                            leaves_per_request=leaves_per_request,
                            cross_count=cross_count_int,
                            k_slots=K,
                            n_kv_groups=self.n_kv_groups,
                            scale=self.scale,
                        )
                    else:
                        out = three_part_attention_paged(
                            q,
                            pool_k_layer, pool_v_layer,
                            cross_loc_2d, cross_mask_view,
                            ttt_cache[0] if has_ttt else None,
                            ttt_cache[1] if has_ttt else None,
                            all_k, all_v,
                            cross_count=cross_count_int,
                            ttt_count=ttt_count_eff,
                            ttt_mask=ttt_mask if has_ttt else None,
                            k_slots=K,
                            n_kv_groups=self.n_kv_groups,
                            scale=self.scale,
                        )
            else:
                cross_k_view = cross_cache[0][:, :, :cross_count, :] if cross_count > 0 else cross_cache[0][:, :, :1, :]
                cross_v_view = cross_cache[1][:, :, :cross_count, :] if cross_count > 0 else cross_cache[1][:, :, :1, :]
                out = three_part_attention(
                    q,
                    cross_k_view, cross_v_view,
                    ttt_cache[0] if has_ttt else None,
                    ttt_cache[1] if has_ttt else None,
                    all_k, all_v,   # GQA-non-expanded; kernel does GQA index via pid_h // n_kv_groups
                    cross_count=cross_count_int,
                    ttt_count=ttt_count_eff,
                    ttt_mask=ttt_mask if has_ttt else None,
                    k_slots=K, n_kv_groups=self.n_kv_groups, scale=self.scale,
                )
            if _deep:
                _ev_attn_e = torch.cuda.Event(enable_timing=True); _ev_attn_e.record()
                _PROFILE_EVENTS.append(("attn_compute", _ev_attn_s, _ev_attn_e))
                _ev_op_s = torch.cuda.Event(enable_timing=True); _ev_op_s.record()
            # Update cross cache (same as legacy path)
            if update_cross_cache:
                new_k = repeat_kv(all_k, self.n_kv_groups)
                new_v = repeat_kv(all_v, self.n_kv_groups)
                if is_paged:
                    _write_cross_kv_paged(cross_cache, new_k, new_v)
                else:
                    self._ensure_cache_buf(cross_cache, cross_count + K, new_k)
                    cross_cache[0][:, :, cross_count:cross_count + K, :] = new_k
                    cross_cache[1][:, :, cross_count:cross_count + K, :] = new_v
                    cross_cache[2] = cross_count + K
            out_proj = self.o_proj(out.transpose(1, 2).contiguous().view(B, K, -1))
            if _deep:
                _ev_op_e = torch.cuda.Event(enable_timing=True); _ev_op_e.record()
                _PROFILE_EVENTS.append(("o_proj", _ev_op_s, _ev_op_e))
            return out_proj, (all_k, all_v)

        # Legacy/SDPA path: needs GQA-expanded current k/v for cat
        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)

        if _deep:
            _ev_cat_s = torch.cuda.Event(enable_timing=True); _ev_cat_s.record()

        # Build full KV: [cross_cache | ttt_cache | current]
        kv_parts_k = []
        kv_parts_v = []

        if cross_count > 0:
            if is_paged:
                _ck, _cv = _read_cross_kv_paged(cross_cache)
                kv_parts_k.append(_ck)
                kv_parts_v.append(_cv)
            else:
                kv_parts_k.append(cross_cache[0][:, :, :cross_count, :])
                kv_parts_v.append(cross_cache[1][:, :, :cross_count, :])

        if has_ttt:
            kv_parts_k.append(repeat_kv(ttt_cache[0], self.n_kv_groups))
            kv_parts_v.append(repeat_kv(ttt_cache[1], self.n_kv_groups))

        C = sum(p.shape[2] for p in kv_parts_k)  # total cached length

        kv_parts_k.append(k)
        kv_parts_v.append(v)

        if len(kv_parts_k) > 1:
            k_full = torch.cat(kv_parts_k, dim=2)  # [B, H, C+K, D]
            v_full = torch.cat(kv_parts_v, dim=2)
        else:
            k_full, v_full = k, v

        if _deep:
            _ev_cat_e = torch.cuda.Event(enable_timing=True); _ev_cat_e.record()
            _PROFILE_EVENTS.append(("kv_cat", _ev_cat_s, _ev_cat_e))
            _ev_attn_s = torch.cuda.Event(enable_timing=True); _ev_attn_s.record()

        use_sdpa_single = os.environ.get('TREE_ATTN_SDPA', '0') == '1'
        use_triton_single = (
            os.environ.get('TREE_ATTN_TRITON', '0') == '1'
            and self.head_dim == 128
        )

        if use_sdpa_single and C > 0:
            # Unified path: everything in [0:C] is "masked region" (cross fully
            # visible, ttt per ttt_mask, or caller-supplied full_kv_mask), then
            # current K is causal within slot.
            if full_kv_mask is not None:
                _kv_mask = full_kv_mask
            elif has_ttt:
                if cross_count > 0:
                    _kv_mask = torch.cat(
                        [ttt_mask.new_ones(B, cross_count), ttt_mask], dim=1,
                    )
                else:
                    _kv_mask = ttt_mask
            else:
                _kv_mask = torch.ones(B, C, dtype=torch.bool, device=q.device)
            # Build additive attention bias [B, 1, K, C+K]: 0 for visible, -inf for masked.
            bias = torch.zeros(B, 1, K, C + K, dtype=q.dtype, device=q.device)
            bias[:, :, :, :C] = torch.where(
                _kv_mask[:, None, None, :], 0.0, float('-inf'),
            ).to(q.dtype)
            # Current K region: causal (upper-triangular -inf) within slots
            bias[:, :, :, C:] = self._get_causal_mask(q.device).to(q.dtype)[None, None]
            out = F.scaled_dot_product_attention(
                q, k_full, v_full,
                attn_mask=bias,
                scale=self.scale,
            )
        elif use_sdpa_single and C == 0:
            # No cache; causal-only within current K
            bias = self._get_causal_mask(q.device).to(q.dtype)[None, None]
            out = F.scaled_dot_product_attention(
                q, k_full, v_full,
                attn_mask=bias,
                scale=self.scale,
            )
        elif use_triton_single:
            from ._tree_attention_triton import single_pos_attention
            if full_kv_mask is not None:
                _kv_mask = full_kv_mask
                _kv_count = full_kv_mask.shape[1]
            elif has_ttt:
                # Build combined cross+ttt mask: ones for cross region, ttt_mask for ttt.
                if cross_count > 0:
                    _kv_mask = torch.cat(
                        [ttt_mask.new_ones(B, cross_count), ttt_mask], dim=1,
                    )
                else:
                    _kv_mask = ttt_mask
                _kv_count = _kv_mask.shape[1]
            else:
                # No TTT: cross region only. Fully visible, represent with all-ones mask.
                _kv_mask = torch.ones(
                    B, cross_count, dtype=torch.bool, device=q.device,
                ) if cross_count > 0 else None
                _kv_count = cross_count
            out = single_pos_attention(
                q, k_full, v_full,
                cross_count=0,
                ttt_count=_kv_count,
                ttt_mask=_kv_mask,
                k_slots=K,
                scale=self.scale,
            )
        else:
            # Single Q @ K^T (merged, 1 matmul instead of 2)
            attn = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale  # [B, H, K, C+K]

            # Apply masks
            if has_ttt:
                # TTT region needs variable mask; cross region is fully visible.
                # Fast path: caller supplied a pre-built combined mask shared across layers.
                if full_kv_mask is not None:
                    full_mask = full_kv_mask
                elif cross_count > 0:
                    full_mask = torch.cat([
                        ttt_mask.new_ones(B, cross_count),
                        ttt_mask,
                    ], dim=1)  # [B, C]
                else:
                    full_mask = ttt_mask
                attn[..., :C].masked_fill_(~full_mask[:, None, None, :], float('-inf'))
            elif full_kv_mask is not None and C > 0:
                # Block-1 graph path: no TTT but caller supplied a cross-only mask
                # (used when cross_cache is padded to a bucket size and the tail is
                # uninitialized — the mask keeps padded positions from contributing).
                attn[..., :C].masked_fill_(~full_kv_mask[:, None, None, :], float('-inf'))
            # Cross-only unmasked: fully visible, no mask needed on cached region

            # StreamingLLM-style cache compression: drop slot_1-3 of far positions.
            # Applied on top of any existing mask (conjunctive). Env-gated.
            stream_cross_mask = _stream_mask(cross_count, K, B, q.device)
            if stream_cross_mask is not None:
                attn[..., :cross_count].masked_fill_(
                    ~stream_cross_mask[:, None, None, :], float('-inf'),
                )

            # Causal mask for current K slots
            attn[..., C:] += self._get_causal_mask(q.device)

            # Single softmax + single attn @ V (1 matmul instead of 2)
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(attn, v_full)

        if _deep:
            _ev_attn_e = torch.cuda.Event(enable_timing=True); _ev_attn_e.record()
            _PROFILE_EVENTS.append(("attn_compute", _ev_attn_s, _ev_attn_e))
            _ev_op_s = torch.cuda.Event(enable_timing=True); _ev_op_s.record()

        # Update cross cache: store all K slots (pre-allocated buffer)
        if update_cross_cache:
            new_k = repeat_kv(all_k, self.n_kv_groups)   # [B, n_heads, K, head_dim]
            new_v = repeat_kv(all_v, self.n_kv_groups)
            if is_paged:
                _write_cross_kv_paged(cross_cache, new_k, new_v)
            else:
                self._ensure_cache_buf(cross_cache, cross_count + K, new_k)
                cross_cache[0][:, :, cross_count:cross_count + K, :] = new_k
                cross_cache[1][:, :, cross_count:cross_count + K, :] = new_v
                cross_cache[2] = cross_count + K

        out_proj = self.o_proj(out.transpose(1, 2).contiguous().view(B, K, -1))
        if _deep:
            _ev_op_e = torch.cuda.Event(enable_timing=True); _ev_op_e.record()
            _PROFILE_EVENTS.append(("o_proj", _ev_op_s, _ev_op_e))
        return out_proj, (all_k, all_v)

    def forward_batch(self, x, cache, pos_ids, max_position: int, N: int,
                      return_last_kv: bool = False,
                      cross_mask: Optional[torch.Tensor] = None,
                      n_per_req: Optional[torch.Tensor] = None,
                      flashinfer_cross=None):
        """Batch forward for N positions (prefill/cache update). No TTT cache.

        Args:
            return_last_kv: If True, also return (last_k, last_v) for the last
                position's K slots (before GQA repeat), for TTT cache seeding.
            cross_mask: Optional [B, max_cross] bool tensor (True = valid cross
                slot, False = padded).  When provided, ``cache[2]`` is treated
                as the *padded* ``max_cross`` length and per-req variable
                cross_counts are masked out via additive -inf bias on the
                attention scores.  Triton fast-path falls back to eager when
                cross_mask is provided (Triton kernel has no per-row mask).
            n_per_req: Optional [B] long tensor giving the *real* N_i per row
                (when N is the padded ``max_N``).  Required if return_last_kv
                is True under padded batching, so last_kv is gathered at
                position ``(N_i - 1) * K`` rather than the padded tail.
            flashinfer_cross: Optional pre-planned :class:`SpecBlockFlashInferCross`
                wrapper.  When provided, cross attention is computed via
                FlashInfer's paged ragged kernel (compute = sum(c_i) instead
                of B*max_cross).  Current K-slot tree-mask attention is
                computed separately (eager) and LSE-merged with cross.
        """
        B, NK, _ = x.shape
        K = self.K

        q_flat, k_flat, v_flat = self._qkv_proj(x)
        q = q_flat.view(B, NK, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_flat.view(B, NK, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v_flat.view(B, NK, self.n_kv_heads, self.head_dim).transpose(1, 2)

        rope_len = max_position + 1
        cos, sin = self.rope(rope_len, dtype=q.dtype, device=q.device)
        q, k = apply_rope(q, k, cos, sin, pos_ids)

        # Save last position's KV before GQA repeat (for TTT cache).
        # Under padded batching (n_per_req provided), gather per-row at the
        # real (N_i - 1)*K position instead of the padded tail.
        last_kv = None
        if return_last_kv:
            if n_per_req is not None:
                _device = k.device
                _Kidx = torch.arange(K, device=_device, dtype=torch.long)
                _offs = (n_per_req.to(_device, dtype=torch.long) - 1) * K
                _gather_pos = _offs.unsqueeze(1) + _Kidx.unsqueeze(0)  # [B, K]
                # Expand for [B, n_kv_heads, K, D] gather along dim=2.
                _idx = (
                    _gather_pos.unsqueeze(1).unsqueeze(-1)
                    .expand(B, k.shape[1], K, k.shape[-1])
                )
                last_kv = (
                    torch.gather(k, dim=2, index=_idx),
                    torch.gather(v, dim=2, index=_idx),
                )
            else:
                last_kv = (k[:, :, -K:, :], v[:, :, -K:, :])

        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)

        cache_count = cache[2]
        is_paged_b = _is_paged_cache(cache)

        use_sdpa = (
            os.environ.get('TREE_ATTN_SDPA', '0') == '1'
        )
        use_triton = (
            os.environ.get('TREE_ATTN_TRITON', '0') == '1'
            and self.head_dim == 128
            and cross_mask is None  # Triton kernel has no per-row mask
        )
        # Cache-count bucketing: attend to bucket-aligned length so torch.compile
        # sees <= max_cache_len/BUCKET distinct shapes instead of one per iter
        # (fixes recompile cascade on long-cache benches like cnn_dm). Tree
        # mask implicitly masks pad region because (curr_kv_pos >= N) → mask.
        # Triton path only (SDPA falls back to canonical).
        # Bucketing requires writing into the cache before the attention
        # call; the paged path does the write *after* attention, so disable
        # bucketing in paged mode and fall through to the cat path.
        cache_bucket = (
            int(os.environ.get('CACHE_COUNT_BUCKET', '0'))
            if (use_triton and not is_paged_b) else 0
        )
        NK = N * K

        # ---------------- Stage D: FlashInfer ragged cross + eager current ----------------
        # Triggered when caller has pre-planned a SpecBlockFlashInferCross
        # wrapper.  Replaces padded matmul cross (compute = B*max_cross) with
        # ragged paged FlashInfer (compute = sum(c_i)) for long-cross workloads
        # where padding overhead dominates.  Current K-slot tree-mask
        # attention is run eager (NK is small).  LSE merge combines.
        if flashinfer_cross is not None and is_paged_b:
            from ._tree_attention_triton import merge_attn_outputs

            # ---- Cross stage via FlashInfer ----
            B_q, n_heads_q, NK_q, D_q = q.shape  # NK_q == NK
            q_flat = (
                q.permute(0, 2, 1, 3).reshape(B_q * NK_q, n_heads_q, D_q)
            )
            kv_pool = cache[0]
            layer_id = cache[1]
            pool_k_layer_3d = kv_pool.get_k(layer_id)
            pool_v_layer_3d = kv_pool.get_v(layer_id)
            out_cross_flat, lse_cross_flat = flashinfer_cross.run_layer(
                q_flat, pool_k_layer_3d, pool_v_layer_3d,
            )
            out_cross = (
                out_cross_flat.reshape(B_q, NK_q, n_heads_q, D_q)
                .permute(0, 2, 1, 3).contiguous()
            )
            lse_cross = (
                lse_cross_flat.reshape(B_q, NK_q, n_heads_q)
                .permute(0, 2, 1).contiguous()
            )

            # ---- Current K-slot tree-mask attention (eager) ----
            batch_mask = _build_cross_position_mask(N, K, str(x.device))
            attn_curr = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_curr = attn_curr + batch_mask[None, None]
            lse_curr = torch.logsumexp(attn_curr, dim=-1)  # [B, H, NK]
            attn_curr_sm = F.softmax(
                attn_curr, dim=-1, dtype=torch.float32,
            ).to(q.dtype)
            out_curr = torch.matmul(attn_curr_sm, v)

            # ---- LSE merge ----
            out = merge_attn_outputs(out_cross, lse_cross, out_curr, lse_curr)
            _kv_already_written = False
            # Cache write below applies (paged set_kv).
        elif cache_count > 0 and cache_bucket > 0 and use_triton:
            # Pre-write new k/v in-place, then attend bucket-aligned slice.
            total_valid = cache_count + NK
            total_attend = ((total_valid + cache_bucket - 1) // cache_bucket) * cache_bucket
            self._ensure_cache_buf(cache, total_attend, k)
            cache[0][:, :, cache_count:cache_count + NK, :] = k[:, :, :NK, :]
            cache[1][:, :, cache_count:cache_count + NK, :] = v[:, :, :NK, :]
            k_full = cache[0][:, :, :total_attend, :]
            v_full = cache[1][:, :, :total_attend, :]
            from ._tree_attention_triton import tree_attention
            out = tree_attention(q, k_full, v_full, cache_count, K, self.scale)
            _kv_already_written = True
        elif cache_count > 0:
            if is_paged_b:
                prev_k, prev_v = _read_cross_kv_paged(cache)
            else:
                prev_k = cache[0][:, :, :cache_count, :]
                prev_v = cache[1][:, :, :cache_count, :]

            # Merge KV: [cached | current] → single Q @ K^T
            k_full = torch.cat([prev_k, k], dim=2)  # [B, H, C+NK, D]
            v_full = torch.cat([prev_v, v], dim=2)

            if use_sdpa:
                batch_mask = _build_cross_position_mask(N, K, str(x.device))
                # Build additive bias: [NK, cache_count+NK] with 0 for visible, -inf for masked.
                # Cross region fully visible → 0 (or per-row -inf for padded slots
                # when cross_mask is provided). Current region has tree mask.
                NK_v = N * K
                B_bias = B if cross_mask is not None else 1
                attn_bias = torch.zeros(
                    B_bias, NK_v, cache_count + NK_v, dtype=q.dtype, device=q.device,
                )
                if cross_mask is not None:
                    cross_bias = torch.where(
                        cross_mask, q.new_zeros(()), q.new_full((), float('-inf')),
                    )  # [B, max_cross]
                    attn_bias[..., :cache_count] = cross_bias[:, None, :]
                attn_bias[..., cache_count:] = batch_mask.to(q.dtype)[None]
                attn_mask = (
                    attn_bias[:, None]
                    if cross_mask is not None
                    else attn_bias[None]
                )
                out = F.scaled_dot_product_attention(
                    q, k_full, v_full,
                    attn_mask=attn_mask,
                    scale=self.scale,
                )
            elif use_triton:
                # Single kernel: tree-masked attention. Shape-invariant over
                # cache_count — no bucket padding / cuBLAS tile drift.
                # (Triton kernel has no per-row cross_mask; gated above.)
                from ._tree_attention_triton import tree_attention
                out = tree_attention(q, k_full, v_full, cache_count, K, self.scale)
            else:
                batch_mask = _build_cross_position_mask(N, K, str(x.device))
                attn = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale
                if cross_mask is not None:
                    cross_bias = torch.where(
                        cross_mask, q.new_zeros(()), q.new_full((), float('-inf')),
                    )
                    attn[..., :cache_count] += cross_bias[:, None, None, :]
                attn[..., cache_count:] += batch_mask[None, None]
                attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
                out = torch.matmul(attn, v_full)
            _kv_already_written = False
        else:
            if use_sdpa:
                batch_mask = _build_cross_position_mask(N, K, str(x.device)).to(q.dtype)
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=batch_mask[None, None],
                    scale=self.scale,
                )
            elif use_triton:
                from ._tree_attention_triton import tree_attention
                out = tree_attention(q, k, v, 0, K, self.scale)
            else:
                batch_mask = _build_cross_position_mask(N, K, str(x.device))
                attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn = attn + batch_mask[None, None]
                attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
                out = torch.matmul(attn, v)
            _kv_already_written = False

        # Store all K slots for all N positions in cross cache
        # k shape: [B, n_heads, N*K, head_dim] (already GQA repeated).
        # Skip if bucket-path already wrote (idempotent but saves 2 copies).
        if not _kv_already_written:
            if is_paged_b:
                # Slice to NK exactly (k may be padded by graph-safe path).
                _write_cross_kv_paged(cache, k[:, :, :NK, :], v[:, :, :NK, :])
            else:
                self._ensure_cache_buf(cache, cache_count + NK, k)
                cache[0][:, :, cache_count:cache_count + NK, :] = k[:, :, :NK, :]
                cache[1][:, :, cache_count:cache_count + NK, :] = v[:, :, :NK, :]
                cache[2] = cache_count + NK
        else:
            cache[2] = cache_count + NK

        result = self.o_proj(out.transpose(1, 2).contiguous().view(B, NK, -1))
        if return_last_kv:
            return result, last_kv
        return result

    def forward_batch_graph_safe(self, x, cache, pos_ids, max_position: int, N: int,
                                 cross_mask=None):
        """Graph-safe batch forward without mutating persistent cross KV.

        Paged caches use a shape-stable Triton kernel that gathers persistent
        slots directly from ``SpecBlockKVPool`` and applies the per-request
        cross mask inside the kernel.  Current accepted-position K/V stay in
        non-expanded GQA layout and are returned for graph-external masked
        scatter into the paged pool.
        """
        B, NK, _ = x.shape
        K = self.K

        q_flat, k_flat, v_flat = self._qkv_proj(x)
        q = q_flat.view(B, NK, self.n_heads, self.head_dim).transpose(1, 2)
        k_pre = k_flat.view(B, NK, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_pre = v_flat.view(B, NK, self.n_kv_heads, self.head_dim).transpose(1, 2)

        rope_len = max_position + 1
        cos, sin = self.rope(rope_len, dtype=q.dtype, device=q.device)
        q, k_pre = apply_rope(q, k_pre, cos, sin, pos_ids)

        cache_count = int(cache[2])
        if _is_paged_cache(cache):
            from ._tree_attention_triton import tree_attention_paged

            cross_loc = cache[3]
            if cross_loc.dim() == 1:
                cross_loc = cross_loc.unsqueeze(0)
            out = tree_attention_paged(
                q,
                cache[0].get_k(cache[1]),
                cache[0].get_v(cache[1]),
                cross_loc,
                cross_mask,
                k_pre,
                v_pre,
                cross_count=cache_count,
                k_slots=K,
                n_kv_groups=self.n_kv_groups,
                scale=self.scale,
            )
        else:
            k = repeat_kv(k_pre, self.n_kv_groups)
            v = repeat_kv(v_pre, self.n_kv_groups)
            batch_mask = _build_cross_position_mask(N, K, str(x.device))
            if cache_count > 0:
                prev_k = cache[0][:, :, :cache_count, :]
                prev_v = cache[1][:, :, :cache_count, :]
                k_full = torch.cat([prev_k, k], dim=2)
                v_full = torch.cat([prev_v, v], dim=2)
                attn = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale
                if cross_mask is not None:
                    attn[..., :cache_count].masked_fill_(
                        ~cross_mask[:, None, None, :], float('-inf'),
                    )
                attn[..., cache_count:] += batch_mask[None, None]
                attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
                out = torch.matmul(attn, v_full)
            else:
                attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn = attn + batch_mask[None, None]
                attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
                out = torch.matmul(attn, v)

        result = self.o_proj(out.transpose(1, 2).contiguous().view(B, NK, -1))
        return result, (k_pre, v_pre)


# ============================================================
#              SpecBlock MLP & Decoder Layer
# ============================================================

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        H, I = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.act = ACT2FN[getattr(config, 'hidden_act', 'silu')]
        # Lazy-cached fused gate_up weight (shape [2I, H]). Built on first fast-path call.
        self._gate_up_weight = None

    def forward(self, x):
        # Fused sgl_kernel path: 1 fused matmul for gate+up, then silu_and_mul kernel.
        # Replaces 2 separate matmuls + manual silu + in-place mul.
        # Disabled when weights are torchao quantized (AffineQuantizedTensor)
        # since cat is unsupported on those tensors. Quantized models fall
        # back to the 2-matmul path where torchao auto-dequantizes per call.
        _deep = os.environ.get("STATIC_DRAFT_PROFILE_DEEP", "0") == "1"
        gate_w = self.gate_proj.weight
        is_quantized = (
            type(gate_w).__name__ in ("AffineQuantizedTensor",)
            or not isinstance(gate_w, torch.Tensor)
        )
        if (
            _SGL_KERNEL_AVAILABLE
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and not is_quantized
        ):
            if self._gate_up_weight is None or self._gate_up_weight.device != x.device:
                self._gate_up_weight = torch.cat(
                    [self.gate_proj.weight, self.up_proj.weight], dim=0
                ).contiguous()
            if _deep:
                _ev_gu_s = torch.cuda.Event(enable_timing=True); _ev_gu_s.record()
            # Single matmul → [..., 2I]
            gate_up = F.linear(x, self._gate_up_weight)
            if _deep:
                _ev_gu_e = torch.cuda.Event(enable_timing=True); _ev_gu_e.record()
                _PROFILE_EVENTS.append(("mlp_gate_up", _ev_gu_s, _ev_gu_e))
            I = self.gate_proj.out_features
            out_shape = gate_up.shape[:-1] + (I,)
            # F.linear output is already contiguous; skip .contiguous() to save a copy.
            gate_up_flat = gate_up.view(-1, 2 * I)
            out_flat = torch.empty(
                gate_up_flat.shape[0], I, dtype=gate_up.dtype, device=gate_up.device
            )
            if _deep:
                _ev_sm_s = torch.cuda.Event(enable_timing=True); _ev_sm_s.record()
            _sgl_silu_and_mul(gate_up_flat, out_flat)
            if _deep:
                _ev_sm_e = torch.cuda.Event(enable_timing=True); _ev_sm_e.record()
                _PROFILE_EVENTS.append(("mlp_silu_mul", _ev_sm_s, _ev_sm_e))
            out = out_flat.view(out_shape)
            if _deep:
                _ev_dp_s = torch.cuda.Event(enable_timing=True); _ev_dp_s.record()
            ret = self.down_proj(out)
            if _deep:
                _ev_dp_e = torch.cuda.Event(enable_timing=True); _ev_dp_e.record()
                _PROFILE_EVENTS.append(("mlp_down_proj", _ev_dp_s, _ev_dp_e))
            return ret
        # Fallback path
        gate = self.act(self.gate_proj(x))
        gate.mul_(self.up_proj(x))  # in-place multiply
        return self.down_proj(gate)


class SpecBlockDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        eps = getattr(config, 'rms_norm_eps', 1e-5)
        self.input_layernorm = RMSNorm(H, eps)
        self.post_attention_layernorm = RMSNorm(H, eps)
        self.self_attn = SpecBlockAttentionWithCache(config)
        self.mlp = MLP(config)

    def forward(self, x, cross_cache, pos_ids, max_position=None,
                ttt_cache=None, ttt_mask=None, update_cross_cache=True,
                full_kv_mask=None):
        """Single position forward. Returns (hidden, all_kv)."""
        _deep = os.environ.get("STATIC_DRAFT_PROFILE_DEEP", "0") == "1"
        residual = x
        if _deep:
            _ev_n1_s = torch.cuda.Event(enable_timing=True); _ev_n1_s.record()
            normed = self.input_layernorm(x)
            _ev_n1_e = torch.cuda.Event(enable_timing=True); _ev_n1_e.record()
            _PROFILE_EVENTS.append(("norm1", _ev_n1_s, _ev_n1_e))
        else:
            normed = self.input_layernorm(x)
        attn_out, all_kv = self.self_attn(
            normed, cross_cache, pos_ids, max_position,
            ttt_cache=ttt_cache, ttt_mask=ttt_mask,
            update_cross_cache=update_cross_cache,
            full_kv_mask=full_kv_mask,
        )
        # Fused (residual + attn_out) → post_attn_layernorm via sgl_kernel.
        # Replaces 2 ops (add + RMSNorm) with 1 kernel; in-place modifies
        # both residual (to residual + attn_out) and attn_out (to normed).
        # Skip if quantized weight or non-CUDA / non-bf16-fp16.
        _use_fused_norm = (
            _SGL_KERNEL_AVAILABLE
            and _sgl_fused_add_rmsnorm is not None
            and attn_out.is_cuda
            and attn_out.dtype in (torch.bfloat16, torch.float16)
            and os.environ.get("FUSED_ADD_NORM", "1") == "1"
        )
        if _use_fused_norm:
            if _deep:
                _ev_n2_s = torch.cuda.Event(enable_timing=True); _ev_n2_s.record()
            orig_shape = attn_out.shape
            attn_flat = attn_out.view(-1, orig_shape[-1])
            res_flat = residual.view(-1, orig_shape[-1])
            _sgl_fused_add_rmsnorm(
                attn_flat, res_flat,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )
            # attn_out is now norm(residual + attn_out); residual is residual + attn_out.
            normed2 = attn_out
            x = residual  # already accumulated to residual + attn_out
            if _deep:
                _ev_n2_e = torch.cuda.Event(enable_timing=True); _ev_n2_e.record()
                _PROFILE_EVENTS.append(("norm2", _ev_n2_s, _ev_n2_e))
                _ev_mlp_s = torch.cuda.Event(enable_timing=True); _ev_mlp_s.record()
            mlp_out = self.mlp(normed2)
            if _deep:
                _ev_mlp_e = torch.cuda.Event(enable_timing=True); _ev_mlp_e.record()
                _PROFILE_EVENTS.append(("mlp", _ev_mlp_s, _ev_mlp_e))
            x = x + mlp_out
        else:
            x = residual + attn_out
            if _deep:
                _ev_n2_s = torch.cuda.Event(enable_timing=True); _ev_n2_s.record()
                normed2 = self.post_attention_layernorm(x)
                _ev_n2_e = torch.cuda.Event(enable_timing=True); _ev_n2_e.record()
                _PROFILE_EVENTS.append(("norm2", _ev_n2_s, _ev_n2_e))
                _ev_mlp_s = torch.cuda.Event(enable_timing=True); _ev_mlp_s.record()
                mlp_out = self.mlp(normed2)
                _ev_mlp_e = torch.cuda.Event(enable_timing=True); _ev_mlp_e.record()
                _PROFILE_EVENTS.append(("mlp", _ev_mlp_s, _ev_mlp_e))
                x = x + mlp_out
            else:
                x = x + self.mlp(self.post_attention_layernorm(x))
        return x, all_kv

    def forward_batch(self, x, cache, pos_ids, max_position: int, N: int,
                      return_last_kv: bool = False,
                      cross_mask: Optional[torch.Tensor] = None,
                      n_per_req: Optional[torch.Tensor] = None,
                      flashinfer_cross=None):
        """Batch forward (prefill/cache update). No TTT cache."""
        if return_last_kv:
            attn_out, last_kv = self.self_attn.forward_batch(
                self.input_layernorm(x), cache, pos_ids, max_position, N,
                return_last_kv=True,
                cross_mask=cross_mask, n_per_req=n_per_req,
                flashinfer_cross=flashinfer_cross,
            )
            x = x + attn_out
            x = x + self.mlp(self.post_attention_layernorm(x))
            return x, last_kv
        else:
            x = x + self.self_attn.forward_batch(
                self.input_layernorm(x), cache, pos_ids, max_position, N,
                cross_mask=cross_mask, n_per_req=n_per_req,
                flashinfer_cross=flashinfer_cross,
            )
            return x + self.mlp(self.post_attention_layernorm(x))

    def forward_batch_graph_safe(self, x, cache, pos_ids, max_position: int, N: int,
                                 cross_mask=None):
        """Graph-safe batch forward. Returns (output, pre_gqa_kv)."""
        attn_out, pre_kv = self.self_attn.forward_batch_graph_safe(
            self.input_layernorm(x), cache, pos_ids, max_position, N, cross_mask,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, pre_kv


# ============================================================
#              SpecBlock Inference Model
# ============================================================

class SpecBlockInferenceModel(nn.Module):
    """SpecBlock model for inference with KV cache and rank head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.K = getattr(config, 'diffspec_draft_token_num', 7)
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = getattr(config, 'draft_vocab_size', 32000)
        self.num_layers = getattr(config, 'num_hidden_layers', 1)
        self.rank_classes = getattr(config, 'rank_classes', 4)

        self.input_layer = SpecBlockInputLayer(config)
        self.layers = nn.ModuleList([SpecBlockDecoderLayer(config) for _ in range(self.num_layers)])
        self.norm = RMSNorm(config.hidden_size, getattr(config, 'rms_norm_eps', 1e-5))
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)
        # Rank Head: normed hidden (H) + top10_lp(10) + gaps(3) + cumprob_top1(1) + entropy(1) = H+15
        self.rank_feature_dim = 10 + 3 + 1 + 1
        self.rank_head = nn.Sequential(
            nn.Linear(config.hidden_size + self.rank_feature_dim, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, self.rank_classes, bias=False),
        )

        self.register_buffer("t2d", torch.ones(self.vocab_size, dtype=torch.bool))
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))

    def _rank_forward(self, normed: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Approximated rank features. RANK_TOPK env (default 10) controls
        the size of the partial-sort. Smaller K → faster topk but
        slightly degraded entropy approximation (rank_head was trained
        with K=20; user accepts mild rank-acc drop for speed).
        """
        _deep = os.environ.get("STATIC_DRAFT_PROFILE_DEEP", "0") == "1"
        _rank_topk = int(os.environ.get("RANK_TOPK", "10"))
        with torch.no_grad():
            if _deep:
                _ev_tk_s = torch.cuda.Event(enable_timing=True); _ev_tk_s.record()
            top_logits, top_indices = logits.detach().topk(_rank_topk, dim=-1)
            top_logits = top_logits.float()
            # Tree expansion consumes the same sorted candidates on the next
            # scheduler iteration.  Retain them so it does not repeat a full-
            # vocabulary top-k over the identical b0 logits.
            self._last_rank_top_indices = top_indices
            top_lse = torch.logsumexp(top_logits, dim=-1, keepdim=True)
            if _deep:
                _ev_tk_e = torch.cuda.Event(enable_timing=True); _ev_tk_e.record()
                _PROFILE_EVENTS.append(("rh_topk_lse", _ev_tk_s, _ev_tk_e))
                _ev_ft_s = torch.cuda.Event(enable_timing=True); _ev_ft_s.record()

            top10_logits = top_logits[..., :10]
            top10_lp = top10_logits - top_lse
            top10_p = top10_lp.exp()
            gap_12 = top10_logits[..., 0:1] - top10_logits[..., 1:2]
            gap_13 = top10_logits[..., 0:1] - top10_logits[..., 2:3]
            gap_15 = top10_logits[..., 0:1] - top10_logits[..., 4:5]
            cumprob_top1 = top10_p[..., 0:1]

            # Approximate entropy with the same partial-sort range.
            # When RANK_TOPK=10, this loses the contribution of ranks 11-20
            # to entropy (~few percent of mass typically); rank_head was
            # trained with K=20 entropy but tolerates the shift in practice.
            top_lp = top_logits - top_lse
            top_p = top_lp.exp()
            entropy = -(top_p * top_lp).nan_to_num(0.0).sum(-1, keepdim=True)

            rank_features = torch.cat(
                [top10_lp, gap_12, gap_13, gap_15, cumprob_top1, entropy], dim=-1,
            ).to(normed.dtype)
            if _deep:
                _ev_ft_e = torch.cuda.Event(enable_timing=True); _ev_ft_e.record()
                _PROFILE_EVENTS.append(("rh_features", _ev_ft_s, _ev_ft_e))
        if _deep:
            _ev_mm_s = torch.cuda.Event(enable_timing=True); _ev_mm_s.record()
        rank_input = torch.cat([normed.detach(), rank_features], dim=-1)
        ret = self.rank_head(rank_input)
        if _deep:
            _ev_mm_e = torch.cuda.Event(enable_timing=True); _ev_mm_e.record()
            _PROFILE_EVENTS.append(("rh_matmul", _ev_mm_s, _ev_mm_e))
        return ret

    # ---- Single position: draft call (with norm + lm_head + rank_head) ----

    def forward_with_cache(self, hidden, input_ids, cache, position_id,
                           use_draft_condition=False, ttt_cache=None,
                           ttt_mask=None, update_cross_cache=True,
                           full_kv_mask=None):
        """Single position forward for draft.

        Args:
            hidden: [B, 1, 3H] (block 1) or [B, 1, H] (block 2+)
            input_ids: [B, 1]
            cache: list of [k_buf, v_buf, count, max_len] per layer (pre-allocated cache)
            position_id: int, position index for RoPE
            use_draft_condition: True for block 2+ (hidden is H, skip condition_proj)
            ttt_cache: per-layer list of (ttt_k, ttt_v) from prior TTT blocks.
                Each (ttt_k, ttt_v) has shape [B, n_kv_heads, C_ttt, head_dim]
                where C_ttt = num_prev_blocks * K (all K slots per block).
            ttt_mask: [B, C_ttt] bool mask, True for valid entries. Required when ttt_cache is set.
            update_cross_cache: if True, append all K slots to cross_cache

        Returns:
            logits: [B, K, draft_vocab_size]
            rank_logits: [B, K, rank_classes]
            draft_hidden: [B, K, H] (pre-norm, for TTT chaining)
            new_ttt_kv: per-layer list of (all_k, all_v) each [B, n_kv_heads, K, head_dim]
        """
        B = hidden.shape[0]
        _deep = os.environ.get("STATIC_DRAFT_PROFILE_DEEP", "0") == "1"
        if _deep:
            _ev_il_s = torch.cuda.Event(enable_timing=True); _ev_il_s.record()
        x = self.input_layer(hidden, input_ids, use_draft_condition=use_draft_condition)
        if _deep:
            _ev_il_e = torch.cuda.Event(enable_timing=True); _ev_il_e.record()
            _PROFILE_EVENTS.append(("fwd_input_layer", _ev_il_s, _ev_il_e))
        # Sequential position IDs: row b's slot_k uses base[b] + 1 + k.
        # Accept int (legacy single-req) or [B] tensor (multi-req batch).
        if torch.is_tensor(position_id):
            base = position_id.to(x.device, dtype=torch.long)
            if base.numel() == 1:
                base_int = int(base.item())
                pos_ids = torch.arange(
                    base_int + 1, base_int + 1 + self.K,
                    device=x.device, dtype=torch.long,
                ).unsqueeze(0).expand(B, -1)
                max_pos = base_int + self.K
            else:
                slot_offsets = torch.arange(
                    self.K, device=x.device, dtype=torch.long,
                )
                pos_ids = (base.unsqueeze(1) + 1) + slot_offsets.unsqueeze(0)
                max_pos = int(base.max().item()) + self.K
        else:
            pos_ids = torch.arange(
                position_id + 1, position_id + 1 + self.K,
                device=x.device, dtype=torch.long,
            ).unsqueeze(0).expand(B, -1)
            max_pos = position_id + self.K

        new_ttt_kv = []
        for layer_idx, layer in enumerate(self.layers):
            layer_ttt = ttt_cache[layer_idx] if ttt_cache is not None else None
            x, all_kv = layer(
                x, cache[layer_idx], pos_ids, max_position=max_pos,
                ttt_cache=layer_ttt, ttt_mask=ttt_mask,
                update_cross_cache=update_cross_cache,
                full_kv_mask=full_kv_mask,
            )
            new_ttt_kv.append(all_kv)

        draft_hidden = x  # [B, K, H] pre-norm
        if _deep:
            _ev_fn_s = torch.cuda.Event(enable_timing=True); _ev_fn_s.record()
        normed = self.norm(x)
        if _deep:
            _ev_fn_e = torch.cuda.Event(enable_timing=True); _ev_fn_e.record()
            _PROFILE_EVENTS.append(("fwd_final_norm", _ev_fn_s, _ev_fn_e))
            _ev_lm_s = torch.cuda.Event(enable_timing=True); _ev_lm_s.record()
        logits = self.lm_head(normed)
        if _deep:
            _ev_lm_e = torch.cuda.Event(enable_timing=True); _ev_lm_e.record()
            _PROFILE_EVENTS.append(("fwd_lm_head", _ev_lm_s, _ev_lm_e))
            _ev_rh_s = torch.cuda.Event(enable_timing=True); _ev_rh_s.record()
        rank_logits = self._rank_forward(normed, logits)
        if _deep:
            _ev_rh_e = torch.cuda.Event(enable_timing=True); _ev_rh_e.record()
            _PROFILE_EVENTS.append(("fwd_rank_head", _ev_rh_s, _ev_rh_e))

        return logits, rank_logits, draft_hidden, new_ttt_kv

    # ---- Batch N positions: prefill / cache update (no norm + lm_head) ----

    def _forward_batch_cache(self, hidden_3h, input_ids, cache, start_position):
        """Batch forward for N positions. Only updates cache, no logits."""
        B, N, _ = hidden_3h.shape
        K = self.K
        x = self.input_layer.forward_batch(hidden_3h, input_ids)

        # Sequential position IDs: position i's slot_k uses (start + i + 1 + k)
        base_pos = torch.arange(
            start_position + 1, start_position + 1 + N, device=x.device
        )
        slot_offsets = torch.arange(K, device=x.device)
        pos_ids = (base_pos.unsqueeze(1) + slot_offsets.unsqueeze(0)).reshape(-1)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)

        max_position = start_position + N + K - 1

        for layer_idx, layer in enumerate(self.layers):
            x = layer.forward_batch(x, cache[layer_idx], pos_ids, max_position, N)

    def prefill(self, hidden_3h, input_ids, chunk_size: int = 64):
        """Batch prefill for S positions with chunking."""
        S = hidden_3h.shape[1]
        # Pre-allocate cache for prefix + generous generation budget
        cache = self.reset_cache(max_cache_len=(S + 2048) * self.K)

        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            self._forward_batch_cache(
                hidden_3h[:, start:end, :],
                input_ids[:, start:end],
                cache,
                start_position=start,
            )

        # Prompt cache compaction options (training-free):
        # 1. STREAM_SNAP_PROMPT=1: SnapKV attention-guided (recommended for long
        #    prompts — cnn_dm/math500; no per-iter maintenance cost).
        # 2. STREAM_COMPACT_PROMPT=1: legacy StreamingLLM sink+near-window
        #    (maintenance cost high on long prompt due to maybe_compact_near_to_far).
        # SnapKV takes precedence when both set.
        if os.environ.get('STREAM_SNAP_PROMPT', '0') == '1':
            obs_win = int(os.environ.get('STREAM_SNAP_OBS_WIN', '32'))
            budget = int(os.environ.get('STREAM_SNAP_BUDGET', '256'))
            sink = int(os.environ.get('STREAM_SNAP_SINK_POS', '8'))
            pool_k = int(os.environ.get('STREAM_SNAP_POOL', '7'))
            self.snapkv_compact_prompt_cache(
                cache, S, obs_win=obs_win, budget=budget,
                sink_pos=sink, pool_kernel=pool_k,
            )
        elif os.environ.get('STREAM_COMPACT_PROMPT', '0') == '1':
            sink = int(os.environ.get('STREAM_COMPACT_SINK_POS', '8'))
            nw = int(os.environ.get('STREAM_COMPACT_NEAR_WIN', '0'))
            self.compact_prompt_cache(cache, S, sink_pos=sink, near_win=nw)

        return cache, S

    def prefill_and_draft(self, hidden_3h, input_ids, draft_hidden_3h, draft_input_id, chunk_size: int = 64):
        """Merged prefill + first draft: fills cache and returns logits for last position.

        Saves one forward pass by appending the first draft position to the last prefill chunk.

        Args:
            hidden_3h: [B, S, 3H] prompt hidden states
            input_ids: [B, S] shifted prompt token ids
            draft_hidden_3h: [B, 1, 3H] hidden state for first draft position
            draft_input_id: [B, 1] first generated token id
            chunk_size: max positions per chunk

        Returns:
            cache, draft_position, logits, rank_logits, draft_hidden, ttt_kv
        """
        S = hidden_3h.shape[1]
        # Max cache len must be large enough to cover (prefill S slots +
        # generated tokens × K draft slots per iter) without realloc.
        # Matches `prefill` which uses (S + 2048) * K. Also large enough
        # for CUDA graph's max_cross_count (view-based capture needs
        # shape[2] >= cc_bucket for all used buckets).
        cache = self.reset_cache(max_cache_len=(S + 2048) * self.K)

        # Prefill all chunks except the last
        last_chunk_start = max(0, S - (S % chunk_size)) if S % chunk_size != 0 else max(0, S - chunk_size)
        for start in range(0, last_chunk_start, chunk_size):
            end = min(start + chunk_size, last_chunk_start)
            if end > start:
                self._forward_batch_cache(
                    hidden_3h[:, start:end, :], input_ids[:, start:end],
                    cache, start_position=start,
                )

        # Last chunk + draft position merged into one forward
        merged_h = torch.cat([hidden_3h[:, last_chunk_start:S, :], draft_hidden_3h], dim=1)
        merged_ids = torch.cat([input_ids[:, last_chunk_start:S], draft_input_id], dim=1)
        logits, rank_logits, draft_hidden, ttt_kv, new_pos = \
            self.update_cache_and_draft(merged_h, merged_ids, cache, last_chunk_start)

        # Optional prompt compaction (same as `prefill`). Compacts only the S prompt
        # positions; the first-draft position (written at end by update_cache_and_draft)
        # is preserved as tail.
        if os.environ.get('STREAM_COMPACT_PROMPT', '0') == '1':
            sink = int(os.environ.get('STREAM_COMPACT_SINK_POS', '8'))
            nw = int(os.environ.get('STREAM_COMPACT_NEAR_WIN', '0'))
            self.compact_prompt_cache(cache, S, sink_pos=sink, near_win=nw)

        return cache, new_pos, logits, rank_logits, draft_hidden, ttt_kv

    def update_cache(self, hidden_3h, input_ids, cache, start_position):
        """Batch cache update for N accepted positions."""
        N = hidden_3h.shape[1]
        self._forward_batch_cache(hidden_3h, input_ids, cache, start_position)
        # Gen-phase demotion maintenance (slot_1..K-1 → slot_0 for aged-out near
        # positions). High cost on long prompts (cnn_dm: observed draft 21→58 ms).
        # STREAM_COMPACT_DECODE_MAINTAIN=0 disables this maintenance — StreamingLLM
        # compact then behaves "prefill-once" (like SnapKV), zero decode overhead.
        near_win = int(os.environ.get('STREAM_COMPACT_NEAR_WIN', '0'))
        if near_win > 0 and os.environ.get('STREAM_COMPACT_DECODE_MAINTAIN', '1') == '1':
            self.maybe_compact_near_to_far(cache, near_win)
        return start_position + N

    def update_cache_and_draft(self, hidden_3h, input_ids, cache, start_position):
        """Merged cache update + block 1 draft in a single forward pass.

        Processes N positions through layers in one batched forward:
        - First N-1 positions: cache update (accepted tokens)
        - Last position: block 1 draft (returns logits/rank/hidden/ttt_kv)

        This saves one full draft forward per iteration compared to
        calling update_cache() then forward_with_cache() separately.

        Args:
            hidden_3h: [B, N, 3H] where N = accept_length + 1
                       (first accept_length for cache update, last 1 for draft)
            input_ids: [B, N] correspondingly shifted tokens
            cache: cross-position cache (updated in-place)
            start_position: position id of the first token

        Returns:
            logits: [B, K, draft_vocab_size] for last position
            rank_logits: [B, K, rank_classes] for last position
            draft_hidden: [B, K, H] pre-norm hidden for last position
            new_ttt_kv: per-layer (k, v) each [B, n_kv_heads, K, head_dim]
            new_position: start_position + N
        """
        B, N, _ = hidden_3h.shape
        K = self.K

        x = self.input_layer.forward_batch(hidden_3h, input_ids)  # [B, N*K, H]

        # Sequential position IDs: position i's slot_k uses (start + i + 1 + k)
        base_pos = torch.arange(
            start_position + 1, start_position + 1 + N, device=x.device
        )
        slot_offsets = torch.arange(K, device=x.device)
        pos_ids = (base_pos.unsqueeze(1) + slot_offsets.unsqueeze(0)).reshape(-1)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)
        max_position = start_position + N + K - 1

        new_ttt_kv = []
        for layer_idx, layer in enumerate(self.layers):
            x, last_kv = layer.forward_batch(
                x, cache[layer_idx], pos_ids, max_position, N,
                return_last_kv=True,
            )
            new_ttt_kv.append(last_kv)

        # lm_head + rank_head only on last position's K slots
        last_x = x[:, -K:, :]
        draft_hidden = last_x
        normed = self.norm(last_x)
        logits = self.lm_head(normed)
        rank_logits = self._rank_forward(normed, logits)

        # Gen-phase demotion maintenance (slot_1..K-1 → slot_0 for aged-out near
        # positions). High cost on long prompts (cnn_dm: observed draft 21→58 ms).
        # STREAM_COMPACT_DECODE_MAINTAIN=0 disables this maintenance — StreamingLLM
        # compact then behaves "prefill-once" (like SnapKV), zero decode overhead.
        near_win = int(os.environ.get('STREAM_COMPACT_NEAR_WIN', '0'))
        if near_win > 0 and os.environ.get('STREAM_COMPACT_DECODE_MAINTAIN', '1') == '1':
            self.maybe_compact_near_to_far(cache, near_win)

        return logits, rank_logits, draft_hidden, new_ttt_kv, start_position + N

    # ---- Cache management ----

    def reset_cache(self, max_cache_len=2048):
        """Reset cross-position cache with pre-allocated buffer capacity.

        Cache format: [k_buf, v_buf, count, max_cache_len, far_end_row]
        k_buf/v_buf are lazily allocated on first write.
        far_end_row (idx 4): first row of "near" region in stream cache mode;
                             None means compaction inactive (all rows are full-K-per-pos).
        """
        return [[None, None, 0, max_cache_len, None] for _ in range(self.num_layers)]

    def pop_cache(self, cache):
        """Remove the last position's K entries from each layer's cross-position cache.

        With pre-allocated buffers, just decrement count (O(1), no copy).
        """
        K = self.K
        for layer_cache in cache:
            count = layer_cache[2]
            if count >= K:
                layer_cache[2] = count - K

    def maybe_compact_near_to_far(self, cache, near_win: int):
        """Demote oldest near positions to far (slot_0 only) when near exceeds near_win.

        Precondition: compact_prompt_cache was called first (cache[4] = far_end_row set).
        Bulk-gathers keep_idx and writes back in place. No-op if near ≤ near_win or
        compaction inactive (cache[4] is None).
        """
        if near_win <= 0:
            return
        K = self.K
        for layer_cache in cache:
            far_end = layer_cache[4] if len(layer_cache) > 4 else None
            if far_end is None:
                continue
            count = layer_cache[2]
            near_rows = count - far_end
            if near_rows <= 0:
                continue
            near_pos = near_rows // K
            if near_pos <= near_win:
                continue
            demote_n = near_pos - near_win
            device = layer_cache[0].device
            # Rows to keep: [0..far_end) + slot_0 of demoted positions + remaining near
            far_idx = torch.arange(far_end, device=device)
            demoted_slot0_idx = far_end + torch.arange(demote_n, device=device) * K
            kept_near_start = far_end + demote_n * K
            kept_near_idx = torch.arange(kept_near_start, count, device=device)
            keep_idx = torch.cat([far_idx, demoted_slot0_idx, kept_near_idx])
            K_buf = layer_cache[0]
            V_buf = layer_cache[1]
            new_K = K_buf.index_select(2, keep_idx)
            new_V = V_buf.index_select(2, keep_idx)
            new_count = keep_idx.shape[0]
            K_buf[:, :, :new_count, :] = new_K
            V_buf[:, :, :new_count, :] = new_V
            layer_cache[2] = new_count
            layer_cache[4] = far_end + demote_n

    def compact_prompt_cache(self, cache, prompt_len: int, sink_pos: int = 8,
                             near_win: int = 0):
        """Physically compact prompt KV matching mask variant B semantics.

        Layout per layer (after compact):
          [sink_pos × K]   [(prompt_len - sink_pos - near_in_prompt) × 1 slot_0]
          [near_in_prompt × K]   [tail]

        where near_in_prompt = min(near_win, prompt_len - sink_pos).

        Matches StreamingLLM: keep **both** sink (first positions) and near (last
        positions) full-K, drop middle to slot_0 only. The previous implementation
        only kept sink, which over-compressed the end-of-prompt region that the
        model relies on heavily (causing ~0.5 acc drop on mtbench/humaneval).
        """
        K = self.K
        if prompt_len <= sink_pos:
            return
        near_in_prompt = max(0, min(near_win, prompt_len - sink_pos))
        far_positions = prompt_len - sink_pos - near_in_prompt
        if far_positions <= 0:
            return  # sink + near covers whole prompt → no compaction

        sink_rows = sink_pos * K
        far_end_pos = sink_pos + far_positions  # position index where near starts
        near_start_row_orig = far_end_pos * K
        prompt_rows = prompt_len * K

        for layer_cache in cache:
            old_count = layer_cache[2]
            if old_count < prompt_rows:
                continue
            device = layer_cache[0].device
            sink_idx = torch.arange(sink_rows, device=device)
            far_idx = torch.arange(sink_pos, far_end_pos, device=device) * K  # slot_0 only
            near_idx = torch.arange(near_start_row_orig, prompt_rows, device=device)
            tail_len = old_count - prompt_rows
            if tail_len > 0:
                tail_idx = torch.arange(prompt_rows, old_count, device=device)
                keep_idx = torch.cat([sink_idx, far_idx, near_idx, tail_idx])
            else:
                keep_idx = torch.cat([sink_idx, far_idx, near_idx])

            K_buf = layer_cache[0]
            V_buf = layer_cache[1]
            new_K = K_buf.index_select(2, keep_idx)
            new_V = V_buf.index_select(2, keep_idx)
            new_len = keep_idx.shape[0]
            K_buf[:, :, :new_len, :] = new_K
            V_buf[:, :, :new_len, :] = new_V
            layer_cache[2] = new_len
            # far_end_row = start of "near" region in new layout = sink_rows + far_positions
            if len(layer_cache) > 4:
                layer_cache[4] = sink_rows + far_positions

    def snapkv_compact_prompt_cache(self, cache, prompt_len: int,
                                    obs_win: int = 32, budget: int = 256,
                                    sink_pos: int = 8, pool_kernel: int = 7):
        """SnapKV-style prompt cache compaction (training-free, attention-guided).

        Reference: SnapKV (Li et al, NeurIPS 2024) https://arxiv.org/abs/2404.14469

        Algorithm (simplified using K-to-K similarity as Q-proxy — we don't store
        Q post-prefill, and K-K similarity is a commonly used fast proxy for
        token importance that still captures "which prompt tokens late tokens
        will need to look back at"):
          1. For each layer, use last `obs_win` positions' K slots as "observer"
          2. Compute similarity scores: observer @ all_prompt_K^T (softmax)
          3. Average score across heads × observers → per-prompt-token importance
          4. 1D avg-pool for smoothness (neighbors share importance)
          5. Keep: sink_pos (first 8) + top-(budget - sink - obs_win) + obs_win
          6. Compact cache in-place; disable `maybe_compact_near_to_far` (set far_end_row=None)

        Advantage over StreamingLLM compact_prompt_cache:
          - No `maybe_compact_near_to_far` maintenance cost per iter (main overhead
            we observed on cnn_dm: draft 21→58 ms when StreamingLLM compact on
            with long prompt due to per-iter index_select/cat cost)
          - Task-aware token selection (not position-based): higher acc retention

        Args:
            cache: [k_buf, v_buf, count, max_len, far_end_row] per-layer list
            prompt_len: number of prompt positions (pre-gen)
            obs_win: observation window (last W positions) used to compute importance
            budget: total positions to keep (including sink + obs_win)
            sink_pos: always-keep prefix positions (attention sink)
            pool_kernel: avg-pool kernel over position axis for smoothing
        """
        K = self.K
        if prompt_len <= sink_pos + obs_win:
            return  # prompt too short, nothing to compact
        budget = min(budget, prompt_len)
        if budget >= prompt_len:
            return  # budget covers whole prompt
        top_b = budget - sink_pos - obs_win
        if top_b <= 0:
            top_b = 0
        prompt_rows = prompt_len * K

        for layer_cache in cache:
            old_count = layer_cache[2]
            if old_count < prompt_rows:
                continue
            K_buf = layer_cache[0]
            V_buf = layer_cache[1]
            device = K_buf.device

            # Observer: last `obs_win` positions' K slots
            obs_start_row = (prompt_len - obs_win) * K
            obs_end_row = prompt_rows
            Q_proxy = K_buf[:, :, obs_start_row:obs_end_row, :]   # [1, Hq, obs_win*K, D]
            K_full = K_buf[:, :, :prompt_rows, :]                 # [1, Hq, prompt_len*K, D]

            # Similarity (K-to-K as Q-proxy; inexact but fast + training-free)
            scale = 1.0 / math.sqrt(K_buf.shape[-1])
            scores = torch.matmul(Q_proxy, K_full.transpose(-2, -1)) * scale
            scores = F.softmax(scores, dim=-1)                    # [1, Hq, obs_win*K, prompt_rows]
            # Aggregate: mean over heads, mean over observer queries → [prompt_rows]
            importance = scores.mean(dim=(0, 1, 2))               # [prompt_rows]
            # Per-position importance (sum over K slots of same position)
            imp_pos = importance.view(prompt_len, K).sum(dim=1)   # [prompt_len]

            # 1D avg-pool for smoothing (Gaussian-like)
            pad = pool_kernel // 2
            padded = F.pad(imp_pos.unsqueeze(0).unsqueeze(0), (pad, pad), mode='replicate')
            smoothed = F.avg_pool1d(padded, kernel_size=pool_kernel,
                                    stride=1, padding=0).squeeze()  # [prompt_len]

            # Selection: sink + top_b (from middle) + obs_win
            sink_idx = torch.arange(sink_pos, device=device)
            obs_idx = torch.arange(prompt_len - obs_win, prompt_len, device=device)
            # Mask out sink + obs from middle selection
            middle_mask = torch.ones(prompt_len, dtype=torch.bool, device=device)
            middle_mask[:sink_pos] = False
            middle_mask[prompt_len - obs_win:] = False
            middle_scores = torch.where(middle_mask, smoothed,
                                        torch.full_like(smoothed, -1e9))
            if top_b > 0 and middle_mask.sum() > top_b:
                _, top_idx = torch.topk(middle_scores, top_b)
            else:
                top_idx = middle_mask.nonzero(as_tuple=True)[0]
            keep_pos = torch.cat([sink_idx, top_idx, obs_idx]).sort().values

            # Expand positions to cache rows (K slots per pos), keep ALL K slots
            K_arange = torch.arange(K, device=device)
            keep_rows = (keep_pos.unsqueeze(1) * K + K_arange.unsqueeze(0)).flatten()
            keep_rows = keep_rows.sort().values

            # Tail (generated tokens if any — usually 0 at end of prefill)
            tail_len = old_count - prompt_rows
            if tail_len > 0:
                tail_idx = torch.arange(prompt_rows, old_count, device=device)
                keep_rows = torch.cat([keep_rows, tail_idx])

            # Compact in-place
            new_K = K_buf.index_select(2, keep_rows)
            new_V = V_buf.index_select(2, keep_rows)
            new_len = keep_rows.shape[0]
            K_buf[:, :, :new_len, :] = new_K
            V_buf[:, :, :new_len, :] = new_V
            layer_cache[2] = new_len
            # Disable maybe_compact_near_to_far — SnapKV fixes cache at prefill end
            # and gen-side positions append normally. far_end_row=None signals
            # "no compaction in progress" to maybe_compact_near_to_far (no-op).
            if len(layer_cache) > 4:
                layer_cache[4] = None

    def prepare_for_inference(self, quantize=None):
        """Compile decoder layers and optionally quantize weights.

        Args:
            quantize: None (default), "int8" (dynamic w8a8), "int8w"
                     (weight-only w8a16 via torchao), or "int4" (weight-only
                     w4a16 via torchao). Quantizes decoder linear weights.
        """
        # torch.compile tried (mode=default on attn+mlp): single forward went
        # 2.59 → 2.81 ms (slightly SLOWER). Compile's per-call graph setup cost
        # exceeds inductor fusion gain at this op granularity. Disabled.
        #
        # torch.compile(mode="reduce-overhead") also tried: inductor's CUDA
        # graph capture asserts `dst.data_ptr() == src.data_ptr()` under
        # varying cross_count shape. Disabling inductor cudagraph + relying
        # on fusion alone was neutral; attention compile with dynamic=True
        # caused heavy recompile thrash (4x regression from shape churn).

        # Weight quantization
        if quantize == "int8":
            self._quantize_weights(torch.int8)
        elif quantize == "int8w":
            self._quantize_int8_weight_only()
        elif quantize == "int4":
            self._quantize_int4()

    def _quantize_weights(self, dtype):
        """Quantize all linear layers to specified dtype using dynamic quantization."""
        try:
            import torch.ao.quantization as quant
            torch.ao.quantization.quantize_dynamic(
                self, {nn.Linear}, dtype=dtype, inplace=True
            )
        except Exception as e:
            print(f"Warning: Dynamic quantization failed, skipping: {e}")

    def _quantize_int8_weight_only(self):
        """Quantize decoder linear weights to INT8 via torchao (W8A16).

        Activations stay bf16; only weights are quantized to int8.
        Replaces nn.Linear with torchao's quantized linear that dequantizes
        weight on the fly. Works alongside the MLP fast-path: but caching
        of self._gate_up_weight (cat'd weights) by sgl_kernel path is
        invalidated since gate_proj/up_proj weights become AffineQuantized
        tensors. We skip MLP wrappers' gate_up_weight caching for quantized
        models — sgl_kernel falls back to standard 2-matmul path.
        """
        try:
            from torchao.quantization import quantize_, int8_weight_only
            for i, layer in enumerate(self.layers):
                quantize_(layer, int8_weight_only())
                # Invalidate sgl_kernel fused gate_up cache (weights now
                # AffineQuantized; can't cat directly).
                if hasattr(layer.mlp, "_gate_up_weight"):
                    layer.mlp._gate_up_weight = None
            print(f"  INT8-W weight-only quantized {len(self.layers)} decoder layers")
        except ImportError:
            print("Warning: torchao not installed, INT8-W skipped. Install with: pip install torchao")
        except Exception as e:
            print(f"Warning: INT8-W quantization failed: {e}")

    def _quantize_int4(self):
        """Quantize linear weights to INT4 using torchao."""
        try:
            from torchao.quantization import quantize_, int4_weight_only
            # Only quantize decoder layers (skip input_layer embeddings and lm_head)
            for i, layer in enumerate(self.layers):
                quantize_(layer, int4_weight_only())
                if hasattr(layer.mlp, "_gate_up_weight"):
                    layer.mlp._gate_up_weight = None
            print(f"  INT4 quantized {len(self.layers)} decoder layers")
        except ImportError:
            print("Warning: torchao not installed, INT4 quantization skipped. Install with: pip install torchao")
        except Exception as e:
            print(f"Warning: INT4 quantization failed: {e}")

    @property
    def embed_tokens(self):
        return self.input_layer.embed_tokens

    def load_vocab_mapping(self, path):
        mapping = torch.load(path, weights_only=False)
        self.t2d.copy_(mapping["t2d"])
        self.d2t.copy_(mapping["d2t"])
