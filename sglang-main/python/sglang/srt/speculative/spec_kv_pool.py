"""SpecBlock-Shift worker-local paged KV pool.

Stores cross-attention K/V for the SpecBlock-Shift draft model in a
single shared paged buffer instead of per-request `[1, n_heads, max_len,
head_dim]` dense buffers.

Allocation strategy: incremental grow on overflow (start small, double-
chunk grow up to ``max_pool_size``).  ``capture_mode()`` ctx forbids
grow inside cuda graph capture (graphs hold tensor pointers; recreate
invalidates the graph).  Caller (graph runner) is responsible for
pre-growing the pool before entering capture.

Storage layout (GQA-expanded)::

    k_buffer: [num_layers, pool_size, n_heads, head_dim]
    v_buffer: [num_layers, pool_size, n_heads, head_dim]

Index 0 is reserved as a "zero sentinel" — read of `pool[0]` always
returns zeros, so callers can pad indices up to a fixed bucket without
adding a separate mask in the cuda-graph hot path.

CUDA graph integration:
    Graph runners enter ``capture_mode()`` ctx during capture; alloc
    inside the ctx raises if it would trigger a grow (caller pre-grows).
    ``pool_version`` is bumped on every actual grow — graph runners
    cache ``(pool_version, ...)`` keys and invalidate on bump.
"""

from __future__ import annotations

import contextlib
import logging
from collections import deque
from typing import Iterator, List, Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_DEFAULT_GROW_CHUNK = 65536  # 65K slots per grow → ~2 GB on Llama-3.1-8B


@triton.jit
def _set_kv_padded_kernel(
    k_pool_ptr,
    v_pool_ptr,
    indices_ptr,
    cache_k_ptr,
    cache_v_ptr,
    valid_slots_ptr,
    slots_per_row,
    stride_pool_slot: tl.constexpr,
    stride_pool_head: tl.constexpr,
    stride_pool_dim: tl.constexpr,
    stride_indices_batch: tl.constexpr,
    stride_indices_slot: tl.constexpr,
    stride_cache_batch: tl.constexpr,
    stride_cache_head: tl.constexpr,
    stride_cache_slot: tl.constexpr,
    stride_cache_dim: tl.constexpr,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = row // slots_per_row
    slot_idx = row - batch_idx * slots_per_row
    valid_slots = tl.load(valid_slots_ptr + batch_idx)
    slot_valid = slot_idx < valid_slots
    pool_idx = tl.load(
        indices_ptr
        + batch_idx * stride_indices_batch
        + slot_idx * stride_indices_slot,
        mask=slot_valid,
        other=0,
    )

    hd = tl.program_id(1) * BLOCK_HD + tl.arange(0, BLOCK_HD)
    head_idx = hd // HEAD_DIM
    dim_idx = hd - head_idx * HEAD_DIM
    mask = slot_valid & (hd < N_HEADS * HEAD_DIM)

    cache_offset = (
        batch_idx * stride_cache_batch
        + head_idx * stride_cache_head
        + slot_idx * stride_cache_slot
        + dim_idx * stride_cache_dim
    )
    pool_offset = (
        pool_idx * stride_pool_slot
        + head_idx * stride_pool_head
        + dim_idx * stride_pool_dim
    )
    cache_k = tl.load(cache_k_ptr + cache_offset, mask=mask, other=0.0)
    cache_v = tl.load(cache_v_ptr + cache_offset, mask=mask, other=0.0)
    tl.store(k_pool_ptr + pool_offset, cache_k, mask=mask)
    tl.store(v_pool_ptr + pool_offset, cache_v, mask=mask)


class SpecBlockKVPool:
    """Worker-local paged KV pool for SpecBlock-Shift cross attention.

    Args:
        initial_pool_size: starting #slots; doubled / chunk-grown on overflow.
        max_pool_size: hard cap (fail-fast on overflow).
        num_layers: draft model #layers (typically 2).
        n_heads: GQA-expanded head count.
        head_dim: per-head dimension.
        dtype: K/V dtype (typically bfloat16).
        device: CUDA device.
    """

    def __init__(
        self,
        initial_pool_size: int,
        max_pool_size: int,
        num_layers: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if initial_pool_size < 2:
            raise ValueError(
                f"initial_pool_size must be >=2 (slot 0 is the zero sentinel), "
                f"got {initial_pool_size}"
            )
        if max_pool_size < initial_pool_size:
            max_pool_size = initial_pool_size

        self.num_layers = num_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_pool_size = max_pool_size

        self._pool_size = initial_pool_size
        # K/V tensors.  [num_layers, pool_size, n_heads, head_dim] keeps
        # per-layer reads contiguous along the slot dim.  Slot 0 is the
        # zero sentinel (kept zero forever for pad-friendly gather).
        self.k_buffer = torch.zeros(
            num_layers, self._pool_size, n_heads, head_dim,
            dtype=dtype, device=self.device,
        )
        self.v_buffer = torch.zeros_like(self.k_buffer)

        # Free list as a stack (LIFO — most-recently-freed slot is
        # popped first, which gives temporal locality for cross_kv that
        # gets re-read shortly after write).  Slot 0 is reserved.
        self._free: deque = deque(range(1, self._pool_size))
        self._n_alloc = 0

        # Stats (high-water mark useful for tuning pool_size).
        self._high_water = 0

        # Graph capture guard.  When True, alloc that would change pool
        # state is allowed (the free list mutates, but tensors don't),
        # while any operation that would replace tensors raises.
        self._in_capture = False

        # Bumped whenever grow replaces the K/V storage so graph runners
        # can invalidate every entry captured against the old pointers.
        self.pool_version = 0

        # Optional flashinfer paged cross attention helper.  Worker
        # attaches its :class:`SpecBlockFlashInferCross` instance here
        # so the model's attention forward can reach it via the cache
        # tuple.  None means "fall back to Triton paged kernel".
        self.flashinfer_cross = None

        logger.info(
            "SpecBlockKVPool initialized: layers=%d, n_heads=%d, "
            "head_dim=%d, dtype=%s, initial_pool_size=%d (~%.2f GB), "
            "max_pool_size=%d",
            num_layers, n_heads, head_dim, str(dtype),
            self._pool_size, self._memory_gb(), max_pool_size,
        )

    # ------------------------------------------------------------
    #  Sizing helpers
    # ------------------------------------------------------------

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def n_alloc(self) -> int:
        return self._n_alloc

    @property
    def n_free(self) -> int:
        return len(self._free)

    @property
    def high_water(self) -> int:
        return self._high_water

    def _memory_gb(self) -> float:
        bytes_per_slot = self.n_heads * self.head_dim * self.k_buffer.element_size()
        total = 2 * self.num_layers * self._pool_size * bytes_per_slot  # k+v
        return total / (1 << 30)

    # ------------------------------------------------------------
    #  Capture mode (cuda graph runner integration)
    # ------------------------------------------------------------

    @contextlib.contextmanager
    def capture_mode(self) -> Iterator[None]:
        """Mark a region as cuda-graph capture.

        Inside this ctx, the pool tensors must not be replaced.  ``alloc``
        and ``free`` may still mutate the Python free list when no grow is
        needed; graphs capture only the underlying tensor data, not this
        bookkeeping.

        Nested capture_mode calls are not allowed.
        """
        if self._in_capture:
            raise RuntimeError(
                "[SpecBlockKVPool] capture_mode is not re-entrant; "
                "an outer cuda graph capture is already active."
            )
        self._in_capture = True
        try:
            yield
        finally:
            self._in_capture = False

    # ------------------------------------------------------------
    #  alloc / free
    # ------------------------------------------------------------

    def alloc(self, n: int) -> torch.Tensor:
        """Allocate ``n`` slots.  Returns a [n] int64 tensor of indices.

        Grows incrementally on overflow up to ``max_pool_size``.  Inside
        a ``capture_mode()`` ctx, grow is forbidden (raises) — caller
        must pre-grow before entering capture.

        Raises:
            RuntimeError: if would grow inside capture_mode, or alloc
                exceeds ``max_pool_size``.
        """
        if n <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        if len(self._free) < n:
            if self._in_capture:
                raise RuntimeError(
                    f"[SpecBlockKVPool] alloc({n}) inside capture_mode would "
                    f"trigger grow (free={len(self._free)}, n_alloc={self._n_alloc}, "
                    f"pool_size={self._pool_size}).  Pre-grow before "
                    f"entering capture or reduce concurrent reqs."
                )
            new_size = self._pool_size + max(n, _DEFAULT_GROW_CHUNK)
            self._grow(new_size)

        # Pop n indices.  Use a list comprehension for speed; deque's
        # `popleft` is O(1).  Sort the popped indices ascending for
        # monotonic memory access pattern when the caller batch-writes.
        indices_list: List[int] = [self._free.popleft() for _ in range(n)]
        indices_list.sort()
        out = torch.tensor(indices_list, dtype=torch.int64, device=self.device)
        self._n_alloc += n
        if self._n_alloc > self._high_water:
            self._high_water = self._n_alloc
        return out

    def free(self, indices: torch.Tensor) -> None:
        """Release ``indices`` back to the free list.  No-op on empty."""
        if indices is None:
            return
        if torch.is_tensor(indices):
            if indices.numel() == 0:
                return
            indices_list = indices.tolist()
        else:
            indices_list = list(indices)
        for i in indices_list:
            ii = int(i)
            if ii == 0:
                # Slot 0 is the reserved zero sentinel; never free.
                continue
            self._free.append(ii)
        self._n_alloc -= len(indices_list)

    def _grow(self, new_size: int) -> None:
        """Grow the pool to ``new_size`` slots (preserves existing data).

        Bumps ``pool_version`` so cuda graph runners caching graphs by
        pool_version can invalidate.
        """
        if new_size > self.max_pool_size:
            raise RuntimeError(
                f"[SpecBlockKVPool] grow request {new_size} exceeds "
                f"max_pool_size {self.max_pool_size}.  Current alloc="
                f"{self._n_alloc}, high_water={self._high_water}.  "
                f"Increase SPECBLOCK_KV_POOL_MAX or reduce concurrent reqs."
            )
        if new_size <= self._pool_size:
            return

        old_size = self._pool_size
        new_k = torch.zeros(
            self.num_layers, new_size, self.n_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        new_v = torch.zeros_like(new_k)
        new_k[:, :old_size, :, :].copy_(self.k_buffer)
        new_v[:, :old_size, :, :].copy_(self.v_buffer)
        self.k_buffer = new_k
        self.v_buffer = new_v
        for i in range(old_size, new_size):
            self._free.append(i)
        self._pool_size = new_size
        self.pool_version += 1
        logger.info(
            "SpecBlockKVPool grew: %d -> %d slots (~%.2f GB), version=%d",
            old_size, new_size, self._memory_gb(), self.pool_version,
        )

    # ------------------------------------------------------------
    #  Read / write
    # ------------------------------------------------------------

    def set_kv(
        self,
        layer_id: int,
        indices: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """Write K/V at ``indices`` of layer ``layer_id``.

        Args:
            layer_id: int in [0, num_layers).
            indices: [N] int64 indices into the pool.
            cache_k: [N, n_heads, head_dim] (must match self.n_heads).
            cache_v: same shape.
        """
        if indices.numel() == 0:
            return
        if cache_k.shape != cache_v.shape:
            raise ValueError(
                f"set_kv: cache_k {tuple(cache_k.shape)} != cache_v "
                f"{tuple(cache_v.shape)}"
            )
        # Cast to pool dtype if needed (most callers already pass bf16).
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        # Index assign.  PyTorch handles int64 index broadcasting.
        self.k_buffer[layer_id, indices] = cache_k
        self.v_buffer[layer_id, indices] = cache_v

    def set_kv_padded(
        self,
        layer_id: int,
        indices: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        valid_slots: torch.Tensor,
    ) -> None:
        """Write only each row's valid prefix from padded batched K/V.

        ``indices`` is ``[B, max_slots]`` and ``cache_k/cache_v`` are
        ``[B, n_heads, max_slots, head_dim]``.  ``valid_slots[b]`` gives the
        number of live entries in row ``b``.  Padding locations are never
        loaded or written, which keeps reserved pool slot 0 zero.
        """
        if indices.numel() == 0:
            return
        if cache_k.shape != cache_v.shape:
            raise ValueError(
                f"set_kv_padded: cache_k {tuple(cache_k.shape)} != cache_v "
                f"{tuple(cache_v.shape)}"
            )
        B, n_heads, max_slots, head_dim = cache_k.shape
        if indices.shape != (B, max_slots):
            raise ValueError(
                f"set_kv_padded: indices {tuple(indices.shape)} != "
                f"({B}, {max_slots})"
            )
        if n_heads != self.n_heads or head_dim != self.head_dim:
            raise ValueError(
                f"set_kv_padded: cache shape heads/dim=({n_heads}, {head_dim}) "
                f"!= pool ({self.n_heads}, {self.head_dim})"
            )
        if valid_slots.shape != (B,):
            raise ValueError(
                f"set_kv_padded: valid_slots {tuple(valid_slots.shape)} != ({B},)"
            )

        k_pool = self.k_buffer[layer_id]
        v_pool = self.v_buffer[layer_id]
        block_hd = 256
        grid = (B * max_slots, triton.cdiv(n_heads * head_dim, block_hd))
        _set_kv_padded_kernel[grid](
            k_pool,
            v_pool,
            indices,
            cache_k,
            cache_v,
            valid_slots,
            max_slots,
            stride_pool_slot=k_pool.stride(0),
            stride_pool_head=k_pool.stride(1),
            stride_pool_dim=k_pool.stride(2),
            stride_indices_batch=indices.stride(0),
            stride_indices_slot=indices.stride(1),
            stride_cache_batch=cache_k.stride(0),
            stride_cache_head=cache_k.stride(1),
            stride_cache_slot=cache_k.stride(2),
            stride_cache_dim=cache_k.stride(3),
            N_HEADS=n_heads,
            HEAD_DIM=head_dim,
            BLOCK_HD=block_hd,
        )

    def get_k(self, layer_id: int) -> torch.Tensor:
        """Return the entire K buffer for ``layer_id`` (no-copy view).

        Shape: [pool_size, n_heads, head_dim].  Callers gather their
        own slots via fancy indexing.
        """
        return self.k_buffer[layer_id]

    def get_v(self, layer_id: int) -> torch.Tensor:
        """Return the entire V buffer for ``layer_id`` (no-copy view)."""
        return self.v_buffer[layer_id]

    # ------------------------------------------------------------
    #  Convenience
    # ------------------------------------------------------------

    def reset(self) -> None:
        """Free all allocated slots (keep the buffer)."""
        self._free.clear()
        self._free.extend(range(1, self._pool_size))
        self._n_alloc = 0
        # Don't zero the buffer — slot 0 stays zero, others get
        # overwritten on next alloc + set_kv.

    def __repr__(self) -> str:
        return (
            f"SpecBlockKVPool(pool_size={self._pool_size}, "
            f"n_alloc={self._n_alloc}, n_free={self.n_free}, "
            f"high_water={self._high_water}, "
            f"layers={self.num_layers}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, dtype={self.dtype})"
        )
