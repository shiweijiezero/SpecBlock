"""GPU-resident fixed-capacity planning for SpecBlock refresh.

This module deliberately owns no scheduler objects, request objects, or KV-pool
allocator. Those remain Python-owned today. It converts the ragged accepted
chains emitted by tree acceptance into fixed-capacity CUDA-graph inputs and
prepares the append-only cross-KV layout consumed by
``SpecBlockKVPool.set_kv_padded``.

It uses only tensor operations: it never calls ``item()``, ``tolist()``, or
loops over requests. Accept and cross-KV capacity overflow are returned as
device booleans instead of being silently truncated. Callers must select a
larger eager/graph bucket before persisting an overflowed row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_refresh_hidden_kernel(
    flat_hidden,
    counts,
    packed_hidden,
    hidden_size: tl.constexpr,
    accept_capacity: tl.constexpr,
    block_hidden: tl.constexpr,
    block_batch: tl.constexpr,
):
    row = tl.program_id(0)
    hidden_block = tl.program_id(1)
    batch_idx = row // accept_capacity
    column = row % accept_capacity
    batch_offsets = tl.arange(0, block_batch)
    prior_counts = tl.load(
        counts + batch_offsets,
        mask=batch_offsets < batch_idx,
        other=-1,
    ).to(tl.int64) + 1
    source_row = tl.sum(prior_counts, axis=0) + column
    count = tl.load(counts + batch_idx).to(tl.int64) + 1
    hidden_offsets = hidden_block * block_hidden + tl.arange(0, block_hidden)
    mask = (column < count) & (hidden_offsets < hidden_size)
    values = tl.load(
        flat_hidden + source_row * hidden_size + hidden_offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(
        packed_hidden + row * hidden_size + hidden_offsets,
        values,
        mask=hidden_offsets < hidden_size,
    )


@triton.jit
def _pack_refresh_metadata_kernel(
    flat_tokens,
    accept_lengths,
    cross_positions,
    existing_cross_counts,
    tokens,
    pos_ids,
    n_per_req,
    chain_mask,
    accept_overflow,
    start_positions,
    cross_counts,
    new_cross_valid_slots,
    K: tl.constexpr,
    accept_capacity: tl.constexpr,
    block_accept: tl.constexpr,
    block_positions: tl.constexpr,
    block_batch: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    accept_length = tl.load(accept_lengths + batch_idx).to(tl.int64)
    count = accept_length + 1
    batch_offsets = tl.arange(0, block_batch)
    prior_counts = tl.load(
        accept_lengths + batch_offsets,
        mask=batch_offsets < batch_idx,
        other=-1,
    ).to(tl.int64) + 1
    source_offset = tl.sum(prior_counts, axis=0)

    columns = tl.arange(0, block_accept)
    valid = columns < count
    token_source = tl.minimum(columns + 1, count - 1)
    token_values = tl.load(
        flat_tokens + source_offset + token_source,
        mask=valid,
        other=0,
    )
    tl.store(tokens + batch_idx * accept_capacity + columns, token_values,
             mask=columns < accept_capacity)
    tl.store(chain_mask + batch_idx * accept_capacity + columns, valid,
             mask=columns < accept_capacity)

    start = tl.load(cross_positions + batch_idx).to(tl.int64) + (
        accept_length > 0
    )
    position_columns = tl.arange(0, block_positions)
    position_values = start + 1 + position_columns // K + position_columns % K
    tl.store(
        pos_ids + batch_idx * accept_capacity * K + position_columns,
        position_values,
        mask=position_columns < accept_capacity * K,
    )

    new_slots = count * K
    tl.store(n_per_req + batch_idx, count)
    tl.store(accept_overflow + batch_idx, count > accept_capacity)
    tl.store(start_positions + batch_idx, start)
    tl.store(
        cross_counts + batch_idx,
        tl.load(existing_cross_counts + batch_idx).to(tl.int64) + new_slots,
    )
    tl.store(new_cross_valid_slots + batch_idx, new_slots)


@dataclass(frozen=True)
class SpecBlockRefreshTensorPlan:
    """Fixed-capacity refresh inputs and persistent cross-KV append plan.

    ``n_per_req = accept_lengths + 1`` includes the bonus token. Valid prefix
    lengths of the returned tensors are thus entirely GPU-resident.
    """

    hidden: torch.Tensor                 # [B, A, 3H]
    tokens: torch.Tensor                 # [B, A]
    pos_ids: torch.Tensor                # [B, A*K]
    n_per_req: torch.Tensor              # [B], int64
    chain_mask: torch.Tensor             # [B, A], bool
    accept_overflow: torch.Tensor        # [B], bool
    start_positions: torch.Tensor        # [B], int64
    # ``cross_loc`` is present only when ``output_cross_capacity`` was
    # requested. Input-only callers can defer the persistent append plan.
    cross_loc: Optional[torch.Tensor]    # [B, C_out]
    cross_counts: torch.Tensor           # [B], int64
    new_cross_valid_slots: torch.Tensor  # [B], int64
    cross_overflow: Optional[torch.Tensor]  # [B], bool


def _check_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape={tuple(tensor.shape)}.")


def _pack_flat_chains(
    flat: torch.Tensor,
    counts: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack contiguous ragged rows using CUDA offsets, gather, and masks.

    The gather index is clamped only to make padded physical reads safe; the
    resulting mask zeroes every padded lane. The caller's bucket-selection
    invariant is ``counts <= capacity``.
    """
    batch_size = counts.shape[0]
    offsets = torch.cumsum(counts, dim=0) - counts
    columns = torch.arange(capacity, device=flat.device)
    valid = columns[None, :] < counts[:, None]
    # A valid refresh always contains its bonus token, so flat has >= B rows.
    # Clamp is for invalid padded gathers only and does not affect valid data.
    gather_idx = (offsets[:, None] + columns[None, :]).clamp_max(flat.shape[0] - 1)
    packed = flat[gather_idx]
    packed = torch.where(
        valid[(...,) + (None,) * (flat.ndim - 1)],
        packed,
        torch.zeros((), dtype=flat.dtype, device=flat.device),
    )
    return packed, valid, offsets


def _build_specblock_refresh_tensor_plan_cuda(
    flat_chain_tokens: torch.Tensor,
    flat_chain_hidden: torch.Tensor,
    accept_lengths: torch.Tensor,
    cross_positions: torch.Tensor,
    existing_cross_counts: torch.Tensor,
    *,
    K: int,
    accept_capacity: int,
) -> SpecBlockRefreshTensorPlan:
    batch_size = accept_lengths.shape[0]
    hidden_size = flat_chain_hidden.shape[1]
    device = flat_chain_tokens.device
    hidden = torch.empty(
        (batch_size, accept_capacity, hidden_size),
        dtype=flat_chain_hidden.dtype,
        device=device,
    )
    tokens = torch.empty(
        (batch_size, accept_capacity),
        dtype=flat_chain_tokens.dtype,
        device=device,
    )
    pos_ids = torch.empty(
        (batch_size, accept_capacity * K),
        dtype=torch.long,
        device=device,
    )
    n_per_req = torch.empty(batch_size, dtype=torch.long, device=device)
    chain_mask = torch.empty(
        (batch_size, accept_capacity), dtype=torch.bool, device=device
    )
    accept_overflow = torch.empty(batch_size, dtype=torch.bool, device=device)
    start_positions = torch.empty(batch_size, dtype=torch.long, device=device)
    cross_counts = torch.empty(batch_size, dtype=torch.long, device=device)
    new_cross_valid_slots = torch.empty(
        batch_size, dtype=torch.long, device=device
    )

    block_batch = triton.next_power_of_2(batch_size)
    block_hidden = 256
    _pack_refresh_hidden_kernel[
        (batch_size * accept_capacity, triton.cdiv(hidden_size, block_hidden))
    ](
        flat_chain_hidden,
        accept_lengths,
        hidden,
        hidden_size=hidden_size,
        accept_capacity=accept_capacity,
        block_hidden=block_hidden,
        block_batch=block_batch,
    )
    _pack_refresh_metadata_kernel[(batch_size,)](
        flat_chain_tokens,
        accept_lengths,
        cross_positions,
        existing_cross_counts,
        tokens,
        pos_ids,
        n_per_req,
        chain_mask,
        accept_overflow,
        start_positions,
        cross_counts,
        new_cross_valid_slots,
        K=K,
        accept_capacity=accept_capacity,
        block_accept=triton.next_power_of_2(accept_capacity),
        block_positions=triton.next_power_of_2(accept_capacity * K),
        block_batch=block_batch,
    )
    return SpecBlockRefreshTensorPlan(
        hidden=hidden,
        tokens=tokens,
        pos_ids=pos_ids,
        n_per_req=n_per_req,
        chain_mask=chain_mask,
        accept_overflow=accept_overflow,
        start_positions=start_positions,
        cross_loc=None,
        cross_counts=cross_counts,
        new_cross_valid_slots=new_cross_valid_slots,
        cross_overflow=None,
    )


def build_specblock_refresh_tensor_plan(
    flat_chain_tokens: torch.Tensor,
    flat_chain_hidden: torch.Tensor,
    accept_lengths: torch.Tensor,
    cross_positions: torch.Tensor,
    existing_cross_loc: torch.Tensor,
    existing_cross_counts: torch.Tensor,
    allocated_cross_loc: torch.Tensor,
    *,
    K: int,
    output_cross_capacity: Optional[int] = None,
) -> SpecBlockRefreshTensorPlan:
    """Build graph-ready refresh tensors without host scalar extraction.

    ``flat_chain_*`` are request-major flattened accepted chains. Each
    ``accept_lengths[b]`` excludes the bonus token, hence
    ``n_per_req[b] = accept_lengths[b] + 1``. ``allocated_cross_loc`` must
    have fixed shape ``[B, A*K]`` and contain sentinel zero in each invalid
    tail; obtaining those slots remains outside this helper because the pool
    allocator currently owns a Python free list.

    The returned ``tokens`` implements the existing refresh convention:
    within an N-token chain it is ``[id_1, ..., id_(N-1), id_(N-1)]``. Thus
    N=1 produces the bonus token once; N>1 duplicates the final token. This
    exactly feeds ``update_cache_and_draft_graph_safe`` while ``hidden`` keeps
    the unshifted target hidden-state chain.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")
    if output_cross_capacity is not None and output_cross_capacity <= 0:
        raise ValueError(
            f"output_cross_capacity must be positive, got {output_cross_capacity}."
        )
    for name, tensor, rank in (
        ("flat_chain_tokens", flat_chain_tokens, 1),
        ("flat_chain_hidden", flat_chain_hidden, 2),
        ("accept_lengths", accept_lengths, 1),
        ("cross_positions", cross_positions, 1),
        ("existing_cross_loc", existing_cross_loc, 2),
        ("existing_cross_counts", existing_cross_counts, 1),
        ("allocated_cross_loc", allocated_cross_loc, 2),
    ):
        _check_rank(name, tensor, rank)
    batch_size = accept_lengths.shape[0]
    if (
        flat_chain_hidden.shape[0] != flat_chain_tokens.shape[0]
        or cross_positions.shape != accept_lengths.shape
        or existing_cross_counts.shape != accept_lengths.shape
        or existing_cross_loc.shape[0] != batch_size
        or allocated_cross_loc.shape[0] != batch_size
    ):
        raise ValueError("Refresh plan batch dimensions must agree.")
    if allocated_cross_loc.shape[1] % K:
        raise ValueError(
            "allocated_cross_loc width must be a multiple of K, got "
            f"width={allocated_cross_loc.shape[1]}, K={K}."
        )
    device = flat_chain_tokens.device
    if any(t.device != device for t in (
        flat_chain_hidden, accept_lengths, cross_positions, existing_cross_loc,
        existing_cross_counts, allocated_cross_loc,
    )):
        raise ValueError("All refresh-plan tensors must share a device.")

    accept_capacity = allocated_cross_loc.shape[1] // K
    if (
        flat_chain_tokens.is_cuda
        and output_cross_capacity is None
        and flat_chain_tokens.is_contiguous()
        and flat_chain_hidden.is_contiguous()
    ):
        return _build_specblock_refresh_tensor_plan_cuda(
            flat_chain_tokens,
            flat_chain_hidden,
            accept_lengths,
            cross_positions,
            existing_cross_counts,
            K=K,
            accept_capacity=accept_capacity,
        )

    n_per_req = accept_lengths.to(torch.long) + 1
    accept_overflow = n_per_req > accept_capacity
    hidden, chain_mask, offsets = _pack_flat_chains(
        flat_chain_hidden, n_per_req, accept_capacity
    )

    # The token stream is shifted by one within each ragged chain, with the
    # final token duplicated. Gather row-wise from the flat request-major data.
    columns = torch.arange(accept_capacity, device=device)
    token_source = torch.minimum(columns[None, :] + 1, n_per_req[:, None] - 1)
    token_indices = (offsets[:, None] + token_source).clamp_max(
        flat_chain_tokens.shape[0] - 1
    )
    tokens = flat_chain_tokens[token_indices]
    tokens = torch.where(chain_mask, tokens, torch.zeros((), dtype=tokens.dtype, device=device))

    # Zero-accept requests retain their persistent position. Positive accepts
    # advance it once before replay, matching the worker's existing semantics.
    start_positions = cross_positions.to(torch.long) + (accept_lengths > 0).to(torch.long)
    # Each accepted-chain position expands to its K-token draft block. The
    # draft's block positions overlap by K-1, so token j / block slot s uses
    # ``start + 1 + j + s`` rather than a flat j*K+s offset.
    pos_columns = (
        torch.arange(accept_capacity, device=device)[:, None]
        + torch.arange(K, device=device)[None, :]
    ).reshape(-1)
    pos_ids = start_positions[:, None] + 1 + pos_columns[None, :]

    new_cross_valid_slots = n_per_req * K
    output_cross_counts = existing_cross_counts.to(torch.long) + new_cross_valid_slots
    cross_loc: Optional[torch.Tensor] = None
    cross_overflow: Optional[torch.Tensor] = None
    if output_cross_capacity is not None:
        cross_overflow = output_cross_counts > output_cross_capacity
        cross_loc = torch.zeros(
            (batch_size, output_cross_capacity),
            dtype=existing_cross_loc.dtype,
            device=device,
        )
        copy_width = min(existing_cross_loc.shape[1], output_cross_capacity)
        cross_loc[:, :copy_width] = existing_cross_loc[:, :copy_width]
        append_columns = existing_cross_counts.to(torch.long)[:, None] + torch.arange(
            allocated_cross_loc.shape[1], device=device
        )[None, :]
        safe_columns = append_columns.clamp_max(output_cross_capacity - 1)
        prior = cross_loc.gather(1, safe_columns)
        append_valid = torch.arange(allocated_cross_loc.shape[1], device=device)[None, :] < new_cross_valid_slots[:, None]
        append_values = torch.where(append_valid, allocated_cross_loc, prior)
        cross_loc.scatter_(1, safe_columns, append_values)

    return SpecBlockRefreshTensorPlan(
        hidden=hidden,
        tokens=tokens,
        pos_ids=pos_ids,
        n_per_req=n_per_req,
        chain_mask=chain_mask,
        accept_overflow=accept_overflow,
        start_positions=start_positions,
        cross_loc=cross_loc,
        cross_counts=output_cross_counts,
        new_cross_valid_slots=new_cross_valid_slots,
        cross_overflow=cross_overflow,
    )
