"""Request-level true batching for HF SpecBlock decoders.

Target prefill and tree verification are batched across active requests.  Draft
prefill and accepted-token updates use a shared ragged KV cache; tree topology and
block-2 expansion remain request-local until their pending leaves are prepared.
"""

from __future__ import annotations

import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, CacheLayerMixin

from .runtime_capabilities import RUNTIME_CAPABILITIES


TreeResult = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list,
    list,
    list,
]


@dataclass
class _RequestState:
    conversation: List[Dict[str, str]]
    prompt_ids: torch.Tensor
    prompt_len: int
    tree_budget: int
    target_prefix_len: int = 0
    draft_cache: Any = None
    draft_position: int = 0
    tree: TreeResult | None = None
    output_tokens: List[int] = field(default_factory=list)
    current_token: torch.Tensor | None = None
    iterations: int = 0
    accept_lengths_raw: List[int] = field(default_factory=list)
    rank_stats_raw: List[list] = field(default_factory=list)
    coverage_raw: List[tuple] = field(default_factory=list)
    finished: bool = False


class _CudaPhaseTimer:
    """Collect CUDA event pairs without synchronizing between phases."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def start(self):
        if not self.enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, name: str, start):
        if not self.enabled:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events[name].append((start, end))

    def elapsed_seconds(self, name: str) -> float:
        if not self.enabled:
            return 0.0
        return sum(start.elapsed_time(end) for start, end in self.events.get(name, ())) / 1000.0


class _DenseTargetCacheLayer(CacheLayerMixin):
    """One fixed-capacity target KV layer with externally controlled writes."""

    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = int(max_cache_len)
        self.attention_width = 0
        self.committed_width = 0
        self.write_start = 0
        self.gap_start = 0
        self.gap_mask = None

    def lazy_initialization(self, key_states: torch.Tensor):
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype = key_states.dtype
        self.device = key_states.device
        shape = (
            self.max_batch_size,
            self.num_heads,
            self.max_cache_len,
            self.head_dim,
        )
        self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        cache_position = None if cache_kwargs is None else cache_kwargs.get(
            "cache_position"
        )
        if cache_position is None:
            raise ValueError("dense target cache requires explicit cache_position")
        if key_states.shape[0] > self.max_batch_size:
            raise ValueError(
                f"cache batch {key_states.shape[0]} exceeds {self.max_batch_size}"
            )
        query_len = key_states.shape[-2]
        if cache_position.shape[0] != query_len:
            raise ValueError("cache_position length must match target query length")
        write_end = self.write_start + query_len
        if write_end > self.max_cache_len or write_end > self.attention_width:
            raise ValueError("target KV write exceeds the prepared cache range")
        rows = slice(0, key_states.shape[0])
        if self.gap_mask is None and self.write_start > self.committed_width:
            gap = slice(self.committed_width, self.write_start)
            # B=1/equal-prefix fast path: only the reserved accepted-path window
            # is masked between committed KV and the staged tree.
            self.keys[rows, :, gap, :].zero_()
            self.values[rows, :, gap, :].zero_()
        elif self.gap_mask is not None:
            # Unequal-prefix batches also have row-local padding before the common
            # tree start.  Clear every masked slot while preserving each row's
            # committed prefix.  Some fused attention reductions still touch
            # masked KV, so stale rejected nodes must remain numerically benign.
            if self.gap_mask.shape[0] != key_states.shape[0]:
                raise ValueError("target KV gap mask does not match active rows")
            gap = slice(self.gap_start, self.write_start)
            mask = self.gap_mask[:, None, :, None]
            self.keys[rows, :, gap, :].masked_fill_(mask, 0)
            self.values[rows, :, gap, :].masked_fill_(mask, 0)
        write_slice = slice(self.write_start, write_end)
        self.keys[rows, :, write_slice, :].copy_(key_states)
        self.values[rows, :, write_slice, :].copy_(value_states)
        return (
            self.keys[rows, :, :self.attention_width, :],
            self.values[rows, :, :self.attention_width, :],
        )

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        return self.attention_width, 0

    def get_seq_length(self) -> int:
        return self.committed_width

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def resize(self, max_cache_len: int, preserve_width: int):
        max_cache_len = int(max_cache_len)
        if max_cache_len <= self.max_cache_len:
            return
        self.max_cache_len = max_cache_len
        if not self.is_initialized:
            return
        shape = (
            self.max_batch_size,
            self.num_heads,
            self.max_cache_len,
            self.head_dim,
        )
        # The reserved commit gap is masked but still participates in some fused
        # attention reductions.  Keep newly exposed slots finite after growth.
        keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        values = torch.zeros(shape, dtype=self.dtype, device=self.device)
        if preserve_width:
            keys[:, :, :preserve_width, :].copy_(
                self.keys[:, :, :preserve_width, :]
            )
            values[:, :, :preserve_width, :].copy_(
                self.values[:, :, :preserve_width, :]
            )
        self.keys = keys
        self.values = values


class _DraftBatchCache:
    """Shared draft KV storage with one logical length per active request."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_cache_len: int,
        device: torch.device,
        original_engine_b1: bool,
    ):
        self.layers = [
            [None, None, None, int(max_cache_len), None, True]
            for _ in range(int(num_layers))
        ]
        self.lengths_host = [0] * int(batch_size)
        self.use_compiled_b1 = bool(original_engine_b1)
        self.lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for layer in self.layers:
            layer[2] = self.lengths

    @property
    def active_batch_size(self) -> int:
        return len(self.lengths_host)

    def _upload_lengths(self):
        if not self.lengths_host:
            self.lengths = self.lengths[:0]
            for layer in self.layers:
                layer[2] = self.lengths
            return
        host = torch.tensor(self.lengths_host, dtype=torch.long, pin_memory=True)
        self.lengths.copy_(host, non_blocking=True)

    def advance(self, valid_lengths: Sequence[int], slots_per_position: int):
        if len(valid_lengths) != self.active_batch_size:
            raise ValueError("draft valid lengths must match active cache rows")
        K = int(slots_per_position)
        for row, valid in enumerate(valid_lengths):
            self.lengths_host[row] += int(valid) * K
        self._upload_lengths()

    def rollback_rows(self, rows: Sequence[int], slots_per_position: int):
        K = int(slots_per_position)
        for row in rows:
            row = int(row)
            if self.lengths_host[row] < K:
                raise ValueError("cannot roll back an empty draft cache row")
            self.lengths_host[row] -= K
        self._upload_lengths()

    def row_view(self, row: int):
        row = int(row)
        count = self.lengths_host[row]
        return [
            [
                layer[0][row:row + 1] if layer[0] is not None else None,
                layer[1][row:row + 1] if layer[1] is not None else None,
                count,
                layer[3],
                None,
            ]
            for layer in self.layers
        ]

    def prefix_layers(self, row_count: int):
        """Return cache views for a non-empty contiguous request prefix."""
        row_count = int(row_count)
        if row_count <= 0 or row_count > self.active_batch_size:
            raise ValueError("draft cache prefix row count is out of range")
        if row_count == self.active_batch_size:
            return self.layers
        prefix = []
        for layer in self.layers:
            if layer[0] is None or layer[1] is None:
                raise RuntimeError(
                    "draft cache must be initialized before slicing inactive rows"
                )
            view = list(layer)
            view[0] = layer[0][:row_count]
            view[1] = layer[1][:row_count]
            view[2] = self.lengths[:row_count]
            prefix.append(view)
        return prefix

    def compact_rows(self, row_indices: Sequence[int]):
        row_indices = [int(index) for index in row_indices]
        if all(index == expected for expected, index in enumerate(row_indices)):
            self.lengths_host = self.lengths_host[:len(row_indices)]
            self.lengths = self.lengths[:len(row_indices)]
            for layer in self.layers:
                if layer[0] is not None:
                    layer[0] = layer[0][:len(row_indices)]
                    layer[1] = layer[1][:len(row_indices)]
                layer[2] = self.lengths
            return
        indices = torch.tensor(row_indices, dtype=torch.long, device=self.lengths.device)
        used_width = max(self.lengths_host[index] for index in row_indices)
        for layer in self.layers:
            if layer[0] is None:
                continue
            selected_keys = layer[0][:, :, :used_width, :].index_select(0, indices)
            selected_values = layer[1][:, :, :used_width, :].index_select(0, indices)
            layer[0][:len(row_indices), :, :used_width, :].copy_(selected_keys)
            layer[1][:len(row_indices), :, :used_width, :].copy_(selected_values)
            layer[0] = layer[0][:len(row_indices)]
            layer[1] = layer[1][:len(row_indices)]
        self.lengths_host = [self.lengths_host[index] for index in row_indices]
        self.lengths = self.lengths.index_select(0, indices)
        for layer in self.layers:
            layer[2] = self.lengths


class _DenseTargetCache(Cache):
    """Persistent dense KV storage for a shrinking request batch.

    Committed request KV always lives at physical ``[0:logical_length)``.  Each
    tree is staged immediately after the current committed tail plus enough room
    for the largest accepted path, so target attention never scans the unused
    generation budget.  Batch rows are compacted only when requests finish.
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        prompt_width: int,
        max_new_tokens: int,
        max_tree_width: int,
    ):
        self.max_batch_capacity = int(max_batch_size)
        self.prompt_width = int(prompt_width)
        self.committed_capacity = self.prompt_width + int(max_new_tokens)
        self.max_tree_width = int(max_tree_width)
        # Staged trees start after a reserved accepted-path window.  Account for
        # both regions up front so late decode iterations never expose an
        # uninitialized masked gap through a mid-generation resize.
        self.capacity = self.committed_capacity + 2 * self.max_tree_width
        self.active_batch_size = self.max_batch_capacity
        self._attention_width = self.prompt_width
        self._committed_width = self.prompt_width
        self._tree_start = self.prompt_width
        self._tree_width = 0
        self._commit_reserve = 0
        super().__init__(
            layers=[
                _DenseTargetCacheLayer(self.capacity) for _ in range(num_layers)
            ]
        )
        self._sync_layer_state(write_start=0)

    def _sync_layer_state(
        self,
        write_start: int,
        gap_start: int = 0,
        gap_mask: torch.Tensor | None = None,
    ):
        for layer in self.layers:
            layer.attention_width = self._attention_width
            layer.committed_width = self._committed_width
            layer.write_start = int(write_start)
            layer.gap_start = int(gap_start)
            layer.gap_mask = gap_mask

    def prepare_prefill(self):
        self.active_batch_size = self.max_batch_capacity
        self._attention_width = self.prompt_width
        self._committed_width = self.prompt_width
        self._tree_start = self.prompt_width
        self._tree_width = 0
        self._commit_reserve = 0
        self._sync_layer_state(write_start=0)

    @property
    def tree_start(self) -> int:
        return self._tree_start

    @property
    def attention_width(self) -> int:
        return self._attention_width

    def prepare_tree(
        self,
        tree_width: int,
        prefix_lengths: Sequence[int],
        commit_reserve: int,
    ):
        tree_width = int(tree_width)
        commit_reserve = int(commit_reserve)
        if len(prefix_lengths) != self.active_batch_size:
            raise ValueError("prefix lengths must match active cache rows")
        if tree_width <= 0:
            raise ValueError("target tree width must be positive")
        if commit_reserve < 0 or commit_reserve > tree_width:
            raise ValueError("target commit reserve must fit inside the tree")
        prefix_lengths = [int(length) for length in prefix_lengths]
        if any(
            length < 0 or length > self.committed_capacity
            for length in prefix_lengths
        ):
            raise ValueError("committed target KV width is out of range")
        committed_width = max(prefix_lengths)

        tree_start = committed_width + commit_reserve
        attention_width = tree_start + tree_width
        self.max_tree_width = max(self.max_tree_width, tree_width)
        if attention_width > self.capacity:
            self.capacity = max(
                attention_width,
                self.committed_capacity + 2 * self.max_tree_width,
            )
            for layer in self.layers:
                layer.resize(self.capacity, preserve_width=committed_width)
        self._attention_width = attention_width
        self._committed_width = committed_width
        self._tree_start = tree_start
        self._tree_width = tree_width
        self._commit_reserve = commit_reserve
        gap_start = committed_width
        gap_mask = None
        if min(prefix_lengths) != committed_width:
            gap_start = min(prefix_lengths)
            device = self.layers[0].keys.device
            positions = torch.arange(
                gap_start,
                tree_start,
                dtype=torch.long,
                device=device,
            )
            row_prefixes = torch.tensor(
                prefix_lengths,
                dtype=torch.long,
                device=device,
            )
            gap_mask = positions.unsqueeze(0) >= row_prefixes.unsqueeze(1)
        self._sync_layer_state(
            write_start=tree_start,
            gap_start=gap_start,
            gap_mask=gap_mask,
        )

    def tree_cache_position(self, tree_width: int, device: torch.device):
        tree_width = int(tree_width)
        if tree_width != self._tree_width:
            raise ValueError("target tree width changed after cache preparation")
        return torch.arange(
            self._tree_start,
            self._tree_start + tree_width,
            dtype=torch.long,
            device=device,
        )

    def commit_paths(
        self,
        requests: Sequence[Tuple[int, int, torch.Tensor]],
    ) -> List[int]:
        if not requests:
            return []

        device = self.layers[0].keys.device
        metadata_values = []
        source_tensors = []
        counts = []
        committed_width = self._committed_width
        for row, logical_prefix_len, selected_tree_indices in requests:
            row = int(row)
            logical_prefix_len = int(logical_prefix_len)
            if row < 0 or row >= self.active_batch_size:
                raise ValueError("accepted target KV row is out of range")
            if logical_prefix_len < 0 or logical_prefix_len > self._committed_width:
                raise ValueError("accepted target KV prefix is out of range")
            if selected_tree_indices.device != device:
                raise ValueError("accepted target KV indices are on the wrong device")
            selected_tree_indices = selected_tree_indices.to(dtype=torch.long)
            count = int(selected_tree_indices.numel())
            committed_end = logical_prefix_len + count
            if count > self._commit_reserve:
                raise ValueError("accepted target KV exceeds the reserved path width")
            if committed_end > self.committed_capacity:
                raise ValueError("accepted target KV exceeds committed cache capacity")
            if committed_end > self._tree_start:
                raise ValueError("accepted target KV overlaps staged tree storage")

            counts.append(count)
            source_tensors.append(selected_tree_indices)
            metadata_values.extend(
                (row, destination)
                for destination in range(logical_prefix_len, committed_end)
            )
            committed_width = max(committed_width, committed_end)

        if not metadata_values:
            self._committed_width = committed_width
            return counts
        metadata_host = torch.tensor(
            metadata_values,
            dtype=torch.long,
            pin_memory=True,
        )
        metadata = metadata_host.to(device=device, non_blocking=True)
        sources = torch.cat(source_tensors).contiguous()
        torch._assert_async(
            ((sources >= 0) & (sources < self._tree_width)).all(),
            "accepted target KV source is out of range",
        )
        sources.add_(self._tree_start)
        from algorithms.target_kv_copy_triton import copy_selected_kv_

        for layer in self.layers:
            copy_selected_kv_(layer.keys, layer.values, metadata, sources)
        self._committed_width = committed_width
        return counts

    def compact_rows(self, row_indices: Sequence[int]):
        row_indices = [int(index) for index in row_indices]
        if all(index == expected for expected, index in enumerate(row_indices)):
            self.active_batch_size = len(row_indices)
            return
        indices = torch.tensor(
            row_indices,
            dtype=torch.long,
            device=self.layers[0].keys.device,
        )
        committed = self._committed_width
        for layer in self.layers:
            selected_keys = layer.keys[:, :, :committed, :].index_select(0, indices)
            selected_values = layer.values[:, :, :committed, :].index_select(0, indices)
            layer.keys[:len(row_indices), :, :committed, :].copy_(selected_keys)
            layer.values[:len(row_indices), :, :committed, :].copy_(selected_values)
        self.active_batch_size = len(row_indices)


def _clone_tree_result(result: TreeResult) -> TreeResult:
    """Own tree tensors before another request can reuse persistent buffers."""
    return (
        result[0].clone(),
        result[1].clone(),
        result[2].clone(),
        result[3].clone(),
        list(result[4]),
        list(result[5]),
        list(result[6]),
    )


def _tree_budget(algorithm, prompt_len: int) -> int:
    chooser = getattr(algorithm, "_batch_tree_budget_for_prompt", None)
    if chooser is None:
        return int(algorithm.total_tokens)
    return int(chooser(prompt_len))


def _with_tree_budget(algorithm, state: _RequestState, fn):
    original = algorithm.total_tokens
    algorithm.total_tokens = state.tree_budget
    try:
        return fn()
    finally:
        algorithm.total_tokens = original


def _set_request_ngram_context(algorithm, state: _RequestState):
    if not getattr(algorithm, "_ngram_cache_on", False):
        return
    full = state.prompt_ids[0].tolist() + state.output_tokens
    algorithm._ngram_prev2 = tuple(full[-2:]) if len(full) >= 2 else (-1, -1)


def _update_ngram_cache(algorithm, state: _RequestState, old_output_len: int):
    if not getattr(algorithm, "_ngram_cache_on", False):
        return
    full = state.prompt_ids[0].tolist() + state.output_tokens
    first_new = state.prompt_len + old_output_len
    start = max(0, first_new - 2)
    for idx in range(start, len(full) - 2):
        a, b, c = (int(full[idx]), int(full[idx + 1]), int(full[idx + 2]))
        table = algorithm._ngram_table.setdefault((a, b), {})
        table[c] = table.get(c, 0) + 1
    algorithm._ngram_prev2 = tuple(full[-2:]) if len(full) >= 2 else (-1, -1)


def _build_initial_draft(algorithm, state: _RequestState, hidden_3h: torch.Tensor):
    first_token = state.current_token
    shifted_ids = torch.cat((state.prompt_ids[:, 1:], first_token), dim=1)
    last_hidden = hidden_3h[:, -1:, :]
    (
        state.draft_cache,
        state.draft_position,
        b0_logits,
        b0_rank_logits,
        b0_draft_hidden,
        b0_ttt_kv,
    ) = algorithm.draft_model.prefill_and_draft(
        hidden_3h, shifted_ids, last_hidden, first_token
    )
    _set_request_ngram_context(algorithm, state)
    result = _with_tree_budget(
        algorithm,
        state,
        lambda: algorithm._build_tree_from_block1_dispatch(
            b0_logits,
            b0_rank_logits,
            b0_draft_hidden,
            b0_ttt_kv,
            first_token,
            state.draft_cache,
            state.draft_position - 1,
            temperature=0.0,
        ),
    )
    state.tree = _clone_tree_result(result)


def _run_ragged_draft_forward(
    algorithm,
    draft_cache: _DraftBatchCache,
    hidden_rows: Sequence[torch.Tensor],
    token_rows: Sequence[torch.Tensor],
    start_positions: Sequence[int],
    valid_lengths: Sequence[int],
):
    batch_size = len(hidden_rows)
    if not (
        len(token_rows) == batch_size
        and len(start_positions) == batch_size
        and len(valid_lengths) == batch_size
        and draft_cache.active_batch_size == batch_size
    ):
        raise ValueError("ragged draft inputs must match active cache rows")
    max_positions = max(int(value) for value in valid_lengths)
    if max_positions <= 0:
        raise ValueError("ragged draft forward requires at least one valid position")
    active_count = sum(int(value) > 0 for value in valid_lengths)
    if any(int(value) <= 0 for value in valid_lengths[:active_count]) or any(
        int(value) > 0 for value in valid_lengths[active_count:]
    ):
        raise ValueError(
            "zero-length ragged draft rows must form a contiguous suffix"
        )
    hidden_size = hidden_rows[0].shape[-1]
    device = hidden_rows[0].device
    hidden_batch = torch.zeros(
        active_count,
        max_positions,
        hidden_size,
        dtype=hidden_rows[0].dtype,
        device=device,
    )
    token_batch = torch.zeros(
        active_count,
        max_positions,
        dtype=torch.long,
        device=device,
    )
    for row, (hidden, tokens, valid) in enumerate(
        zip(
            hidden_rows[:active_count],
            token_rows[:active_count],
            valid_lengths[:active_count],
        )
    ):
        valid = int(valid)
        hidden_batch[row, :valid].copy_(hidden[0, :valid])
        token_batch[row, :valid].copy_(tokens[0, :valid])

    starts = torch.tensor(
        start_positions[:active_count], dtype=torch.long, device=device
    )
    lengths = torch.tensor(
        valid_lengths[:active_count], dtype=torch.long, device=device
    )
    K = int(algorithm.draft_model.K)
    # Padded rows still construct position IDs across ``max_positions``; size
    # the RoPE cache for those harmless tail IDs as well as the valid queries.
    max_position = (
        max(int(start) for start in start_positions[:active_count])
        + max_positions
        + K
        - 1
    )
    max_total_slots = max(
        count + int(valid) * K
        for count, valid in zip(
            draft_cache.lengths_host[:active_count],
            valid_lengths[:active_count],
        )
    )
    cache_capacity = min(int(layer[3]) for layer in draft_cache.layers)
    if max_total_slots > cache_capacity:
        raise RuntimeError(
            "draft cache prefix views cannot resize their owner storage"
        )
    ragged_forward = algorithm.draft_model.update_cache_and_draft_ragged
    if draft_cache.use_compiled_b1:
        ragged_forward = getattr(
            algorithm.draft_model,
            "update_cache_and_draft_ragged_b1",
            ragged_forward,
        )
    outputs = ragged_forward(
        hidden_batch,
        token_batch,
        draft_cache.prefix_layers(active_count),
        starts,
        lengths,
        max_position,
        max_total_slots,
    )
    draft_cache.advance(valid_lengths, K)
    return outputs


def _project_initial_ragged_condition(
    algorithm,
    hidden_sources: Sequence[torch.Tensor],
    source_rows: torch.Tensor,
    source_lengths: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    max_positions: int,
) -> torch.Tensor:
    active_count = starts.numel()
    weight = algorithm.draft_model.input_layer.condition_proj.weight
    if not RUNTIME_CAPABILITIES.needs_ragged_condition_fallback:
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            matmul_persistent_concat3_ragged,
        )

        return matmul_persistent_concat3_ragged(
            hidden_sources[0],
            hidden_sources[1],
            hidden_sources[2],
            weight,
            source_rows[:active_count],
            source_lengths[:active_count],
            starts,
            lengths,
            max_positions,
        )

    # The MetaX persistent kernel returns incorrect values for nonzero chunk
    # offsets. This explicit gather only runs during initial prompt prefill.
    positions = starts[:, None] + torch.arange(
        max_positions,
        device=starts.device,
        dtype=torch.long,
    )[None, :]
    positions = torch.minimum(
        positions,
        source_lengths[:active_count, None] - 1,
    )
    rows = source_rows[:active_count, None].expand_as(positions)
    condition_input = torch.cat(
        [source[rows, positions] for source in hidden_sources],
        dim=-1,
    )
    return F.linear(condition_input, weight)


def _run_initial_ragged_draft_forward(
    algorithm,
    draft_cache: _DraftBatchCache,
    hidden_sources: Sequence[torch.Tensor],
    source_rows: torch.Tensor,
    source_lengths: torch.Tensor,
    token_rows: Sequence[torch.Tensor],
    start_positions: Sequence[int],
    valid_lengths: Sequence[int],
):
    batch_size = len(token_rows)
    if not (
        len(hidden_sources) == 3
        and source_rows.shape == (batch_size,)
        and source_lengths.shape == (batch_size,)
        and len(start_positions) == batch_size
        and len(valid_lengths) == batch_size
        and draft_cache.active_batch_size == batch_size
    ):
        raise ValueError("initial ragged draft inputs must match active cache rows")
    max_positions = max(int(value) for value in valid_lengths)
    if max_positions <= 0:
        raise ValueError("initial ragged draft forward requires valid positions")
    active_count = sum(int(value) > 0 for value in valid_lengths)
    if any(int(value) <= 0 for value in valid_lengths[:active_count]) or any(
        int(value) > 0 for value in valid_lengths[active_count:]
    ):
        raise ValueError(
            "zero-length initial draft rows must form a contiguous suffix"
        )

    device = hidden_sources[0].device
    token_batch = torch.zeros(
        active_count,
        max_positions,
        dtype=torch.long,
        device=device,
    )
    for row, (tokens, valid) in enumerate(
        zip(token_rows[:active_count], valid_lengths[:active_count])
    ):
        valid = int(valid)
        token_batch[row, :valid].copy_(tokens[0, :valid])

    starts = torch.tensor(
        start_positions[:active_count], dtype=torch.long, device=device
    )
    lengths = torch.tensor(
        valid_lengths[:active_count], dtype=torch.long, device=device
    )
    condition = _project_initial_ragged_condition(
        algorithm,
        hidden_sources,
        source_rows,
        source_lengths,
        starts,
        lengths,
        max_positions,
    )

    K = int(algorithm.draft_model.K)
    max_position = (
        max(int(start) for start in start_positions[:active_count])
        + max_positions
        + K
        - 1
    )
    max_total_slots = max(
        count + int(valid) * K
        for count, valid in zip(
            draft_cache.lengths_host[:active_count],
            valid_lengths[:active_count],
        )
    )
    cache_capacity = min(int(layer[3]) for layer in draft_cache.layers)
    if max_total_slots > cache_capacity:
        raise RuntimeError(
            "draft cache prefix views cannot resize their owner storage"
        )
    ragged_forward = (
        algorithm.draft_model.update_cache_and_draft_ragged_from_condition
    )
    if draft_cache.use_compiled_b1:
        ragged_forward = getattr(
            algorithm.draft_model,
            "update_cache_and_draft_ragged_from_condition_b1",
            ragged_forward,
        )
    outputs = ragged_forward(
        condition,
        token_batch,
        draft_cache.prefix_layers(active_count),
        starts,
        lengths,
        max_position,
        max_total_slots,
    )
    draft_cache.advance(valid_lengths, K)
    return outputs


def _build_initial_drafts_batched(
    algorithm,
    entries: Sequence[Tuple[int, _RequestState]],
    hidden_sources: Sequence[torch.Tensor],
    max_new_tokens: int,
    original_engine_b1: bool,
    temperature: float,
) -> _DraftBatchCache:
    if len(hidden_sources) != 3:
        raise ValueError("initial draft requires exactly three hidden sources")
    states = [state for _, state in entries]
    K = int(algorithm.draft_model.K)
    max_prompt = max(state.prompt_len for state in states)
    device = hidden_sources[0].device
    draft_cache = _DraftBatchCache(
        num_layers=int(algorithm.draft_model.num_layers),
        batch_size=len(states),
        max_cache_len=(max_prompt + int(max_new_tokens)) * K,
        device=device,
        original_engine_b1=original_engine_b1,
    )

    source_rows = torch.tensor(
        [source_idx for source_idx, _state in entries],
        dtype=torch.long,
        device=device,
    )
    source_lengths = torch.tensor(
        [state.prompt_len for state in states],
        dtype=torch.long,
        device=device,
    )
    token_rows = []
    total_lengths = []
    for state in states:
        shifted_ids = torch.cat((state.prompt_ids[:, 1:], state.current_token), dim=1)
        token_rows.append(torch.cat((shifted_ids, state.current_token), dim=1))
        total_lengths.append(state.prompt_len + 1)

    final_outputs = None
    final_ttt = None
    chunk_size = 64
    max_total = max(total_lengths)
    for start in range(0, max_total, chunk_size):
        valid_lengths = [
            max(0, min(chunk_size, total - start)) for total in total_lengths
        ]
        chunk_tokens = [row[:, start:start + chunk_size] for row in token_rows]
        logits, rank_logits, draft_hidden, ttt_kv = (
            _run_initial_ragged_draft_forward(
                algorithm,
                draft_cache,
                hidden_sources,
                source_rows,
                source_lengths,
                chunk_tokens,
                [start] * len(states),
                valid_lengths,
            )
        )
        if final_outputs is None:
            final_outputs = [
                torch.empty(
                    len(states), *tensor.shape[1:],
                    dtype=tensor.dtype, device=tensor.device,
                )
                for tensor in (logits, rank_logits, draft_hidden)
            ]
            final_ttt = [
                (
                    torch.empty(
                        len(states), *keys.shape[1:],
                        dtype=keys.dtype, device=keys.device,
                    ),
                    torch.empty(
                        len(states), *values.shape[1:],
                        dtype=values.dtype, device=values.device,
                    ),
                )
                for keys, values in ttt_kv
            ]
        finishing = [
            row for row, total in enumerate(total_lengths)
            if start < total <= start + chunk_size
        ]
        if finishing:
            if finishing[-1] >= logits.shape[0]:
                raise RuntimeError(
                    "finishing draft rows must belong to the computed active prefix"
                )
            indices = torch.tensor(finishing, dtype=torch.long, device=logits.device)
            for destination, source in zip(
                final_outputs, (logits, rank_logits, draft_hidden)
            ):
                destination.index_copy_(0, indices, source.index_select(0, indices))
            for layer_idx, (keys, values) in enumerate(ttt_kv):
                final_ttt[layer_idx][0].index_copy_(
                    0, indices, keys.index_select(0, indices)
                )
                final_ttt[layer_idx][1].index_copy_(
                    0, indices, values.index_select(0, indices)
                )

    for row, state in enumerate(states):
        state.draft_position = total_lengths[row]
        state.draft_cache = draft_cache.row_view(row)
        _set_request_ngram_context(algorithm, state)
    results = algorithm._build_trees_from_block1_batched(
        final_outputs[0],
        final_outputs[1],
        final_outputs[2],
        [(keys, values) for keys, values in final_ttt],
        torch.cat([state.current_token for state in states], dim=0),
        draft_cache,
        [state.draft_position - 1 for state in states],
        [state.tree_budget for state in states],
        temperature=temperature,
    )
    for state, result in zip(states, results):
        state.tree = _clone_tree_result(result)
    return draft_cache


def _pack_trees(states: Sequence[_RequestState], pad_token_id: int):
    widths = [int(state.tree[0].shape[1]) for state in states]
    tree_width = max(widths)
    batch_size = len(states)
    device = states[0].tree[0].device
    input_ids = torch.full(
        (batch_size, tree_width), pad_token_id, dtype=torch.long, device=device
    )
    position_ids = torch.zeros((batch_size, tree_width), dtype=torch.long, device=device)
    for idx, (state, width) in enumerate(zip(states, widths)):
        tokens, _mask, depth_ids = state.tree[:3]
        input_ids[idx, :width] = tokens[0]
        logical_prefix = state.target_prefix_len
        position_ids[idx, :width] = depth_ids[:width] + logical_prefix
        if width < tree_width:
            position_ids[idx, width:] = logical_prefix
    return input_ids, position_ids, widths, tree_width


def _build_additive_mask(
    states: Sequence[_RequestState],
    prefix_lengths: Sequence[int],
    tree_start: int,
    tree_widths: Sequence[int],
    tree_width: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Build [B,1,Q,tree_start+Q] bias over committed KV plus staged trees."""
    tree_start = int(tree_start)
    if tree_start < max(int(length) for length in prefix_lengths):
        raise ValueError("staged target tree overlaps committed KV")
    mask = torch.full(
        (len(states), 1, tree_width, tree_start + tree_width),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    for idx, (state, prefix_len, width) in enumerate(
        zip(states, prefix_lengths, tree_widths)
    ):
        mask[idx, 0, :width, :prefix_len] = 0.0
        tree_visible = state.tree[1][0, 0, :width, :width].bool()
        tree_block = mask[idx, 0, :width, tree_start:tree_start + width]
        tree_block.masked_fill_(tree_visible, 0.0)
        # Padded query rows are ignored, but give each one a private key to avoid
        # all--inf softmax rows producing NaNs in attention implementations.
        for query_idx in range(width, tree_width):
            mask[idx, 0, query_idx, tree_start + query_idx] = 0.0
    return mask


def _run_target_backbone(algorithm, **kwargs):
    """Run the target backbone while retaining only the three draft features."""
    layers = algorithm.target_model.model.layers
    hidden_indices = tuple(int(index) for index in algorithm.hidden_layer_indices)
    if len(hidden_indices) != 3:
        raise ValueError("target draft conditioning requires exactly three layers")
    layer_ids = tuple(index - 1 for index in hidden_indices)
    if any(layer_id < 0 or layer_id >= len(layers) for layer_id in layer_ids):
        raise ValueError(
            f"target hidden indices {hidden_indices} are outside the decoder"
        )

    captured = [None] * len(layer_ids)
    handles = []

    def make_hook(slot):
        def capture(_module, _inputs, output):
            captured[slot] = output[0] if isinstance(output, tuple) else output

        return capture

    try:
        for slot, layer_id in enumerate(layer_ids):
            handles.append(layers[layer_id].register_forward_hook(make_hook(slot)))
        outputs = algorithm.target_model.model(
            **kwargs,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    if any(hidden is None for hidden in captured):
        missing = [
            hidden_indices[slot]
            for slot, hidden in enumerate(captured)
            if hidden is None
        ]
        raise RuntimeError(f"target hidden hooks did not capture layers {missing}")
    return outputs.last_hidden_state, tuple(captured)


def _project_tree_decisions(
    algorithm,
    last_hidden,
    tree_widths,
    return_sampling_logits: bool = False,
):
    """Project real tree nodes and retain logits only for B=1 sampling."""
    if last_hidden.ndim != 3 or last_hidden.shape[0] != len(tree_widths):
        raise ValueError("target tree hidden states do not match active requests")
    tree_width = last_hidden.shape[1]
    if any(int(width) <= 0 or int(width) > tree_width for width in tree_widths):
        raise ValueError("target tree widths are outside the staged query range")
    unpadded_b1 = len(tree_widths) == 1 and int(tree_widths[0]) == tree_width
    if unpadded_b1:
        valid = None
        packed_hidden = last_hidden[0]
    else:
        widths = torch.tensor(
            tree_widths, dtype=torch.long, device=last_hidden.device
        )
        valid = (
            torch.arange(tree_width, device=last_hidden.device)[None, :]
            < widths[:, None]
        )
        packed_hidden = last_hidden[valid]
    packed_logits = algorithm.target_model.lm_head(packed_hidden)

    accept_topk = int(os.environ.get("ACCEPT_TOPK", "1"))
    vocab_size = packed_logits.shape[-1]
    if accept_topk <= 0 or accept_topk > vocab_size:
        raise ValueError(
            f"ACCEPT_TOPK must be in [1, {vocab_size}], got {accept_topk}"
        )

    packed_argmax = torch.argmax(packed_logits, dim=-1)
    if unpadded_b1:
        node_argmax = packed_argmax.unsqueeze(0)
    else:
        node_argmax = torch.zeros(
            last_hidden.shape[:2], dtype=torch.long, device=last_hidden.device
        )
        node_argmax[valid] = packed_argmax
    node_topk = None
    if accept_topk > 1:
        packed_topk = torch.topk(
            packed_logits, accept_topk, dim=-1
        ).indices
        if unpadded_b1:
            node_topk = packed_topk.unsqueeze(0)
        else:
            node_topk = torch.zeros(
                (*last_hidden.shape[:2], accept_topk),
                dtype=torch.long,
                device=last_hidden.device,
            )
            node_topk[valid] = packed_topk

    sampling_logits = None
    if return_sampling_logits:
        if len(tree_widths) != 1 or int(tree_widths[0]) != tree_width:
            raise ValueError("sampling logits require one unpadded tree row")
        sampling_logits = packed_logits
    return (
        node_argmax,
        node_topk,
        sampling_logits,
        int(packed_hidden.shape[0]),
    )


def _update_logit_correction(algorithm, target_argmax, candidates):
    if not getattr(algorithm, "_logit_correct_on", False):
        return
    vocab_size = int(algorithm.target_model.lm_head.out_features)
    if algorithm._target_want_count is None:
        algorithm._target_want_count = torch.zeros(
            vocab_size, device=target_argmax.device, dtype=torch.float32
        )
        algorithm._draft_want_count = torch.zeros_like(algorithm._target_want_count)
    wanted = target_argmax.flatten()
    drafted = candidates[:, 1:].flatten().to(wanted.device)
    valid = drafted >= 0
    wanted = wanted[valid]
    drafted = drafted[valid]
    algorithm._target_want_count.scatter_add_(
        0, wanted, torch.ones_like(wanted, dtype=torch.float32)
    )
    algorithm._draft_want_count.scatter_add_(
        0, drafted, torch.ones_like(drafted, dtype=torch.float32)
    )


def _accept_active_batch(
    algorithm,
    active,
    tree_input_ids,
    node_argmax,
    node_topk,
    hidden_sources,
):
    """Compute greedy acceptance and selected hidden paths as one padded batch."""
    batch_size, tree_width = tree_input_ids.shape
    if node_argmax.shape != (batch_size, tree_width):
        raise ValueError("target argmax decisions must match the staged tree")
    if len(hidden_sources) != 3 or any(
        hidden.shape[:2] != (batch_size, tree_width)
        for hidden in hidden_sources
    ):
        raise ValueError("target hidden sources must match the staged tree")
    retrieve_shapes = [tuple(state.tree[3].shape) for state in active]
    max_paths = max(shape[0] for shape in retrieve_shapes)
    max_path_width = max(shape[1] for shape in retrieve_shapes)
    retrieve_batch = torch.full(
        (batch_size, max_paths, max_path_width),
        -1,
        dtype=torch.long,
        device=tree_input_ids.device,
    )
    for batch_idx, (state, shape) in enumerate(zip(active, retrieve_shapes)):
        num_paths, path_width = shape
        retrieve_batch[batch_idx, :num_paths, :path_width] = state.tree[3]

    last_node_indices = retrieve_batch.new_tensor([
        state.tree[0].shape[1] - 1 for state in active
    ])
    safe_retrieve = torch.where(
        retrieve_batch >= 0,
        retrieve_batch,
        last_node_indices[:, None, None],
    )
    path_argmax = node_argmax[:, None, :].expand(
        -1, max_paths, -1
    ).gather(2, safe_retrieve)
    target_choices = path_argmax[:, :, :-1]

    candidate_source = torch.cat((
        tree_input_ids,
        tree_input_ids.new_full((batch_size, 1), -1),
    ), dim=1)
    candidate_indices = torch.where(
        retrieve_batch >= 0,
        retrieve_batch,
        retrieve_batch.new_full((), tree_width),
    )
    candidates = candidate_source[:, None, :].expand(
        -1, max_paths, -1
    ).gather(2, candidate_indices)

    accept_topk = int(os.environ.get("ACCEPT_TOPK", "1"))
    if accept_topk > 1:
        if node_topk is None or node_topk.shape != (
            batch_size, tree_width, accept_topk
        ):
            raise ValueError("target top-k decisions must match ACCEPT_TOPK")
        topk_indices = safe_retrieve[:, :, :-1, None].expand(
            -1, -1, -1, accept_topk
        )
        path_topk = node_topk[:, None, :, :].expand(
            -1, max_paths, -1, -1
        ).gather(2, topk_indices)
        posterior = (
            candidates[:, :, 1:, None] == path_topk
        ).any(dim=-1).int()
    else:
        posterior = (candidates[:, :, 1:] == target_choices).int()
    path_accept_lengths = torch.cumprod(posterior, dim=2).sum(dim=2)
    accept_lengths = path_accept_lengths.max(dim=1).values.to(torch.long)
    best_candidates = torch.argmax(path_accept_lengths, dim=1).to(torch.long)

    selected_paths = retrieve_batch.gather(
        1,
        best_candidates[:, None, None].expand(-1, 1, max_path_width),
    ).squeeze(1)
    bonus_indices = selected_paths.gather(1, accept_lengths[:, None]).squeeze(1)
    next_tokens = node_argmax.gather(
        1, bonus_indices.clamp_min(0)[:, None]
    )

    accepted_indices = selected_paths[:, 1:]
    accepted_token_ids = tree_input_ids.gather(
        1, accepted_indices.clamp_min(0)
    )
    accepted_positions = torch.arange(
        accepted_indices.shape[1],
        device=accepted_indices.device,
        dtype=torch.long,
    )[None, :]
    accepted_token_ids = torch.where(
        (accepted_positions < accept_lengths[:, None]) & (accepted_indices >= 0),
        accepted_token_ids,
        accepted_token_ids.new_full((), -1),
    )

    safe_selected_paths = selected_paths.clamp_min(0)
    selected_hidden = torch.cat([
        hidden[:, :tree_width].gather(
            1,
            safe_selected_paths[:, :, None].expand(
                -1, -1, hidden.shape[-1]
            ),
        )
        for hidden in hidden_sources
    ], dim=-1)

    metadata = torch.cat((
        accept_lengths[:, None],
        best_candidates[:, None],
        next_tokens,
        selected_paths[:, :1],
        accepted_token_ids,
    ), dim=1).cpu()
    algorithm._batched_acceptance_readbacks = (
        getattr(algorithm, "_batched_acceptance_readbacks", 0) + 1
    )
    algorithm._batched_acceptance_gpu_calls = (
        getattr(algorithm, "_batched_acceptance_gpu_calls", 0) + 1
    )

    accepted_data = []
    for batch_idx, (state, shape, row_metadata) in enumerate(
        zip(active, retrieve_shapes, metadata)
    ):
        num_paths, path_width = shape
        request_target_choices = target_choices[
            batch_idx, :num_paths, :path_width - 1
        ]
        request_candidates = candidates[batch_idx, :num_paths, :path_width]
        _update_logit_correction(
            algorithm,
            request_target_choices,
            request_candidates,
        )

        accept_length = int(row_metadata[0])
        best_candidate = int(row_metadata[1])
        next_token_id = int(row_metadata[2])
        if int(row_metadata[3]) < 0:
            raise RuntimeError("accepted path contains padded tree indices")
        accepted_token_ids_host = [
            int(token) for token in row_metadata[4:4 + accept_length]
        ]
        if any(token < 0 for token in accepted_token_ids_host):
            raise RuntimeError("accepted path contains padded token metadata")

        selected_path_indices = selected_paths[
            batch_idx, :accept_length + 1
        ]
        accepted_tokens = accepted_token_ids[batch_idx, :accept_length]
        hidden_3h = selected_hidden[
            batch_idx:batch_idx + 1, :accept_length + 1
        ]
        accepted_data.append((
            accept_length,
            accepted_tokens,
            next_tokens[batch_idx:batch_idx + 1],
            hidden_3h[:, -1:, :],
            hidden_3h if accept_length > 0 else None,
            best_candidate,
            (
                state.tree[0],
                request_target_choices,
                state.tree[3],
                state.tree[6],
            ),
            selected_path_indices,
            accepted_token_ids_host,
            next_token_id,
        ))
    return accepted_data


def _accept_single_sampling(
    algorithm,
    active,
    tree_input_ids,
    node_hidden,
    hidden_sources,
    temperature: float,
):
    """Apply the original SpecBlock speculative-sampling rule inside the batch core."""
    if len(active) != 1 or tree_input_ids.shape[0] != 1:
        raise ValueError("SpecBlock sampling requires exactly one active request")
    if temperature <= 0:
        raise ValueError("sampling temperature must be positive")
    if node_hidden.shape[:2] != tree_input_ids.shape:
        raise ValueError("sampling hidden states must match the unpadded tree width")
    if len(hidden_sources) != 3 or any(
        hidden.shape[:2] != tree_input_ids.shape
        for hidden in hidden_sources
    ):
        raise ValueError("target hidden sources must match the sampled tree")

    state = active[0]
    retrieve_indices = state.tree[3]
    candidate_source = torch.cat((
        tree_input_ids[0],
        tree_input_ids.new_full((1,), -1),
    ))
    candidate_indices = torch.where(
        retrieve_indices >= 0,
        retrieve_indices,
        retrieve_indices.new_full((), tree_input_ids.shape[1]),
    )
    candidates = candidate_source[candidate_indices]
    packed_path_metadata = torch.cat((
        candidates.reshape(-1),
        retrieve_indices.reshape(-1),
    )).cpu()
    candidate_count = candidates.numel()
    candidates_host = packed_path_metadata[:candidate_count].reshape(
        candidates.shape
    ).tolist()
    retrieve_host = packed_path_metadata[candidate_count:].reshape(
        retrieve_indices.shape
    ).tolist()
    acceptance_readbacks = 1
    projected_rows = 0
    projected_logits = {}

    def project_path_node(path_idx: int, depth_idx: int):
        nonlocal projected_rows
        tree_index = int(retrieve_host[path_idx][depth_idx])
        if tree_index < 0:
            raise RuntimeError("cannot project a padded sampled path node")
        if tree_index not in projected_logits:
            projected_logits[tree_index] = algorithm.target_model.lm_head(
                node_hidden[0, tree_index]
            )
            projected_rows += 1
        return projected_logits[tree_index]

    accepted_width = 1
    accepted_prefix = [int(candidates_host[0][0])]
    best_candidate = 0
    adjusted = False
    sample_p = None
    for depth in range(1, candidates.shape[1]):
        if depth != accepted_width:
            break
        matching_rows = [
            path_idx
            for path_idx, path in enumerate(candidates_host)
            if path[:accepted_width] == accepted_prefix
        ]
        if not matching_rows:
            raise RuntimeError("sampled prefix has no matching tree path")
        first_match = matching_rows[0]
        sample_logits = project_path_node(first_match, depth - 1)
        sample_p = torch.softmax(
            sample_logits.float() / temperature,
            dim=0,
        )

        seen_tokens = set()
        choices = []
        for path_idx in matching_rows:
            token = int(candidates_host[path_idx][depth])
            if token < 0 or token in seen_tokens:
                continue
            seen_tokens.add(token)
            choices.append((path_idx, token))

        choice_probabilities = []
        if choices:
            choice_tokens = torch.tensor(
                [token for _path_idx, token in choices],
                dtype=torch.long,
                device=sample_p.device,
            )
            choice_probabilities = sample_p.index_select(
                0, choice_tokens
            ).cpu().tolist()
            acceptance_readbacks += 1

        remaining_probability = 1.0
        rejected_tokens = []
        adjusted = False
        accepted = False
        for (path_idx, token), probability in zip(
            choices, choice_probabilities
        ):
            conditional_probability = (
                probability / remaining_probability
                if remaining_probability > 0.0
                else 0.0
            )
            if random.random() <= conditional_probability:
                accepted_prefix.append(token)
                accepted_width += 1
                best_candidate = path_idx
                accepted = True
                break
            rejected_tokens.append(token)
            remaining_probability = max(
                0.0, remaining_probability - probability
            )
            adjusted = True

        if not accepted and adjusted:
            if remaining_probability > 0.0:
                rejected = torch.tensor(
                    rejected_tokens,
                    dtype=torch.long,
                    device=sample_p.device,
                )
                # torch.multinomial accepts unnormalized non-negative weights;
                # zeroing rejected mass preserves the residual distribution and
                # avoids an extra full-vocabulary normalization kernel.
                sample_p.index_fill_(0, rejected, 0.0)
            else:
                sample_p = torch.softmax(
                    sample_logits.float() / temperature,
                    dim=0,
                )

    accept_length = accepted_width - 1
    if not adjusted or accepted_width == candidates.shape[1]:
        bonus_logits = project_path_node(best_candidate, accept_length)
        sample_p = torch.softmax(
            bonus_logits.float() / temperature,
            dim=0,
        )
    if sample_p is None:
        bonus_logits = project_path_node(best_candidate, accept_length)
        sample_p = torch.softmax(
            bonus_logits.float() / temperature,
            dim=0,
        )
    next_token = torch.multinomial(sample_p.unsqueeze(0), num_samples=1)

    selected_path_host = retrieve_host[
        best_candidate
    ][:accept_length + 1]
    if any(index < 0 for index in selected_path_host):
        raise RuntimeError("sampled path contains padded tree indices")
    selected_path_indices = retrieve_indices[
        best_candidate, :accept_length + 1
    ]
    accepted_indices = selected_path_indices[1:]
    accepted_tokens = tree_input_ids[0].index_select(0, accepted_indices)
    selected_hidden = torch.cat([
        hidden[0:1].index_select(1, selected_path_indices)
        for hidden in hidden_sources
    ], dim=-1)
    last_hidden_3h = selected_hidden[:, -1:, :]
    lazy_hidden_3h = selected_hidden if accept_length > 0 else None
    accepted_token_ids = accepted_prefix[1:]
    next_token_id = int(next_token[0, 0])
    acceptance_readbacks += 1

    algorithm._batched_acceptance_readbacks = (
        getattr(algorithm, "_batched_acceptance_readbacks", 0)
        + acceptance_readbacks
    )
    algorithm._batched_acceptance_gpu_calls = (
        getattr(algorithm, "_batched_acceptance_gpu_calls", 0) + 1
    )
    algorithm._sampling_lm_head_rows = (
        getattr(algorithm, "_sampling_lm_head_rows", 0) + projected_rows
    )
    return [(
        accept_length,
        accepted_tokens,
        next_token,
        last_hidden_3h,
        lazy_hidden_3h,
        best_candidate,
        None,
        selected_path_indices,
        accepted_token_ids,
        next_token_id,
    )]


def _accept_sampling_batch(
    algorithm,
    active,
    tree_input_ids,
    node_hidden,
    hidden_sources,
    temperature: float,
):
    """Sample a ragged request batch with depth-synchronous GPU decisions.

    Each request follows the scalar sequential candidate rule: sibling proposals
    are tested in retrieve-row order, rejection removes their probability mass,
    and an all-rejected level samples from that residual distribution.  The
    sequential dependency is only across siblings of a *single* request; every
    LM-head projection, probability calculation, and random draw is batched
    over the active request rows.  This is deliberately not a serial B1 loop.
    """
    batch_size, tree_width = tree_input_ids.shape
    if len(active) != batch_size or temperature <= 0:
        raise ValueError("sampling batch and positive temperature are required")
    if node_hidden.shape[:2] != tree_input_ids.shape:
        raise ValueError("sampling hidden states must match the staged tree")
    if len(hidden_sources) != 3 or any(
        hidden.shape[:2] != tree_input_ids.shape for hidden in hidden_sources
    ):
        raise ValueError("target hidden sources must match the sampled tree")

    path_counts = torch.tensor(
        [state.tree[3].shape[0] for state in active],
        dtype=torch.long, device=tree_input_ids.device,
    )
    path_widths = torch.tensor(
        [state.tree[3].shape[1] for state in active],
        dtype=torch.long, device=tree_input_ids.device,
    )
    max_paths = int(path_counts.max())
    max_path_width = int(path_widths.max())
    retrieve = torch.full(
        (batch_size, max_paths, max_path_width), -1, dtype=torch.long,
        device=tree_input_ids.device,
    )
    for row, state in enumerate(active):
        paths, width = state.tree[3].shape
        retrieve[row, :paths, :width] = state.tree[3]
    valid_path = torch.arange(max_paths, device=retrieve.device)[None] < path_counts[:, None]
    local_tree_widths = torch.tensor(
        [state.tree[0].shape[1] for state in active],
        dtype=torch.long, device=tree_input_ids.device,
    )
    # A negative retrieve entry is path padding.  A stale/out-of-row index is
    # equally unusable, but must never reach gather: it is converted to the
    # explicit final -1 sentinel below rather than clamped to a real node.
    valid_retrieve = (retrieve >= 0) & (
        retrieve < local_tree_widths[:, None, None]
    )
    malformed_retrieve = valid_path[:, :, None] & (retrieve >= 0) & ~valid_retrieve
    if os.environ.get("SPECBLOCK_SAMPLING_DEBUG", "0") == "1" and malformed_retrieve.any():
        bad = torch.nonzero(malformed_retrieve, as_tuple=False)[0].cpu().tolist()
        batch_idx, path_idx, depth_idx = (int(value) for value in bad)
        raise RuntimeError(
            "SpecBlock sampled tree contains an out-of-range retrieve index: "
            f"row={batch_idx}, path={path_idx}, depth={depth_idx}, "
            f"retrieve={int(retrieve[batch_idx, path_idx, depth_idx].cpu())}, "
            f"local_tree_width={int(local_tree_widths[batch_idx].cpu())}"
        )
    source = torch.cat((tree_input_ids, tree_input_ids.new_full((batch_size, 1), -1)), dim=1)
    candidates = source[:, None, :].expand(-1, max_paths, -1).gather(
        2, torch.where(valid_retrieve, retrieve, retrieve.new_full((), tree_width))
    )
    rows = torch.arange(batch_size, device=retrieve.device)
    accepted_width = torch.ones(batch_size, dtype=torch.long, device=retrieve.device)
    best_candidate = torch.zeros(batch_size, dtype=torch.long, device=retrieve.device)
    next_tokens = torch.empty((batch_size, 1), dtype=torch.long, device=retrieve.device)
    terminal = torch.zeros(batch_size, dtype=torch.bool, device=retrieve.device)
    projected_rows = 0

    def project(row_ids, node_ids):
        nonlocal projected_rows
        # ``node_ids`` comes from a matching, non-padding retrieve path.
        projected_rows += int(row_ids.numel())
        return algorithm.target_model.lm_head(node_hidden[row_ids, node_ids])

    def finish(row_ids, probabilities):
        if not row_ids.numel():
            return
        next_tokens[row_ids] = torch.multinomial(probabilities, 1)
        terminal[row_ids] = True

    for depth in range(1, max_path_width):
        eligible = (~terminal) & (accepted_width == depth) & (path_widths > depth)
        if not eligible.any():
            continue
        row_ids = torch.nonzero(eligible, as_tuple=False).squeeze(1)
        reference = candidates[rows, best_candidate, :depth]
        matching = valid_path & (
            candidates[:, :, :depth] == reference[:, None, :]
        ).all(dim=-1)
        matching &= eligible[:, None]
        first_match = matching.to(torch.long).argmax(dim=1)
        current_nodes = retrieve[rows, first_match, depth - 1]
        logits = project(row_ids, current_nodes[row_ids])
        probabilities = torch.softmax(logits.float() / temperature, dim=-1)
        # ``probabilities`` is compacted to eligible rows.  Residual sampling
        # receives global batch rows later in this depth, so map those rows back
        # to compact LM-head rows without materializing a B×vocab copy.
        global_to_local = torch.full(
            (batch_size,), -1, dtype=torch.long, device=retrieve.device
        )
        global_to_local[row_ids] = torch.arange(
            row_ids.numel(), dtype=torch.long, device=retrieve.device
        )

        sibling_tokens = candidates[:, :, depth]
        # Preserve scalar retrieve-row order: a row is a proposal only if no
        # earlier matching row proposed the same (non-padding) token.
        same_token = sibling_tokens[:, :, None] == sibling_tokens[:, None, :]
        earlier = torch.tril(
            torch.ones((max_paths, max_paths), dtype=torch.bool, device=retrieve.device),
            diagonal=-1,
        )
        seen_earlier = (same_token & matching[:, None, :] & earlier[None]).any(dim=2)
        sibling_nodes = retrieve[:, :, depth]
        sibling_node_valid = valid_retrieve[:, :, depth]
        vocab_size = probabilities.shape[-1]
        token_in_vocab = (sibling_tokens >= 0) & (sibling_tokens < vocab_size)
        malformed = matching & sibling_node_valid & ~(
            (sibling_tokens == -1) | token_in_vocab
        )
        if os.environ.get("SPECBLOCK_SAMPLING_DEBUG", "0") == "1" and malformed.any():
            # A synchronized, actionable failure for CUDA gates.  Do not clamp:
            # report the request/path/depth, retrieve index, and offending token.
            bad = torch.nonzero(malformed, as_tuple=False)[0].cpu().tolist()
            batch_idx, path_idx = (int(value) for value in bad)
            raise RuntimeError(
                "SpecBlock sampled tree contains an out-of-vocabulary token: "
                f"row={batch_idx}, path={path_idx}, depth={depth}, "
                f"retrieve={int(sibling_nodes[batch_idx, path_idx].cpu())}, "
                f"token={int(sibling_tokens[batch_idx, path_idx].cpu())}, "
                f"vocab_size={vocab_size}, local_tree_width="
                f"{int(local_tree_widths[batch_idx].cpu())}"
            )
        # Padded retrieve nodes and explicit -1 token sentinels are not draft
        # proposals.  Values outside target vocab are malformed and excluded
        # (with the debug gate above making their provenance fail-fast); they
        # are never silently clamped into a different target token.
        proposal = matching & sibling_node_valid & token_in_vocab & ~seen_earlier
        proposal_probs = probabilities.new_zeros((batch_size, max_paths))
        safe_tokens = torch.where(proposal, sibling_tokens, sibling_tokens.new_zeros(()))
        proposal_probs[eligible] = probabilities.gather(1, safe_tokens[eligible])

        remaining = torch.ones(batch_size, dtype=probabilities.dtype, device=retrieve.device)
        chosen = torch.zeros(batch_size, dtype=torch.bool, device=retrieve.device)
        chosen_path = best_candidate.clone()
        rejected = torch.zeros_like(proposal)
        # This loop is over the (small) padded tree fanout only.  Every operation
        # is a B-row tensor operation; requests never fall back to serial decode.
        uniforms = torch.rand(
            (batch_size, max_paths), device=retrieve.device, dtype=probabilities.dtype
        )
        for path_idx in range(max_paths):
            offered = eligible & proposal[:, path_idx] & ~chosen
            conditional = proposal_probs[:, path_idx] / remaining.clamp_min(
                torch.finfo(probabilities.dtype).tiny
            )
            take = offered & (uniforms[:, path_idx] <= conditional)
            reject = offered & ~take
            chosen |= take
            chosen_path = torch.where(
                take, chosen_path.new_full((), path_idx), chosen_path
            )
            rejected[:, path_idx] = reject
            remaining = (remaining - torch.where(
                reject, proposal_probs[:, path_idx], 0.0
            )).clamp_min(0.0)

        accepted_width = accepted_width + chosen.to(torch.long)
        best_candidate = torch.where(chosen, chosen_path, best_candidate)
        failed = eligible & ~chosen
        if failed.any():
            failed_rows = torch.nonzero(failed, as_tuple=False).squeeze(1)
            failed_local = global_to_local[failed_rows]
            torch._assert_async(
                (failed_local >= 0).all(),
                "failed sampling row is absent from compact logits",
            )
            failed_probs = probabilities[failed_local].clone()
            failed_rejected = rejected[failed_rows]
            failed_tokens = safe_tokens[failed_rows]
            failed_batch = torch.arange(
                failed_rows.numel(), device=retrieve.device
            )[:, None].expand_as(failed_tokens)
            # Index only actual sequential rejections.  Scattering all paths
            # would let a later duplicate (which is not a proposal) restore the
            # mass zeroed by its first-occurrence proposal.
            failed_probs[
                failed_batch[failed_rejected], failed_tokens[failed_rejected]
            ] = 0.0
            mass = failed_probs.sum(dim=-1, keepdim=True)
            # torch.multinomial accepts unnormalized weights.  Keeping the
            # scalar residual scale is observable with fixed RNG streams; only
            # restore the original target distribution if all mass vanished.
            failed_probs = torch.where(
                mass > 0, failed_probs, probabilities[failed_local]
            )
            finish(failed_rows, failed_probs)

    unfinished = ~terminal
    if unfinished.any():
        unfinished_rows = torch.nonzero(unfinished, as_tuple=False).squeeze(1)
        node_depth = accepted_width[unfinished_rows] - 1
        node_ids = retrieve[
            unfinished_rows, best_candidate[unfinished_rows], node_depth
        ]
        finish(unfinished_rows, torch.softmax(
            project(unfinished_rows, node_ids).float() / temperature, dim=-1
        ))

    accept_lengths = accepted_width - 1
    selected_paths = retrieve[rows, best_candidate]
    selected_hidden = torch.cat([
        hidden.gather(1, selected_paths.clamp_min(0)[:, :, None].expand(
            -1, -1, hidden.shape[-1]
        )) for hidden in hidden_sources
    ], dim=-1)
    accepted_indices = selected_paths[:, 1:]
    accepted_tokens_all = tree_input_ids.gather(1, accepted_indices.clamp_min(0))
    algorithm._batched_acceptance_gpu_calls = (
        getattr(algorithm, "_batched_acceptance_gpu_calls", 0) + 1
    )
    algorithm._batched_acceptance_readbacks = (
        getattr(algorithm, "_batched_acceptance_readbacks", 0) + 1
    )
    algorithm._sampling_lm_head_rows = (
        getattr(algorithm, "_sampling_lm_head_rows", 0) + projected_rows
    )

    accepted = []
    # One packed host transfer after the batched GPU decision phase.  In
    # particular, B32 does not synchronize once per request to assemble output.
    metadata_width = 3
    host_metadata = torch.cat((
        accept_lengths[:, None], best_candidate[:, None], next_tokens,
        selected_paths, accepted_tokens_all,
    ), dim=1).cpu()
    for row, (state, row_metadata) in enumerate(zip(active, host_metadata)):
        accept_length = int(row_metadata[0])
        path_idx = int(row_metadata[1])
        next_token_id = int(row_metadata[2])
        path_host = row_metadata[metadata_width:metadata_width + max_path_width]
        if (path_host[:accept_length + 1] < 0).any():
            raise RuntimeError("sampled path contains padded tree indices")
        accepted_token_ids = [int(token) for token in row_metadata[
            metadata_width + max_path_width:metadata_width + max_path_width + accept_length
        ]]
        path = selected_paths[row, :accept_length + 1]
        hidden_3h = selected_hidden[row:row + 1, :accept_length + 1]
        accepted.append((
            accept_length,
            accepted_tokens_all[row, :accept_length],
            next_tokens[row:row + 1],
            hidden_3h[:, -1:, :],
            hidden_3h if accept_length else None,
            path_idx,
            None,
            path,
            accepted_token_ids,
            next_token_id,
        ))
    return accepted


def _materialize_coverage_batch(algorithm, states):
    """Read all deferred coverage diagnostics with one post-decode transfer."""
    flat_tensors = []
    state_specs = []
    for state in states:
        coverage_specs = []
        for draft_tokens, target_choices, retrieve_indices, node_block_slots in (
            state.coverage_raw
        ):
            tensors = [draft_tokens, target_choices, retrieve_indices]
            coverage_slots_host = node_block_slots
            if isinstance(node_block_slots, torch.Tensor) and node_block_slots.is_cuda:
                tensors.append(node_block_slots)
                coverage_slots_host = None
            shapes = [tuple(tensor.shape) for tensor in tensors]
            flat_tensors.extend(
                tensor.reshape(-1).to(dtype=torch.long) for tensor in tensors
            )
            coverage_specs.append((shapes, coverage_slots_host))
        state_specs.append(coverage_specs)

    if not flat_tensors:
        return
    packed = torch.cat(flat_tensors).cpu()
    algorithm._batched_coverage_readbacks = (
        getattr(algorithm, "_batched_coverage_readbacks", 0) + 1
    )

    cursor = 0
    for state, coverage_specs in zip(states, state_specs):
        materialized = []
        for shapes, coverage_slots_host in coverage_specs:
            tensors = []
            for shape in shapes:
                numel = math.prod(shape)
                tensors.append(packed[cursor:cursor + numel].reshape(shape))
                cursor += numel
            if coverage_slots_host is None:
                draft_cpu, target_cpu, retrieve_cpu, coverage_slots = tensors
            else:
                draft_cpu, target_cpu, retrieve_cpu = tensors
                coverage_slots = coverage_slots_host
            materialized.append((
                draft_cpu,
                target_cpu,
                retrieve_cpu,
                coverage_slots,
            ))
        state.coverage_raw = materialized


def _eos_ids(algorithm) -> set[int]:
    values = []
    values.append(getattr(algorithm.tokenizer, "eos_token_id", None))
    generation_config = getattr(algorithm.target_model, "generation_config", None)
    values.append(getattr(generation_config, "eos_token_id", None))
    result: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(token) for token in value)
        else:
            result.add(int(value))
    return result


def _build_next_draft(
    algorithm,
    state: _RequestState,
    accept_length: int,
    accepted_tokens: torch.Tensor,
    next_token: torch.Tensor,
    last_hidden_3h: torch.Tensor,
    lazy_hidden_3h: torch.Tensor | None,
):
    _set_request_ngram_context(algorithm, state)
    if accept_length == 0:
        algorithm.draft_model.pop_cache(state.draft_cache)
        result = _with_tree_budget(
            algorithm,
            state,
            lambda: algorithm._build_draft_tree(
                last_hidden_3h,
                next_token,
                state.draft_cache,
                state.draft_position,
                temperature=0.0,
            ),
        )
    else:
        if accept_length == 1:
            batch_tokens = torch.cat((next_token, next_token), dim=1)
        else:
            batch_tokens = torch.cat((
                accepted_tokens[1:].unsqueeze(0), next_token, next_token
            ), dim=1)
        (
            logits,
            rank_logits,
            draft_hidden,
            ttt_kv,
            state.draft_position,
        ) = algorithm.draft_model.update_cache_and_draft(
            lazy_hidden_3h,
            batch_tokens,
            state.draft_cache,
            state.draft_position + 1,
        )
        result = _with_tree_budget(
            algorithm,
            state,
            lambda: algorithm._build_tree_from_block1_dispatch(
                logits,
                rank_logits,
                draft_hidden,
                ttt_kv,
                next_token,
                state.draft_cache,
                state.draft_position - 1,
                temperature=0.0,
            ),
        )
    state.current_token = next_token
    state.tree = _clone_tree_result(result)


def _build_next_drafts_batched(
    algorithm,
    draft_cache: _DraftBatchCache,
    requests: Sequence[Tuple[
        _RequestState,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]],
    temperature: float,
):
    K = int(algorithm.draft_model.K)
    rollback_rows = [
        row for row, request in enumerate(requests) if int(request[1]) == 0
    ]
    if rollback_rows:
        draft_cache.rollback_rows(rollback_rows, K)

    hidden_rows = []
    token_rows = []
    start_positions = []
    valid_lengths = []
    for state, accept_length, accepted_tokens, next_token, last_hidden, lazy_hidden in requests:
        accept_length = int(accept_length)
        if accept_length == 0:
            hidden_rows.append(last_hidden)
            token_rows.append(next_token)
            start_positions.append(state.draft_position)
            valid_lengths.append(1)
            continue
        if accept_length == 1:
            batch_tokens = torch.cat((next_token, next_token), dim=1)
        else:
            batch_tokens = torch.cat((
                accepted_tokens[1:].unsqueeze(0), next_token, next_token
            ), dim=1)
        hidden_rows.append(lazy_hidden)
        token_rows.append(batch_tokens)
        start_positions.append(state.draft_position + 1)
        valid_lengths.append(accept_length + 1)

    logits, rank_logits, draft_hidden, ttt_kv = _run_ragged_draft_forward(
        algorithm,
        draft_cache,
        hidden_rows,
        token_rows,
        start_positions,
        valid_lengths,
    )

    tree_positions = []
    for row, request in enumerate(requests):
        state, accept_length, _accepted_tokens, _next_token, _last_hidden, _lazy_hidden = request
        if int(accept_length) > 0:
            state.draft_position = start_positions[row] + valid_lengths[row]
        state.draft_cache = draft_cache.row_view(row)
        _set_request_ngram_context(algorithm, state)
        tree_positions.append(
            state.draft_position if int(accept_length) == 0
            else state.draft_position - 1
        )
    results = algorithm._build_trees_from_block1_batched(
        logits,
        rank_logits,
        draft_hidden,
        ttt_kv,
        torch.cat([request[3] for request in requests], dim=0),
        draft_cache,
        tree_positions,
        [request[0].tree_budget for request in requests],
        temperature=temperature,
    )
    for request, result in zip(requests, results):
        state, _accept_length, _accepted_tokens, next_token, _last_hidden, _lazy_hidden = request
        state.current_token = next_token
        state.tree = _clone_tree_result(result)


def _empty_results(algorithm, conversations: Sequence[List[Dict[str, str]]]):
    results = []
    for _ in conversations:
        results.append({
            "output": "",
            "metrics": {
                "total_tokens": 0,
                "output_token_ids": [],
                "wall_time": 0.0,
                "tokens_per_second": 0.0,
                "accept_length": 0.0,
                "iterations": 0,
                "accept_lengths_raw": [],
                "prefill_time": 0.0,
                "draft_time": 0.0,
                "target_time": 0.0,
                "verify_time": 0.0,
                "other_time": 0.0,
            },
        })
    algorithm.last_batch_metrics = {
        "batch_wall_time": 0.0,
        "batch_prefill_time": 0.0,
        "batch_draft_time": 0.0,
        "batch_target_time": 0.0,
        "batch_verify_time": 0.0,
        "batch_decode_rounds": 0,
        "active_sizes": [],
        "wall_time": 0.0,
        "prefill_time": 0.0,
        "draft_time": 0.0,
        "target_time": 0.0,
        "verify_time": 0.0,
        "iterations": 0,
        "engine_batch_size": len(conversations),
    }
    return results


@torch.inference_mode()
def generate_conversations(
    algorithm,
    conversations: Sequence[List[Dict[str, str]]],
    max_new_tokens: int,
    temperature: float = 0.0,
    **kwargs,
) -> List[Dict]:
    """Generate a request batch with one target tree forward per active round."""
    sampling = temperature > 1e-5
    use_legacy_compiled_b1 = (
        len(conversations) == 1
        and not sampling
        and os.environ.get("DRAFT_COMPILE", "0") in {"1", "2"}
        and os.environ.get("SPECBLOCK_HYBRID_B1", "0") == "1"
    )
    if getattr(algorithm, "_adapt_hooks", None) is not None:
        raise NotImplementedError(
            "online adaptation hooks are not request-batch aware yet"
        )
    draft_device = next(algorithm.draft_model.parameters()).device
    draft_head_dim = int(algorithm.draft_model.layers[0].self_attn.head_dim)
    if draft_device.type != "cuda" or draft_head_dim != 128:
        raise NotImplementedError(
            "request-batched SpecBlock draft requires CUDA with head_dim=128"
        )
    if os.environ.get("TREE_ATTN_3WAY", "1") != "1" or os.environ.get(
        "TREE_ATTN_SDPA", "0"
    ) == "1":
        raise NotImplementedError(
            "request-batched SpecBlock draft requires TREE_ATTN_3WAY=1 "
            "and TREE_ATTN_SDPA=0"
        )
    unsupported_cache_modes = (
        "STREAM_SNAP_PROMPT",
        "STREAM_COMPACT_PROMPT",
        "STREAM_COMPACT_DECODE_MAINTAIN",
    )
    enabled_cache_modes = [
        name for name in unsupported_cache_modes
        if os.environ.get(name, "0") == "1"
    ]
    if (
        int(os.environ.get("STREAM_COMPACT_NEAR_WIN", "0")) > 0
        and os.environ.get("STREAM_COMPACT_DECODE_MAINTAIN", "1") == "1"
        and "STREAM_COMPACT_DECODE_MAINTAIN" not in enabled_cache_modes
    ):
        enabled_cache_modes.append("STREAM_COMPACT_DECODE_MAINTAIN")
    if enabled_cache_modes:
        raise NotImplementedError(
            "request-batched SpecBlock draft does not support cache compaction: "
            + ", ".join(enabled_cache_modes)
        )
    if not conversations:
        algorithm.last_batch_metrics = {
            "batch_wall_time": 0.0,
            "batch_prefill_time": 0.0,
            "batch_draft_time": 0.0,
            "batch_target_time": 0.0,
            "batch_verify_time": 0.0,
            "batch_decode_rounds": 0,
            "active_sizes": [],
            "wall_time": 0.0,
            "prefill_time": 0.0,
            "draft_time": 0.0,
            "target_time": 0.0,
            "verify_time": 0.0,
            "iterations": 0,
            "engine_batch_size": 0,
        }
        return []
    if max_new_tokens <= 0:
        return _empty_results(algorithm, conversations)
    if len(conversations) > 1 and getattr(algorithm, "draft_quantize", None) is not None:
        raise NotImplementedError(
            "request-batched SpecBlock requires an unquantized draft model; "
            "grouped block-2 projections require tensor weights"
        )

    cuda_timing = torch.cuda.is_available() and str(algorithm.device).startswith("cuda")
    if cuda_timing:
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    phase_timer = _CudaPhaseTimer(cuda_timing)
    algorithm._block_forward_events = defaultdict(list)
    block2_packed_before = getattr(
        algorithm, "_batched_block2_packed_leaves", 0
    )
    block2_padded_before = getattr(
        algorithm, "_batched_block2_padded_capacity", 0
    )

    chat_template_kwargs = algorithm._chat_template_kwargs()
    prompts = [
        algorithm.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        for conversation in conversations
    ]
    tokenizer = algorithm.tokenizer
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(algorithm.device)
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    prompt_lengths = attention_mask.sum(dim=1).tolist()
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    states = [
        _RequestState(
            conversation=list(conversation),
            prompt_ids=input_ids[idx:idx + 1, :int(prompt_lengths[idx])].clone(),
            prompt_len=int(prompt_lengths[idx]),
            tree_budget=_tree_budget(algorithm, int(prompt_lengths[idx])),
            target_prefix_len=int(prompt_lengths[idx]),
        )
        for idx, conversation in enumerate(conversations)
    ]
    prompt_width = int(input_ids.shape[1])
    max_tree_width = max(state.tree_budget for state in states) + 1
    target_cache = _DenseTargetCache(
        num_layers=int(algorithm.target_model.config.num_hidden_layers),
        max_batch_size=len(states),
        prompt_width=prompt_width,
        max_new_tokens=max_new_tokens,
        max_tree_width=max_tree_width,
    )
    target_cache.prepare_prefill()
    prefill_cache_position = torch.arange(
        prompt_width, dtype=torch.long, device=input_ids.device
    )

    prefill_start = phase_timer.start()
    prefill_hidden, hidden_sources = _run_target_backbone(
        algorithm,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=target_cache,
        cache_position=prefill_cache_position,
        use_cache=True,
    )
    prompt_ends = torch.tensor(
        [length - 1 for length in prompt_lengths],
        dtype=torch.long,
        device=prefill_hidden.device,
    )
    prompt_rows = torch.arange(len(states), device=prefill_hidden.device)
    first_logits = algorithm.target_model.lm_head(
        prefill_hidden[prompt_rows, prompt_ends]
    )
    if sampling:
        first_tokens = torch.multinomial(
            torch.softmax(first_logits.float() / temperature, dim=-1),
            num_samples=1,
        )
    else:
        first_tokens = torch.argmax(first_logits, dim=-1, keepdim=True)
    del first_logits, prefill_hidden
    eos_ids = _eos_ids(algorithm)
    phase_timer.stop("prefill", prefill_start)

    initial_draft_start = phase_timer.start()
    first_token_ids = first_tokens.flatten().cpu()
    algorithm._batched_prefill_readbacks = (
        getattr(algorithm, "_batched_prefill_readbacks", 0) + 1
    )
    initial_entries = []
    for idx, (state, first_id) in enumerate(zip(states, first_token_ids)):
        first_token = first_tokens[idx:idx + 1]
        state.current_token = first_token
        first_id = int(first_id)
        state.output_tokens.append(first_id)
        if first_id in eos_ids or max_new_tokens == 1:
            state.finished = True
            continue
        initial_entries.append((idx, state))
    initial_chunk_counts = [
        (state.prompt_len + 64) // 64 for _source_idx, state in initial_entries
    ]
    if any(
        left < right
        for left, right in zip(initial_chunk_counts, initial_chunk_counts[1:])
    ):
        initial_entries.sort(
            key=lambda entry: (entry[1].prompt_len + 64) // 64,
            reverse=True,
        )
    del first_tokens, first_token_ids
    draft_cache = None
    if initial_entries:
        if use_legacy_compiled_b1:
            if len(initial_entries) != 1:
                raise RuntimeError("compiled B1 draft requires exactly one active request")
            source_idx, state = initial_entries[0]
            prompt_hidden = torch.cat(
                [
                    source[source_idx:source_idx + 1, :state.prompt_len]
                    for source in hidden_sources
                ],
                dim=-1,
            )
            _build_initial_draft(algorithm, state, prompt_hidden)
        else:
            draft_cache = _build_initial_drafts_batched(
                algorithm,
                initial_entries,
                hidden_sources,
                max_new_tokens,
                original_engine_b1=len(conversations) == 1,
                temperature=temperature,
            )
    del hidden_sources
    phase_timer.stop("draft", initial_draft_start)

    active = [state for _source_idx, state in initial_entries]
    if active:
        target_cache.compact_rows(
            [source_idx for source_idx, _state in initial_entries]
        )
    active_sizes: List[int] = []
    target_attention_widths: List[int] = []
    target_fixed_attention_widths: List[int] = []
    target_committed_widths: List[int] = []
    target_tree_starts: List[int] = []
    target_tree_widths: List[int] = []
    target_commit_reserves: List[int] = []
    target_verify_lm_head_rows = 0
    target_verify_lm_head_capacity = 0
    while active:
        active_sizes.append(len(active))
        prefix_lengths = [state.target_prefix_len for state in active]
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id or 0
        tree_input_ids, tree_position_ids, tree_widths, tree_width = _pack_trees(
            active, int(pad_token_id)
        )
        # retrieve_indices may keep a fixed column capacity padded with -1.
        # A committed path can never contain more entries than the materialized
        # tree, so cap the reserve without scanning GPU values or synchronizing.
        commit_reserve = min(
            tree_width,
            max(int(state.tree[3].shape[1]) for state in active),
        )
        target_cache.prepare_tree(
            tree_width,
            prefix_lengths,
            commit_reserve,
        )
        tree_cache_position = target_cache.tree_cache_position(
            tree_width, tree_input_ids.device
        )
        model_dtype = next(algorithm.target_model.parameters()).dtype
        additive_mask = _build_additive_mask(
            active,
            prefix_lengths,
            target_cache.tree_start,
            tree_widths,
            tree_width,
            model_dtype,
            tree_input_ids.device,
        )
        target_attention_widths.append(target_cache.attention_width)
        target_fixed_attention_widths.append(
            target_cache.committed_capacity + tree_width
        )
        target_committed_widths.append(max(prefix_lengths))
        target_tree_starts.append(target_cache.tree_start)
        target_tree_widths.append(tree_width)
        target_commit_reserves.append(commit_reserve)
        target_start = phase_timer.start()
        target_backbone_start = phase_timer.start()
        verify_hidden, verify_hidden_sources = _run_target_backbone(
            algorithm,
            input_ids=tree_input_ids,
            past_key_values=target_cache,
            cache_position=tree_cache_position,
            position_ids=tree_position_ids,
            attention_mask=additive_mask,
            use_cache=True,
        )
        phase_timer.stop("target_backbone", target_backbone_start)
        if sampling:
            node_argmax = None
            node_topk = None
            sampling_rows_before = getattr(
                algorithm, "_sampling_lm_head_rows", 0
            )
        else:
            target_lm_head_start = phase_timer.start()
            (
                node_argmax,
                node_topk,
                _sampling_logits,
                projected_rows,
            ) = _project_tree_decisions(
                algorithm,
                verify_hidden,
                tree_widths,
                return_sampling_logits=False,
            )
            target_verify_lm_head_rows += projected_rows
            phase_timer.stop("target_lm_head", target_lm_head_start)
        target_verify_lm_head_capacity += len(active) * tree_width
        phase_timer.stop("target", target_start)

        verify_start = phase_timer.start()
        if sampling:
            accepted_data = _accept_sampling_batch(
                algorithm,
                active,
                tree_input_ids,
                verify_hidden,
                verify_hidden_sources,
                temperature,
            )
            target_verify_lm_head_rows += (
                getattr(algorithm, "_sampling_lm_head_rows", 0)
                - sampling_rows_before
            )
        else:
            accepted_data = _accept_active_batch(
                algorithm,
                active,
                tree_input_ids,
                node_argmax,
                node_topk,
                verify_hidden_sources,
            )
        del node_argmax, node_topk, verify_hidden, verify_hidden_sources
        phase_timer.stop("verify", verify_start)

        next_active = []
        next_active_rows = []
        pending_commits = []
        for active_idx, (state, accepted) in enumerate(zip(active, accepted_data)):
            (
                accept_length,
                accepted_tokens,
                next_token,
                last_hidden_3h,
                lazy_hidden_3h,
                best_candidate,
                coverage_data,
                selected_cache_indices,
                accepted_token_ids,
                next_token_id,
            ) = accepted
            state.rank_stats_raw.append(state.tree[4])
            if coverage_data is not None:
                state.coverage_raw.append(coverage_data)

            proposed = accepted_token_ids + [next_token_id]
            remaining = max_new_tokens - len(state.output_tokens)
            taken = []
            for token in proposed[:remaining]:
                taken.append(int(token))
                if int(token) in eos_ids:
                    break
            old_output_len = len(state.output_tokens)
            state.output_tokens.extend(taken)
            effective_accept = min(accept_length, len(taken))
            state.accept_lengths_raw.append(effective_accept)
            state.iterations += 1
            _update_ngram_cache(algorithm, state, old_output_len)

            state.finished = (
                len(state.output_tokens) >= max_new_tokens
                or (taken and taken[-1] in eos_ids)
                or len(taken) < len(proposed)
            )
            if state.finished:
                state.draft_cache = None
                state.tree = None
                state.current_token = None
            else:
                logical_prefix_len = int(prefix_lengths[active_idx])
                committed = int(selected_cache_indices.numel())
                pending_commits.append((
                    active_idx,
                    logical_prefix_len,
                    selected_cache_indices,
                ))
                state.target_prefix_len = logical_prefix_len + committed
                next_active_rows.append(active_idx)
                next_active.append((
                    state,
                    accept_length,
                    accepted_tokens,
                    next_token,
                    last_hidden_3h,
                    lazy_hidden_3h,
                ))

        commit_start = phase_timer.start()
        target_cache.commit_paths(pending_commits)
        if next_active:
            target_cache.compact_rows(next_active_rows)
        phase_timer.stop("verify", commit_start)
        if next_active:
            draft_start = phase_timer.start()
            if use_legacy_compiled_b1:
                if len(next_active) != 1:
                    raise RuntimeError(
                        "compiled B1 draft rebuild requires one active request"
                    )
                _build_next_draft(algorithm, *next_active[0])
            else:
                draft_cache.compact_rows(next_active_rows)
                _build_next_drafts_batched(
                    algorithm,
                    draft_cache,
                    next_active,
                    temperature=temperature,
                )
            phase_timer.stop("draft", draft_start)
        active = [args[0] for args in next_active]

    _materialize_coverage_batch(algorithm, states)
    if cuda_timing:
        torch.cuda.synchronize()
    batch_wall_time = time.perf_counter() - wall_start
    prefill_time = phase_timer.elapsed_seconds("prefill")
    draft_time = phase_timer.elapsed_seconds("draft")
    target_time = phase_timer.elapsed_seconds("target")
    target_backbone_time = phase_timer.elapsed_seconds("target_backbone")
    target_lm_head_time = phase_timer.elapsed_seconds("target_lm_head")
    verify_time = phase_timer.elapsed_seconds("verify")
    other_time = max(
        0.0, batch_wall_time - prefill_time - draft_time - target_time - verify_time
    )
    block2_packed_leaves = (
        getattr(algorithm, "_batched_block2_packed_leaves", 0)
        - block2_packed_before
    )
    block2_padded_capacity = (
        getattr(algorithm, "_batched_block2_padded_capacity", 0)
        - block2_padded_before
    )
    if not 0 <= block2_packed_leaves <= block2_padded_capacity:
        raise RuntimeError(
            "packed block-2 accounting is inconsistent: "
            f"packed={block2_packed_leaves} padded={block2_padded_capacity}"
        )

    algorithm.last_batch_metrics = {
        "batch_wall_time": batch_wall_time,
        "batch_prefill_time": prefill_time,
        "batch_draft_time": draft_time,
        "batch_target_time": target_time,
        "batch_target_backbone_time": target_backbone_time,
        "batch_target_lm_head_time": target_lm_head_time,
        "batch_verify_time": verify_time,
        "batch_other_time": other_time,
        "batch_decode_rounds": len(active_sizes),
        "active_sizes": active_sizes,
        "target_attention_widths": target_attention_widths,
        "target_fixed_attention_widths": target_fixed_attention_widths,
        "target_committed_widths": target_committed_widths,
        "target_tree_starts": target_tree_starts,
        "target_tree_widths": target_tree_widths,
        "target_commit_reserves": target_commit_reserves,
        "target_prefill_lm_head_rows": len(states),
        "target_prefill_lm_head_capacity": len(states) * prompt_width,
        "target_verify_lm_head_rows": target_verify_lm_head_rows,
        "target_verify_lm_head_capacity": target_verify_lm_head_capacity,
        "target_lm_head_rows_removed": (
            len(states) * prompt_width
            + target_verify_lm_head_capacity
            - len(states)
            - target_verify_lm_head_rows
        ),
        "block2_packed_leaves": block2_packed_leaves,
        "block2_padded_capacity": block2_padded_capacity,
        "block2_padding_removed": (
            block2_padded_capacity - block2_packed_leaves
        ),
        "block2_padding_removed_pct": (
            100.0
            * (block2_padded_capacity - block2_packed_leaves)
            / block2_padded_capacity
            if block2_padded_capacity > 0
            else 0.0
        ),
        "batch_size": len(states),
        # Standard runner fields.
        "wall_time": batch_wall_time,
        "prefill_time": prefill_time,
        "draft_time": draft_time,
        "target_time": target_time,
        "target_backbone_time": target_backbone_time,
        "target_lm_head_time": target_lm_head_time,
        "verify_time": verify_time,
        "iterations": len(active_sizes),
        "engine_batch_size": len(states),
    }

    draft_forward_times = {}
    if cuda_timing:
        for depth, events in algorithm._block_forward_events.items():
            draft_forward_times[depth] = sum(
                start.elapsed_time(end) for start, end in events
            ) / 1000.0
    algorithm.last_batch_metrics["draft_forward_times"] = draft_forward_times

    results = []
    for state in states:
        output_tensor = torch.tensor(
            state.output_tokens, dtype=torch.long, device=algorithm.device
        )
        output_text = tokenizer.decode(output_tensor, skip_special_tokens=True)
        coverage_stats = [
            algorithm._compute_coverage_stats(draft, target, retrieve, slots)
            for draft, target, retrieve, slots in state.coverage_raw
        ]
        rank_stats = algorithm._aggregate_rank_stats(
            state.rank_stats_raw, state.accept_lengths_raw
        )
        block_pos_stats = algorithm._aggregate_block_pos_stats(coverage_stats)
        num_tokens = len(state.output_tokens)
        results.append({
            "output": output_text,
            "metrics": {
                "total_tokens": num_tokens,
                "output_token_ids": list(state.output_tokens),
                "wall_time": batch_wall_time,
                "tokens_per_second": num_tokens / batch_wall_time if batch_wall_time > 0 else 0.0,
                "accept_length": algorithm.compute_accept_length(state.accept_lengths_raw),
                "iterations": state.iterations,
                "accept_lengths_raw": list(state.accept_lengths_raw),
                "prefill_time": prefill_time,
                "draft_time": draft_time,
                "target_time": target_time,
                "verify_time": verify_time,
                "other_time": other_time,
                "draft_pct": draft_time / batch_wall_time * 100 if batch_wall_time > 0 else 0.0,
                "target_pct": target_time / batch_wall_time * 100 if batch_wall_time > 0 else 0.0,
                "verify_pct": verify_time / batch_wall_time * 100 if batch_wall_time > 0 else 0.0,
                "draft_forward_times": {},
                "rank_stats": rank_stats,
                "block_pos_stats": block_pos_stats,
            },
        })
    return results
