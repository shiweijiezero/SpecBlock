"""SpecBlock-Shift speculative decoding spec_info.

Two independent dataclasses, both directly subclassing :class:`SpecInput`
(NOT :class:`EagleDraftInput`) so that SGLang's scheduler treats SpecBlock-
Shift via duck-typed dispatch on ``filter_batch`` / ``merge_batch`` /
``prepare_for_verify`` / ``prepare_extend_after_decode`` / etc.  This is
the same polymorphic pattern Medusa and Ngram already follow.

Lifecycle::

    iter 1 prefill:
        forward_target_extend  -- target captures 3-layer aux hidden
        _draft_extend_for_prefill -- draft consumes 3H, produces b0_*
        => SpecBlockDraftInput populated for iter 2

    iter >=2 decode:
        draft -- per-req specblock_tree_builder.build_tree(...) then
                 concat -> SpecBlockVerifyInput
        target_verify -- runs native indexed root-to-node target attention
        verify -- greedy / sampling tree accept (re-uses EAGLE kernels
                  verify_tree_greedy and tree_speculative_sampling_target_only)
        prepare_extend_after_decode -- batch shape rebuild for next iter
        _draft_extend_for_decode -- new b0_* state from accepted hidden
        => SpecBlockDraftInput populated for iter k+1

State that lives across iters (per-request) is held *inside* this class:
    - hidden_states (B, 3H) target 3-layer concat (cross condition input)
    - cross_loc:        per-req K cross attention pool indices (1D int64)
    - ttt_k / ttt_v:     per-req K TTT (test-time training) KV tensors
    - b0_logits / b0_hidden / b0_input_id: cached block-0 outputs

Filter/merge propagate the per-req lists in lockstep with batch.reqs so
that scheduler split / merge / chunked prefill / preempt all keep the
SpecBlock state consistent.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    TREE_SPEC_KERNEL_AVAILABLE,
    assign_req_to_token_pool,
    create_extend_after_decode_spec_info,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, next_power_of_2

logger = logging.getLogger(__name__)

_is_npu = is_npu()

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )


# ============================================================
#  Topology helpers (parents -> EAGLE-format BFS pointers)
# ============================================================

def specblock_tree_max_depth(K: int, max_blocks: int) -> int:
    """Return the tree builder's root-inclusive parent-walk depth bound."""
    K = int(K)
    max_blocks = int(max_blocks)
    if K < 1:
        raise ValueError(f"SpecBlock tree K must be positive, got {K}.")
    if max_blocks < 1:
        raise ValueError(
            f"SpecBlock tree max_blocks must be positive, got {max_blocks}."
        )
    # Each block may append a K-token parent chain to an existing path.
    return K * max_blocks


def parents_to_next_token_and_sibling(
    parents: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a flat parents array to EAGLE's (next_token, next_sibling).

    The EAGLE verify kernel requires three integer arrays per batch:

      retrive_index[b, i]:        global tree-token offset of node i
      retrive_next_token[b, i]:   index of node i's first child (-1 if leaf)
      retrive_next_sibling[b, i]: index of next sibling (-1 if last child)

    SpecBlock-Shift's tree builder yields ``parents`` in BFS order with
    parents[0] = -1 (root).  We reconstruct the BFS first-child / next-
    sibling pointers by a single forward pass over ``parents``.

    Args:
        parents: int64 tensor of shape [N], parents[i] in [-1, N-1].

    Returns:
        next_token:   int64 tensor [N], -1 if leaf.
        next_sibling: int64 tensor [N], -1 if last child.
    """
    n = parents.shape[0]
    device = parents.device
    next_token = torch.full((n,), -1, dtype=torch.int64, device=device)
    next_sibling = torch.full((n,), -1, dtype=torch.int64, device=device)

    # Build incidence on CPU for a single linear pass (n is small, <=600).
    parents_cpu = parents.tolist()
    last_child_cpu: List[int] = [-1] * n
    next_token_cpu: List[int] = [-1] * n
    next_sibling_cpu: List[int] = [-1] * n

    for i in range(n):
        p = parents_cpu[i]
        if p < 0:
            continue
        prev_last = last_child_cpu[p]
        if prev_last < 0:
            next_token_cpu[p] = i
        else:
            next_sibling_cpu[prev_last] = i
        last_child_cpu[p] = i

    next_token.copy_(torch.tensor(next_token_cpu, dtype=torch.int64))
    next_sibling.copy_(torch.tensor(next_sibling_cpu, dtype=torch.int64))
    return next_token, next_sibling


# ============================================================
#  SpecBlockDraftInput
# ============================================================

@dataclass
class SpecBlockDraftInput(SpecInput):
    """Per-iteration draft state for SpecBlock-Shift.

    Required tensors (matching forward_batch_info.py:923-940 hardcoded
    field accesses so SGLang's CUDA-graph padding path stays valid):

        hidden_states  (B, 3*hidden_size) target 3-layer concat hidden
                       captured at prefill / verify time. Fed into the
                       draft input_layer for the *next* iteration.
        verified_id    (B,) last accepted token id per request.
        accept_length  (B,) accept-chain length per request (excluding
                       bonus). Used by forward_batch_info.py:980 in the
                       draft-extend forward mode.
        topk_p / topk_index : intentionally None.  SpecBlock-Shift does
                       not use top-k chain candidates; the pad path at
                       forward_batch_info.py:928-933 checks ``is not
                       None`` before padding so leaving them None is safe.

    SpecBlock-specific per-request lists (length B; element shapes per req):

        cross_loc              cross-attention K/V indices into the worker's
                               :class:`SpecBlockKVPool`.  list[B] of int64
                               tensors of shape [count_i].  All draft layers
                               share the same indices (the pool stores one
                               K/V slot per (layer, position) tuple).
        ttt_k, ttt_v           per-layer TTT KV (length K=4 each).
                               list[B] of list[L_layers] of
                               (1, n_kv_heads, K, head_dim).
        b0_logits              (B, K, V_draft) cached block-0 logits.
        b0_hidden              (B, K, hidden_size) cached block-0 draft
                               hidden -- input to block-1 forward.
        b0_input_id            (B, K) tokens chosen at block-0 (greedy or
                               beam, depending on tree config).
        b0_rank_logits         (B, K, rank_classes) rank-head logits.

    capture_hidden_mode:
        CaptureHiddenMode.FULL because the draft consumes the target's
        3-layer aux hidden (concat of layer-1 / mid / N-4).  The model
        file llama_specblock.py registers
        ``aux_hidden_state_layer_ids`` so the ModelRunner emits the FULL
        3-layer tensor in logits_output.hidden_states.
    """

    # ---- forward_batch_info.py hardcoded fields (must exist) ----
    hidden_states: torch.Tensor = None
    verified_id: torch.Tensor = None
    accept_length: torch.Tensor = None
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None

    # ---- SpecBlock-specific per-req state (length-B lists) ----
    # cross_loc[i] is a 1-D int64 tensor of indices into the worker-local
    # :class:`SpecBlockKVPool` (worker.spec_kv_pool).  All draft layers
    # share these indices; the pool stores one (k, v) per (layer, slot).
    # Length matches cross_count[i] exactly.
    cross_loc: List[torch.Tensor] = field(default_factory=list)
    # cross_count[i] is the number of cross slots currently live for req i.
    # Equal to len(cross_loc[i]); kept as an int for fast access in tree
    # builder pad/gather hot paths (avoids `.numel()` per iter).
    cross_count: List[int] = field(default_factory=list)
    # cross_position[i] is the persistent HF-style draft_position returned
    # by prefill_and_draft / update_cache_and_draft.  The precomputed block-0
    # tree expands at cross_position[i] - 1; zero-accept rebuilds keep this
    # persistent value unchanged.
    cross_position: List[int] = field(default_factory=list)
    ttt_k: List[List[torch.Tensor]] = field(default_factory=list)
    ttt_v: List[List[torch.Tensor]] = field(default_factory=list)
    b0_logits: torch.Tensor = None
    b0_hidden: torch.Tensor = None
    b0_input_id: torch.Tensor = None
    b0_rank_logits: torch.Tensor = None
    # Sorted candidates already computed by the rank head for these logits.
    # Reused by one-block tree expansion to avoid duplicate vocab top-k.
    b0_top_indices: Optional[torch.Tensor] = None

    # ---- Worker-injected references (NOT serialized / deepcopied) ----
    # The worker injects its :class:`SpecBlockKVPool` here so that
    # ``filter_batch`` can release dropped reqs' indices back to the
    # pool without going through the worker.  Skipped in __deepcopy__
    # because the pool holds GPU buffers that must not be cloned.
    kv_pool: Optional[Any] = None

    # ---- Capture / metadata ----
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # ---- iter >=2 extend metadata (set by prepare_extend_after_decode) ----
    accept_length_cpu: Optional[List[int]] = None
    seq_lens_for_draft_extend: Optional[torch.Tensor] = None
    seq_lens_for_draft_extend_cpu: Optional[torch.Tensor] = None
    req_pool_indices_for_draft_extend: Optional[torch.Tensor] = None

    # ---- Spec V2 schema (mirrors EagleDraftInputV2Mixin) ----
    # The scheduler reads these on V2 path; SpecBlock-Shift now supports
    # the same overlap path so we plumb them through.
    #
    # ALLOC_LEN_PER_DECODE: tokens to over-allocate per req per decode
    # iter so target's KV pool grows by chunks rather than per-token.
    # For SpecBlock-Shift, max accept length per iter is tree depth =
    # max_blocks * K (default 2 * 4 = 8); +1 for the bonus token.
    ALLOC_LEN_PER_DECODE: int = 9
    new_seq_lens: Optional[torch.Tensor] = None
    verify_done: Optional[torch.cuda.Event] = None
    # FutureIndices is a small dataclass written by the scheduler
    # before verify; we just need to hold it through.  Typed as Any
    # to avoid an import cycle.
    future_indices: Optional[Any] = None

    def __post_init__(self):
        super().__init__(SpecInputType.SPECBLOCK_SHIFT_DRAFT)
        if self.hidden_states is not None:
            self.device = self.hidden_states.device
        elif self.verified_id is not None:
            self.device = self.verified_id.device
        else:
            self.device = None

    # ------------------------------------------------------------
    #  ABC / interface
    # ------------------------------------------------------------

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """Draft input does not consume tokens itself."""
        return 0, 0

    def release_resources(self) -> None:
        """Return every live cross-attention slot to the worker pool."""
        if self.kv_pool is not None:
            for loc in self.cross_loc:
                if loc is not None and torch.is_tensor(loc) and loc.numel() > 0:
                    self.kv_pool.free(loc)
        self.cross_loc = []
        self.cross_count = []
        self.cross_position = []
        self.ttt_k = []
        self.ttt_v = []

    @classmethod
    def create_idle_input(
        cls,
        device: str,
        hidden_size: int,
        target_hidden_concat_dim: int,
        K: int,
        num_layers: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "SpecBlockDraftInput":
        """Empty draft input for CUDA graph capture / idle batches."""
        del num_layers  # ttt is length-0 list in idle state
        return cls(
            hidden_states=torch.empty(
                (0, target_hidden_concat_dim), device=device, dtype=dtype
            ),
            verified_id=torch.empty((0,), device=device, dtype=torch.int64),
            accept_length=torch.empty((0,), device=device, dtype=torch.int32),
            cross_loc=[],
            cross_count=[],
            cross_position=[],
            ttt_k=[],
            ttt_v=[],
            b0_logits=torch.empty((0, K, 1), device=device, dtype=dtype),
            b0_hidden=torch.empty((0, K, hidden_size), device=device, dtype=dtype),
            b0_input_id=torch.empty((0, K), device=device, dtype=torch.int64),
            b0_rank_logits=torch.empty((0, K, 1), device=device, dtype=dtype),
            b0_top_indices=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

    # ------------------------------------------------------------
    #  __deepcopy__ : skip the GPU pool reference
    # ------------------------------------------------------------

    def __deepcopy__(self, memo):
        """Deep-copy without pulling the kv_pool's GPU buffers along.

        SGLang's scheduler occasionally deepcopies spec_info when
        forking (e.g. eos-overlap path).  The pool holds large GPU
        tensors and must not be duplicated; the cloned spec_info shares
        the same pool reference so its cross_loc indices remain valid.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for fname in self.__dataclass_fields__:
            if fname == "kv_pool":
                # Share, do not clone.
                setattr(new, fname, self.kv_pool)
                continue
            value = getattr(self, fname)
            try:
                setattr(new, fname, deepcopy(value, memo))
            except Exception:
                # Fall back to share-by-ref for unpickleable internals
                # (e.g. nested torch.Tensor on a non-default device).
                setattr(new, fname, value)
        # Restore parent ABC fields the dataclass does not own.
        new.spec_input_type = getattr(self, "spec_input_type", None)
        new.device = getattr(self, "device", None)
        return new

    # ------------------------------------------------------------
    #  Filter / merge (scheduler dynamic-batching plumbing)
    # ------------------------------------------------------------

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        """Filter per-req state to the surviving requests.

        SpecBlock-Shift always uses *gather* indexing (not the EAGLE-style
        ``[:B]`` truncation fast-path).  Reason: ``_refresh_draft_state``
        does NOT pre-filter to ``unfinished_index`` like EAGLE does in
        ``eagle_info.py``, so the per-req lists/tensors stored here are
        in original-bs order.  When the scheduler then calls with
        ``has_been_filtered=True`` and a sparse keep list (e.g.
        ``[0, 2, 3]`` after req 1 finishes mid-batch), the previous
        ``[:B]`` fast-path silently kept the wrong reqs and leaked /
        misaligned cross_loc → catastrophic accept drop at bs>=2 with
        mtbench-style heterogeneous response lengths.

        Treat ``has_been_filtered`` as a no-op for correctness.  The
        microsecond cost of ``index_select`` vs slice at bs<=16 is
        negligible vs the accept correctness it buys back.
        """
        if torch.is_tensor(new_indices):
            keep_idx = new_indices.tolist()
            keep_tensor = new_indices
        else:
            keep_idx = list(new_indices)
            keep_tensor = torch.tensor(
                keep_idx, dtype=torch.long, device=self.device,
            ) if self.device is not None else torch.tensor(
                keep_idx, dtype=torch.long,
            )
        del has_been_filtered  # always gather

        # V2 plan_stream path: tensors are re-fetched from FutureMap via
        # resolve_future on next iter — only reindex future_indices here,
        # leaving the tensor fields as-is (they'll be overwritten anyway).
        # Mirrors EagleDraftInput.filter_batch's early-return.
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[keep_tensor]
            # NOTE: continue to list filter below; FutureMap doesn't track
            # cross_loc/cross_count/cross_position/ttt_k/ttt_v lists.
        else:
            # V1 path: index_select tensors directly.
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[keep_tensor]
            if self.verified_id is not None:
                self.verified_id = self.verified_id[keep_tensor]
            if self.accept_length is not None:
                self.accept_length = self.accept_length[keep_tensor]
            if self.b0_logits is not None:
                self.b0_logits = self.b0_logits[keep_tensor]
            if self.b0_hidden is not None:
                self.b0_hidden = self.b0_hidden[keep_tensor]
            if self.b0_input_id is not None:
                self.b0_input_id = self.b0_input_id[keep_tensor]
            if self.b0_rank_logits is not None:
                self.b0_rank_logits = self.b0_rank_logits[keep_tensor]
            if self.b0_top_indices is not None:
                self.b0_top_indices = self.b0_top_indices[keep_tensor]

        # List fields gathered by Python slicing in lockstep with batch.reqs.
        # cross_loc holds GPU pool indices -- release the indices for
        # dropped reqs back to the pool BEFORE slicing the list, so the
        # surviving reqs' indices stay valid.
        if self.cross_loc:
            keep_set = set(keep_idx)
            old_n = len(self.cross_loc)
            if self.kv_pool is not None:
                for old_idx in range(old_n):
                    if old_idx in keep_set:
                        continue
                    loc = self.cross_loc[old_idx]
                    if loc is not None and torch.is_tensor(loc) and loc.numel() > 0:
                        self.kv_pool.free(loc)
            self.cross_loc = [self.cross_loc[i] for i in keep_idx]
        if self.cross_count:
            self.cross_count = [self.cross_count[i] for i in keep_idx]
        if self.cross_position:
            self.cross_position = [self.cross_position[i] for i in keep_idx]
        if self.ttt_k:
            self.ttt_k = [self.ttt_k[i] for i in keep_idx]
        if self.ttt_v:
            self.ttt_v = [self.ttt_v[i] for i in keep_idx]

    def merge_batch(self, other: "SpecBlockDraftInput"):
        """Merge another batch's state into self (cat tensors, extend lists).

        V2 plan_stream path: when ``future_indices`` is set, the tensor
        fields tracked by ``FutureMap._SPECBLOCK_FIELDS`` (b0_logits,
        b0_rank_logits, b0_hidden, b0_input_id, verified_id,
        new_seq_lens) are RE-FETCHED from the buffer by
        ``resolve_future`` on the next iter using the merged indices.
        Concatenating those tensors here is wasted (and stale).  We only
        need to (a) merge future_indices and (b) merge fields NOT tracked
        by the FutureMap (cross_loc/count/position, ttt_k/v lists, plus
        accept_length / hidden_states which can be useful for cross-iter
        carry).  Mirrors EagleDraftInput.merge_batch's early-return.
        """
        if other is None:
            return

        # V2 plan_stream path: only merge future_indices + lists; future-
        # buffered tensor fields will be re-fetched in resolve_future.
        if self.future_indices is not None:
            # Both running and other must have future_indices when V2 is
            # active (scheduler sets it on every batch_result via
            # batch.spec_info.future_indices = future_indices in
            # scheduler.py:2211).
            if other.future_indices is not None:
                from sglang.srt.managers.overlap_utils import FutureIndices
                merged_indices = torch.cat(
                    [self.future_indices.indices,
                     other.future_indices.indices]
                )
                self.future_indices = FutureIndices(
                    indices=merged_indices,
                    interval=slice(
                        int(merged_indices[0].item()),
                        int(merged_indices[-1].item()) + 1,
                    ),
                )
            # Merge per-req lists (not in FutureMap, so must concat here).
            if self.kv_pool is None and other.kv_pool is not None:
                self.kv_pool = other.kv_pool
            for attr in (
                "cross_loc", "cross_count", "cross_position",
                "ttt_k", "ttt_v",
            ):
                mine = getattr(self, attr)
                theirs = getattr(other, attr)
                if not mine:
                    setattr(self, attr, theirs if theirs else [])
                elif theirs:
                    setattr(self, attr, mine + theirs)
            return

        # V1 path (no future_indices): cat all tensor fields + extend lists.
        for attr in (
            "hidden_states",
            "verified_id",
            "accept_length",
            "b0_logits",
            "b0_hidden",
            "b0_input_id",
            "b0_rank_logits",
            "b0_top_indices",
        ):
            mine = getattr(self, attr)
            theirs = getattr(other, attr)
            if mine is None:
                setattr(self, attr, theirs)
            elif theirs is not None:
                setattr(self, attr, torch.cat([mine, theirs], dim=0))

        # Lists extend.  cross_loc carries GPU pool indices: when self
        # has no pool reference but other does, adopt other's pool.
        # (The scheduler may merge a fresh prefill batch into an
        # in-flight decode batch where only one side has been wired.)
        if self.kv_pool is None and other.kv_pool is not None:
            self.kv_pool = other.kv_pool
        for attr in (
            "cross_loc", "cross_count", "cross_position",
            "ttt_k", "ttt_v",
        ):
            mine = getattr(self, attr)
            theirs = getattr(other, attr)
            if not mine:
                setattr(self, attr, theirs if theirs else [])
            elif theirs:
                setattr(self, attr, mine + theirs)

    # ------------------------------------------------------------
    #  prepare_for_extend (iter 1 prefill setup)
    # ------------------------------------------------------------

    def prepare_for_extend(self, batch: ScheduleBatch):
        """Iter 1 prefill: ensure the target captures 3-layer aux hidden.

        The draft input does not exist yet at this point in the very
        first prefill (scheduler creates it after); this hook is mostly
        used to set capture_hidden_mode on the batch.  Keep symmetric
        with EAGLE/Medusa for scheduler compatibility.
        """
        if batch.forward_mode.is_idle():
            return
        # Mark batch so the target ModelRunner captures FULL aux hidden.
        # See model_runner.py:1870 + llama_eagle3.py:178-183 for analog.
        batch.capture_hidden_mode = CaptureHiddenMode.FULL

    # ------------------------------------------------------------
    #  prepare_for_decode (Spec V2 path)
    # ------------------------------------------------------------

    def prepare_for_decode(self, batch: ScheduleBatch):
        """V2 decode prep — over-allocate target KV slots for the next iter.

        Mirrors :meth:`EagleDraftInputV2Mixin.prepare_for_decode` but with
        SpecBlock-Shift's tree size budget.  ``ALLOC_LEN_PER_DECODE`` is
        sized for max_blocks * K + 1 (worst-case accept).

        The scheduler calls this after the previous iter's verify_done
        event signals; at that point we know per-req kv_committed_len and
        can compute the next iter's KV slot allocation.
        """
        if batch.tree_cache.supports_swa() and batch.tree_cache.is_chunk_cache():
            for req in batch.reqs:
                batch.tree_cache.evict_swa(req, req.seqlen - 1)

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()
        # Wait for the previous iter's verify to complete (verify_done
        # event was set on the main stream during verify).
        batch.maybe_wait_verify_done()

        page_size = batch.token_to_kv_pool_allocator.page_size
        cur_kv_lens_cpu = []
        nxt_kv_lens_cpu = []
        num_needed_tokens = 0
        for r in batch.reqs:
            x = (
                r.kv_committed_len
                + 2 * self.ALLOC_LEN_PER_DECODE
                - r.kv_allocated_len
            )
            cur_kv_lens_cpu.append(r.kv_allocated_len)
            nxt_kv_lens_cpu.append(r.kv_allocated_len + x)
            num_needed_tokens += x
            r.kv_allocated_len += x

        cur_kv_lens_cpu = torch.tensor(
            cur_kv_lens_cpu, dtype=torch.int32, device="cpu",
        )
        nxt_kv_lens_cpu = torch.tensor(
            nxt_kv_lens_cpu, dtype=torch.int32, device="cpu",
        )

        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            cur_kv_lens = cur_kv_lens_cpu.to(device=batch.device)
            nxt_kv_lens = nxt_kv_lens_cpu.to(device=batch.device)
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                nxt_kv_lens,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_cpu.to(device=batch.device),
            nxt_kv_lens_cpu.to(device=batch.device),
            out_cache_loc,
            bs,
        )

        # Sync seq_lens_cpu (V2 schedulers expect this set after
        # prepare_for_decode; matches EagleDraftInputV2Mixin behavior).
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

    # ------------------------------------------------------------
    #  prepare_extend_after_decode (iter >=2 batch reshape)
    # ------------------------------------------------------------

    def prepare_extend_after_decode(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
    ):
        """After verify, rebuild ``batch`` so the target's draft-extend
        forward replays the accepted chain into KV cache.

        Mirrors EAGLE's eagle_info.py:696-727 exactly.  The hot input is
        ``self.verified_id`` -- a 1D tensor over sum_i (accept_len_i + 1)
        produced by VerifyInput.verify (= predict[accept_index]).  The
        kernel splits it back per-req into:

            self.positions     [sum_i (accept_len_i+1)]  per-token absolute pos
            self.verified_id   [bs]  overwritten in-place with each req's
                                     LAST chain token (the "next input id"
                                     fed to the draft model).

        After this returns ``batch.input_ids`` is the FLAT chain (which
        the target ModelRunner reads to replay the KV writes).  The
        per-req ``verified_id`` lives on as ``self.verified_id`` for the
        next iter's draft.
        """
        if batch.forward_mode.is_idle():
            return

        bs = batch.batch_size()

        # Stash the *prefix* state for the draft extend forward (the
        # draft needs to know the prefix end to read existing KV).
        self.seq_lens_for_draft_extend = batch.seq_lens.clone()
        self.seq_lens_for_draft_extend_cpu = (
            batch.seq_lens_cpu.clone() if batch.seq_lens_cpu is not None else None
        )
        self.req_pool_indices_for_draft_extend = batch.req_pool_indices.clone()

        # Cache cpu copy for downstream extend_lens construction.
        accept_length_cpu = self.accept_length.cpu()
        self.accept_length_cpu = accept_length_cpu.tolist()

        # batch.input_ids = the accepted chain (flat over all reqs)
        batch.input_ids = self.verified_id
        batch.extend_lens = [x + 1 for x in self.accept_length_cpu]
        batch.extend_num_tokens = sum(batch.extend_lens)
        # Note: seq_lens here is the OLD prefix end (we just stashed it).
        # The scheduler / target runner will derive new seq_lens after
        # the extend forward writes the chain into cache.
        batch.seq_lens = self.seq_lens_for_draft_extend
        batch.seq_lens_cpu = self.seq_lens_for_draft_extend_cpu
        batch.req_pool_indices = self.req_pool_indices_for_draft_extend
        batch.return_logprob = False
        batch.return_hidden_states = False

        # accept_length now includes the bonus token => +1.
        self.accept_length.add_(1)
        # Allocate output buffers for the kernel.
        self.positions = torch.empty_like(batch.input_ids, dtype=torch.long)
        new_verified_id = torch.empty_like(self.accept_length, dtype=torch.int32)

        create_extend_after_decode_spec_info[(bs,)](
            batch.input_ids,             # input: accepted chain flat
            batch.seq_lens,              # input: prefix end (per req)
            self.accept_length,          # input: accept length (with bonus)
            self.positions,              # output: per-token absolute pos
            new_verified_id,             # output: per-req last chain token
            next_power_of_2(max(speculative_num_steps + 1, bs)),
        )
        # Overwrite verified_id with the per-req last token (the
        # "next-iter input_id" given to the draft model).
        self.verified_id = new_verified_id

        # SpecBlock-Shift draft consumes the target's 3-layer aux hidden;
        # request the target verify forward to capture FULL again so the
        # next iter's draft has fresh cross-condition input.
        self.capture_hidden_mode = CaptureHiddenMode.FULL


# ============================================================
#  SpecBlockVerifyInput
# ============================================================

@dataclass
class SpecBlockVerifyInput(SpecInput):
    """Per-iteration verify state for SpecBlock-Shift.

    Built fresh by ``draft()`` each decode step from per-req trees.  The
    verify forward and acceptance run on a single concatenated batch.

    Required tensors (matching forward_batch_info.py:974/984 hardcoded):
        draft_token_num     dynamic per-step max tree size across batch
        num_tokens_per_batch alias of draft_token_num for the
                             draft_extend_v2 path

    Tree-flat 1D fields (sum_i N_i, where N_i is each req's tree size):
        draft_token         int64, the candidate token IDs per node.
        positions           int64, BFS position-id per node.
        custom_mask         bool [sum_i N_i * N_i], request-packed tree-only
                            ancestor masks; prefix visibility is implicit.

    Per-batch 2D BFS pointer fields (B, N_max):
        retrive_index       int64, global flat offset of each tree node.
        retrive_next_token  int64, first-child index (-1 if leaf).
        retrive_next_sibling int64, next-sibling (-1 if last child).
        Padding rows beyond N_i are filled with -1.

    SpecBlock-specific:
        tree_parents        int64 [B, N_max], BFS parent per node.
        tree_depth          int64 [sum_i N_i], depth per node in its tree.
        tree_max_depth      configured maximum token-parent depth emitted by
                            the tree builder; target native decode uses it as
                            a fail-stop Triton parent-walk bound.
        tree_lps            float32 [sum_i N_i], path log-prob (for
                            beam-search-style sorting if needed).

    Carry-through state:
        verified_id         (B,) last accepted token (==
                            DraftInput.verified_id for the *next* iter).
        accept_length       (B,) accept-chain length (set by verify()).
        accept_index        (B, N_max) -1-padded accepted node indices.
        predict             tree_token_num+1 1D, kernel-filled gold IDs.

    SpecBlock 3-layer hidden carry:
        hidden_states       (B, 3H) used to seed next iter's draft.

    capture_hidden_mode = FULL because the target verify forward also
    needs to emit 3-layer aux hidden so the next iter's draft has fresh
    cross-condition input.
    """

    # ---- Required by hardcoded fields ----
    draft_token: torch.Tensor = None
    custom_mask: torch.Tensor = None
    positions: torch.Tensor = None
    retrive_index: torch.Tensor = None
    retrive_next_token: torch.Tensor = None
    retrive_next_sibling: torch.Tensor = None
    draft_token_num: int = 0          # fwd_batch_info.py:974
    num_tokens_per_batch: int = 0     # fwd_batch_info.py:984

    # ---- SpecBlock-specific tree metadata ----
    tree_parents: torch.Tensor = None
    tree_depth: torch.Tensor = None
    tree_max_depth: int = 0
    tree_lps: torch.Tensor = None
    tree_sizes_cpu: Optional[List[int]] = None  # per-req actual N_i

    # ---- Carried per-req SpecBlock state (mirrors SpecBlockDraftInput
    # fields so it survives the draft -> verify -> refresh hop). The
    # scheduler's merge_batch / filter_batch on spec_info propagates them
    # in lockstep with batch.reqs, which is why we keep them on the
    # spec_info instead of self.draft_states[rid] / self._prev_lists. ----
    cross_loc: List[torch.Tensor] = field(default_factory=list)
    cross_count: List[int] = field(default_factory=list)
    cross_position: List[int] = field(default_factory=list)
    ttt_k: List[List[torch.Tensor]] = field(default_factory=list)
    ttt_v: List[List[torch.Tensor]] = field(default_factory=list)
    # Worker-injected KV pool reference (skipped in __deepcopy__).
    kv_pool: Optional[Any] = None

    # ---- Carry-through hidden / accept ----
    hidden_states: torch.Tensor = None
    verified_id: torch.Tensor = None
    accept_length: torch.Tensor = None
    accept_index: torch.Tensor = None
    predict: torch.Tensor = None

    # ---- Sequence metadata ----
    seq_lens_sum: int = 0
    seq_lens_cpu: torch.Tensor = None

    # ---- Capture ----
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # ---- KV alloc bookkeeping (set by prepare_for_verify) ----
    last_loc: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(SpecInputType.SPECBLOCK_SHIFT_VERIFY)
        if self.draft_token is not None:
            self.device = self.draft_token.device
        elif self.hidden_states is not None:
            self.device = self.hidden_states.device
        else:
            self.device = None
        # Keep the fwd_batch_info alias in sync.
        if self.num_tokens_per_batch == 0:
            self.num_tokens_per_batch = self.draft_token_num

    def _trace(self, label: str, **kw) -> None:
        """V1-vs-V2 divergence trace via env SPECBLOCK_V2_TRACE.

        Logs verify-related tensors for first 30 iters so a side-by-side
        diff can isolate the layer where V2 path diverges from V1.
        """
        import os as _os
        if _os.environ.get("SPECBLOCK_V2_TRACE", "0") != "1":
            return
        cls = type(self)
        n = getattr(cls, "_TRACE_N", 0)
        if n >= 30:
            return
        cls._TRACE_N = n + 1
        parts = [f"[V2TRACE.{cls._TRACE_N}.{label}]"]
        for k, v in kw.items():
            if isinstance(v, torch.Tensor):
                t = v.detach().cpu()
                if t.numel() <= 16:
                    parts.append(f"{k}={t.tolist()}")
                else:
                    parts.append(
                        f"{k}.shape={list(t.shape)} "
                        f"first8={t.flatten()[:8].tolist()}"
                    )
            else:
                parts.append(f"{k}={v}")
        logger.info(" ".join(parts))

    # ------------------------------------------------------------
    #  ABC / interface
    # ------------------------------------------------------------

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """Verify forward emits draft_token_num tokens per req."""
        return self.draft_token_num, self.draft_token_num

    def release_resources(self) -> None:
        """Return every carried cross-attention slot to the worker pool."""
        if self.kv_pool is not None:
            for loc in self.cross_loc:
                if loc is not None and torch.is_tensor(loc) and loc.numel() > 0:
                    self.kv_pool.free(loc)
        self.cross_loc = []
        self.cross_count = []
        self.cross_position = []
        self.ttt_k = []
        self.ttt_v = []

    @classmethod
    def create_idle_input(
        cls, draft_token_num: int, device: str = "cuda"
    ) -> "SpecBlockVerifyInput":
        """Empty verify input for CUDA-graph capture."""
        return cls(
            draft_token=torch.empty((0,), dtype=torch.int64, device=device),
            custom_mask=torch.empty((0,), dtype=torch.bool, device=device),
            positions=torch.empty((0,), dtype=torch.int64, device=device),
            retrive_index=torch.full(
                (0, draft_token_num), -1, dtype=torch.int64, device=device
            ),
            retrive_next_token=torch.full(
                (0, draft_token_num), -1, dtype=torch.int64, device=device
            ),
            retrive_next_sibling=torch.full(
                (0, draft_token_num), -1, dtype=torch.int64, device=device
            ),
            draft_token_num=draft_token_num,
            num_tokens_per_batch=draft_token_num,
            tree_parents=torch.empty(
                (0, draft_token_num), dtype=torch.int64, device=device
            ),
            tree_depth=torch.empty((0,), dtype=torch.int64, device=device),
            tree_lps=torch.empty((0,), dtype=torch.float32, device=device),
            tree_sizes_cpu=[],
            hidden_states=None,
            verified_id=None,
            accept_length=None,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

    # ------------------------------------------------------------
    #  __deepcopy__ : skip the GPU pool reference (mirrors DraftInput)
    # ------------------------------------------------------------

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for fname in self.__dataclass_fields__:
            if fname == "kv_pool":
                setattr(new, fname, self.kv_pool)
                continue
            value = getattr(self, fname)
            try:
                setattr(new, fname, deepcopy(value, memo))
            except Exception:
                setattr(new, fname, value)
        new.spec_input_type = getattr(self, "spec_input_type", None)
        new.device = getattr(self, "device", None)
        return new

    # ------------------------------------------------------------
    #  Filter / merge
    # ------------------------------------------------------------

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        """The verify input is rebuilt every iteration, but the scheduler
        may still call filter on it after a request finishes mid-verify.
        We filter the (B, N_max) and per-req-block tensors; flat 1D
        tensors are tricky because they pack variable-length trees, so
        the worker rebuilds them next iter rather than re-packing here.

        We DO filter the carried per-req SpecBlock state (cross_*, ttt_*,
        cross_count, cross_position) -- those are the long-lived state.
        """
        # Same gather-always policy as DraftInput: SpecBlock-Shift's
        # _refresh_draft_state does NOT pre-filter to unfinished_index
        # like EAGLE does, so the [:B] fast-path silently misaligns
        # state when keep_indices is sparse (e.g. [0, 2, 3]).  Force
        # gather indexing in both branches.
        if torch.is_tensor(new_indices):
            keep_tensor = new_indices
            keep_idx = new_indices.tolist()
            B = int(new_indices.numel())
        else:
            keep_idx = list(new_indices)
            keep_tensor = torch.tensor(keep_idx, dtype=torch.long, device=self.device)
            B = len(keep_idx)
        del has_been_filtered  # always gather

        for attr in (
            "retrive_index",
            "retrive_next_token",
            "retrive_next_sibling",
            "tree_parents",
        ):
            t = getattr(self, attr)
            if t is not None and t.shape[0] >= B:
                setattr(self, attr, t[keep_tensor])
        if self.hidden_states is not None and self.hidden_states.shape[0] >= B:
            self.hidden_states = self.hidden_states[keep_tensor]

        # Carried SpecBlock state (per-req lists) -- gather lockstep.
        if self.cross_loc:
            keep_set = set(keep_idx)
            old_n = len(self.cross_loc)
            if self.kv_pool is not None:
                for old_idx in range(old_n):
                    if old_idx in keep_set:
                        continue
                    loc = self.cross_loc[old_idx]
                    if loc is not None and torch.is_tensor(loc) and loc.numel() > 0:
                        self.kv_pool.free(loc)
        for attr in (
            "cross_loc", "cross_count", "cross_position",
            "ttt_k", "ttt_v",
        ):
            lst = getattr(self, attr)
            if lst:
                setattr(self, attr, [lst[i] for i in keep_idx])

        if self.tree_sizes_cpu is not None:
            self.tree_sizes_cpu = [self.tree_sizes_cpu[i] for i in keep_idx]
        # NOTE: draft_token / positions / custom_mask / tree_depth / tree_lps
        # are flattened across requests with variable-length per-req sizes.
        # Filtering them in place would require a costly per-req rescan.
        # In practice the scheduler only calls filter_batch *after* verify
        # has consumed these flat fields, so we leave them stale.

    def merge_batch(self, other: "SpecBlockVerifyInput"):
        """Merge another batch's verify state into self.

        Tree-related flat fields (draft_token, positions, custom_mask,
        retrive_*) are batch-specific and rebuilt at next draft(), so
        we don't merge them. We DO merge hidden_states (continuity) and
        the carried per-req SpecBlock state (cross_*, ttt_*).
        """
        if other is None:
            return
        if self.hidden_states is None:
            self.hidden_states = other.hidden_states
        elif other.hidden_states is not None:
            self.hidden_states = torch.cat(
                [self.hidden_states, other.hidden_states], dim=0
            )
        # Carried SpecBlock state (per-req lists) -- extend lockstep.
        if self.kv_pool is None and other.kv_pool is not None:
            self.kv_pool = other.kv_pool
        for attr in (
            "cross_loc", "cross_count", "cross_position",
            "ttt_k", "ttt_v",
        ):
            mine = getattr(self, attr)
            theirs = getattr(other, attr)
            if not mine:
                setattr(self, attr, theirs if theirs else [])
            elif theirs:
                setattr(self, attr, mine + theirs)

    # ------------------------------------------------------------
    #  prepare_for_verify (KV alloc + req_to_token mapping)
    # ------------------------------------------------------------

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        """Allocate KV cache slots for verify forward.

        Identical structure to Medusa's implementation (medusa_info.py:217)
        and EAGLE's prepare_for_verify -- the math doesn't depend on the
        algorithm, just on the tree's draft_token_num.

        After this:
            batch.input_ids   = self.draft_token (1D)
            batch.out_cache_loc = freshly allocated slots
            batch.req_to_token mapping updated for verify forward
        """
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                len(batch.input_ids),
            )
            end_offset = batch.seq_lens + self.draft_token_num
            for req in batch.reqs:
                req.kv_allocated_len += 1
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            self.last_loc = last_loc

        # Map draft tokens into req_to_token pool so verify forward sees them.
        bs = batch.batch_size()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        raise RuntimeError(
            "SpecBlock Shift target verification requires native indexed tree "
            "attention; legacy packed-mask prefill arguments are unsupported."
        )

    # ------------------------------------------------------------
    #  Verify (greedy + sampling)
    # ------------------------------------------------------------

    def _greedy_verify(self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput):
        """Greedy tree accept via sgl_kernel verify_tree_greedy.

        We re-use the EAGLE3 acceptance kernel verbatim.  Inputs are the
        per-batch retrive_*  pointers already constructed at draft time.
        """
        bs = batch.batch_size()

        expected_logits_len = bs * self.draft_token_num
        actual_logits_len = logits_output.next_token_logits.shape[0]
        if actual_logits_len != expected_logits_len:
            raise RuntimeError(
                f"[SpecBlockVerifyInput] logits shape mismatch: "
                f"expected {expected_logits_len} (bs={bs} * "
                f"draft_token_num={self.draft_token_num}), got "
                f"{actual_logits_len}. Did num_draft_tokens drift?"
            )

        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, self.draft_token_num)
        candidates = self.draft_token.reshape(bs, self.draft_token_num)

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        # Zero-init: V2 path's accept_index may reference uninit positions
        # (V1's _fill_requests truncates accept_index at finish points;
        # V2's skip_finalize does not). zero avoids OOB on embed lookup.
        self.predict = torch.empty(predict_shape, dtype=torch.int32, device=self.device)

        # Bonus: reserve one extra accept slot for the bonus token.
        accept_capacity = int(self.tree_max_depth) + 1
        if accept_capacity > self.draft_token_num:
            raise RuntimeError(
                "SpecBlock accept path capacity exceeds the verification tree: "
                f"capacity={accept_capacity}, tree_width={self.draft_token_num}."
            )
        self.accept_index = torch.full(
            (bs, accept_capacity), -1, dtype=torch.int32, device=self.device
        )
        self.accept_length = torch.empty((bs,), dtype=torch.int32, device=self.device)

        # Trace inputs to verify_tree_greedy (compare V1 vs V2 same prompt).
        self._trace(
            "verify_in",
            bs=bs, draft_token_num=self.draft_token_num,
            candidates=candidates,
            target_predict=target_predict,
            root_candidates=candidates[:, 0],
            root_target_predict=target_predict[:, 0],
            retrive_index=self.retrive_index,
            retrive_next_token=self.retrive_next_token,
            retrive_next_sibling=self.retrive_next_sibling,
            seq_lens=batch.seq_lens,
        )

        verify_tree_greedy_func(
            predicts=self.predict,
            accept_index=self.accept_index,
            accept_token_num=self.accept_length,
            candidates=candidates,
            retrive_index=self.retrive_index,
            retrive_next_token=self.retrive_next_token,
            retrive_next_sibling=self.retrive_next_sibling,
            target_predict=target_predict,
            topk=1,
        )

        # Trace outputs.
        self._trace(
            "verify_out",
            accept_index=self.accept_index,
            accept_index_head=self.accept_index[:, :4],
            accept_length=self.accept_length,
            predict=self.predict,
        )

    def _sampling_verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ):
        """Sampling tree accept via tree_speculative_sampling_target_only.

        SpecBlock-Shift's draft model does not emit per-token probabilities
        tied to the verify tree (we use rank-class branching, not topk),
        so we set draft_probs = 0 to fall back to target-only acceptance
        (= reject only when target prob below threshold).  EAGLE uses the
        same trick when running with target_only.
        """
        bs = batch.batch_size()
        candidates = self.draft_token.reshape(bs, self.draft_token_num)

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        # See _greedy_verify above for why zero-init.
        self.predict = torch.empty(predict_shape, dtype=torch.int32, device=self.device)

        accept_capacity = int(self.tree_max_depth) + 1
        if accept_capacity > self.draft_token_num:
            raise RuntimeError(
                "SpecBlock accept path capacity exceeds the verification tree: "
                f"capacity={accept_capacity}, tree_width={self.draft_token_num}."
            )
        self.accept_index = torch.full(
            (bs, accept_capacity), -1, dtype=torch.int32, device=self.device
        )
        self.accept_length = torch.empty((bs,), dtype=torch.int32, device=self.device)

        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, self.draft_token_num, dim=0
        )
        target_probs = F.softmax(
            logits_output.next_token_logits / expanded_temperature, dim=-1
        )
        if sampling_info.need_top_k_sampling:
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )
        if sampling_info.need_top_p_sampling:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
        target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

        # Target-only acceptance -> draft_probs=0 (same trick as Medusa).
        draft_probs = torch.zeros(
            target_probs.shape, dtype=torch.float32, device=self.device
        )
        coins = torch.rand_like(candidates, dtype=torch.float32, device=self.device)
        coins_for_final = torch.rand((bs,), dtype=torch.float32, device=self.device)

        tree_speculative_sampling_target_only(
            predicts=self.predict,
            accept_index=self.accept_index,
            accept_token_num=self.accept_length,
            candidates=candidates,
            retrive_index=self.retrive_index,
            retrive_next_token=self.retrive_next_token,
            retrive_next_sibling=self.retrive_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=get_global_server_args().speculative_accept_threshold_single,
            threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
            deterministic=True,
        )

    def _fill_requests(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        skip_finalize: bool = False,
    ) -> Optional[torch.Tensor]:
        """Append accepted tokens and compact the verified tree path.

        V1 needs the accepted IDs on the host to run per-token stop and grammar
        handling. Copy only the compact accepted paths, rather than copying the
        full tree-shaped ``accept_index`` and ``predict`` tensors. The returned
        CPU tensor contains bonus-inclusive path lengths and is also reused by
        ``verify`` for cache accounting, avoiding a second length D2H copy.

        ``skip_finalize`` (V2 path) leaves Python-side request finalization to
        the scheduler. It intentionally performs no D2H transfer.
        """
        # Keep the 2-D form until V1 stop handling has decided whether an
        # accepted path needs truncation. These indices also preserve the
        # non-contiguous tree path selected by target acceptance.
        compact_accept_index = self.accept_index[self.accept_index != -1]
        compact_verified_id = self.predict[compact_accept_index]
        accept_lens_cpu = None

        if not skip_finalize:
            # accept_length excludes the bonus token; every accepted path has
            # exactly one bonus token. These are the only V1 D2H payloads.
            accept_lens_cpu = (self.accept_length + 1).to(
                "cpu", non_blocking=True
            )
            compact_verified_id_cpu = compact_verified_id.to(
                "cpu", non_blocking=True
            )
            path_lens = accept_lens_cpu.tolist()
            verified_ids = compact_verified_id_cpu.tolist()

            has_finished = False
            final_path_lens = []
            offset = 0
            for req, path_len in zip(batch.reqs, path_lens):
                # Count emitted tokens independently of req.output_ids, whose
                # prefix belongs to previous decode iterations.
                accepted_this_req = 0
                for token_id in verified_ids[offset : offset + path_len]:
                    accepted_this_req += 1
                    req.output_ids.append(token_id)
                    req.check_finished()
                    if req.finished():
                        has_finished = True
                        break
                    if req.grammar is not None:
                        req.grammar.accept_token(token_id)

                req.spec_verify_ct += 1
                # Preserve V1 accounting: this records target-tree acceptance
                # before request-local stopping truncates the committed path.
                req.spec_accepted_tokens += path_len - 1
                final_path_lens.append(accepted_this_req)
                offset += path_len

            if has_finished:
                # Rebuild the bonus-inclusive accepted length vector from the
                # CPU decisions, then truncate the GPU paths in one vectorized
                # operation. This keeps KV freeing, hidden/logit compaction,
                # and scheduler lengths aligned with early stop/EOS handling.
                final_path_lens_cpu = torch.tensor(
                    final_path_lens, dtype=self.accept_length.dtype
                )
                final_path_lens_gpu = final_path_lens_cpu.to(
                    self.device, non_blocking=True
                )
                positions = torch.arange(
                    self.accept_index.shape[1], device=self.device
                )
                self.accept_index.masked_fill_(
                    positions.unsqueeze(0) >= final_path_lens_gpu.unsqueeze(1),
                    -1,
                )
                self.accept_length = final_path_lens_gpu - 1
                accept_lens_cpu = final_path_lens_cpu

                # The compact path may have shrunk, so gather it again before
                # using it for logits, hidden states, and KV bookkeeping.
                compact_accept_index = self.accept_index[
                    self.accept_index != -1
                ]
                compact_verified_id = self.predict[compact_accept_index]

        self.accept_index = compact_accept_index
        logits_output.next_token_logits = logits_output.next_token_logits[
            self.accept_index
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                self.accept_index
            ]
        self.verified_id = compact_verified_id
        return accept_lens_cpu

    def _free_cache(
        self, batch: ScheduleBatch, page_size: int, accept_length_cpu: torch.Tensor
    ):
        """Release KV slots that were allocated for rejected draft tokens."""
        bs = batch.batch_size()

        if page_size == 1:
            evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
            evict_mask[self.accept_index] = False
            batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            batch.out_cache_loc = batch.out_cache_loc[self.accept_index]
        else:
            src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                batch.seq_lens,
                batch.out_cache_loc,
                self.accept_index,
                self.accept_length,
                self.draft_token_num,
                page_size,
            )
            to_free_slots = torch.empty(
                (to_free_num_slots.sum().item(),),
                dtype=torch.int32,
                device=self.device,
            )
            get_target_cache_loc[(bs,)](
                tgt_cache_loc,
                to_free_slots,
                self.accept_length,
                to_free_num_slots,
                batch.out_cache_loc,
                self.draft_token_num,
                next_power_of_2(self.draft_token_num),
                next_power_of_2(bs),
            )
            batch.token_to_kv_pool_allocator.free(to_free_slots)
            batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                tgt_cache_loc, src_cache_loc
            )
            batch.out_cache_loc = tgt_cache_loc

        accept_length_list = accept_length_cpu.tolist()
        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_length_list[i] + 1
            req.kv_allocated_len = req.kv_committed_len

        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.accept_length + 1,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,
        *,
        skip_free_cache: bool = False,
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, torch.Tensor]:
        """End-to-end verify pipeline.

        1. Apply logit bias / penalty / grammar (same as EAGLE / Medusa).
        2. Greedy or sampling tree-accept.
        3. _fill_requests: append accepted tokens, flatten accept_index.
        4. _free_cache: release rejected KV.
        5. Update batch.seq_lens.

        Returns ``(logits_output, verified_id, accept_lens_cpu)``.

        ``accept_lens_cpu`` is a [bs] int32 CPU tensor of bonus-inclusive
        accepted path lengths. V1 creates it while compact accepted IDs are
        finalized; V2 copies it after GPU-only path compaction.

        V1's host finalization synchronizes only the accepted-path IDs and
        per-request lengths; KV and scheduler accounting reuse that length
        tensor.
        """
        del vocab_mask  # SpecBlock-Shift currently has no grammar mask path
        bs = batch.batch_size()
        sampling_info = batch.sampling_info

        if bs != len(sampling_info):
            sampling_info = deepcopy(sampling_info)
            indices = list(range(bs))
            sampling_info.filter_batch(indices, indices)

        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        if (
            (
                sampling_info.penalizer_orchestrator is not None
                and sampling_info.penalizer_orchestrator.is_required
            )
            or sampling_info.logit_bias is not None
        ):
            linear_penalty = torch.zeros(
                (bs, logits_output.next_token_logits.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            sampling_info.apply_logits_bias(linear_penalty)
            logits_output.next_token_logits.add_(
                torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
            )

        is_all_greedy = sampling_info.is_all_greedy
        if not is_all_greedy and (not TREE_SPEC_KERNEL_AVAILABLE or _is_npu):
            raise RuntimeError(
                "SPECBLOCK_SHIFT sampling requires the tree speculative "
                "sampling kernel on CUDA."
            )

        if is_all_greedy:
            self._greedy_verify(batch, logits_output)
        else:
            self._sampling_verify(batch, logits_output, sampling_info)

        # V2 path: skip Python-side req.output_ids / check_finished updates.
        # V2 scheduler does them in process_batch_result_decode.
        accept_lens_cpu = self._fill_requests(
            batch, logits_output, skip_finalize=skip_free_cache
        )
        # V2 path skips _free_cache: the V2 plan-stream over-allocates
        # 2 * ALLOC_LEN_PER_DECODE per req (prepare_for_decode), and the
        # remainder stays in the pool until the req finishes — matches
        # EAGLE V2's pattern (eagle_worker_v2.py:682+ verify never calls
        # _free_cache). V1 path still frees rejected slots inline since
        # V1 prepare_for_verify allocates exactly bs*draft_token_num.
        if skip_free_cache:
            # V2 scheduler consumes bonus-inclusive lengths directly.
            accept_lens_cpu = (self.accept_length + 1).to(
                "cpu", non_blocking=True
            )
        else:
            # _fill_requests has already copied the final bonus-inclusive
            # lengths alongside compact verified IDs. Reuse that transfer for
            # V1 KV accounting rather than making another D2H copy.
            assert accept_lens_cpu is not None
            accept_count_cpu = accept_lens_cpu - 1
            self._free_cache(batch, page_size, accept_count_cpu)

        batch.seq_lens.add_(self.accept_length + 1)
        batch.seq_lens_cpu.add_(accept_lens_cpu)

        return logits_output, self.verified_id, accept_lens_cpu
