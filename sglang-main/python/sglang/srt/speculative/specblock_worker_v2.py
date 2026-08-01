"""SpecBlock-Shift native V2 worker — mirrors EAGLEWorkerV2 pattern.

P2.6 (2026-05-09): replaces the P2.5 adapter approach (delegate V1 via
ModelWorkerBatch-as-ScheduleBatch shim) which delivered acc parity but
zero bs=4 throughput gain because V1's monolithic forward_batch_generation
runs sequentially without exposing prepare-vs-forward seams the overlap
scheduler can exploit.

This file mirrors :class:`EAGLEWorkerV2` (eagle_worker_v2.py:575-860):
  - forward_batch_generation dispatches three modes (extend / idle / decode)
    and explicitly orchestrates: target_forward → _draft_extend_for_prefill
    (extend), draft_worker.draft → verify → _draft_extend_for_decode
    (decode).
  - verify uses plan_stream_ctx to prepare verify_forward_batch on the
    plan stream while the previous iter's draft is still running on
    the main stream — the actual sched gap closure mechanism.
  - GenerationBatchResult.next_draft_input is the next-iter spec_info
    handoff (vs V1's batch.spec_info in-place mutation).

Heavy lifting (model loading, attention backend, spec_kv_pool, tree
builder) still comes from :class:`SpecBlockWorker` to avoid
duplicating ~300 LOC of init.  V1 sub-methods (forward_target_extend,
forward_draft_extend, draft, _verify_and_accept, _refresh_draft_state)
are called directly from V2 with MWB-shaped args.

Design notes: ``notes/p25_spec_v2_integration_plan.md`` (now superseded
by the inline plan in this docstring).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import (
    BaseDraftWorker,
    BaseSpecWorker,
)
from sglang.srt.speculative.specblock_info import (
    SpecBlockDraftInput,
    SpecBlockVerifyInput,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context


logger = logging.getLogger(__name__)


def _pack_verified_ids_for_scheduler(
    verified_ids: torch.Tensor,
    accept_lens: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Pack flattened accepted paths into V2's fixed per-request stride."""
    if verified_ids.ndim != 1 or accept_lens.ndim != 1:
        raise RuntimeError(
            "SpecBlock V2 scheduler packing expects flat verified IDs and "
            "one accept length per request."
        )
    if stride <= 0:
        raise RuntimeError(f"Invalid SpecBlock V2 scheduler stride {stride}.")

    accept_lens = accept_lens.to(device=verified_ids.device, dtype=torch.int64)
    source = torch.arange(
        verified_ids.numel(), device=verified_ids.device, dtype=torch.int64
    )
    ends = torch.cumsum(accept_lens, dim=0)
    rows = torch.searchsorted(ends, source, right=True)
    starts = ends - accept_lens
    destinations = rows * stride + source - starts[rows]
    packed = torch.zeros(
        accept_lens.numel() * stride,
        device=verified_ids.device,
        dtype=torch.int64,
    )
    packed.scatter_(0, destinations, verified_ids.to(torch.int64))
    return packed


def _get_plan_stream(device: str) -> Tuple[Optional[torch.cuda.Stream], object]:
    """Create the optional verification-plan stream used by Spec V2.

    Match EAGLE V2's feature gate: when overlap planning is disabled, the
    preparation code still runs in-order under a null context.  Keeping this
    branch explicit makes the no-overlap path a correctness baseline.
    """
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        device_module = torch.get_device_module(device)
        stream = device_module.Stream()
        return stream, device_module.stream(stream)
    from contextlib import nullcontext

    return None, nullcontext()


# ============================================================
#  ScheduleBatch shim over ModelWorkerBatch
#
#  V1 sub-methods (forward_draft_extend, draft, _verify_and_accept,
#  _refresh_draft_state) accept ScheduleBatch.  We pass a minimal shim
#  to call them with V2's ModelWorkerBatch.  Same shape as P2.5 adapter
#  but only used inside individual sub-method calls — V2 outer worker
#  drives orchestration directly.
# ============================================================


class _MWBAsScheduleBatch:
    __slots__ = (
        "_mwb", "return_hidden_states",
        # ScheduleBatch attrs forwarded from worker (not on MWB):
        "token_to_kv_pool_allocator", "req_to_token_pool",
    )

    def __init__(
        self,
        mwb: ModelWorkerBatch,
        token_to_kv_pool_allocator=None,
        req_to_token_pool=None,
    ):
        object.__setattr__(self, "_mwb", mwb)
        object.__setattr__(self, "return_hidden_states", False)
        # V2 worker passes these so V1's _verify_and_accept / _free_cache
        # can release rejected KV slots without going through ScheduleBatch.
        object.__setattr__(self, "token_to_kv_pool_allocator", token_to_kv_pool_allocator)
        object.__setattr__(self, "req_to_token_pool", req_to_token_pool)

    def batch_size(self) -> int:
        return len(self._mwb.seq_lens)

    def get_model_worker_batch(self, seq_lens_cpu_cache=None):
        if seq_lens_cpu_cache is not None:
            self._mwb.seq_lens_cpu = seq_lens_cpu_cache
        return self._mwb

    def __getattr__(self, name: str):
        if name == "extend_lens":
            return self._mwb.extend_seq_lens
        return getattr(self._mwb, name)

    def __setattr__(self, name: str, value):
        if name in (
            "return_hidden_states",
            "token_to_kv_pool_allocator",
            "req_to_token_pool",
        ):
            object.__setattr__(self, name, value)
            return
        if name == "extend_lens":
            self._mwb.extend_seq_lens = value
            return
        setattr(self._mwb, name, value)


# ============================================================
#  Draft worker — owns V1 worker for setup, exposes V2 phase methods
# ============================================================


class SpecBlockDraftWorker(BaseDraftWorker):
    """V2 draft worker with explicit phase methods.

    Owns a :class:`SpecBlockWorker` instance for model + attention
    backend + spec_kv_pool setup but does not delegate
    ``forward_batch_generation`` — V2 outer worker calls
    :meth:`_draft_extend_for_prefill` / :meth:`draft` /
    :meth:`_draft_extend_for_decode` separately so the overlap scheduler
    can interleave with the main forward stream.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        from sglang.srt.speculative.specblock_worker import (
            SpecBlockWorker,
        )

        # V1 worker handles model loading, attention backend, spec_kv_pool,
        # tree builder, etc.  We do NOT call its forward_batch_generation;
        # instead we reach into its sub-methods (forward_draft_extend,
        # draft, _verify_and_accept, _refresh_draft_state) directly.
        self._v1 = SpecBlockWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )

        self.server_args = server_args
        self.device = server_args.device
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.target_worker = target_worker

        # Pass-through attrs used by parent V2 worker / scheduler.
        self.draft_runner = self._v1.model_runner
        self.draft_tp_context = self._v1.draft_tp_context
        self.K = self._v1.K
        self.num_layers = self._v1.num_layers
        self.spec_kv_pool = self._v1.spec_kv_pool
        # Cuda graph runner for draft extend stays None — Phase 3 work.
        self.cuda_graph_runner_for_draft_extend = None

    # --------------------------------------------------------
    #  BaseDraftWorker interface
    # --------------------------------------------------------

    def draft(self, model_worker_batch: ModelWorkerBatch) -> SpecBlockVerifyInput:
        """Build a verify input by running draft trees.  Wraps V1.draft
        which expects a ScheduleBatch-shaped input.
        """
        sb = _MWBAsScheduleBatch(model_worker_batch)
        return self._v1.draft(sb)

    def draft_extend(self):
        """No-op — V2 routes extend through explicit
        _draft_extend_for_prefill / _draft_extend_for_decode below.
        """
        pass

    # --------------------------------------------------------
    #  V2 phase methods
    # --------------------------------------------------------

    def _draft_extend_for_prefill(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> SpecBlockDraftInput:
        """Prefill phase: build initial SpecBlockDraftInput from target
        prefill outputs.  V1's forward_draft_extend reads
        logits_output.hidden_states + next_token_ids from the target run,
        so we shim a thin LogitsOutputLike wrapper here.

        Mirrors EagleDraftWorker._draft_extend_for_prefill (eagle_worker
        _v2.py:455-501) interface contract.
        """

        # V1 forward_draft_extend expects logits_output.hidden_states.
        class _LogitsOutputShim:
            __slots__ = ("hidden_states",)

            def __init__(self, hs):
                self.hidden_states = hs

        sb = _MWBAsScheduleBatch(batch)
        # V1 reads batch.seq_lens / batch.input_ids / batch.extend_lens
        # which all come through the shim; seq_lens_cpu defaults to None
        # which V1 doesn't actually need.
        self._v1.forward_draft_extend(
            sb,
            _LogitsOutputShim(target_hidden_states),
            next_token_ids,
            seq_lens_cpu=batch.seq_lens_cpu,
        )
        # V1 writes the new spec_info onto sb (shim writes through to MWB).
        return batch.spec_info

    def _draft_extend_for_decode(
        self,
        batch: ModelWorkerBatch,
        batch_result: GenerationBatchResult,
    ) -> None:
        """Decode-step refresh phase: take verified hidden states, run
        draft model to produce new b0_*, ttt_kv, cross_loc state for the
        next iter.  V1's _refresh_draft_state does this; we wrap it.

        Sets batch_result.next_draft_input to the refreshed
        SpecBlockDraftInput so the scheduler can stash it for next iter.
        """
        sb = _MWBAsScheduleBatch(batch)
        # V1._refresh_draft_state needs:
        #  - batch (ScheduleBatch shim)
        #  - verify_info: SpecBlockVerifyInput (the spec_info from draft())
        #  - logits_output: target verify forward output (hidden_states cropped)
        #  - verified_id: accepted token ids flat
        #  - accept_length_cpu: List[int]
        # GenerationBatchResult carries logits_output, next_token_ids
        # (verified_ids), accept_length_per_req_cpu.  verify_info we pull
        # from the *current* batch.spec_info (set by draft).
        verify_info = batch.spec_info
        # IMPORTANT: V1's _refresh_draft_state expects ``verified_id`` in
        # the filtered shape (sum_i (accept_i + 1)) — not the unfiltered
        # bs*stride predict tensor that the V2 scheduler consumes via
        # ``batch_result.next_token_ids``.  Pull the filtered tensor from
        # the outer V2 worker's stash (set in verify()).
        outer = getattr(self, "_outer_worker_v2", None)
        verified_id_filtered = (
            outer._last_verified_ids_filtered
            if outer is not None and outer._last_verified_ids_filtered is not None
            else batch_result.next_token_ids
        )
        self._v1._refresh_draft_state(
            sb,
            verify_info,
            batch_result.logits_output,
            verified_id_filtered,
            batch_result.accept_length_per_req_cpu,
            accept_lengths_gpu=verify_info.accept_length,
        )
        # V1 sets sb.spec_info = new_spec_info (shim writes through).
        next_draft_input = batch.spec_info
        # Record verify_done event AFTER _refresh_draft_state populates
        # next_draft_input.b0_* — so plan-stream's maybe_wait_verify_done()
        # blocks until fresh b0_* are visible.  Mirror EAGLE V2 ordering.
        outer = getattr(self, "_outer_worker_v2", None)
        if outer is not None and outer._verify_done is not None:
            outer._verify_done.record()
            next_draft_input.verify_done = outer._verify_done
        batch_result.next_draft_input = next_draft_input


# ============================================================
#  Outer V2 worker — orchestrates target_forward / draft / verify /
#  draft_extend_for_decode with explicit plan_stream prepare overlap.
# ============================================================


class SpecBlockWorkerV2(BaseSpecWorker):
    """SpecBlock-Shift V2 worker.

    Mirrors :class:`EAGLEWorkerV2` (eagle_worker_v2.py:575).  Native V2
    contract: accepts ModelWorkerBatch directly, returns
    GenerationBatchResult with next_draft_input populated.  plan_stream
    used to prepare verify forward batch concurrently with the previous
    iter's draft on the main stream.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # SpecBlock-Shift reuses the EAGLE-style topk argument as its slot-0
        # beam width. It is not EAGLE's per-step tree semantics, but the V1
        # worker and GPU tree builder consume it, so V2 must preserve the value.

        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.device = server_args.device
        self._target_worker = target_worker
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        # Mirror EAGLEWorkerV2:99-102 — scheduler reads these on V2 path
        # for accept-length / cache-loc carve-out math.
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = (
            server_args.speculative_num_draft_tokens
        )

        # Set ALLOC_LEN_PER_DECODE before constructing draft worker (so
        # prepare_for_decode allocates enough slots for verify forward).
        # Mirrors EAGLEWorkerV2:108.  For SpecBlock-Shift, num_draft_tokens
        # is the tree budget (default 90) and dominates over num_steps*topk.
        SpecBlockDraftInput.ALLOC_LEN_PER_DECODE = max(
            server_args.speculative_num_steps
            * server_args.speculative_eagle_topk,
            server_args.speculative_num_draft_tokens,
        )

        # Share the allocator with target worker (mirrors EAGLEWorkerV2:119).
        # req_to_token_pool is needed for V2 verify's out_cache_loc carve-out
        # via assign_extend_cache_locs_func.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self._draft_worker = SpecBlockDraftWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        # Back-ref so draft_worker._draft_extend_for_decode can read
        # outer worker's _verify_done event after verify.
        self._draft_worker._outer_worker_v2 = self

        # Stashes filtered verified_ids tensor from each verify call so
        # _draft_extend_for_decode can pass it to _refresh_draft_state
        # (which needs filtered, not the V2 unfiltered next_token_ids).
        self._last_verified_ids_filtered: Optional[torch.Tensor] = None

        # plan_stream for prepare-stage overlap (mirror EAGLEWorkerV2:171).
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)
        logger.info(
            "SpecBlockWorkerV2 initialized (plan_stream_enabled=%s, "
            "steps=%d, beam_width=%d, verify_tokens=%d)",
            self.plan_stream is not None,
            self.speculative_num_steps,
            self.topk,
            self.speculative_num_draft_tokens,
        )

    # --------------------------------------------------------
    #  BaseSpecWorker interface
    # --------------------------------------------------------

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> SpecBlockDraftWorker:
        return self._draft_worker

    def clear_cache_pool(self):
        self._draft_worker._v1.clear_cache_pool()

    # --------------------------------------------------------
    #  forward_batch_generation orchestration
    # --------------------------------------------------------

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch,
    ) -> GenerationBatchResult:
        """Mirror EAGLEWorkerV2.forward_batch_generation (line 632-680).

        Three paths:
          1. extend (prefill): target prefill → _draft_extend_for_prefill.
          2. idle: passthrough target forward.
          3. decode: draft.draft → verify (plan_stream prepare) →
             _draft_extend_for_decode.
        """
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Path 1: prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self._target_worker.forward_batch_generation(
                model_worker_batch
            )
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            with self._draft_worker.draft_tp_context(
                self._draft_worker.draft_runner.tp_group
            ):
                batch_output.next_draft_input = (
                    self._draft_worker._draft_extend_for_prefill(
                        model_worker_batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                    )
                )
            return batch_output

        # Path 2: a scheduler padding/dummy idle batch has no real request
        # state.  In particular, a zero-row SpecBlockDraftInput cannot enter
        # the draft CUDA graph (its active batch size must be positive).  Match
        # the target worker's idle semantics and leave draft/verify/refresh
        # untouched; there is no next-draft handoff to construct.
        if model_worker_batch.forward_mode.is_idle():
            return self._target_worker.forward_batch_generation(model_worker_batch)

        # Path 3: decode. Preserve the legacy empty-input construction for a
        # non-idle caller; normal V2 batches receive real state via
        # next_draft_input from prefill or the preceding refresh.
        if model_worker_batch.spec_info is None:
            target_cfg = self._target_worker.model_runner.model.config
            draft_dtype = next(
                self._draft_worker.draft_runner.model.parameters()
            ).dtype
            model_worker_batch.spec_info = (
                SpecBlockDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=target_cfg.hidden_size,
                    target_hidden_concat_dim=3 * target_cfg.hidden_size,
                    K=self._draft_worker.K,
                    num_layers=self._draft_worker.num_layers,
                    dtype=draft_dtype,
                )
            )

        # Phase A: draft → produces verify_input
        with self._draft_worker.draft_tp_context(
            self._draft_worker.draft_runner.tp_group
        ):
            verify_input = self._draft_worker.draft(model_worker_batch)

        assert verify_input.is_verify_input(), (
            "draft must return a SpecBlockVerifyInput"
        )
        model_worker_batch.spec_info = verify_input

        # Phase B: target verify (with plan_stream prepare overlap)
        batch_output = self.verify(model_worker_batch)

        # Phase C: draft_extend_for_decode → fills next_draft_input
        with self._draft_worker.draft_tp_context(
            self._draft_worker.draft_runner.tp_group
        ):
            self._draft_worker._draft_extend_for_decode(
                model_worker_batch, batch_output
            )

        return batch_output

    # --------------------------------------------------------
    #  verify with plan_stream prepare overlap
    # --------------------------------------------------------

    def _prepare_verify_forward_batch(
        self,
        batch: ModelWorkerBatch,
        spec_info: SpecBlockVerifyInput,
    ) -> Tuple[ForwardBatch, bool]:
        """Prepare target verification metadata on the plan stream.

        This is the SpecBlock counterpart of EAGLE's ``prepare_for_v2_verify``.
        It deliberately stops before target execution: the caller orders the
        plan stream against the draft stream, repairs draft-dependent graph
        buffers, and only then launches the target forward on the main stream.
        """
        from sglang.srt.speculative.eagle_info_v2 import (
            assign_extend_cache_locs_func,
        )

        if not batch.forward_mode.is_idle():
            bs = len(batch.req_pool_indices)
            batch.input_ids = spec_info.draft_token
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=self.req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + spec_info.draft_token_num,
                batch_size=bs,
                draft_token_num=spec_info.draft_token_num,
                device=batch.input_ids.device,
            )

        # Dynamic SpecBlock tree widths must be visible before the attention
        # backend creates its target-verify metadata.
        target_attn = self._target_worker.model_runner.attn_backend
        if hasattr(target_attn, "num_draft_tokens"):
            target_attn.num_draft_tokens = spec_info.draft_token_num

        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.return_hidden_states = False
        verify_forward_batch = ForwardBatch.init_new(
            batch, self._target_worker.model_runner
        )

        graph_runner = self._target_worker.model_runner.graph_runner
        can_run_cuda_graph = bool(
            graph_runner and graph_runner.can_run(verify_forward_batch)
        )
        if can_run_cuda_graph:
            graph_runner.replay_prepare(verify_forward_batch)
        elif not batch.forward_mode.is_idle():
            target_attn.init_forward_metadata(verify_forward_batch)

        return verify_forward_batch, can_run_cuda_graph

    def verify(self, batch: ModelWorkerBatch) -> GenerationBatchResult:
        """Run target verify on the draft tree.

        Mirrors EAGLEWorkerV2.verify (line 682-798) overlap pattern but
        uses V1's _verify_and_accept under the hood (which contains the
        SpecBlock-specific tree mask / accept logic).

        plan_stream prepares the verify ScheduleBatch / target forward
        metadata while main stream finishes the prior draft iter.  Save
        is ~5ms on bs=4 — modest but real (bigger gains require P3 cuda
        graph capture).
        """
        spec_info: SpecBlockVerifyInput = batch.spec_info  # type: ignore[assignment]
        if not isinstance(spec_info, SpecBlockVerifyInput):
            raise RuntimeError(
                "SpecBlock V2 verification requires SpecBlockVerifyInput; "
                f"got {type(spec_info).__name__}."
            )

        # The draft tree is still executing on the main stream when Python
        # reaches this point.  Build cache locations, ForwardBatch, and target
        # attention metadata on plan_stream so that work overlaps the tail of
        # tree construction rather than serializing behind it.
        if self.plan_stream is not None:
            batch.seq_lens.record_stream(
                torch.get_device_module(self.device).current_stream()
            )
        with self.plan_stream_ctx:
            verify_forward_batch, prepared_can_run_cuda_graph = (
                self._prepare_verify_forward_batch(batch, spec_info)
            )

        # ``init_forward_metadata`` / graph replay preparation can read the
        # tree topology before the draft stream has finished.  The EAGLE V2
        # correction hook refreshes exactly those draft-dependent buffers after
        # the GPU-only stream join, without repeating the whole plan stage.
        if self.plan_stream is not None:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
            graph_runner = self._target_worker.model_runner.graph_runner
            target_attn = self._target_worker.model_runner.attn_backend
            target_attn.update_verify_buffers_to_fill_after_draft(
                spec_info,
                graph_runner.bs if prepared_can_run_cuda_graph else None,
            )

        sb = _MWBAsScheduleBatch(
            batch,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
        )
        (
            logits_output,
            verified_ids,
            num_accepted,
            accept_length_cpu,
            can_run_cuda_graph,
        ) = self._draft_worker._v1._verify_and_accept(
            sb,
            skip_prepare=True,
            skip_free_cache=True,
            prepared_forward_batch=verify_forward_batch,
            prepared_can_run_cuda_graph=prepared_can_run_cuda_graph,
        )
        # Stash the filtered accepted chains so _draft_extend_for_decode can
        # feed V1's refresh path. The same tensor is packed below for the V2
        # scheduler; raw tree-node-indexed predict entries are not contiguous.
        self._last_verified_ids_filtered = verified_ids
        # NOTE: verify_done event must be recorded AFTER
        # _refresh_draft_state writes new b0_* (so plan-stream sees fresh
        # values). Done in _draft_extend_for_decode below.
        # Pre-create the event here so _draft_extend_for_decode can record.
        self._verify_done = (
            torch.get_device_module(self.device).Event()
            if self.plan_stream is not None
            else None
        )

        # V2 scheduler reads result.accept_lens (Tensor on cpu) to derive
        # accept_length_per_req_cpu and num_accepted_tokens.  Stashed on
        # _v1 by _verify_and_accept above.
        accept_lens_t = self._draft_worker._v1._last_accept_lens_cpu

        # The V2 scheduler slices one accepted path from each fixed-width row.
        # ``spec_info.predict`` is tree-node-indexed and therefore contains
        # gaps whenever the accepted path does not follow contiguous BFS nodes;
        # exposing its raw prefix can commit stale tokens or a false EOS. Pack
        # the already-filtered accepted chains into the scheduler contract.
        stride = self.speculative_num_draft_tokens
        accept_lens_gpu = spec_info.accept_length + 1
        predict_flat = _pack_verified_ids_for_scheduler(
            verified_ids,
            accept_lens_gpu,
            stride,
        ).to("cpu", non_blocking=True)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict_flat,
            num_accepted_tokens=num_accepted,
            accept_length_per_req_cpu=accept_length_cpu,
            accept_lens=accept_lens_t,
            can_run_cuda_graph=can_run_cuda_graph,
        )
