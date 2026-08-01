"""SpecBlock-Shift speculative decoding worker (redesigned).

This worker is a complete rewrite of the previous (deleted) port that
relied on EagleDraftInput placeholders + ``self.draft_states[rid]`` dict
and broke under SGLang scheduler's dynamic-batching paths.

State management policy (key correctness fix vs the old port):

    ALL per-request SpecBlock state lives on ``batch.spec_info`` as
    :class:`SpecBlockDraftInput` attributes (cross_k/v, ttt_k/v,
    b0_*).  ``filter_batch`` / ``merge_batch`` propagate them in lockstep
    with batch.reqs, so preempt / chunked prefill / dynamic batching all
    work without per-worker bookkeeping.

Lifecycle (one full speculative iter)::

    forward_mode == EXTEND (prefill or first-step):
        forward_target_extend()
            -> target.forward_batch_generation(capture=FULL)
            -> logits_output.hidden_states is [sum_S, 3H] (3-layer aux concat)
        forward_draft_extend()
            -> per-req: draft_model.prefill_and_draft(hidden_3h, shifted_ids,
                                                     last_hidden, first_token)
            -> batch.spec_info = SpecBlockDraftInput(b0_*, cross_kv,
                                                         ttt_kv, ...)
        return GenerationBatchResult(logits_output, next_token_ids, num_acc=0)

    forward_mode == DECODE  (next-iter draft):
        draft()
            -> per-req: build_tree(...)
            -> concat per-req trees into SpecBlockVerifyInput.
            -> batch.spec_info = SpecBlockVerifyInput
            -> batch.forward_mode = TARGET_VERIFY
        (scheduler then calls forward_batch_generation again)

    forward_mode == TARGET_VERIFY:
        target.forward_batch_generation(is_verify=True)
            -> logits_output (next_token_logits, hidden_states 3H)
        spec_info.verify(...)
            -> greedy / sampling tree accept (re-uses EAGLE kernels).
            -> spec_info.verified_id is now a [sum (accept_len+1)] chain.
        _refresh_draft_state()
            -> per-req: draft_model.update_cache_and_draft(verified_h3,
                                                          verified_ids, cache, pos+1)
            -> next batch.spec_info = SpecBlockDraftInput populated
                                      for iter k+1.
        batch.forward_mode = DECODE
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map
from sglang.srt.speculative.specblock_info import (
    SpecBlockDraftInput,
    SpecBlockVerifyInput,
    parents_to_next_token_and_sibling,
    specblock_tree_max_depth,
)
from sglang.srt.speculative.spec_kv_pool import SpecBlockKVPool
from sglang.srt.speculative.specblock_refresh_tensor_plan import (
    build_specblock_refresh_tensor_plan,
)
from sglang.srt.speculative.spec_flashinfer_cross import SpecBlockFlashInferCross
from sglang.srt.utils import empty_context

logger = logging.getLogger(__name__)


# Profiling toggle (env SPECBLOCK_SGL_PROFILE=1).  Off by default.
_PROFILE = os.environ.get("SPECBLOCK_SGL_PROFILE", "0") == "1"
# Deep sub-phase profiling (env SPECBLOCK_DEEP_PROFILE=1) — finer split
# of draft + refresh into named sub-phases.  Adds torch.cuda.synchronize()
# at each step so it's lossy for prod throughput but useful for finding
# the actual hot sub-phase.
_DEEP_PROFILE = os.environ.get("SPECBLOCK_DEEP_PROFILE", "0") == "1"
_DEEP_PROF_STATE: dict = {"n": 0}

_RETIRED_GRAPH_ENV_VARS = (
    "SPECBLOCK_GRAPH",
    "SPECBLOCK_DRAFT_CUDA_GRAPH",
    "SPECBLOCK_REFRESH_CUDA_GRAPH",
)


def _reject_retired_graph_env() -> None:
    present = [name for name in _RETIRED_GRAPH_ENV_VARS if name in os.environ]
    if present:
        raise RuntimeError(
            "SpecBlock no longer supports the retired graph environment "
            f"variables {present}. The draft graph runner is unconditional; "
            "--disable-cuda-graph controls only SGLang target-side graphs."
        )


class SpecBlockWorker(TpModelWorker):
    """SpecBlock-Shift speculative decoding worker.

    Sub-classes :class:`TpModelWorker` (same as EAGLEWorker v1) so the
    draft model is loaded into a SGLang ModelRunner and the worker
    exposes get_memory_pool / forward_batch_generation / etc.
    """

    # ============================================================
    #  __init__
    # ============================================================

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
        enable_overlap: bool = False,
    ):
        _reject_retired_graph_env()

        # ---- Parse server args ---------------------------------------------
        self.server_args = server_args
        self._target_worker = target_worker
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.page_size = server_args.page_size
        self.enable_overlap = enable_overlap
        self.enable_nan_detection = server_args.enable_nan_detection
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # SpecBlock-Shift hyperparams.  Real values populated post-load
        # from the draft model config; placeholders here.
        self.K: int = 0
        self.num_layers: int = 0
        self.max_blocks: int = 2
        self.beam_width: int = int(server_args.speculative_eagle_topk)
        verify_tokens = int(server_args.speculative_num_draft_tokens)
        if verify_tokens < 2:
            raise ValueError(
                "SpecBlock requires at least 2 root-inclusive verify tokens; "
                f"got {verify_tokens}."
            )
        # Tree finalizers take a non-root budget and prepend the verified root.
        # Keep the resulting width identical to SGLang's generic speculative
        # token count so target CUDA-graph buffers and benchmark labels agree.
        self.total_tokens: int = verify_tokens - 1
        self.rank_classes: int = 4

        # Match target's context length.
        server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )

        # ---- CUDA graph: disabled for SpecBlock-Shift -----------------------
        # The draft model uses SpecBlockAttentionWithCache (flex_attention +
        # external KV tuples) which has not been graph-captured.  Phase 3
        # may revisit with a custom SpecBlockDraftCudaGraphRunner.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # ---- Share KV pool with target --------------------------------------
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # ---- Optional vocab mapping (hot tokens) ----------------------------
        self.hot_token_id = None
        if server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )

        # ---- Load draft model via TpModelWorker -----------------------------
        if server_args.enable_dp_attention:
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()

        with ctx, speculative_moe_backend_context():
            super().__init__(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        # ---- Read SpecBlock hyperparams from loaded draft model ------------
        draft_model = self.model_runner.model
        if hasattr(draft_model, "K"):
            self.K = int(draft_model.K)
        if hasattr(draft_model, "num_layers"):
            self.num_layers = int(draft_model.num_layers)
        if self.K <= 0 or (self.K & (self.K - 1)) != 0:
            raise ValueError(
                "SpecBlock SGLang requires K to be a positive power of two "
                f"for its Triton tree kernels; got K={self.K}."
            )
        if hasattr(draft_model, "config"):
            cfg = draft_model.config
            self.rank_classes = int(getattr(cfg, "rank_classes", 4))
            self.max_blocks = int(getattr(cfg, "num_ttt_blocks", 2))

        # The serving configuration, not the training checkpoint default,
        # controls how many block forwards are used at inference.  This keeps
        # --speculative-num-steps aligned with HF's config-list semantics.
        requested_blocks = int(server_args.speculative_num_steps)
        if requested_blocks > 0:
            self.max_blocks = requested_blocks
        _mb_env = os.environ.get("SPECBLOCK_MAX_BLOCKS")
        if _mb_env is not None:
            self.max_blocks = int(_mb_env)
        if self.max_blocks not in (1, 2):
            raise ValueError(
                "SpecBlock SGLang currently supports max_blocks=1 or 2; "
                f"got {self.max_blocks}."
            )

        # rank_slot_topk + rank_to_factor: read defaults from tree builder.
        # Used by tree-build kernels (capturable region needs them on
        # worker so cuda graph runner can read).
        from sglang.srt.speculative.specblock_tree_builder import (
            _RANK_TO_FACTOR_DEFAULT,
            _RANK_SLOT_TOPK_DEFAULT,
        )
        self.rank_to_factor = _RANK_TO_FACTOR_DEFAULT
        self.rank_slot_topk = _RANK_SLOT_TOPK_DEFAULT

        # ---- Share embedding + lm_head from target -------------------------
        embed, head = self._target_worker.model_runner.model.get_embed_and_head()

        if hasattr(draft_model, "get_hot_token_id"):
            model_hot = draft_model.get_hot_token_id()
            if model_hot is not None and self.hot_token_id is None:
                self.hot_token_id = model_hot.to(embed.device)

        if self.hot_token_id is not None:
            head = head.clone()
            self.hot_token_id = self.hot_token_id.to(head.device)
            head.data = head.data[self.hot_token_id]

        if hasattr(draft_model, "set_embed_and_head"):
            draft_model.set_embed_and_head(embed, head)
        elif hasattr(draft_model, "set_embed"):
            draft_model.set_embed(embed)

        # ---- Register target's 3-layer aux hidden capture ------------------
        # SpecBlock-Shift consumes target hidden at three layers
        # ([2, num_layers // 2, num_layers - 3] by EAGLE3 default) which
        # the target model concatenates into a 3H tensor in
        # logits_output.hidden_states (see llama_eagle3.py:178-183 analog).
        target_model = self._target_worker.model_runner.model
        if hasattr(target_model, "set_eagle3_layers_to_capture"):
            target_model.set_eagle3_layers_to_capture()
            logger.info(
                "SpecBlock-Shift: registered target eagle3 3-layer capture."
            )
        else:
            logger.warning(
                "SpecBlock-Shift: target model lacks set_eagle3_layers_to_capture; "
                "forward_target_extend may fall back to last-layer hidden only."
            )

        # ---- Restore + finalise --------------------------------------------
        self.model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )

        with self.draft_tp_context(self.model_runner.tp_group), \
                speculative_moe_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        # ---- Worker-local paged KV pool for cross-attention --------------
        # Stores GQA-expanded K/V per (layer, slot).  Replaces per-req
        # `[1, n_heads, max_len, D]` dense buffers which over-allocate to
        # context_length=131072 -- big OOM trigger under cuda graph buckets.
        # The pool is mandatory (no fallback): cross_loc is the only
        # cross-cache state path the rest of the worker / tree builder /
        # spec_info know about.
        target_cfg = self._target_worker.model_runner.model.config
        n_heads = target_cfg.num_attention_heads
        head_dim = target_cfg.hidden_size // n_heads
        # Pool starts at SPECBLOCK_KV_POOL_SIZE (default 16K) and grows
        # incrementally on overflow up to SPECBLOCK_KV_POOL_MAX (default
        # 524288).  capture_mode() ctx (graph runner) forbids grow
        # during capture; caller must pre-grow before entering.
        initial_pool_size = int(
            os.environ.get("SPECBLOCK_KV_POOL_SIZE", "16384")
        )
        max_pool_size = int(
            os.environ.get("SPECBLOCK_KV_POOL_MAX", str(1 << 19))  # 524288
        )
        draft_dtype = next(draft_model.parameters()).dtype
        self.spec_kv_pool = SpecBlockKVPool(
            initial_pool_size=initial_pool_size,
            max_pool_size=max_pool_size,
            num_layers=self.num_layers,
            n_heads=n_heads,  # GQA-expanded
            head_dim=head_dim,
            dtype=draft_dtype,
            device=self.device,
        )

        # ---- Optional flashinfer paged cross attention (Stage D) ----
        # When SPECBLOCK_FLASHINFER=1 the attention forward's paged
        # branch routes the cross region through flashinfer's
        # BatchPrefillWithPagedKVCacheWrapper (vLLM/SGLang main path
        # kernel, highly tuned), then merges with the ttt+curr region
        # via online softmax.  Off by default so installations without
        # flashinfer keep working.
        self.flashinfer_cross = None
        if os.environ.get("SPECBLOCK_FLASHINFER", "0") == "1":
            try:
                self.flashinfer_cross = SpecBlockFlashInferCross(
                    n_heads=n_heads,
                    head_dim=head_dim,
                    dtype=draft_dtype,
                    device=self.device,
                )
                # Attach to the pool so the model's attention forward
                # can reach the wrapper through the cache tuple's
                # kv_pool element (cache[0]).
                self.spec_kv_pool.flashinfer_cross = self.flashinfer_cross
            except Exception as e:
                logger.warning(
                    "[SpecBlock] flashinfer cross init failed (%s); "
                    "falling back to Triton paged kernel.", e,
                )
                self.flashinfer_cross = None

        # Optional perf accumulator (env SPECBLOCK_SGL_PROFILE=1).
        if _PROFILE:
            self._prof = {
                "draft": 0.0, "verify_setup": 0.0, "verify_fwd": 0.0,
                "verify_accept": 0.0, "refresh": 0.0, "n": 0,
            }

        # ---- Sole custom draft CUDA graph runner ----
        # Captures build_tree_gpu's GPU-only chain for B=1.  Unsupported
        # request shapes fail explicitly until this same runner is extended.
        from sglang.srt.speculative.specblock_draft_cuda_graph_runner import (
            SpecBlockDraftCudaGraphRunner,
        )
        self.draft_cuda_graph_runner = SpecBlockDraftCudaGraphRunner(self)

        from sglang.srt.speculative.specblock_refresh_cuda_graph_runner import (
            SpecBlockRefreshCudaGraphRunner,
        )
        self.refresh_cuda_graph_runner = SpecBlockRefreshCudaGraphRunner(self)
        self.refresh_cuda_graph_runner.precapture_up_to(
            int(os.environ.get("SPECBLOCK_REFRESH_PRECAPTURE_MAX_CROSS", "0"))
        )

        logger.info(
            f"SpecBlock-Shift worker initialized: "
            f"K={self.K}, num_layers={self.num_layers}, "
            f"max_blocks={self.max_blocks}, beam_width={self.beam_width}, "
            f"verify_tokens={self.total_tokens + 1}, "
            f"non_root_budget={self.total_tokens}, "
            f"rank_classes={self.rank_classes}, "
            f"hot_vocab="
            f"{None if self.hot_token_id is None else len(self.hot_token_id)}"
        )

    # ============================================================
    #  BaseSpecWorker contract (TpModelWorker is the actual base, but
    #  scheduler treats us as a spec worker via duck-typing on these
    #  properties + clear_cache_pool + forward_batch_generation).
    # ============================================================

    @property
    def tree_max_depth(self) -> int:
        """Configured maximum parent walk for every tree this worker emits."""
        return specblock_tree_max_depth(self.K, self.max_blocks)

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        # SpecBlock-Shift collapses the draft + spec roles into a single
        # worker (same as EAGLE v1).  Returning self is consistent with
        # the duck-typed interface expected by scheduler.
        return self

    def clear_cache_pool(self):
        """Release every worker-local cross-attention KV slot."""
        self.spec_kv_pool.reset()

    # ------------------------------------------------------------
    #  Backend init: SpecBlock-Shift uses its own internal attention
    #  + external KV tuples, so SGLang's draft_attn_backend stays None.
    # ------------------------------------------------------------

    def init_attention_backend(self):
        self.draft_attn_backend = None

    def init_cuda_graphs(self):
        # CUDA graph for SpecBlock-Shift draft is Phase 3 work.
        return

    # ============================================================
    #  Paged cache helpers
    # ============================================================

    def _dense_to_paged(
        self, cache_dense: List[List], count: int
    ) -> torch.Tensor:
        """Transfer a per-layer dense ``[k_buf, v_buf, count, max_len]`` cache
        produced by ``prefill_and_draft`` into the worker's paged pool.

        Returns the [count] int64 indices for this req's cross_kv slots.
        The dense ``cache_dense`` buffers can be released by the caller
        once this returns (the paged pool now owns the data).
        """
        if count <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        indices = self.spec_kv_pool.alloc(count)
        for L in range(self.num_layers):
            # cache_dense[L][0] : [1, n_heads, max_len, head_dim]
            k_dense = cache_dense[L][0][0, :, :count, :].permute(1, 0, 2).contiguous()
            v_dense = cache_dense[L][1][0, :, :count, :].permute(1, 0, 2).contiguous()
            self.spec_kv_pool.set_kv(L, indices, k_dense, v_dense)
        return indices

    def _build_paged_cache_for_decode(
        self,
        cross_loc: torch.Tensor,
        new_cross_loc: torch.Tensor,
    ) -> List[List]:
        """Build a per-layer paged cache list that
        :class:`SpecBlockAttentionWithCache` can consume in-place.

        Layout per layer (matching ``_is_paged_cache`` detection)::

            [kv_pool, layer_id, count, cross_loc, new_cross_loc]
        """
        count = int(cross_loc.numel())
        cache = []
        for L in range(self.num_layers):
            cache.append(
                [self.spec_kv_pool, L, count, cross_loc, new_cross_loc]
            )
        return cache

    # ============================================================
    #  Hot path -- prefill (forward_mode == EXTEND)
    # ============================================================

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, torch.Tensor]:
        """Run target prefill with CaptureHiddenMode.FULL.

        ``set_eagle3_layers_to_capture`` was registered on the target
        model in __init__, so the target's LogitsProcessor concatenates
        the 3 aux layers into ``logits_output.hidden_states`` of shape
        [sum_S, 3H].
        """
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self._target_worker.forward_batch_generation(model_worker_batch)
        return (
            batch_result.logits_output,
            batch_result.next_token_ids,
            model_worker_batch.seq_lens_cpu,
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ):
        """Initialize SpecBlockDraftInput from prefill outputs.

        Per-req:
          1. Slice [sum_S, 3H] hidden by extend_lens to get [1, S_i, 3H].
          2. Build ``shifted_ids`` = [prompt[1:], next_token]; this is the
             standard SpecBlock-Shift input shift used by prefill_and_draft.
          3. Call draft_model.prefill_and_draft(hidden, shifted_ids,
             last_hidden, first_token) -> (cache, position, b0_logits,
             b0_rank, b0_hidden, b0_ttt_kv).
          4. Aggregate per-req tensors into a single SpecBlockDraftInput.
        """
        del seq_lens_cpu  # implicit through batch.seq_lens

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "[SpecBlockWorker.forward_draft_extend] target prefill "
                "returned None hidden -- did set_eagle3_layers_to_capture run?"
            )
        H = self._target_worker.model_runner.model.config.hidden_size
        if hidden.shape[-1] != 3 * H:
            raise RuntimeError(
                f"[SpecBlockWorker] expected aux hidden shape (*, 3H={3 * H}); "
                f"got {tuple(hidden.shape)}.  capture_aux_hidden_states="
                f"{getattr(self._target_worker.model_runner.model, 'capture_aux_hidden_states', '?')}"
            )

        bs = batch.batch_size()
        draft_model = self.model_runner.model.inner
        draft_dtype = next(draft_model.parameters()).dtype
        if hidden.dtype != draft_dtype:
            hidden = hidden.to(draft_dtype)

        extend_lens = batch.extend_lens

        # Per-req results we will assemble into the new spec_info.
        cross_loc_list: List[torch.Tensor] = []
        cross_count_list: List[int] = []
        cross_position_list: List[int] = []
        ttt_k_list: List[List[torch.Tensor]] = []
        ttt_v_list: List[List[torch.Tensor]] = []
        b0_logits_list: List[torch.Tensor] = []
        b0_rank_list: List[torch.Tensor] = []
        b0_hidden_list: List[torch.Tensor] = []
        b0_input_ids_list: List[torch.Tensor] = []
        b0_top_indices_list: List[torch.Tensor] = []
        last_hidden_list: List[torch.Tensor] = []

        offset = 0
        for i in range(bs):
            S_i = int(extend_lens[i])
            hidden_i = hidden[offset:offset + S_i].unsqueeze(0).contiguous()  # [1, S_i, 3H]
            prompt_ids_i = (
                batch.input_ids[offset:offset + S_i].unsqueeze(0).contiguous()
            )
            offset += S_i

            last_hidden_i = hidden_i[:, -1:, :]                  # [1, 1, 3H]
            first_token_i = next_token_ids[i:i + 1].view(1, 1)   # [1, 1]
            shifted_ids_i = torch.cat([prompt_ids_i[:, 1:], first_token_i], dim=1)

            # prefill_and_draft self-allocates a dense [1, n_heads, (S+2048)*K, D]
            # cache that lives only for this prefill scope; we transfer it
            # into the worker-local paged pool immediately after.  The dense
            # buffer is then GC'd, freeing the over-allocated tail.
            cache_i, pos_i, b0_lg, b0_rk, b0_h, b0_kv = draft_model.prefill_and_draft(
                hidden_i, shifted_ids_i, last_hidden_i, first_token_i,
            )
            cross_count_per_req = int(cache_i[0][2]) if cache_i else 0
            cross_loc_per_req = self._dense_to_paged(cache_i, cross_count_per_req)
            # Drop the dense reference so PyTorch can free the over-alloc tail.
            cache_i = None
            cross_loc_list.append(cross_loc_per_req)
            cross_count_list.append(cross_count_per_req)

            # b0_kv is per-layer (k, v) tensors (TTT KV at K slots).
            tk_per_layer = [kv[0] for kv in b0_kv]
            tv_per_layer = [kv[1] for kv in b0_kv]
            ttt_k_list.append(tk_per_layer)
            ttt_v_list.append(tv_per_layer)

            b0_logits_list.append(b0_lg)        # [1, K, V_draft]
            b0_rank_list.append(b0_rk)          # [1, K, rank_classes]
            b0_hidden_list.append(b0_h)         # [1, K, H]
            b0_input_ids_list.append(first_token_i)  # [1, 1]
            b0_top_indices_list.append(draft_model._last_rank_top_indices)
            last_hidden_list.append(last_hidden_i.squeeze(0).squeeze(0))  # [3H]
            cross_position_list.append(int(pos_i))

        # Stack per-req tensors along batch dim 0 so the spec_info has
        # standard (B, ...) shapes.
        b0_logits = torch.cat(b0_logits_list, dim=0)             # [B, K, V_draft]
        b0_rank = torch.cat(b0_rank_list, dim=0)                  # [B, K, rank_classes]
        b0_hidden = torch.cat(b0_hidden_list, dim=0)              # [B, K, H]
        b0_input_id = torch.cat(b0_input_ids_list, dim=0)         # [B, 1]  (last verified)
        b0_top_indices = torch.cat(b0_top_indices_list, dim=0)
        last_hidden_3h = torch.stack(last_hidden_list, dim=0)     # [B, 3H]

        # Build accept_length placeholder = 0 (no accept yet at prefill end).
        zero_acc = torch.zeros(bs, dtype=torch.int32, device=self.device)

        spec_info = SpecBlockDraftInput(
            hidden_states=last_hidden_3h,
            verified_id=next_token_ids.to(torch.int64),
            accept_length=zero_acc,
            cross_loc=cross_loc_list,
            cross_count=cross_count_list,
            cross_position=cross_position_list,
            ttt_k=ttt_k_list,
            ttt_v=ttt_v_list,
            b0_logits=b0_logits,
            b0_hidden=b0_hidden,
            b0_input_id=b0_input_id,
            b0_rank_logits=b0_rank,
            b0_top_indices=b0_top_indices,
            new_seq_lens=batch.seq_lens.clone(),  # V2 future-buffer
            capture_hidden_mode=CaptureHiddenMode.FULL,
            kv_pool=self.spec_kv_pool,
        )
        batch.spec_info = spec_info
        batch.return_hidden_states = False

    # ============================================================
    #  Hot path -- decode (forward_mode == DECODE -> TARGET_VERIFY)
    # ============================================================

    def _require_draft_graph_bucket(
        self, spec_info: SpecBlockDraftInput, bs: int
    ) -> Tuple[int, int, int]:
        bucket = self.draft_cuda_graph_runner.resolve_buckets(spec_info, bs)
        if bucket is None:
            max_cross = max(spec_info.cross_count) if spec_info.cross_count else 0
            raise RuntimeError(
                "SpecBlock draft inputs exceed the supported CUDA Graph buckets: "
                f"batch_size={bs}, max_cross_count={max_cross}."
            )
        return bucket

    def draft(self, batch: ScheduleBatch) -> SpecBlockVerifyInput:
        """Build per-req trees via batched draft, then concat into one
        :class:`SpecBlockVerifyInput`.

        Reads cached b0_* / cross_kv / ttt_kv from
        ``batch.spec_info`` (a :class:`SpecBlockDraftInput`) and routes
        through the sole :class:`SpecBlockDraftCudaGraphRunner` path.

        Single-pass, GPU-only post-processing: parents → topology link
        list via ``build_retrieve_links_gpu``; per-req prefix mask via a
        batched ones+cat instead of Python for-loops; positions /
        seq_lens / sum stay GPU tensors (caller derives ints lazily).
        """
        from sglang.srt.speculative.specblock_tree_kernels import (
            build_retrieve_links_gpu,
        )

        bs = batch.batch_size()
        spec_info: SpecBlockDraftInput = batch.spec_info  # type: ignore[assignment]

        # ---- V2 audit: detect spec_info size mismatch with batch size ----
        if os.environ.get("SPECBLOCK_V2_SIZE_AUDIT", "0") == "1":
            import logging as _lg
            _l = _lg.getLogger(__name__)
            cls = SpecBlockWorker
            n = getattr(cls, "_V2_SIZE_AUDIT_N", 0)
            if n < 30 and spec_info is not None:
                cls._V2_SIZE_AUDIT_N = n + 1
                b0_shape = (
                    list(spec_info.b0_logits.shape) if spec_info.b0_logits is not None else None
                )
                _l.info(
                    f"[V2SIZE.{n+1}] bs={bs} "
                    f"len(cross_loc)={len(spec_info.cross_loc) if spec_info.cross_loc else 0} "
                    f"len(cross_count)={len(spec_info.cross_count) if spec_info.cross_count else 0} "
                    f"len(cross_position)={len(spec_info.cross_position) if spec_info.cross_position else 0} "
                    f"len(ttt_k)={len(spec_info.ttt_k) if spec_info.ttt_k else 0} "
                    f"len(ttt_v)={len(spec_info.ttt_v) if spec_info.ttt_v else 0} "
                    f"b0_logits.shape={b0_shape}"
                )

        # ---- DEEP profile: time draft sub-phases ----
        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t0 = time.perf_counter()

        # V2 trace: log b0_input_id at draft entry (first 30 iters).
        if os.environ.get("SPECBLOCK_V2_TRACE", "0") == "1":
            import logging as _lg
            _l = _lg.getLogger(__name__)
            cls = SpecBlockDraftInput
            n = getattr(cls, "_DRAFT_TRACE_N", 0)
            if n < 30:
                cls._DRAFT_TRACE_N = n + 1
                bid = spec_info.b0_input_id
                hid = spec_info.b0_hidden
                bls = spec_info.b0_logits
                vid = spec_info.verified_id
                if bid is not None:
                    _l.info(
                        f"[V2TRACE.draft.{n+1}] "
                        f"b0_input_id={bid.flatten()[:8].tolist()} "
                        f"verified_id={vid.flatten()[:8].tolist() if vid is not None else None} "
                        f"b0_logits.first8="
                        f"{bls.flatten()[:8].tolist() if bls is not None else None} "
                        f"b0_hidden.first8="
                        f"{hid.flatten()[:8].tolist() if hid is not None else None} "
                        f"cross_count={list(spec_info.cross_count)} "
                        f"cross_position={list(spec_info.cross_position)}"
                    )
        if not isinstance(spec_info, SpecBlockDraftInput):
            raise RuntimeError(
                "[SpecBlockWorker.draft] expected batch.spec_info to be "
                f"SpecBlockDraftInput; got {type(spec_info).__name__}."
            )
        runner = self.draft_cuda_graph_runner
        bucket = self._require_draft_graph_bucket(spec_info, bs)
        if not runner.can_run(*bucket):
            with self.spec_kv_pool.capture_mode():
                # Seed warmup with real inputs; degenerate all-zero paged
                # attention inputs are not a valid capture workload.
                runner.capture_one(*bucket, spec_info=spec_info)
        trees = runner.replay(spec_info, bs, *bucket)

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_setup = time.perf_counter()

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_build_tree = time.perf_counter()

        fixed_outputs = trees[0].get("_batched_fixed_outputs")
        N_full = int(trees[0]["draft_token"].shape[0])
        device = self.device

        if (
            fixed_outputs is not None
            and os.environ.get("SPECBLOCK_TREE_AUDIT", "0") == "1"
        ):
            audit_n = getattr(type(self), "_TREE_AUDIT_N", 0)
            if audit_n < 10:
                type(self)._TREE_AUDIT_N = audit_n + 1
                audit_tokens = fixed_outputs["tokens"]
                audit_parents = fixed_outputs["parents"]
                audit_lps = fixed_outputs["lps"]
                audit_valid = torch.isfinite(audit_lps)
                nonroot = torch.arange(N_full, device=device) > 0
                same_sibling = (
                    (audit_tokens[:, :, None] == audit_tokens[:, None, :])
                    & (audit_parents[:, :, None] == audit_parents[:, None, :])
                    & audit_valid[:, :, None]
                    & audit_valid[:, None, :]
                    & nonroot[None, :, None]
                    & nonroot[None, None, :]
                )
                upper = torch.triu(
                    torch.ones(
                        N_full, N_full, dtype=torch.bool, device=device,
                    ),
                    diagonal=1,
                )
                duplicate_pairs = same_sibling & upper
                duplicate_count = duplicate_pairs.sum(dim=(1, 2))
                audit_depth = fixed_outputs["depth"]
                duplicate_depths = [
                    audit_depth[b, duplicate_pairs[b].nonzero()[:, 0]]
                    .cpu()
                    .tolist()
                    for b in range(audit_tokens.shape[0])
                ]
                invalid_count = (~audit_valid[:, 1:]).sum(dim=1)
                logger.info(
                    "[SpecBlockTreeAudit] iter=%d invalid=%s "
                    "duplicate_siblings=%s duplicate_depths=%s",
                    audit_n + 1,
                    invalid_count.cpu().tolist(),
                    duplicate_count.cpu().tolist(),
                    duplicate_depths,
                )

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_pad = time.perf_counter()

        # ---- Pack per-req trees into batched verify spec_info ----
        # The fixed-width finalizer already owns contiguous [B, N] outputs;
        # consume zero-copy flattened views rather than cat/stack copies.
        if fixed_outputs is not None:
            draft_token_cat = fixed_outputs["tokens"].reshape(-1)
            parents_stack = fixed_outputs["topo_parents"]
            depth_stack = fixed_outputs["depth"]
            positions_t = fixed_outputs["positions"].reshape(-1)
            tree_lps_cat = fixed_outputs["lps"].reshape(-1)
        else:
            draft_token_cat = torch.cat(
                [t["draft_token"] for t in trees], dim=0
            )
            raw_parents_stack = torch.stack(
                [t["parents"].to(torch.int64) for t in trees], dim=0
            )
            depth_stack = torch.stack(
                [t["depth"].to(torch.int64) for t in trees], dim=0
            )
            tree_lps_cat = torch.cat(
                [t["cum_log_prob"] for t in trees], dim=0
            )
            # Generic finalizers store non-root parents in raw-tree coordinates.
            parents_stack = torch.empty_like(raw_parents_stack)
            parents_stack[:, 0] = -1
            parents_stack[:, 1:] = torch.where(
                raw_parents_stack[:, 1:] >= 0,
                raw_parents_stack[:, 1:] + 1,
                0,
            )
            positions_t = (
                depth_stack + batch.seq_lens.to(torch.int64).unsqueeze(1)
            ).reshape(-1)
        tree_depth_cat = depth_stack.reshape(-1)

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_cat = time.perf_counter()

        # Prefix attention is implicitly all-visible in the grouped verifier;
        # only carry the root-to-node tree mask emitted by the GPU finalizer.
        seq_lens_cpu = batch.seq_lens_cpu
        if seq_lens_cpu is None:
            raise RuntimeError(
                "SPECBLOCK_SHIFT requires ScheduleBatch.seq_lens_cpu during decode."
            )
        if fixed_outputs is not None:
            custom_mask = fixed_outputs["mask"].reshape(-1)
        else:
            tree_masks = torch.stack(
                [tree["tree_mask"] for tree in trees], dim=0,
            )
            custom_mask = tree_masks.to(torch.bool).reshape(-1).contiguous()

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_metadata = time.perf_counter()

        if fixed_outputs is not None:
            retrive_index = fixed_outputs["retrieve_index"]
        else:
            # retrive_index: row i = arange(N_full) + i * N_full.
            ar_n = torch.arange(N_full, dtype=torch.long, device=device)
            ar_b = torch.arange(bs, dtype=torch.long, device=device)
            retrive_index = (
                ar_n.unsqueeze(0).expand(bs, -1) + ar_b.unsqueeze(1) * N_full
            ).contiguous()

        # The one-block fixed-width finalizer emits topology links directly.
        # Multi-block / variable-width trees retain the generic GPU conversion.
        if fixed_outputs is not None:
            retrive_next_token = fixed_outputs["next_token"]
            retrive_next_sibling = fixed_outputs["next_sibling"]
        elif all("retrive_next_token" in tree for tree in trees):
            retrive_next_token = torch.stack(
                [tree["retrive_next_token"] for tree in trees], dim=0,
            )
            retrive_next_sibling = torch.stack(
                [tree["retrive_next_sibling"] for tree in trees], dim=0,
            )
        else:
            retrive_next_token, retrive_next_sibling = build_retrieve_links_gpu(
                trees, device=device, N_full=N_full,
            )

        verify_info = SpecBlockVerifyInput(
            draft_token=draft_token_cat,
            custom_mask=custom_mask,
            positions=positions_t,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            draft_token_num=N_full,
            num_tokens_per_batch=N_full,
            tree_parents=parents_stack,
            tree_depth=tree_depth_cat,
            # Target native decode compiles this tree-builder contract rather
            # than using the much larger tree-width allocation as a walk bound.
            tree_max_depth=self.tree_max_depth,
            tree_lps=tree_lps_cat,
            tree_sizes_cpu=[N_full] * bs,
            hidden_states=spec_info.hidden_states,  # carry through 3H
            verified_id=spec_info.verified_id,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            # Carry per-req SpecBlock state through draft -> verify ->
            # refresh hop, so scheduler filter_batch / merge_batch keeps
            # them in sync (they live on batch.spec_info, not on self).
            cross_loc=list(spec_info.cross_loc),
            cross_count=list(spec_info.cross_count),
            cross_position=list(spec_info.cross_position),
            ttt_k=spec_info.ttt_k,
            ttt_v=spec_info.ttt_v,
            kv_pool=self.spec_kv_pool,
        )
        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_end = time.perf_counter()
            cls = SpecBlockWorker
            st = getattr(cls, "_DEEP_PROF_DRAFT", None)
            if st is None:
                st = {
                    "n": 0, "setup": 0.0, "build_tree": 0.0,
                    "pad": 0.0, "cat": 0.0, "metadata": 0.0, "tail": 0.0,
                }
                cls._DEEP_PROF_DRAFT = st
            st["n"] += 1
            st["setup"] += _dp_t_setup - _dp_t0
            st["build_tree"] += _dp_t_build_tree - _dp_t_setup
            st["pad"] += _dp_t_pad - _dp_t_build_tree
            st["cat"] += _dp_t_cat - _dp_t_pad
            st["metadata"] += _dp_t_metadata - _dp_t_cat
            st["tail"] += _dp_t_end - _dp_t_metadata
            if st["n"] % 50 == 0:
                n = st["n"]
                logger.info(
                    "[DEEPprof:draft] n=%d setup=%.2fms build_tree=%.2fms "
                    "pad=%.2fms cat=%.2fms metadata=%.2fms tail=%.2fms total=%.2fms",
                    n,
                    st["setup"]/n*1000, st["build_tree"]/n*1000,
                    st["pad"]/n*1000, st["cat"]/n*1000,
                    st["metadata"]/n*1000, st["tail"]/n*1000,
                    (st["setup"]+st["build_tree"]+st["pad"]+st["cat"]+st["metadata"]+st["tail"])/n*1000,
                )

        return verify_info

    # ============================================================
    #  Hot path -- TARGET_VERIFY
    # ============================================================

    def _verify_and_accept(
        self,
        batch: ScheduleBatch,
        *,
        skip_prepare: bool = False,
        skip_free_cache: bool = False,
        prepared_forward_batch: Optional[ForwardBatch] = None,
        prepared_can_run_cuda_graph: Optional[bool] = None,
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int, List[int], bool]:
        """Run target verify forward then accept; return raw fields for
        the GenerationBatchResult assembly in forward_batch_generation.

        ``skip_prepare`` (V2 path): skip ``prepare_for_verify``; out_cache_loc
        and req_to_token mapping were already set up on the plan stream.
        ``prepared_forward_batch`` keeps that plan-stream preparation live through
        target execution, rather than rebuilding a ModelWorkerBatch on the main
        stream.  It is V2-only; the V1 path remains unchanged.
        """
        spec_info: SpecBlockVerifyInput = batch.spec_info  # type: ignore[assignment]
        if not isinstance(spec_info, SpecBlockVerifyInput):
            raise RuntimeError(
                "[SpecBlockWorker._verify_and_accept] expected "
                "SpecBlockVerifyInput; got "
                f"{type(spec_info).__name__}."
            )

        # Allocate KV slots for verify forward (V1 path; V2 plan-stream
        # has already done this in prepare_for_decode).
        if not skip_prepare:
            spec_info.prepare_for_verify(batch, self.page_size)

        # ---- bs > 1: tell target attn backend the actual draft_token_num.
        # TritonAttnBackend.num_draft_tokens is normally set once from
        # server_args.speculative_num_draft_tokens, but our tree size is
        # dynamic.  See specblock_info.py / docs for details.
        try:
            tgt_attn = self._target_worker.model_runner.attn_backend
            if hasattr(tgt_attn, "num_draft_tokens"):
                tgt_attn.num_draft_tokens = spec_info.draft_token_num
        except AttributeError:
            pass

        batch.return_hidden_states = False
        batch.forward_mode = ForwardMode.TARGET_VERIFY

        if _PROFILE:
            torch.cuda.synchronize()
            _t_vf_s = time.perf_counter()
        if prepared_forward_batch is None:
            model_worker_batch = batch.get_model_worker_batch(
                seq_lens_cpu_cache=spec_info.seq_lens_cpu
            )
            batch_result = self._target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
        else:
            if prepared_can_run_cuda_graph is None:
                raise RuntimeError(
                    "Prepared SpecBlock V2 verification requires its CUDA "
                    "graph eligibility result."
                )
            batch_result = self._target_worker.forward_batch_generation(
                model_worker_batch=None,
                forward_batch=prepared_forward_batch,
                is_verify=True,
                skip_attn_backend_init=True,
            )
        if _PROFILE:
            torch.cuda.synchronize()
            _t_vf_e = time.perf_counter()
            self._prof.setdefault("verify_fwd", 0.0)
            self._prof["verify_fwd"] += _t_vf_e - _t_vf_s
            self._prof.setdefault("draft_token_num_sum", 0)
            self._prof["draft_token_num_sum"] += int(spec_info.draft_token_num)
        logits_output = batch_result.logits_output
        can_run_cuda_graph = batch_result.can_run_cuda_graph

        # Stash hidden 3H so verify() can carry them through to next iter.
        spec_info.hidden_states = logits_output.hidden_states

        if _PROFILE:
            torch.cuda.synchronize()
            _t_va_s = time.perf_counter()
        logits_output, verified_ids, accept_lens_cpu = spec_info.verify(
            batch, logits_output, self.page_size,
            skip_free_cache=skip_free_cache,
        )
        if _PROFILE:
            torch.cuda.synchronize()
            _t_va_e = time.perf_counter()
            self._prof.setdefault("verify_accept", 0.0)
            self._prof["verify_accept"] += _t_va_e - _t_va_s

        # Single sync — derive both list and num_accepted from the cpu
        # tensor's tolist.  ``accept_lens_cpu`` from spec_info.verify is
        # accept_count + 1 (includes bonus).  V1 ``accept_length_cpu``
        # downstream interface expects accept_count (no bonus).
        accept_lens_with_bonus = accept_lens_cpu.tolist()
        accept_length_cpu = [x - 1 for x in accept_lens_with_bonus]
        num_accepted = sum(accept_length_cpu)
        # Stash tensor on self so V2 worker can plumb to GenerationBatchResult
        # (V2 scheduler reads result.accept_lens.is_cpu — needs accept+1).
        self._last_accept_lens_cpu = accept_lens_cpu
        return (
            logits_output,
            verified_ids,
            num_accepted,
            accept_length_cpu,
            can_run_cuda_graph,
        )

    # ============================================================
    #  After verify -- refresh draft state for iter k+1
    # ============================================================

    def _refresh_draft_state(
        self,
        batch: ScheduleBatch,
        verify_info: SpecBlockVerifyInput,
        logits_output: LogitsProcessorOutput,
        verified_id: torch.Tensor,
        accept_length_cpu: List[int],
        accept_lengths_gpu: Optional[torch.Tensor] = None,
    ):
        """Padded batched draft_model.update_cache_and_draft to seed iter k+1.

        Single forward call over all reqs (vs V1 per-req loop).  Pads
        per-req variable N (= accept_len+1) and cross_count to (max_N,
        max_cross), masks cross padding via cross_mask, and gathers per-req
        real-N_i last-K outputs via n_per_req.

        Pad slots in new_indices use sentinel index 0, but the paged KV write
        masks them with each row's real slot count so slot 0 remains zero.

        At entry:
            logits_output.hidden_states is [sum_i (accept_len_i+1), 3H],
                cropped to accepted indices by spec_info._fill_requests.
            verified_id is [sum_i (accept_len_i+1)] flat chain.

        Result: ``batch.spec_info`` becomes a fresh SpecBlockDraftInput
        populated with new b0_*, updated cross_kv, ttt_kv etc.
        """
        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _r_t0 = time.perf_counter()

        verified_h3 = logits_output.hidden_states  # [cum_len, 3H]
        if verified_h3 is None:
            raise RuntimeError(
                "[SpecBlockWorker._refresh_draft_state] hidden_states "
                "is None -- target verify forward did not capture FULL aux."
            )

        draft_model = self.model_runner.model.inner
        draft_dtype = next(draft_model.parameters()).dtype
        if verified_h3.dtype != draft_dtype:
            verified_h3 = verified_h3.to(draft_dtype)

        cross_loc_list = list(verify_info.cross_loc)
        cross_count_list = list(verify_info.cross_count)
        cross_position_list = list(verify_info.cross_position)

        bs = len(accept_length_cpu)
        K = self.K

        # HF semantics for a rejected root: discard that root's block-0 K
        # slots, then rebuild block-0 from the target bonus at the next
        # position.  Keeping the failed block in persistent cross KV changes
        # every later draft proposal and leaks stale state indefinitely.
        for ridx, accept_len in enumerate(accept_length_cpu):
            if int(accept_len) != 0:
                continue
            if cross_count_list[ridx] < K:
                raise RuntimeError(
                    "SpecBlock zero-accept rollback requires the current "
                    f"block-0 K={K} slots, got cross_count="
                    f"{cross_count_list[ridx]} for request {ridx}."
                )
            old_loc = cross_loc_list[ridx]
            dropped = old_loc[-K:]
            self.spec_kv_pool.free(dropped)
            cross_loc_list[ridx] = old_loc[:-K]
            cross_count_list[ridx] -= K

        Ns = [int(accept_length_cpu[i]) + 1 for i in range(bs)]
        max_N = max(Ns)
        max_cross = max(cross_count_list) if cross_count_list else 0
        refresh_bucket = self.refresh_cuda_graph_runner.resolve_buckets(
            bs, max_cross, max_N,
        )
        if refresh_bucket is None:
            raise RuntimeError(
                "SpecBlock refresh request exceeds supported CUDA Graph "
                f"buckets: bs={bs}, max_cross={max_cross}, max_N={max_N}."
            )
        bcap, cross_bucket, accept_bucket = refresh_bucket

        # ---- Build fixed-width paged cross input ----
        # Slot 0 is the immutable zero sentinel.  Every bucket tail remains 0
        # and is excluded by cross_mask inside the captured Triton kernel.
        cross_loc_padded = torch.zeros(
            bs, cross_bucket, dtype=torch.long, device=self.device,
        )
        cross_mask = torch.zeros(
            bs, cross_bucket, dtype=torch.bool, device=self.device,
        )
        for ridx in range(bs):
            c_i = cross_count_list[ridx]
            if c_i > 0:
                cross_loc_padded[ridx, :c_i] = cross_loc_list[ridx].to(
                    self.device, dtype=torch.long,
                )
                cross_mask[ridx, :c_i] = True

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _r_t_pad = time.perf_counter()

        # ---- Allocate live paged indices; graph-bucket tail stays slot 0 ----
        new_indices_padded = torch.zeros(
            bs, accept_bucket * K, dtype=torch.long, device=self.device,
        )
        new_indices_per_req: List[torch.Tensor] = []
        for ridx in range(bs):
            N_i = Ns[ridx]
            alloc_i = self.spec_kv_pool.alloc(N_i * K)
            new_indices_padded[ridx, :N_i * K] = alloc_i
            new_indices_per_req.append(alloc_i)

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _r_t_alloc = time.perf_counter()

        # Tensorize the ragged accepted chains without extracting CUDA scalar
        # lengths. Pool allocation / rollback and persistent list ownership
        # remain deliberately outside this helper for V1/V2 scheduler safety.
        # V1 passes the original acceptance tensor, so refresh avoids a
        # CPU-list -> GPU round trip. The fallback keeps the V2 compatibility
        # shim safe until it forwards this tensor as well.
        accept_lengths = (
            accept_lengths_gpu.to(device=self.device, dtype=torch.int32)
            if accept_lengths_gpu is not None
            else torch.tensor(
                accept_length_cpu, dtype=torch.int32, device=self.device,
            )
        )
        cross_positions = torch.tensor(
            cross_position_list, dtype=torch.long, device=self.device,
        )
        existing_cross_counts = torch.tensor(
            cross_count_list, dtype=torch.long, device=self.device,
        )
        refresh_plan = build_specblock_refresh_tensor_plan(
            verified_id.to(self.device, dtype=torch.long),
            verified_h3,
            accept_lengths,
            cross_positions,
            cross_loc_padded,
            existing_cross_counts,
            new_indices_padded,
            K=K,
        )
        # Bucket selection above guarantees this without a host synchronizing
        # ``any()``: max(Ns) <= accept_bucket.
        hidden_padded = refresh_plan.hidden
        tokens_padded = refresh_plan.tokens
        pos_ids = refresh_plan.pos_ids
        n_per_req = refresh_plan.n_per_req
        valid_new_slots = refresh_plan.new_cross_valid_slots
        start_position = refresh_plan.start_positions
        start_positions_cpu = [
            p if int(accept_length_cpu[i]) == 0 else p + 1
            for i, p in enumerate(cross_position_list)
        ]

        # ---- V2 audit trace ----
        if os.environ.get("SPECBLOCK_V2_AUDIT", "0") == "1":
            import logging as _lg
            _l = _lg.getLogger(__name__)
            cls = SpecBlockWorker
            n = getattr(cls, "_REFRESH_AUDIT_N", 0)
            if n < 50:
                cls._REFRESH_AUDIT_N = n + 1
                pool = self.spec_kv_pool
                cl_max_per_req = []
                for ridx in range(bs):
                    if cross_count_list[ridx] > 0:
                        cl_max_per_req.append(
                            int(cross_loc_padded[ridx, :cross_count_list[ridx]].max().item())
                        )
                    else:
                        cl_max_per_req.append(-1)
                _l.info(
                    f"[V2AUDIT.refresh.{n+1}] bs={bs} Ns={Ns} max_N={max_N} "
                    f"K={K} max_cross={max_cross} "
                    f"cross_count_list={list(cross_count_list)} "
                    f"cl_max_per_req={cl_max_per_req} "
                    f"pool_size={pool.pool_size} n_alloc={pool.n_alloc} "
                    f"n_free={pool.n_free} pool_version={pool.pool_version} "
                    f"new_indices_max="
                    f"{int(new_indices_padded.max().item()) if new_indices_padded.numel() else 0} "
                    f"start_position={start_position.tolist()} "
                    f"cross_position_list={list(cross_position_list)}"
                )

        # Allocation above may grow the paged pool and invalidate captured
        # pointers. Resolve capture only after every live slot is allocated.
        if not self.refresh_cuda_graph_runner.can_run(
            bcap, cross_bucket, accept_bucket,
        ):
            self.refresh_cuda_graph_runner.capture_one(
                bcap, cross_bucket, accept_bucket,
            )

        # ---- Replay the sole refresh model path ----
        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _r_t_plan = time.perf_counter()

        (
            logits,
            rank_logits,
            draft_hidden,
            new_cross_kv,
            ttt_kv,
            b0_top_indices,
        ) = self.refresh_cuda_graph_runner.replay(
            bcap,
            cross_bucket,
            accept_bucket,
            bs,
            hidden_padded,
            tokens_padded,
            pos_ids,
            cross_loc_padded,
            cross_mask,
            n_per_req,
        )

        # Persistent paged KV mutation stays outside capture.  The Triton
        # scatter writes only each row's real N_i*K prefix, so bucket padding
        # never overwrites the reserved zero sentinel.
        for layer_idx, (layer_k, layer_v) in enumerate(new_cross_kv):
            self.spec_kv_pool.set_kv_padded(
                layer_idx,
                new_indices_padded,
                layer_k,
                layer_v,
                valid_new_slots,
            )

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _r_t_fwd = time.perf_counter()

        # ---- Per-req post-process ----
        new_cross_loc: List[torch.Tensor] = []
        new_cross_count: List[int] = []
        new_cross_position: List[int] = []
        new_ttt_k: List[List[torch.Tensor]] = [[] for _ in range(bs)]
        new_ttt_v: List[List[torch.Tensor]] = [[] for _ in range(bs)]

        for ridx in range(bs):
            N_i = Ns[ridx]
            c_i = cross_count_list[ridx]
            old_loc = cross_loc_list[ridx].to(
                self.device, dtype=torch.long,
            )
            updated_loc = torch.cat(
                [old_loc, new_indices_per_req[ridx]], dim=0,
            )
            new_cross_loc.append(updated_loc)
            new_cross_count.append(c_i + N_i * K)
            # A zero-accept rebuild replaces block-0 at the current
            # draft_position; unlike a positive update, it must not advance
            # the persistent position used by the next iteration.
            if int(accept_length_cpu[ridx]) == 0:
                new_cross_position.append(int(cross_position_list[ridx]))
            else:
                new_cross_position.append(
                    int(start_positions_cpu[ridx] + N_i)
                )

        for layer_idx in range(self.num_layers):
            layer_k, layer_v = ttt_kv[layer_idx]  # [B, n_kv_heads, K, D]
            for ridx in range(bs):
                new_ttt_k[ridx].append(layer_k[ridx:ridx + 1])
                new_ttt_v[ridx].append(layer_v[ridx:ridx + 1])

        # ``n_per_req`` is GPU-resident; gather each row's actual chain tail
        # from the fixed-capacity refresh plan instead of preserving Python
        # per-request tensor slices.
        tail_idx = (n_per_req - 1).unsqueeze(1)
        b0_input_id = tokens_padded.gather(1, tail_idx)  # [B, 1]
        last_hidden_3h = hidden_padded[
            torch.arange(bs, device=self.device), n_per_req - 1
        ]  # [B, 3H]

        b0_logits = logits  # [B, K, vocab]
        b0_rank = rank_logits  # [B, K, rank_classes]
        b0_hidden = draft_hidden  # [B, K, H]

        accept_length_t = torch.tensor(
            accept_length_cpu, dtype=torch.int32, device=self.device,
        )
        per_req_last_id = b0_input_id.squeeze(-1).clone()  # [B]

        new_spec_info = SpecBlockDraftInput(
            hidden_states=last_hidden_3h,
            verified_id=per_req_last_id,
            accept_length=accept_length_t,
            cross_loc=new_cross_loc,
            cross_count=new_cross_count,
            cross_position=new_cross_position,
            ttt_k=new_ttt_k,
            ttt_v=new_ttt_v,
            b0_logits=b0_logits,
            b0_hidden=b0_hidden,
            b0_input_id=b0_input_id,
            b0_rank_logits=b0_rank,
            b0_top_indices=b0_top_indices,
            new_seq_lens=batch.seq_lens.clone(),  # V2 future-buffer
            capture_hidden_mode=CaptureHiddenMode.FULL,
            kv_pool=self.spec_kv_pool,
        )
        batch.spec_info = new_spec_info

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _r_t_end = time.perf_counter()
            cls = SpecBlockWorker
            st = getattr(cls, "_DEEP_PROF_REFRESH", None)
            if st is None:
                st = {
                    "n": 0, "pad": 0.0, "alloc": 0.0, "plan": 0.0,
                    "fwd": 0.0, "tail": 0.0,
                }
                cls._DEEP_PROF_REFRESH = st
            st["n"] += 1
            st["pad"] += _r_t_pad - _r_t0
            st["alloc"] += _r_t_alloc - _r_t_pad
            st["plan"] += _r_t_plan - _r_t_alloc
            st["fwd"] += _r_t_fwd - _r_t_plan
            st["tail"] += _r_t_end - _r_t_fwd
            if st["n"] % 50 == 0:
                n = st["n"]
                logger.info(
                    "[DEEPprof:refresh] n=%d pad=%.2fms alloc=%.2fms "
                    "plan=%.2fms fwd=%.2fms tail=%.2fms total=%.2fms",
                    n,
                    st["pad"]/n*1000, st["alloc"]/n*1000,
                    st["plan"]/n*1000, st["fwd"]/n*1000,
                    st["tail"]/n*1000,
                    (st["pad"]+st["alloc"]+st["plan"]+st["fwd"]+st["tail"])/n*1000,
                )

    # ============================================================
    #  forward_batch_generation -- main scheduler entry
    # ============================================================

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """Single-step driver invoked by the scheduler each tick."""
        if _PROFILE:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        # ---- EXTEND (prefill or resumed extend) ----
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(
                batch
            )
            self.forward_draft_extend(
                batch, logits_output, next_token_ids, seq_lens_cpu
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )

        # ---- DECODE (build trees -> set TARGET_VERIFY) ----
        if batch.forward_mode.is_decode():
            verify_info = self.draft(batch)
            if _PROFILE:
                torch.cuda.synchronize()
                _t1 = time.perf_counter()
                self._prof["draft"] += _t1 - _t0
            batch.spec_info = verify_info
            # Fall through to TARGET_VERIFY in this same call.
            batch.forward_mode = ForwardMode.TARGET_VERIFY

        # ---- TARGET_VERIFY -----
        if batch.forward_mode.is_target_verify():
            if _PROFILE:
                torch.cuda.synchronize()
                _ts = time.perf_counter()
            # Stash the verify_info reference *before* _verify_and_accept
            # mutates batch.spec_info.  We need it in _refresh_draft_state
            # to read carried per-req cross_*/ttt_* state.
            verify_info = batch.spec_info
            (
                logits_output,
                verified_ids,
                num_accepted,
                accept_length_cpu,
                can_run_cuda_graph,
            ) = self._verify_and_accept(batch)
            if _PROFILE:
                torch.cuda.synchronize()
                _tv = time.perf_counter()
                self._prof["verify_setup"] += _tv - _ts

            self._refresh_draft_state(
                batch,
                verify_info,
                logits_output,
                verified_ids,
                accept_length_cpu,
                accept_lengths_gpu=verify_info.accept_length,
            )
            if _PROFILE:
                torch.cuda.synchronize()
                _tr = time.perf_counter()
                self._prof["refresh"] += _tr - _tv

            # Reset to DECODE so the scheduler does NOT mis-classify next iter
            # as an extend batch (TARGET_VERIFY's is_extend() returns True
            # because of how ForwardMode is enumerated).
            batch.forward_mode = ForwardMode.DECODE

            if _PROFILE:
                self._prof["n"] += 1
                if self._prof["n"] % 50 == 0:
                    p = self._prof
                    n = p["n"]
                    avg_dn = p.get("draft_token_num_sum", 0) / max(n, 1)
                    # verify_fwd / verify_accept are *inside* verify_setup;
                    # worker hot path = draft + verify_setup + refresh.
                    worker_hot_ms = (
                        (p['draft'] + p['verify_setup'] + p['refresh']) / n * 1000
                    )
                    logger.info(
                        f"[SBSprof] iter={n} avg_tree={avg_dn:.1f} | "
                        f"draft={p['draft'] / n * 1000:.2f}ms "
                        f"verify_setup={p['verify_setup'] / n * 1000:.2f}ms "
                        f"(verify_fwd={p.get('verify_fwd', 0.0) / n * 1000:.2f}ms "
                        f"accept={p.get('verify_accept', 0.0) / n * 1000:.2f}ms) "
                        f"refresh={p['refresh'] / n * 1000:.2f}ms "
                        f"worker_hot={worker_hot_ms:.2f}ms"
                    )
                    # Drain attention-level _PROFILE_EVENTS (set via
                    # STATIC_DRAFT_PROFILE_DEEP=1) and aggregate by label.
                    try:
                        from sglang.srt.models import _specblock_inference as _sim
                        events = _sim._PROFILE_EVENTS
                        if events:
                            torch.cuda.synchronize()
                            agg = {}
                            for label, evs, eve in events:
                                t = evs.elapsed_time(eve)  # ms
                                a = agg.setdefault(label, [0.0, 0])
                                a[0] += t
                                a[1] += 1
                            parts = []
                            for label, (total_ms, k) in sorted(
                                agg.items(), key=lambda kv: -kv[1][0]
                            ):
                                parts.append(
                                    f"{label}={total_ms / k:.2f}ms*{k // n}"
                                )
                            logger.info("[SBSprof:deep] " + " ".join(parts))
                            events.clear()
                    except Exception as e:
                        logger.warning(f"[SBSprof] deep drain failed: {e}")
                    # Drain tree-builder phase profile (set via
                    # SPECBLOCK_TREE_PROFILE=1).
                    try:
                        from sglang.srt.speculative import specblock_tree_builder as _tb
                        tps = _tb._BUILDTREE_PROFILE_STATE
                        if tps["n"] > 0:
                            tn = tps["n"]
                            logger.info(
                                "[SBSprof:tree] iter=%d "
                                "block1=%.2fms b2_fwd=%.2fms finalize=%.2fms "
                                "total=%.2fms",
                                tn,
                                tps["block1"] / tn * 1000,
                                tps["b2_fwd"] / tn * 1000,
                                tps["finalize"] / tn * 1000,
                                (tps["block1"] + tps["b2_fwd"] + tps["finalize"]) / tn * 1000,
                            )
                    except Exception as e:
                        logger.warning(f"[SBSprof] tree drain failed: {e}")

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verified_ids,
                num_accepted_tokens=num_accepted,
                accept_length_per_req_cpu=accept_length_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        # ---- Fallback (idle / unsupported mode) ----
        model_worker_batch = batch.get_model_worker_batch()
        return self._target_worker.forward_batch_generation(model_worker_batch)
