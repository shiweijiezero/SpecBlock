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
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map
from sglang.srt.speculative.specblock_info import (
    SpecBlockDraftInput,
    SpecBlockVerifyInput,
    parents_to_next_token_and_sibling,
)
from sglang.srt.speculative.spec_kv_pool import SpecBlockKVPool
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
        self.beam_width: int = int(os.environ.get("SPECBLOCK_BEAM_WIDTH", "10"))
        self.total_tokens: int = int(os.environ.get("SPECBLOCK_TOTAL_TOKENS", "90"))
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
        if hasattr(draft_model, "config"):
            cfg = draft_model.config
            self.rank_classes = int(getattr(cfg, "rank_classes", 4))
            self.max_blocks = int(getattr(cfg, "num_ttt_blocks", 2))
        _mb_env = os.environ.get("SPECBLOCK_MAX_BLOCKS")
        if _mb_env is not None:
            self.max_blocks = int(_mb_env)

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

        # ---- CUDA graph runners (production default ON) ----
        # Draft runner captures build_tree_gpu's GPU-only chain per
        # (bs=1, cross_bucket, pend_bucket).  Refresh runner captures
        # _refresh_draft_state's batched cache-and-draft per (bs,
        # cross_bucket, accept_max).  Both lazy-capture on first miss.
        from sglang.srt.speculative.specblock_draft_cuda_graph_runner import (
            SpecBlockDraftCudaGraphRunner,
        )
        from sglang.srt.speculative.specblock_refresh_cuda_graph_runner import (
            SpecBlockRefreshCudaGraphRunner,
        )
        self.draft_cuda_graph_runner = SpecBlockDraftCudaGraphRunner(self)
        self.refresh_cuda_graph_runner = SpecBlockRefreshCudaGraphRunner(self)

        logger.info(
            f"SpecBlock-Shift worker initialized: "
            f"K={self.K}, num_layers={self.num_layers}, "
            f"max_blocks={self.max_blocks}, beam_width={self.beam_width}, "
            f"total_tokens={self.total_tokens}, rank_classes={self.rank_classes}, "
            f"hot_vocab="
            f"{None if self.hot_token_id is None else len(self.hot_token_id)}"
        )

    # ============================================================
    #  BaseSpecWorker contract (TpModelWorker is the actual base, but
    #  scheduler treats us as a spec worker via duck-typing on these
    #  properties + clear_cache_pool + forward_batch_generation).
    # ============================================================

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
        """The KV pool is owned by target_worker; nothing to clear here."""
        return

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
            last_hidden_list.append(last_hidden_i.squeeze(0).squeeze(0))  # [3H]
            cross_position_list.append(int(pos_i))

        # Stack per-req tensors along batch dim 0 so the spec_info has
        # standard (B, ...) shapes.
        b0_logits = torch.cat(b0_logits_list, dim=0)             # [B, K, V_draft]
        b0_rank = torch.cat(b0_rank_list, dim=0)                  # [B, K, rank_classes]
        b0_hidden = torch.cat(b0_hidden_list, dim=0)              # [B, K, H]
        b0_input_id = torch.cat(b0_input_ids_list, dim=0)         # [B, 1]  (last verified)
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
            new_seq_lens=batch.seq_lens.clone(),  # V2 future-buffer
            capture_hidden_mode=CaptureHiddenMode.FULL,
            kv_pool=self.spec_kv_pool,
        )
        batch.spec_info = spec_info
        batch.return_hidden_states = False

    # ============================================================
    #  Hot path -- decode (forward_mode == DECODE -> TARGET_VERIFY)
    # ============================================================

    def draft(self, batch: ScheduleBatch) -> SpecBlockVerifyInput:
        """Build per-req trees via batched draft, then concat into one
        :class:`SpecBlockVerifyInput`.

        Reads cached b0_* / cross_kv / ttt_kv from
        ``batch.spec_info`` (a :class:`SpecBlockDraftInput`).
        Uses :func:`build_tree_batched` so the GPU-bottleneck block-2
        forward runs ONCE for all bs reqs (vs B sequential calls).

        Single-pass, GPU-only post-processing: parents → topology link
        list via ``build_retrieve_links_gpu``; per-req prefix mask via a
        batched ones+cat instead of Python for-loops; positions /
        seq_lens / sum stay GPU tensors (caller derives ints lazily).
        """
        from sglang.srt.speculative.specblock_tree_builder import (
            build_tree_batched,
        )
        from sglang.srt.speculative.specblock_tree_kernels import (
            build_retrieve_links_gpu,
            pad_tree_gpu,
        )

        bs = batch.batch_size()
        draft_model = self.model_runner.model.inner
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

        # Build paged cross_kv + ttt_kv as List[B] of List[L] of meta.
        # The list-comprehension is GPU-zero-cost (only Python references)
        # and feeds straight into build_tree_batched's batched forward.
        num_layers = self.num_layers
        cross_kv_cache_b: List[List[List]] = [
            [
                [self.spec_kv_pool, L, int(spec_info.cross_count[ridx]),
                 spec_info.cross_loc[ridx], None]
                for L in range(num_layers)
            ]
            for ridx in range(bs)
        ]
        ttt_kv_b: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [
            [
                (spec_info.ttt_k[ridx][L], spec_info.ttt_v[ridx][L])
                for L in range(num_layers)
            ]
            for ridx in range(bs)
        ]

        # CUDA graph fast path (B=1 only, when bucket fits).  Captures the
        # build_tree_gpu chain on first miss; replays subsequently.  ~50x
        # less kernel launch overhead vs eager build_tree_batched.
        runner = self.draft_cuda_graph_runner
        replay_tree = None
        if bs == 1 and runner.enabled:
            bucket = runner.resolve_buckets(spec_info, bs)
            if bucket is not None:
                if not runner.can_run(*bucket):
                    with self.spec_kv_pool.capture_mode():
                        # Pass spec_info so capture warmup uses real inputs
                        # (avoids triton attn OOB on all-zero static buffers).
                        runner.capture_one(*bucket, spec_info=spec_info)
                replay_tree = runner.replay(spec_info, *bucket)

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_setup = time.perf_counter()

        if replay_tree is not None:
            trees = [replay_tree]
        else:
            trees = build_tree_batched(
                draft_model=draft_model,
                block1_logits_b=spec_info.b0_logits,
                block1_rank_logits_b=spec_info.b0_rank_logits,
                block1_hidden_b=spec_info.b0_hidden,
                block1_ttt_kv_b=ttt_kv_b,
                initial_input_id_b=spec_info.b0_input_id,
                cross_kv_cache_b=cross_kv_cache_b,
                cross_position_b=list(spec_info.cross_position),
                K=self.K,
                max_blocks=self.max_blocks,
                beam_width=self.beam_width,
                total_tokens=self.total_tokens,
                rank_classes=self.rank_classes,
                d2t=getattr(draft_model, "d2t", None),
            )

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_build_tree = time.perf_counter()

        # Pad all trees to the same N for batched verify forward.  GPU
        # vectorised pad (no Python for-loop over rows).
        if bs > 1:
            N_max = max(int(t["draft_token"].shape[0]) for t in trees)
            trees = [pad_tree_gpu(t, N_max) for t in trees]
        N_full = int(trees[0]["draft_token"].shape[0])
        device = self.device

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_pad = time.perf_counter()

        # ---- Concat per-req trees into batched verify spec_info ----
        # Flat 1D fields (everything stays on GPU).
        draft_token_cat = torch.cat(
            [t["draft_token"] for t in trees], dim=0
        )  # [bs * N_full]
        depth_stack = torch.stack(
            [t["depth"].to(torch.int64) for t in trees], dim=0
        )  # [bs, N_full]
        # positions: each row's depth + that row's seq_lens.  No host sync.
        positions_t = (
            depth_stack + batch.seq_lens.to(torch.int64).unsqueeze(1)
        ).reshape(-1)  # [bs * N_full]
        tree_depth_cat = depth_stack.reshape(-1)
        tree_lps_cat = torch.cat(
            [t["cum_log_prob"] for t in trees], dim=0
        )

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_cat = time.perf_counter()

        # custom_mask: per req contributes (P_i + N_full) * N_full bools.
        # Per-req prefix length P_i differs, so ragged cat is unavoidable —
        # but each piece is a single GPU view with no host sync.
        seq_lens_cpu = batch.seq_lens.cpu()  # one async transfer; .tolist() below is the actual sync
        seq_lens_cpu_list = seq_lens_cpu.tolist()
        mask_chunks = []
        for ridx in range(bs):
            P_i = int(seq_lens_cpu_list[ridx])
            tree_mask = trees[ridx]["tree_mask"].bool()
            prefix_part = torch.ones(
                (N_full, P_i), dtype=torch.bool, device=device
            )
            full_2d = torch.cat([prefix_part, tree_mask], dim=1)
            mask_chunks.append(full_2d.reshape(-1))
        custom_mask = torch.cat(mask_chunks, dim=0).contiguous()

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _dp_t_mask = time.perf_counter()

        # retrive_index: row i = arange(N_full) + i * N_full.
        ar_n = torch.arange(N_full, dtype=torch.long, device=device)
        ar_b = torch.arange(bs, dtype=torch.long, device=device)
        retrive_index = (
            ar_n.unsqueeze(0).expand(bs, -1) + ar_b.unsqueeze(1) * N_full
        ).contiguous()

        # retrive_next_token / next_sibling: GPU-only batched conversion.
        # build_retrieve_links_gpu does the parents (raw) -> topo (root=0)
        # mapping inline + scatter_reduce.amin sibling pointers in one pass.
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
            tree_depth=tree_depth_cat,
            tree_lps=tree_lps_cat,
            tree_sizes_cpu=[N_full] * bs,
            hidden_states=spec_info.hidden_states,  # carry through 3H
            verified_id=spec_info.verified_id,
            seq_lens_sum=int(seq_lens_cpu.sum().item()),  # piggyback on the .cpu() above
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
                    "pad": 0.0, "cat": 0.0, "mask": 0.0, "tail": 0.0,
                }
                cls._DEEP_PROF_DRAFT = st
            st["n"] += 1
            st["setup"] += _dp_t_setup - _dp_t0
            st["build_tree"] += _dp_t_build_tree - _dp_t_setup
            st["pad"] += _dp_t_pad - _dp_t_build_tree
            st["cat"] += _dp_t_cat - _dp_t_pad
            st["mask"] += _dp_t_mask - _dp_t_cat
            st["tail"] += _dp_t_end - _dp_t_mask
            if st["n"] % 50 == 0:
                n = st["n"]
                logger.info(
                    "[DEEPprof:draft] n=%d setup=%.2fms build_tree=%.2fms "
                    "pad=%.2fms cat=%.2fms mask=%.2fms tail=%.2fms total=%.2fms",
                    n,
                    st["setup"]/n*1000, st["build_tree"]/n*1000,
                    st["pad"]/n*1000, st["cat"]/n*1000,
                    st["mask"]/n*1000, st["tail"]/n*1000,
                    (st["setup"]+st["build_tree"]+st["pad"]+st["cat"]+st["mask"]+st["tail"])/n*1000,
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
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int, List[int], bool]:
        """Run target verify forward then accept; return raw fields for
        the GenerationBatchResult assembly in forward_batch_generation.

        ``skip_prepare`` (V2 path): skip ``prepare_for_verify``; out_cache_loc
        and req_to_token mapping were already set up on plan-stream.
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

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )

        if _PROFILE:
            torch.cuda.synchronize()
            _t_vf_s = time.perf_counter()
        batch_result = self._target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
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

        # Force greedy verify regardless of T.  Reason: SpecBlock-Shift
        # tree branches via rank-class, so we don't expose per-token
        # vocab-wide draft probs to the SGLang sampling kernel.  Using
        # tree_speculative_sampling_target_only with draft_probs=0 then
        # rejects all spec tokens (accept_length collapses to 1).
        # Workaround: run greedy accept (target argmax matches draft).
        # Bonus token at last accepted pos is also target argmax (no T).
        # Output diversity at T>0 is therefore reduced -- known limitation.
        sampling_info = batch.sampling_info
        if sampling_info is not None and not sampling_info.is_all_greedy:
            sampling_info.is_all_greedy = True

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
    ):
        """Padded batched draft_model.update_cache_and_draft to seed iter k+1.

        Single forward call over all reqs (vs V1 per-req loop).  Pads
        per-req variable N (= accept_len+1) and cross_count to (max_N,
        max_cross), masks cross padding via cross_mask, and gathers per-req
        real-N_i last-K outputs via n_per_req.

        Pad slots in new_indices write to sentinel paged slot 0; caller
        never appends those to cross_loc, so future reads never see padded
        garbage.

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

        cross_loc_list = verify_info.cross_loc
        cross_count_list = verify_info.cross_count
        cross_position_list = verify_info.cross_position

        bs = len(accept_length_cpu)
        K = self.K

        Ns = [int(accept_length_cpu[i]) + 1 for i in range(bs)]
        max_N = max(Ns)
        max_cross = max(cross_count_list) if cross_count_list else 0

        # ---- Build padded inputs: hidden / tokens ----
        H3 = verified_h3.shape[-1]
        hidden_padded = verified_h3.new_zeros((bs, max_N, H3))
        tokens_padded = torch.zeros(
            bs, max_N, dtype=torch.long, device=self.device,
        )

        last_hidden_3h_list: List[torch.Tensor] = []
        next_token_list: List[torch.Tensor] = []
        offset = 0
        for ridx in range(bs):
            N_i = Ns[ridx]
            h_i = verified_h3[offset:offset + N_i]                # [N_i, 3H]
            id_i = verified_id[offset:offset + N_i].to(
                self.device, dtype=torch.long,
            )
            offset += N_i

            hidden_padded[ridx, :N_i] = h_i
            next_tok = id_i[-1:]  # [1]
            if N_i == 1:
                tok_seq = next_tok
            elif N_i == 2:
                tok_seq = torch.cat([next_tok, next_tok])
            else:
                mid = id_i[1:-1]
                tok_seq = torch.cat([mid, next_tok, next_tok])
            tokens_padded[ridx, :N_i] = tok_seq

            last_hidden_3h_list.append(h_i[-1])
            next_token_list.append(next_tok)

        # ---- Build padded cross_loc / cross_mask ----
        if max_cross > 0:
            cross_loc_padded = torch.zeros(
                bs, max_cross, dtype=torch.long, device=self.device,
            )
            cross_mask = torch.zeros(
                bs, max_cross, dtype=torch.bool, device=self.device,
            )
            for ridx in range(bs):
                c_i = cross_count_list[ridx]
                if c_i > 0:
                    cross_loc_padded[ridx, :c_i] = (
                        cross_loc_list[ridx].to(
                            self.device, dtype=torch.long,
                        )
                    )
                    cross_mask[ridx, :c_i] = True
        else:
            # No cross history yet (first decode step right after extend).
            cross_loc_padded = torch.zeros(
                bs, 0, dtype=torch.long, device=self.device,
            )
            cross_mask = None  # forward_batch falls into cache_count==0 path.

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _r_t_pad = time.perf_counter()

        # ---- Allocate new paged indices per-req; pad to slot 0 ----
        new_indices_padded = torch.zeros(
            bs, max_N * K, dtype=torch.long, device=self.device,
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

        # +1 to align with V1: V1 called update_cache_and_draft(... old_pos + 1).
        # Inside the model, base_pos = start_position + 1 + arange(N), so the
        # absolute token positions become old_pos + 2, ..., old_pos + 1 + N.
        # Matching V1 keeps cross_position semantics (caller stored old_pos + 1 + N).
        start_position = torch.tensor(
            [p + 1 for p in cross_position_list],
            dtype=torch.long, device=self.device,
        )
        n_per_req = torch.tensor(
            Ns, dtype=torch.long, device=self.device,
        )

        cache_i = [
            [
                self.spec_kv_pool,
                L_idx,
                max_cross,
                cross_loc_padded,
                new_indices_padded,
            ]
            for L_idx in range(self.num_layers)
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

        # ---- FlashInfer ragged cross plan (replaces padded matmul cross) ----
        # Plan ONCE per decode step, reused across all layers (cross_loc /
        # cross_counts are layer-invariant).  When max_cross == 0 (first decode
        # iter after extend), skip plan + fall through to padded path which
        # handles cache_count==0 cleanly.
        flashinfer_cross_arg = None
        if self.flashinfer_cross is not None and max_cross > 0:
            scale = draft_model.layers[0].self_attn.scale
            self.flashinfer_cross.plan_step(
                cross_loc_padded=cross_loc_padded,
                cross_counts=list(cross_count_list),
                K=max_N * K,
                scale=scale,
                device=self.device,
                dtype=draft_dtype,
            )
            flashinfer_cross_arg = self.flashinfer_cross

        if _DEEP_PROFILE:
            torch.cuda.synchronize()
            _r_t_plan = time.perf_counter()

        # ---- Single batched forward ----
        # max_position_hint = max(cross_position_list)+1 spares the GPU->CPU
        # sync that .max().item() would force inside update_cache_and_draft.
        max_position_hint = max(cross_position_list) + 1
        logits, rank_logits, draft_hidden, ttt_kv, new_pos = (
            draft_model.update_cache_and_draft(
                hidden_padded, tokens_padded, cache_i, start_position,
                cross_mask=cross_mask, n_per_req=n_per_req,
                flashinfer_cross=flashinfer_cross_arg,
                max_position_hint=max_position_hint,
            )
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

        new_pos_cpu = new_pos.tolist()
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
            new_cross_position.append(int(new_pos_cpu[ridx]))

        for layer_idx in range(self.num_layers):
            layer_k, layer_v = ttt_kv[layer_idx]  # [B, n_kv_heads, K, D]
            for ridx in range(bs):
                new_ttt_k[ridx].append(layer_k[ridx:ridx + 1])
                new_ttt_v[ridx].append(layer_v[ridx:ridx + 1])

        b0_input_id = torch.stack(next_token_list, dim=0)  # [B, 1]
        last_hidden_3h = torch.stack(last_hidden_3h_list, dim=0)  # [B, 3H]

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
                batch, verify_info, logits_output, verified_ids, accept_length_cpu
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
