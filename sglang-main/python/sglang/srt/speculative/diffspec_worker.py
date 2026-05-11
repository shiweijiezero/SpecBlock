"""
DiffSpec Standard Speculative Decoding Worker.

Following EAGLE3's strategy:
- Recursive multi-step draft generation
- Dynamic tree building with top-k selection
- Tree-based verification

Key differences from EAGLE3:
- Multi-condition tokens (M historical hidden states)
- 2-step iterative refinement (optional, can be disabled for standard mode)
- Uses historical hidden states only (not current position)
"""

import logging
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    detect_nan,
    draft_tp_context,
    fast_topk,
    load_token_map,
    select_top_k_tokens,
)
from sglang.srt.utils import (
    empty_context,
    get_available_gpu_memory,
    next_power_of_2,
)

logger = logging.getLogger(__name__)


class DiffSpecWorker(TpModelWorker):
    """
    DiffSpec Standard Worker - Following EAGLE3's decoding strategy.

    This is the standard version (diffspec_standard) that follows EAGLE3's approach:
    1. Recursive multi-step draft generation with top-k selection
    2. Dynamic tree building
    3. Tree-based verification with target model

    DiffSpec-specific features:
    - Multi-condition tokens (M=3 by default): uses [h_{i-3}, h_{i-2}, h_{i-1}]
    - Iterative refinement (optional): can refine predictions in multiple iterations
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
        enable_overlap: bool = False,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.enable_overlap = enable_overlap
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # DiffSpec-specific: number of condition tokens (M)
        # Will be updated from model config after loading
        self.num_condition_tokens = 3  # Default, overridden by model config

        # Override context length
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Disable cuda graph during init
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share allocator with target worker
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids if specified
        self.hot_token_id = None
        if server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )

        # Init draft worker
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

        # Share embedding from target model
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        # Check if model has hot_token_id (for vocab mapping)
        if hasattr(self.draft_model_runner.model, 'get_hot_token_id'):
            hot_token_id = self.draft_model_runner.model.get_hot_token_id()
            if hot_token_id is not None:
                self.hot_token_id = hot_token_id.to(embed.device)

        if self.hot_token_id is not None:
            head = head.clone()
            self.hot_token_id = self.hot_token_id.to(head.device)
            head.data = head.data[self.hot_token_id]

        # Set embedding and head
        if hasattr(self.draft_model_runner.model, 'set_embed_and_head'):
            self.draft_model_runner.model.set_embed_and_head(embed, head)
        elif hasattr(self.draft_model_runner.model, 'set_embed'):
            self.draft_model_runner.model.set_embed(embed)

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )

        with self.draft_tp_context(self.draft_model_runner.tp_group), speculative_moe_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        # Dummy tensors for page allocation
        self.num_new_pages_per_topk = torch.empty((), dtype=torch.int64, device=self.device)
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        logger.info(
            f"DiffSpec worker initialized: "
            f"topk={self.topk}, "
            f"speculative_num_steps={self.speculative_num_steps}, "
            f"speculative_num_draft_tokens={self.speculative_num_draft_tokens}"
        )

    def init_attention_backend(self):
        """Initialize attention backends for multi-step draft."""
        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend
        self.draft_extend_attn_backend = draft_backend_factory.create_draft_extend_backend()

        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    def init_cuda_graphs(self):
        """Capture cuda graphs (simplified - no graph capture for now)."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        # TODO: Implement CUDA graph capture for DiffSpec
        # For now, we run without CUDA graphs
        logger.info("DiffSpec: CUDA graph capture not implemented yet, running without graphs")

    @property
    def draft_model_runner(self):
        return self.model_runner

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """
        Run DiffSpec speculative decoding forward.

        Workflow:
        1. EXTEND: Target model processes input, then call forward_draft_extend to setup topk
        2. DECODE: Draft generates candidates, verify with target model
        """
        # Handle EXTEND mode (first generation, like prefill)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(batch)

            # Setup spec_info with topk from target model logits
            # This is needed for DiffSpec's initial candidate generation
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context():
                self.forward_draft_extend(
                    batch, logits_output, next_token_ids, seq_lens_cpu
                )

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            # Handle DECODE mode (subsequent generations)
            with self.draft_tp_context(self.draft_model_runner.tp_group), speculative_moe_backend_context():
                spec_info = self.draft(batch)

            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )

            # Update hidden states for next iteration
            with self.draft_tp_context(self.draft_model_runner.tp_group), speculative_moe_backend_context():
                if (
                    self.server_args.enable_dp_attention
                    or batch.spec_info.verified_id.shape[0] > 0
                ):
                    batch.last_hidden_states = logits_output.hidden_states
                    batch.last_next_token_ids = verify_output.verified_id
                    batch.last_logits_output = logits_output

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, torch.Tensor]:
        """
        Run target model extend (prefill).

        Captures full hidden states for draft model.
        """
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
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
        """
        Forward draft model after target extend.

        For DiffSpec:
        1. Get hidden states from target model
        2. Save hidden states and verified_id (Medusa pattern)

        Note: Unlike the previous approach, we don't save topk here.
        Instead, we regenerate topk in draft() from hidden_states to handle
        batch merging correctly (when multiple EXTEND batches are combined).
        """
        # Get hidden states from target model
        hidden_states = logits_output.hidden_states

        # Save to batch.last_hidden_states (Medusa pattern)
        # This ensures hidden states are available even after scheduler operations
        batch.last_hidden_states = hidden_states

        batch.return_hidden_states = False

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        """Preprocess for decode: allocate cache, setup positions."""
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        if self.page_size == 1:
            for req in batch.reqs:
                req.kv_allocated_len += self.speculative_num_steps * self.topk
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * self.speculative_num_steps * self.topk,
                backup_state=True,
            )
        else:
            # For page_size > 1
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            seq_lens = prefix_lens + self.speculative_num_steps * self.topk
            seq_lens_cpu = prefix_lens_cpu + self.speculative_num_steps * self.topk
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            out_cache_loc, token_to_kv_pool_state_backup = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                num_seqs * self.speculative_num_steps * self.topk,
                backup_state=True,
            )

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
        )

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

        # Restore allocator state (rollback the speculative allocation)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

        # Also restore kv_allocated_len (it was increased earlier, now roll it back)
        if self.page_size == 1:
            for req in batch.reqs:
                req.kv_allocated_len -= self.speculative_num_steps * self.topk

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        """Create idle input for empty batch."""
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self.model_config.hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """
        Generate draft tokens using iterative denoising.

        DiffSpec workflow:
        1. Regenerate topk from last_hidden_states (saved in EXTEND mode)
        2. For each step, DiffSpec model refines candidates
        3. Build dynamic tree from parent-child relationships
        """
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Get hidden_states and verified_id from last EXTEND
        hidden_states = batch.last_hidden_states
        verified_id = batch.input_ids

        # Create spec_info WITHOUT topk (will be generated by draft model's first forward)
        # Unlike EAGLE3, DiffSpec doesn't need topk from target model
        # The draft model will generate initial candidates in its first iteration
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=verified_id,
            num_tokens_per_batch=1,
            num_tokens_for_logprob_per_batch=1,
            topk_p=None,  # Will be generated by draft model
            topk_index=None,  # Will be generated by draft model
        )

        self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_batch = self.topk
        spec_info.num_tokens_for_logprob_per_batch = self.topk
        batch.return_hidden_states = False

        # Get forward batch
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST

        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.can_run_dp_cuda_graph = False

        if not forward_batch.forward_mode.is_idle() and self.speculative_num_steps > 1:
            self.draft_attn_backend.init_forward_metadata(forward_batch)

        # Run draft forward
        parent_list, top_scores_index, draft_tokens = self.draft_forward(forward_batch)

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree
        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        """
        DiffSpec draft forward - following EAGLE3's recursive strategy.

        For each step:
        1. Select top-k tokens from current candidates
        2. Run draft model forward
        3. Get new top-k predictions
        4. Track parent-child relationships for tree building
        """
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values for tree building
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            # Select top-k tokens and track tree info
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # Last step: don't need to run forward
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs for next forward
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run draft model forward
            logits_output, _ = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            )

            if self.enable_nan_detection:
                detect_nan(logits_output)

            # Get top-k for next step
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)

            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]

            hidden_states = logits_output.hidden_states

        # Organize results for tree building
        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        """Verify draft tokens using target model."""
        spec_info.prepare_for_verify(batch, self.page_size)
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        # Forward with target model
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        if self.enable_nan_detection:
            detect_nan(logits_output)

        spec_info.hidden_states = logits_output.hidden_states

        # Verify using tree verification
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask=None,
        )

        # Post process: pick accepted indices
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        # Prepare for next draft
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        """
        For DiffSpec, we don't need to do anything here.

        Unlike EAGLE3 which runs draft model after decode to prepare for next iteration,
        DiffSpec does all draft work in the draft() method during denoising iterations.
        This method is called from forward_batch_generation but DiffSpec doesn't need it.
        """
        pass

    def clear_cache_pool(self):
        """Clear cache pool (shared with target worker)."""
        pass
