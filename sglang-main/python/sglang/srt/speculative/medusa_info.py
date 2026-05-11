"""Medusa speculative decoding data structures.

Medusa uses parallel heads to predict multiple future tokens in a single forward pass.
Unlike EAGLE's recursive TTT approach, Medusa is simpler:
- No Transformer layers (num_hidden_layers=0)
- Fixed prediction pattern (4 parallel heads → tree structure)
- Tree-based verification (same kernel as EAGLE)

Architecture:
- MedusaDraftInput: Stores hidden states between iterations (similar to EagleDraftInput)
- MedusaVerifyInput: Tree structure for verification (similar to EagleVerifyInput)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
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
    assign_req_to_token_pool_func,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, next_power_of_2

_is_npu = is_npu()

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )

logger = logging.getLogger(__name__)


@dataclass
class MedusaDraftInput(SpecInput):
    """Draft input for Medusa speculative decoding.

    Similar to EagleDraftInput but simpler:
    - No topk_p/topk_index (Medusa generates directly from hidden states)
    - Only stores hidden states and verified_id for next draft generation

    This is maintained between iterations:
    1. After EXTEND: Created with initial hidden states
    2. After VERIFY: Updated with last token's hidden states
    3. Before next DECODE: Used to generate draft tokens
    """

    # Core tensors
    hidden_states: torch.Tensor = None  # (bs, hidden_size) - last token's hidden state per request
    verified_id: torch.Tensor = None     # (bs,) - last accepted token per request

    # For compatibility with spec_info interface
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST

    def __post_init__(self):
        super().__init__(SpecInputType.MEDUSA_DRAFT)
        if self.hidden_states is not None:
            self.device = self.hidden_states.device
        else:
            self.device = None

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """Return (max_draft_tokens, avg_draft_tokens)."""
        # For draft input, we don't generate tokens yet, so return 0
        return 0, 0

    @classmethod
    def create_idle_input(cls, device: str, hidden_size: int, dtype: torch.dtype):
        """Create an idle input for CUDA graph capture or empty batches."""
        return cls(
            hidden_states=torch.empty((0, hidden_size), device=device, dtype=dtype),
            verified_id=torch.empty((0,), device=device, dtype=torch.int32),
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        """Filter batch when some requests are finished.

        Args:
            new_indices: Indices of requests to keep
            has_been_filtered: If True, new_indices is just length (already filtered in verify)
                              If False, new_indices contains actual indices to select
        """
        if self.hidden_states is None:
            return

        if has_been_filtered:
            # Already filtered in verify, just truncate to new length
            self.hidden_states = self.hidden_states[: len(new_indices)]
            if self.verified_id is not None:
                self.verified_id = self.verified_id[: len(new_indices)]
        else:
            # Need to actually filter by indices
            self.hidden_states = self.hidden_states[new_indices]
            if self.verified_id is not None:
                self.verified_id = self.verified_id[new_indices]

    def merge_batch(self, spec_info: "MedusaDraftInput"):
        """Merge another batch into this one.

        Used when combining batches in the scheduler.
        """
        if spec_info is None:
            return

        # Merge hidden states
        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
        elif spec_info.hidden_states is not None:
            self.hidden_states = torch.cat(
                [self.hidden_states, spec_info.hidden_states], dim=0
            )

        # Merge verified_id
        if self.verified_id is None:
            self.verified_id = spec_info.verified_id
        elif spec_info.verified_id is not None:
            self.verified_id = torch.cat(
                [self.verified_id, spec_info.verified_id], dim=0
            )


@dataclass
class MedusaVerifyInput(SpecInput):
    """Verification input for Medusa tree-based speculative decoding.

    Now uses tree structure similar to EAGLE3:
    - Draft tokens arranged in tree (60 nodes with mc_sim_7b_60)
    - Tree attention mask for efficient verification
    - Top-k sampling from each Medusa head
    """

    # Core tensors (same as EAGLE3)
    draft_token: torch.Tensor  # (bs * draft_token_num,) flattened
    custom_mask: torch.Tensor  # Tree attention mask
    positions: torch.Tensor  # (bs * draft_token_num,) position IDs
    retrive_index: torch.Tensor  # (bs, draft_token_num) tree node indices
    retrive_next_token: torch.Tensor  # (bs, draft_token_num) next token indices
    retrive_next_sibling: torch.Tensor  # (bs, draft_token_num) sibling indices
    # Configuration
    num_heads: int  # Number of Medusa heads (typically 4)
    draft_token_num: int  # Total draft tokens (typically 60 for mc_sim_7b_60)
    topk: int  # Top-k per head (typically 10)
    # Optional fields with defaults (same as EAGLE3)
    capture_hidden_mode: object = None  # CaptureHiddenMode.LAST
    seq_lens_sum: int = 0  # Sum of sequence lengths
    seq_lens_cpu: torch.Tensor = None  # CPU copy of seq_lens
    # Hidden states for next draft generation
    hidden_states: torch.Tensor = None

    def __post_init__(self):
        super().__init__(SpecInputType.MEDUSA_VERIFY)
        if self.capture_hidden_mode is None:
            self.capture_hidden_mode = CaptureHiddenMode.LAST
        # Handle None draft_token during CUDA graph capture
        if self.draft_token is not None:
            self.device = self.draft_token.device
        else:
            self.device = None

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """Return (max_draft_tokens, avg_draft_tokens)."""
        return self.draft_token_num, self.draft_token_num

    @classmethod
    def create_idle_input(cls, topk: int, num_heads: int, num_verify_tokens: int):
        """Create an idle input for CUDA graph capture or empty batches."""
        if not _is_npu:
            device = "cuda"
        else:
            device = "npu"
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device=device),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device=device),
            positions=torch.empty((0,), dtype=torch.int64, device=device),
            retrive_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrive_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrive_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            num_heads=num_heads,
            topk=topk,
            draft_token_num=num_verify_tokens,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
        )

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        """Prepare KV cache allocation for verification phase.

        Completely reuses EAGLE3's logic for tree-based KV cache allocation.
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
            # Update kv_allocated_len for block mode (same as EAGLE)
            # This accounts for the verified_id that's prepended to draft_tokens
            for req in batch.reqs:
                req.kv_allocated_len += 1
        else:
            # Paged KV cache allocation (same as EAGLE3)
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

        # Call assign_req_to_token_pool to map draft tokens (same as Ngram)
        # This is needed for the verification forward pass
        # Use direct kernel call to avoid wrapper function issues
        bs = batch.batch_size()
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool
        import triton

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
        """Generate attention arguments for tree-based verification.

        Simplified implementation that avoids complex kernels.
        Uses pure PyTorch operations instead of Triton kernels.
        """
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        # QO indices: each sample has draft_token_num query tokens (tree structure)
        qo_indptr = torch.arange(
            0,
            (1 + bs) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        # KV indices: paged_kernel_lens + draft_token_num for each sample
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # Build kv_indices using Triton kernel (same as Ngram/EAGLE)
        total_kv_len = cum_kv_seq_len[-1].item()
        kv_indices = torch.empty(total_kv_len, dtype=torch.int32, device=device)

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,  # kv_start_idx
            kv_indices,
            req_to_token.size(1),
        )

        # Return custom_mask for tree attention (same as EAGLE)
        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def _greedy_verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
    ):
        """Greedy verification using tree kernel (completely reuses EAGLE3's implementation)."""
        bs = batch.batch_size()

        # Validate shapes before reshape
        expected_logits_len = bs * self.draft_token_num
        actual_logits_len = logits_output.next_token_logits.shape[0]

        if actual_logits_len != expected_logits_len:
            raise RuntimeError(
                f"Logits shape mismatch: expected {expected_logits_len} "
                f"(bs={bs} * draft_token_num={self.draft_token_num}), "
                f"got {actual_logits_len}"
            )

        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, self.draft_token_num)

        candidates = self.draft_token.reshape(bs, self.draft_token_num)

        # Prepare predict tensor - 1D array with length = bs * draft_token_num + bs
        # This will be filled by verify kernel with accepted token IDs
        # +1 for the next token position (same as EAGLE)
        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        self.predict = torch.empty(predict_shape, dtype=torch.int32, device=self.device)

        # Prepare accept tensors - same as EAGLE
        # CRITICAL: accept_index size should be num_speculative_steps, NOT num_nodes!
        # For Medusa, draft_token_num includes verified_id (+1), but speculative steps
        # only count new predictions (excluding verified_id).
        # So accept_index should be (bs, draft_token_num - 1) to avoid buffer overflow!
        num_spec_steps = self.draft_token_num - 1  # Exclude verified_id from spec steps
        self.accept_index = torch.full(
            (bs, num_spec_steps), -1, dtype=torch.int32, device=self.device
        )
        self.accept_length = torch.empty((bs,), dtype=torch.int32, device=self.device)


        # Use tree indices from worker (built using reconstruct_indices_from_tree_mask)
        # These are already in the correct format for verify_tree_greedy
        # EXACTLY same call signature as EAGLE3

        # Validate tree indices before calling kernel
        max_retrive_idx = self.retrive_index.max().item()
        max_next_token = self.retrive_next_token.max().item()
        max_next_sibling = self.retrive_next_sibling.max().item()
        max_valid = self.draft_token_num - 1

        if max_retrive_idx > bs * self.draft_token_num - 1:
            raise RuntimeError(
                f"Invalid retrive_index: max={max_retrive_idx}, "
                f"valid range=[0, {bs * self.draft_token_num - 1}]"
            )
        if max_next_token >= self.draft_token_num and max_next_token != -1:
            raise RuntimeError(
                f"Invalid retrive_next_token: max={max_next_token}, "
                f"valid range=[-1, {max_valid}]"
            )
        if max_next_sibling >= self.draft_token_num and max_next_sibling != -1:
            raise RuntimeError(
                f"Invalid retrive_next_sibling: max={max_next_sibling}, "
                f"valid range=[-1, {max_valid}]"
            )

        verify_tree_greedy_func(
            predicts=self.predict,  # mutable
            accept_index=self.accept_index,  # mutable
            accept_token_num=self.accept_length,  # mutable
            candidates=candidates,
            retrive_index=self.retrive_index,  # From tree building
            retrive_next_token=self.retrive_next_token,  # From tree building
            retrive_next_sibling=self.retrive_next_sibling,  # From tree building
            target_predict=target_predict,
            topk=self.topk,  # Must pass topk (same as EAGLE3)
        )

    def _sampling_verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ):
        """Sampling verification with temperature support.

        Fully implements tree_speculative_sampling_target_only like EAGLE.
        """
        bs = batch.batch_size()
        candidates = self.draft_token.reshape(bs, self.draft_token_num)

        # Prepare predict tensor (+1 for next token)
        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        self.predict = torch.empty(predict_shape, dtype=torch.int32, device=self.device)

        # Prepare accept tensors
        # Shape is (bs, draft_token_num) to match retrive_index dimensions
        self.accept_index = torch.full(
            (bs, self.draft_token_num), -1, dtype=torch.int32, device=self.device
        )
        self.accept_length = torch.empty((bs,), dtype=torch.int32, device=self.device)

        # Apply temperature and get target probs
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, self.draft_token_num, dim=0
        )  # (bs * draft_token_num, 1)

        target_probs = F.softmax(
            logits_output.next_token_logits / expanded_temperature, dim=-1
        )  # (bs * draft_token_num, vocab_size)

        # Apply top-k renormalization
        target_probs = top_k_renorm_prob(
            target_probs,
            torch.repeat_interleave(
                sampling_info.top_ks, self.draft_token_num, dim=0
            ),
        )  # (bs * draft_token_num, vocab_size)

        # Apply top-p renormalization if needed
        if not torch.all(sampling_info.top_ps == 1.0):
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )

        target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

        # Medusa doesn't have draft probs (no probability from draft model)
        # Use zeros like EAGLE's target-only mode
        draft_probs = torch.zeros(
            target_probs.shape, dtype=torch.float32, device=self.device
        )

        # Coins for rejection sampling
        coins = torch.rand_like(
            candidates, dtype=torch.float32, device=self.device
        )
        # Coins for final sampling
        coins_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=self.device
        )

        # Call tree speculative sampling kernel
        tree_speculative_sampling_target_only(
            predicts=self.predict,  # mutable
            accept_index=self.accept_index,  # mutable
            accept_token_num=self.accept_length,  # mutable
            candidates=candidates,
            retrive_index=self.retrive_index,
            retrive_next_token=self.retrive_next_token,
            retrive_next_sibling=self.retrive_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
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
    ):
        """Fill accepted tokens into request output_ids and filter logits.

        Following EAGLE's pattern exactly:
        1. Use 2D accept_index to iterate
        2. Use predict[idx] to get token id (idx is value in accept_index)
        3. Flatten accept_index at the end
        """
        bs = batch.batch_size()
        has_finished = False

        # Convert to CPU for iteration - same as EAGLE
        accept_index_cpu = self.accept_index.tolist()
        predict_cpu = self.predict.tolist()

        # Debug: validate indices
        max_accept_idx = max(
            (idx for row in accept_index_cpu for idx in row if idx != -1),
            default=-1
        )
        if max_accept_idx >= len(predict_cpu):
            raise RuntimeError(
                f"accept_index out of range: max index={max_accept_idx}, "
                f"but predict length={len(predict_cpu)}. "
                f"accept_index shape={self.accept_index.shape}, "
                f"draft_token_num={self.draft_token_num}, "
                f"bs={bs}"
            )

        # Iterate every accepted token and check if req has finished after appending the token
        # Same pattern as EAGLE
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                token_id = predict_cpu[idx]
                req.output_ids.append(token_id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    self.accept_index[i, j + 1:] = -1
                    break
                else:
                    if req.grammar is not None:
                        req.grammar.accept_token(token_id)
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += (
                sum(1 for idx in accept_index_row if idx != -1) - 1
            )

        if has_finished:
            self.accept_length = (self.accept_index != -1).sum(dim=1) - 1

        # Flatten accept_index - same as EAGLE
        self.accept_index = self.accept_index[self.accept_index != -1]

        # Filter logits and hidden states using flattened accept_index
        logits_output.next_token_logits = logits_output.next_token_logits[
            self.accept_index
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                self.accept_index
            ]

        # Set verified_id using predict and accept_index - same as EAGLE
        self.verified_id = self.predict[self.accept_index]

    def _free_cache(
        self, batch: ScheduleBatch, page_size: int, accept_length_cpu: torch.Tensor
    ):
        """Free KV cache for unaccepted tokens.

        Follows Ngram's implementation with separate handling for block and paged modes.
        """
        bs = batch.batch_size()

        # Free the KV cache for unaccepted tokens (same as Ngram)
        if page_size == 1:
            # Block mode: simple boolean masking
            evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
            evict_mask[self.accept_index] = False
            batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            batch.out_cache_loc = batch.out_cache_loc[self.accept_index]
        else:
            # Paged mode: use get_src_tgt_cache_loc and get_target_cache_loc kernels
            from sglang.srt.speculative.spec_utils import get_src_tgt_cache_loc, get_target_cache_loc
            from sglang.srt.utils import next_power_of_2

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

            # Free the kv cache
            batch.token_to_kv_pool_allocator.free(to_free_slots)

            # Copy the kv cache
            batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                tgt_cache_loc, src_cache_loc
            )
            batch.out_cache_loc = tgt_cache_loc

        # Update request KV lengths
        accept_length_list = accept_length_cpu.tolist()
        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_length_list[i] + 1
            req.kv_allocated_len = req.kv_committed_len

        # Update req_to_token_pool mapping AFTER filtering out_cache_loc
        # Use direct kernel call like Ngram (not the wrapper function)
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool
        import triton

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
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int]:
        """Verify draft tokens and compute acceptance.

        Following EAGLE's pattern:
        1. Apply logit processors and penalties
        2. Greedy or sampling verification
        3. Fill accepted tokens into requests (this also flattens accept_index)
        4. Free KV cache
        5. Update sequence lengths

        Returns:
            (logits_output, verified_ids, num_accepted_tokens)
        """
        bs = batch.batch_size()
        sampling_info = batch.sampling_info

        # Handle batch size mismatch - must filter sampling_info!
        # This can happen when some requests finish and batch is filtered
        if bs != len(sampling_info):
            sampling_info = deepcopy(sampling_info)
            # Create indices for filtering: 0, 1, 2, ..., bs-1
            # This assumes batch has already been filtered by scheduler
            indices = list(range(bs))
            sampling_info.filter_batch(indices, indices)

        # Apply custom logit processors if registered
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        # Apply penalty
        if (
            sampling_info.penalizer_orchestrator.is_required
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

        # Apply grammar mask
        if vocab_mask is not None:
            # Grammar masking for Medusa (if needed)
            pass

        # Choose verification strategy
        is_all_greedy = sampling_info.is_all_greedy
        if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
            logger.warning(
                "Tree speculative sampling kernel unavailable (likely AMD/HIP build). "
                "Falling back to greedy verification."
            )

        if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE or _is_npu:
            self._greedy_verify(batch, logits_output)
        else:
            # Sampling verification with temperature support
            self._sampling_verify(batch, logits_output, sampling_info)

        # Fill accepted tokens and filter logits
        # This also flattens accept_index and sets verified_id
        self._fill_requests(batch, logits_output)

        # Get accept_length_cpu before any modifications
        accept_length_cpu = self.accept_length.cpu()
        num_accepted_tokens = accept_length_cpu.sum().item()

        # Free KV cache for unaccepted tokens
        self._free_cache(batch, page_size, accept_length_cpu)

        # Update sequence lengths (accept_length accepted drafts + 1 next token)
        batch.seq_lens.add_(self.accept_length + 1)
        batch.seq_lens_cpu.add_(accept_length_cpu + 1)

        return logits_output, self.verified_id, num_accepted_tokens

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        """Filter batch when some requests are finished.

        Similar to EAGLE's implementation but simpler since Medusa doesn't have
        complex draft input state.
        """
        if self.hidden_states is not None:
            if has_been_filtered:
                self.hidden_states = self.hidden_states[: len(new_indices)]
            else:
                self.hidden_states = self.hidden_states[new_indices]

        if self.draft_token is not None:
            # Reshape to (bs, draft_token_num) for filtering
            if len(self.draft_token) > 0:
                bs = len(new_indices)
                old_bs = len(self.draft_token) // self.draft_token_num
                if old_bs > bs:
                    draft_token_2d = self.draft_token.reshape(old_bs, self.draft_token_num)
                    if has_been_filtered:
                        self.draft_token = draft_token_2d[:bs].flatten()
                    else:
                        self.draft_token = draft_token_2d[new_indices].flatten()

        # DO NOT filter tree indices - they contain global flattened indices that become invalid after filtering!
        # The tree indices will be rebuilt fresh in medusa_worker._build_tree_from_tokens() on next iteration.
        # This matches Ngram's approach which also rebuilds indices every iteration.

    def merge_batch(self, spec_info: MedusaVerifyInput):
        """Merge another batch into this one.

        NOTE: MedusaVerifyInput is recreated for each forward pass, so merge_batch
        should rarely be called. Unlike EagleDraftInput which accumulates state,
        MedusaVerifyInput is ephemeral.

        We only merge hidden_states for compatibility with the scheduler.
        Tree-related fields should NOT be merged as they're batch-specific.
        """
        if spec_info is None:
            return

        # Only merge hidden states (state between iterations)
        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
        elif spec_info.hidden_states is not None:
            self.hidden_states = torch.cat(
                [self.hidden_states, spec_info.hidden_states], dim=0
            )

        # NOTE: Do NOT merge tree-related fields (retrive_index, draft_token, positions, custom_mask)
        # These are specific to each batch's tree structure and should not be concatenated
        # Each batch should maintain its own tree structure independently
