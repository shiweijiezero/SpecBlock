"""Medusa speculative decoding worker.

Medusa uses multiple parallel heads to predict future tokens in a single forward pass.
This is simpler than EAGLE's recursive TTT approach:
- No iterative expansion
- Fixed draft pattern (num_heads tokens)
- Simpler verification (greedy matching)
"""

import logging
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.medusa_choices import get_medusa_choices
from sglang.srt.speculative.medusa_info import MedusaVerifyInput
from sglang.srt.speculative.medusa_utils import (
    build_medusa_parent_and_scores,
    get_num_draft_tokens_from_choices,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class MedusaWorker(TpModelWorker):
    """
    Medusa speculative decoding worker.

    Inherits from TpModelWorker like EAGLEWorker to properly handle model loading
    and dtype conversion.

    Workflow:
    1. Draft phase: Run draft model (Medusa heads) to predict num_heads future tokens
    2. Verify phase: Run target model to verify draft predictions
    3. Accept phase: Greedily match draft tokens with target predictions

    Key differences from EAGLE:
    - No TTT recursive expansion
    - Fixed number of draft tokens (= num_heads)
    - Simpler tree structure (linear sequence)
    - Faster draft generation (single forward pass)
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
        # Parse arguments
        self.server_args = server_args
        self.speculative_num_steps = server_args.speculative_num_steps
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Pre-load config to get medusa_num_heads BEFORE super().__init__()
        # This ensures server_args.speculative_num_draft_tokens is correct for CUDA graph capture
        from transformers import AutoConfig
        draft_config = AutoConfig.from_pretrained(
            server_args.speculative_draft_model_path,
            trust_remote_code=server_args.trust_remote_code,
        )
        model_num_heads = getattr(draft_config, "medusa_num_heads", 4)

        # Load Medusa tree structure configuration
        self.medusa_choices = get_medusa_choices(server_args.medusa_choices)
        self.topk = server_args.medusa_topk
        self.num_draft_tokens = get_num_draft_tokens_from_choices(server_args.medusa_choices)

        # Validate that medusa_choices is compatible with topk
        max_topk_idx = max(choice[-1] for choice in self.medusa_choices)
        if max_topk_idx >= self.topk:
            raise ValueError(
                f"medusa_choices '{server_args.medusa_choices}' requires topk >= {max_topk_idx + 1}, "
                f"but got topk={self.topk}. "
                f"For topk=1, use 'mc_linear_4_topk1' instead of 'mc_linear_4'."
            )

        logger.info(
            f"Medusa tree configuration: {server_args.medusa_choices}, "
            f"num_heads={model_num_heads}, topk={self.topk}, draft_tokens={self.num_draft_tokens}, "
            f"max_topk_idx={max_topk_idx}"
        )

        # Override server_args.speculative_num_draft_tokens to match tree size + 1
        # This is critical for CUDA graph capture
        # NOTE: build_tree_kernel_efficient prepends verified_id to draft_tokens,
        # so actual input_ids length is (num_draft_tokens + 1) * bs
        # This matches EAGLE's behavior when topk=1 (see server_args.py:1637-1642)
        actual_num_tokens = self.num_draft_tokens + 1
        if server_args.speculative_num_draft_tokens != actual_num_tokens:
            logger.warning(
                f"Overriding --speculative-num-draft-tokens from {server_args.speculative_num_draft_tokens} "
                f"to {actual_num_tokens} (num_draft_tokens={self.num_draft_tokens} + 1 for verified_id)"
            )
            server_args.speculative_num_draft_tokens = actual_num_tokens

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later if needed.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with target worker.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Init draft worker - this loads the Medusa model properly
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

        # Restore cuda graph setting
        server_args.disable_cuda_graph = backup_disable_cuda_graph

        # Get Medusa-specific parameters from the loaded model
        self.draft_model = self.model_runner.model
        # For Medusa, num_heads is determined by the model architecture, not user config
        # The model's medusa_num_heads defines how many parallel heads are available
        self.num_heads = model_num_heads

        self.draft_vocab_size = getattr(
            self.draft_model.config,
            "draft_vocab_size",
            self.draft_model.config.vocab_size,
        )

        # Share embedding with target model
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        self.draft_model.set_embed_and_head(embed, head)

        # Configure target model to return hidden states
        self._configure_hidden_state_capture()

        # Preallocate tensors for draft generation
        self.max_batch_size = target_worker.max_running_requests
        self._init_preallocated_tensors()

        # Build static tree indices for all possible batch sizes
        self._build_tree_indices()

        logger.info(
            f"Initialized MedusaWorker with {self.num_heads} heads, "
            f"topk={self.topk}, draft_tokens={self.num_draft_tokens}, "
            f"draft_vocab_size={self.draft_vocab_size}"
        )

    def _configure_hidden_state_capture(self):
        """Configure target model to return hidden states for Medusa.

        Original Medusa only uses the last layer's hidden states.
        No auxiliary layers needed (unlike EAGLE3).
        """
        # Original Medusa: Only need last layer hidden states
        # No need to call set_eagle3_layers_to_capture()
        self.use_aux_layers = False
        logger.info(
            "Medusa uses only last layer hidden states from target model "
            "(original Medusa design, no auxiliary layers)"
        )

    def _init_preallocated_tensors(self):
        """Preallocate tensors for draft generation."""
        # Preallocate for different batch sizes (tree-based, so use num_draft_tokens)
        # num_nodes = num_draft_tokens + 1 (including verified_id)
        num_nodes = self.num_draft_tokens + 1

        self.draft_tokens_batch = {}
        self.positions_batch = {}
        self.retrive_index_batch = {}
        self.retrive_next_token_batch = {}
        self.retrive_next_sibling_batch = {}

        for bs in range(1, self.max_batch_size + 1):
            total_draft = bs * self.num_draft_tokens
            total_nodes = bs * num_nodes

            self.draft_tokens_batch[bs] = torch.zeros(
                total_draft, dtype=torch.int64, device=self.device
            )
            self.positions_batch[bs] = torch.zeros(
                total_nodes, dtype=torch.int64, device=self.device
            )
            # CRITICAL: Initialize tree indices with -1 (invalid marker)
            # These will be filled by reconstruct_indices_from_tree_mask
            self.retrive_index_batch[bs] = torch.full(
                (bs, num_nodes), -1, dtype=torch.int64, device=self.device
            )
            self.retrive_next_token_batch[bs] = torch.full(
                (bs, num_nodes), -1, dtype=torch.int64, device=self.device
            )
            self.retrive_next_sibling_batch[bs] = torch.full(
                (bs, num_nodes), -1, dtype=torch.int64, device=self.device
            )

    def _build_tree_indices(self):
        """
        Build static tree structure from medusa_choices.

        Unlike EAGLE3's dynamic tree building, Medusa's tree is predetermined.
        We precompute the tree_mask once during initialization and reuse it.
        """
        # Calculate tree depth (max depth of any choice)
        # For mc_linear_4: [[0], [1], [2], [3]] -> depth = 1
        # For mc_sim_7b_60: [[0], [0,0], [1], [0,1], ...] -> depth = max(len(choice))
        self.tree_depth = max(len(choice) for choice in self.medusa_choices)

        # Generate static tree mask (this is the KEY difference from EAGLE3!)
        # Shape: (num_draft_tokens+1, num_draft_tokens+1) for single batch
        # This mask is shared across all batches
        from sglang.srt.speculative.medusa_utils import generate_medusa_tree_mask
        tree_mask_single = generate_medusa_tree_mask(
            self.medusa_choices,
            device=self.device,
        )
        self.tree_mask_template = tree_mask_single  # Shape: (num_nodes, num_nodes)

        # For tree verification, we also need token indices
        # Build mapping from choice to token index for draft token selection
        sorted_choices = sorted(self.medusa_choices, key=lambda x: (len(x), x))
        self.sorted_choices = sorted_choices
        self.choice_to_idx = {tuple(choice): i for i, choice in enumerate(sorted_choices)}

        logger.info(
            f"Built Medusa static tree: tree_depth={self.tree_depth}, "
            f"tree_mask shape {self.tree_mask_template.shape}, "
            f"{self.num_draft_tokens} draft nodes (+1 verified node)"
        )

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        """Generate draft tokens and prepare for verification using tree structure."""
        bs = batch.batch_size()

        # Generate draft tokens using tree structure (similar to EAGLE)
        draft_tokens, tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling = \
            self._generate_draft_tokens_tree(batch)

        # Switch to verify mode
        batch.spec_algorithm = SpeculativeAlgorithm.MEDUSA
        batch.forward_mode = ForwardMode.TARGET_VERIFY

        # Create MedusaVerifyInput with tree indices (similar to NgramVerifyInput)
        # CRITICAL: draft_tokens contains (bs * (num_draft_tokens + 1)) elements (verified_id prepended),
        # but draft_token_num should be num_draft_tokens (NOT +1) to match Ngram/EAGLE3 semantics!
        # The +1 verified_id is already included in draft_tokens, but draft_token_num describes
        # the tree structure size for kernel indexing purposes.
        batch.spec_info = MedusaVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            num_heads=self.num_heads,
            draft_token_num=self.num_draft_tokens + 1,  # Total nodes including verified_id
            topk=self.topk,
            seq_lens_sum=batch.seq_lens_sum,  # Required by EAGLE kernel
            seq_lens_cpu=batch.seq_lens_cpu,  # Required by get_model_worker_batch
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def _generate_draft_tokens_tree(self, batch: ScheduleBatch):
        """
        Generate draft tokens using Medusa tree structure.

        Similar to EAGLE's approach but simpler:
        - No TTT recursive expansion (single forward pass)
        - Use predefined tree structure (medusa_choices)
        - Top-k sampling from each head

        Returns:
            Tuple of (draft_tokens, tree_mask, positions, retrive_index,
                     retrive_next_token, retrive_next_sibling)
        """
        bs = batch.batch_size()

        # Check if hidden states are available
        if not hasattr(batch, "last_hidden_states") or batch.last_hidden_states is None:
            # Fallback: use zero tokens
            draft_tokens = torch.zeros(
                bs * self.num_draft_tokens, dtype=torch.int64, device=self.device
            )
            # Build tree with zero tokens (will likely be rejected, but prevents crash)
            return self._build_tree_from_tokens(batch, draft_tokens)

        hidden_states = batch.last_hidden_states

        # Validate hidden states shape
        if hidden_states.shape[0] != bs:
            raise RuntimeError(
                f"Hidden states batch size mismatch: expected {bs}, got {hidden_states.shape[0]}"
            )

        # Generate logits from all Medusa heads
        from sglang.srt.speculative.spec_utils import draft_tp_context
        from sglang.srt.layers.moe.utils import speculative_moe_backend_context

        with draft_tp_context(self.model_runner.tp_group), speculative_moe_backend_context():
            medusa_logits = self.draft_model.compute_logits(hidden_states)
            # Shape: (num_heads, bs, vocab_size)

        # Get top-k candidates from each head
        # Shape: (num_heads, bs, topk)
        topk_tokens = torch.topk(medusa_logits, self.topk, dim=-1).indices

        # Select tokens based on medusa_choices
        draft_tokens = self._select_tokens_from_choices(topk_tokens, bs)

        # Build tree structure using EAGLE's kernel
        return self._build_tree_from_tokens(batch, draft_tokens)

    def _select_tokens_from_choices(self, topk_tokens: torch.Tensor, bs: int) -> torch.Tensor:
        """
        Select draft tokens based on medusa_choices and topk candidates.

        Args:
            topk_tokens: Tensor of shape (num_heads, bs, topk) with top-k token IDs
            bs: Batch size

        Returns:
            draft_tokens: Tensor of shape (bs, num_draft_tokens) with selected token IDs
        """
        sorted_choices = sorted(self.medusa_choices, key=lambda x: (len(x), x))
        draft_tokens = torch.zeros(
            (bs, self.num_draft_tokens), dtype=torch.int64, device=self.device
        )

        for choice_idx, choice in enumerate(sorted_choices):
            # For each choice, select the corresponding token from the appropriate head's topk
            # In Medusa, each head predicts a different future position:
            # - Head 0 predicts position i+1 (depth 1)
            # - Head 1 predicts position i+2 (depth 2)
            # - Head 2 predicts position i+3 (depth 3)
            # - Head 3 predicts position i+4 (depth 4)
            #
            # The choice path [a, b, c] means:
            # - depth 1: select topk[a] from head 0
            # - depth 2: select topk[b] from head 1
            # - depth 3: select topk[c] from head 2
            #
            # For the tree node at depth d, we select from head (d-1)

            depth = len(choice)
            head_idx = min(depth - 1, self.num_heads - 1)
            topk_idx = choice[-1]  # The last element is the topk index for this depth

            tokens = topk_tokens[head_idx, :, topk_idx]  # (bs,)

            draft_tokens[:, choice_idx] = tokens

        return draft_tokens.flatten()  # (bs * num_draft_tokens,)

    def _build_tree_from_tokens(self, batch: ScheduleBatch, draft_tokens: torch.Tensor):
        """
        Build tree structure from draft tokens using precomputed tree mask.

        Unlike EAGLE3, Medusa's tree is static, so we use the precomputed mask
        and reconstruct indices from it.

        Args:
            batch: Current batch
            draft_tokens: Draft tokens (bs * num_draft_tokens,)

        Returns:
            Tuple of (draft_tokens_with_verified, tree_mask, positions, retrive_index,
                     retrive_next_token, retrive_next_sibling)
        """
        bs = batch.batch_size()
        num_nodes = self.num_draft_tokens + 1  # +1 for verified_id

        # Get verified_id (last accepted token from each request)
        verified_id = torch.zeros(bs, dtype=torch.int64, device=self.device)
        for i, req in enumerate(batch.reqs):
            if req.output_ids:
                verified_id[i] = req.output_ids[-1]
            else:
                # Use last input token if no output yet
                verified_id[i] = req.origin_input_ids[-1]

        # Prepend verified_id to draft_tokens
        # draft_tokens: (bs, num_draft_tokens) -> (bs, num_nodes)
        draft_tokens_2d = draft_tokens.reshape(bs, self.num_draft_tokens)
        draft_tokens_with_verified = torch.cat(
            [verified_id.unsqueeze(1), draft_tokens_2d], dim=1
        )  # Shape: (bs, num_nodes)
        draft_tokens_out = draft_tokens_with_verified.flatten()  # (bs * num_nodes,)

        # Expand tree_mask for current batch
        # tree_mask_template: (num_nodes, num_nodes), dtype=bool
        # Need to flatten and repeat for batch
        # Kernel expects: (bs * num_nodes * num_nodes,) bool tensor
        tree_mask_single_flat = self.tree_mask_template.flatten()  # (num_nodes^2,)
        tree_mask_batched = tree_mask_single_flat.unsqueeze(0).expand(bs, -1).contiguous()  # (bs, num_nodes^2)
        tree_mask_batched = tree_mask_batched.flatten()  # (bs * num_nodes^2,)

        # Ensure it's contiguous and correct dtype
        if not tree_mask_batched.is_contiguous():
            tree_mask_batched = tree_mask_batched.contiguous()
        assert tree_mask_batched.dtype == torch.bool, f"tree_mask must be bool, got {tree_mask_batched.dtype}"

        # Debug: check tree_mask shape
        expected_mask_len = bs * num_nodes * num_nodes
        if tree_mask_batched.shape[0] != expected_mask_len:
            raise RuntimeError(
                f"tree_mask_batched shape mismatch! Expected {expected_mask_len} (bs={bs}, num_nodes={num_nodes}), "
                f"got {tree_mask_batched.shape[0]}"
            )

        # Use reconstruct_indices_from_tree_mask to get tree indices
        from sgl_kernel import reconstruct_indices_from_tree_mask

        # Use preallocated buffers (like Ngram)
        # These buffers are initialized with -1 during __init__
        # and reused across iterations to avoid GPU memory allocation overhead
        seq_lens = batch.seq_lens  # (bs,)
        positions = self.positions_batch[bs]
        retrive_index = self.retrive_index_batch[bs]
        retrive_next_token = self.retrive_next_token_batch[bs]
        retrive_next_sibling = self.retrive_next_sibling_batch[bs]

        # Clear positions to 0 before kernel writes
        positions.zero_()
        # Tree indices are already -1 from initialization, but let's ensure it
        retrive_index.fill_(-1)
        retrive_next_token.fill_(-1)
        retrive_next_sibling.fill_(-1)

        # Call kernel to reconstruct indices
        # CRITICAL: Pass num_nodes (including verified_id) as draft_token_num parameter!
        # The kernel expects the full tree size, not just draft tokens.
        # This matches our tree_mask_batched size: (bs * num_nodes * num_nodes)
        reconstruct_indices_from_tree_mask(
            tree_mask_batched,  # (bs * num_nodes * num_nodes,)
            seq_lens,  # (bs,)
            positions,  # (bs * num_nodes,) - output
            retrive_index,  # (bs, num_nodes) - output
            retrive_next_token,  # (bs, num_nodes) - output
            retrive_next_sibling,  # (bs, num_nodes) - output
            bs,
            num_nodes,  # Total nodes including verified_id
        )

        return draft_tokens_out, tree_mask_batched, positions, retrive_index, retrive_next_token, retrive_next_sibling

    def forward_batch_generation(self, batch: ScheduleBatch):
        """Main entry point for Medusa speculative decoding.

        Handles EXTEND (first generation) and DECODE (subsequent generations) modes.
        Similar to EAGLE3's architecture.
        """
        # Handle EXTEND mode (first generation, like prefill)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Run target model to get hidden states
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

            # Store hidden states for next decode step
            # CaptureHiddenMode.LAST returns the last token's hidden state for each request
            # Shape should be (bs, hidden_size)
            batch.last_hidden_states = batch_result.logits_output.hidden_states
            return batch_result

        # Handle DECODE mode (subsequent generations)
        if batch.forward_mode.is_decode():
            self._prepare_for_speculative_decoding(batch)
            # Now batch.forward_mode is TARGET_VERIFY
        elif batch.forward_mode.is_target_verify():
            # CRITICAL: If forward_mode is already TARGET_VERIFY (e.g., CUDA graph reusing batch),
            # we must still regenerate spec_info! Otherwise we'll use stale retrive_index
            # with garbage values, causing illegal memory access.
            self._prepare_for_speculative_decoding(batch)

        # Handle TARGET_VERIFY mode
        if batch.forward_mode.is_target_verify():
            return self._verify_and_accept(batch)

        # Fallback for other modes
        model_worker_batch = batch.get_model_worker_batch()
        return self.target_worker.forward_batch_generation(model_worker_batch)

    def _verify_and_accept(self, batch: ScheduleBatch):
        """Verify draft tokens with target model and accept matching ones.

        Similar to Ngram worker's implementation:
        1. Run target model with is_verify=True
        2. Call verify() which handles everything (accept + sample next token)
        3. Return the result directly
        """
        # Convert ScheduleBatch to ModelWorkerBatch
        model_worker_batch = batch.get_model_worker_batch()

        # Run target model forward pass with is_verify=True
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output = batch_result.logits_output
        can_run_cuda_graph = batch_result.can_run_cuda_graph

        # Verify and compute acceptance
        # verify() handles:
        # 1. Greedy/sampling verification of draft tokens
        # 2. Fill accepted tokens into req.output_ids
        # 3. Sample next token and add to req.output_ids
        # 4. Free unused KV cache
        # 5. Filter logits to next token position
        # Returns: (logits_output, verified_ids, num_accepted)
        # Note: verified_ids contains ALL tokens (accepted drafts + next token)
        logits_output, next_token_ids, num_accepted = batch.spec_info.verify(
            batch, logits_output, self.page_size
        )

        # Process logprobs if requested
        if batch.return_logprob:
            self._process_logprobs(batch, logits_output, next_token_ids)

        # Store hidden states for next draft generation
        # logits_output.hidden_states is (num_accepted_total, hidden_size) after _fill_requests
        # We need to extract the last token's hidden state for each request
        if hasattr(logits_output, "hidden_states") and logits_output.hidden_states is not None:
            bs = batch.batch_size()
            accept_length_cpu = batch.spec_info.accept_length.cpu().tolist()

            # Calculate cumulative indices to find the last token for each request
            # Each request has (accept_length + 1) tokens in verified_id
            last_token_indices = []
            cumsum = 0
            for acc_len in accept_length_cpu:
                # Last token index for this request is cumsum + acc_len
                # (acc_len accepted drafts + 1 next token - 1 for 0-indexing)
                last_token_indices.append(cumsum + acc_len)
                cumsum += acc_len + 1

            last_token_indices = torch.tensor(
                last_token_indices, dtype=torch.int64, device=self.device
            )
            # Store in both places for compatibility:
            # - batch.last_hidden_states: for _generate_draft_tokens_tree
            # - batch.spec_info.hidden_states: for filter_batch/merge_batch
            last_hidden = logits_output.hidden_states[last_token_indices]
            batch.last_hidden_states = last_hidden
            batch.spec_info.hidden_states = last_hidden

        # Switch back to DECODE mode for next iteration
        batch.forward_mode = ForwardMode.DECODE

        # CRITICAL: Speculative verification cannot use CUDA graph because:
        # 1. Tree structure changes between iterations
        # 2. Batch size may vary (requests completing)
        # 3. Accept/reject decisions affect control flow
        # Always return can_run_cuda_graph=False for verification passes
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted,
            can_run_cuda_graph=False,  # Disable CUDA graph for speculative verification
        )

    def _process_logprobs(
        self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput, verified_ids: torch.Tensor
    ):
        """Process log probabilities for accepted tokens.

        Similar to Ngram's implementation but adapted for Medusa's structure.
        """
        from sglang.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs

        bs = batch.batch_size()
        accept_length_per_req_cpu = batch.spec_info.accept_length.cpu().tolist()

        # num_tokens_per_req includes accepted tokens + 1 next token
        num_tokens_per_req = [accept + 1 for accept in accept_length_per_req_cpu]

        # Get sampling info
        top_logprobs_nums = [
            req.top_logprobs_num if req.return_logprob else 0
            for req in batch.reqs
        ]
        token_ids_logprobs = [
            req.token_ids_logprob if req.return_logprob else None
            for req in batch.reqs
        ]

        # Compute temperature-adjusted logprobs
        # Note: logits_output.next_token_logits has been filtered to (bs, vocab_size)
        # We need to expand it for all accepted + next tokens
        temperatures = batch.sampling_info.temperatures

        # For accepted tokens, we need to use the original logits from verify
        # But those have been discarded. For simplicity, we'll use the filtered logits
        # which only contain the next token position logits.
        # This means we can only provide logprobs for the next token, not accepted tokens.

        # TODO: If logprobs for accepted tokens are needed, we need to:
        # 1. Store original logits before filtering in verify()
        # 2. Compute logprobs for all accepted positions

        # For now, just handle the next token (similar to normal decode)
        from sglang.srt.environ import envs
        if envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
            logprobs = torch.nn.functional.log_softmax(
                logits_output.next_token_logits, dim=-1
            )
        else:
            logprobs = torch.nn.functional.log_softmax(
                logits_output.next_token_logits / temperatures, dim=-1
            )

        # Get next token ids (last token in each request)
        next_token_ids_list = []
        for i, req in enumerate(batch.reqs):
            if req.output_ids:
                next_token_ids_list.append(req.output_ids[-1])
            else:
                next_token_ids_list.append(0)  # Shouldn't happen

        next_token_ids_tensor = torch.tensor(
            next_token_ids_list, dtype=torch.int64, device=self.device
        )

        logits_output.next_token_logprobs = logprobs[
            torch.arange(bs, device=self.device),
            next_token_ids_tensor,
        ]

        # Handle top_logprobs if requested
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums)

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

        # Add logprobs to requests (only for next token, not accepted tokens)
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        for i, req in enumerate(batch.reqs):
            if req.return_logprob:
                # Only add logprob for the next token (last one in output_ids)
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_ids_list[i])

                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        logits_output.next_token_top_logprobs_idx[i]
                    )

    def get_memory_pool(self):
        """Return memory pool shared with target worker."""
        return self.req_to_token_pool, self.token_to_kv_pool_allocator

    def update_weights_from_tensor(self, *args, **kwargs):
        """Forward weight update to target worker."""
        return self.target_worker.update_weights_from_tensor(*args, **kwargs)
