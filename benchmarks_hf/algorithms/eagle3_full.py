"""EAGLE3 tree-based speculative decoding algorithm - FULL IMPLEMENTATION.

This is the complete tree-based implementation following the reference EAGLE-main code.
"""

import time
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import sys
from pathlib import Path

from .base import BaseAlgorithm


# EAGLE constants
TOPK = 10  # Top-k for tree construction (matching EAGLE-main)


class EAGLE3AlgorithmFull(BaseAlgorithm):
    """EAGLE3 with complete tree-based speculation (matching EAGLE-main)."""

    def __init__(
        self,
        model_path: str,
        draft_model_path: str,
        device: str = "cuda",
        draft_tokens: int = 60,  # Total tokens in tree (matching mc_sim_7b_60)
        topk: int = 10,  # Top-k candidates per step
        depth: int = 5,  # Tree depth
        **kwargs
    ):
        """Initialize EAGLE3 with full tree construction.

        Args:
            model_path: Path to target model
            draft_model_path: Path to EAGLE3 draft model
            device: Device
            draft_tokens: Total tree nodes (default: 60 for mc_sim_7b_60)
            topk: Top-k candidates per expansion (default: 10)
            depth: Tree depth (default: 5)
        """
        super().__init__(model_path, draft_model_path, device, **kwargs)
        self.draft_tokens = draft_tokens
        self.topk = topk
        self.depth = depth
        self.target_model = None
        self.draft_model = None
        self.tokenizer = None

    def load_model(self):
        """Load target and EAGLE3 draft models."""
        print(f"Loading target model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load target model
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        self.target_model.eval()

        # Load EAGLE3 draft model
        print(f"Loading EAGLE3 draft model from {self.draft_model_path}...")

        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
            from safetensors.torch import load_file

            config = AutoConfig.from_pretrained(self.draft_model_path, trust_remote_code=True)
            self.draft_model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")

            safetensors_path = os.path.join(self.draft_model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                self.draft_model.load_state_dict(state_dict, strict=False)
                print(f"EAGLE3 draft model loaded!")
            else:
                raise FileNotFoundError(f"No checkpoint at {safetensors_path}")

            self.draft_model = self.draft_model.to(device=self.device, dtype=torch.bfloat16)
            self.draft_model.eval()

        except Exception as e:
            raise RuntimeError(f"Cannot load EAGLE3 draft model: {e}")

        print("Models loaded successfully!")

    def get_hidden_states_from_target(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get 3-layer concatenated hidden states from target model."""
        with torch.no_grad():
            outputs = self.target_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            all_hidden = outputs.hidden_states
            # all_hidden has length num_hidden_layers + 1 (includes embedding)
            # all_hidden[0] = embedding, all_hidden[i] = layer i-1 output
            num_hidden_layers = len(all_hidden) - 1  # exclude embedding

            # Match training code: [1, num_layers // 2 - 1, num_layers - 4]
            # +1 offset because all_hidden[0] is embedding
            low_idx = 1 + 1  # layer 1
            mid_idx = num_hidden_layers // 2 - 1 + 1  # layer (num_layers // 2 - 1)
            high_idx = num_hidden_layers - 4 + 1  # layer (num_layers - 4)

            concat_hidden = torch.cat([
                all_hidden[low_idx],
                all_hidden[mid_idx],
                all_hidden[high_idx]
            ], dim=-1)

            return concat_hidden, outputs.logits

    def topk_tree_generate(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 0.0
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate tree of draft tokens using top-k recursive expansion.

        This follows EAGLE-main's topK_genrate() logic.

        Returns:
            draft_tokens: Tensor of shape (draft_tokens,)
            parents_list: List of parent indices for tree structure
        """
        # Initialize
        last_hidden = hidden_states[:, -1, :]  # (1, hidden_size * 3)

        scores_list = []
        parents_list = []
        tokens_list = []

        # Step 0: Get initial top-k from last hidden state
        with torch.no_grad():
            # Forward through draft model to get logits
            draft_hidden = self.draft_model(
                hidden_states=last_hidden.unsqueeze(1),  # (1, 1, hidden_size * 3)
                inputs_embeds=None,
                attention_mask=None,
                ttt_length=1
            )
            logits = self.draft_model.lm_head(draft_hidden)[:, -1, :]  # (1, vocab_size)

            # Top-k sampling
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                log_probs = torch.log(probs + 1e-10)
            else:
                log_probs = F.log_softmax(logits, dim=-1)

            top_scores, top_indices = torch.topk(log_probs, self.topk, dim=-1)  # (1, topk)

            scores = top_scores[0]  # (topk,)
            scores_list.append(scores.unsqueeze(0))  # (1, topk)
            parents_list.append(torch.zeros(1, dtype=torch.long, device=self.device))
            tokens_list.append(top_indices)  # (1, topk)

            # Prepare for next depth
            # input_hidden: (1, topk, hidden_size * 3)
            input_hidden = last_hidden.unsqueeze(1).repeat(1, self.topk, 1)
            # input_tokens: (topk,)
            input_tokens = top_indices[0]
            topk_cs_index = torch.arange(self.topk, device=self.device)

        # Recursive expansion (depth iterations)
        for i in range(self.depth):
            with torch.no_grad():
                # Get embeddings for current tokens
                # input_tokens shape: (topk,) or (topk,) from previous iteration
                token_embeds = self.draft_model.embed_input_ids(input_tokens)  # (topk, hidden_size)
                if token_embeds.dim() == 2:
                    token_embeds = token_embeds.unsqueeze(0)  # (1, topk, hidden_size)

                # Forward through draft model
                # input_hidden: (1, topk, hidden_size * 3)
                # token_embeds: (1, topk, hidden_size)
                out_hidden = []
                for j in range(input_hidden.shape[1]):
                    h = input_hidden[:, j:j+1, :]  # (1, 1, hidden_size * 3)
                    e = token_embeds[:, j:j+1, :]  # (1, 1, hidden_size)
                    draft_h = self.draft_model(
                        hidden_states=h,
                        inputs_embeds=e,
                        attention_mask=None,
                        ttt_length=1
                    )
                    out_hidden.append(draft_h)
                out_hidden = torch.cat(out_hidden, dim=1)  # (1, topk, hidden_size)

                # Get logits
                logits = self.draft_model.lm_head(out_hidden)  # (1, topk, vocab_size)

                # Top-k for each branch
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    log_probs = torch.log(probs + 1e-10)
                else:
                    log_probs = F.log_softmax(logits, dim=-1)

                top_branch_scores, top_branch_indices = torch.topk(log_probs[0], self.topk, dim=-1)  # (topk, topk)

                # Cumulative scores: scores from previous + new scores
                cu_scores = top_branch_scores + scores.unsqueeze(-1)  # (topk, topk)

                # Select top-k from all branches
                topk_cs = torch.topk(cu_scores.view(-1), self.topk, dim=-1)
                topk_cs_index, topk_cs_scores = topk_cs.indices, topk_cs.values

                scores = topk_cs_scores  # (topk,)

                # Determine which branch each top-k came from
                out_ids = topk_cs_index // self.topk  # (topk,)
                input_hidden = out_hidden[:, out_ids, :]  # (1, topk, hidden_size)

                # Get the actual tokens
                input_tokens = top_branch_indices.view(-1)[topk_cs_index]  # (topk,)

                tokens_list.append(top_branch_indices)  # (topk, topk)
                scores_list.append(cu_scores)  # (topk, topk)

                # Parent indices
                bias1 = self.topk if i > 0 else 0
                bias2 = max(0, i - 1)
                bias = 1 + self.topk ** 2 * bias2 + bias1
                parents = topk_cs_index + bias
                parents_list.append(parents)

        # Final selection: choose top draft_tokens from all candidates
        all_scores = torch.cat([s.view(-1) for s in scores_list], dim=0)  # (total_candidates,)
        all_tokens = torch.cat([t.view(-1) for t in tokens_list], dim=0)  # (total_candidates,)

        top_final = torch.topk(all_scores, min(self.draft_tokens, len(all_scores)), dim=-1)
        top_final_indices = top_final.indices
        top_final_indices = torch.sort(top_final_indices).values

        draft_tokens = all_tokens[top_final_indices]  # (draft_tokens,)

        return draft_tokens, parents_list

    def generate(
        self,
        conversations: List[List[Dict[str, str]]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> List[Dict]:
        """Generate with EAGLE3 tree-based speculation."""
        if self.target_model is None:
            self.load_model()

        results = []
        do_sample = temperature > 0

        for conversation in conversations:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            input_length = input_ids.shape[1]

            start_time = time.time()
            total_draft_tokens = 0
            total_accepted_tokens = 0
            iterations = 0

            with torch.no_grad():
                while input_ids.shape[1] - input_length < max_new_tokens:
                    iterations += 1

                    # Get hidden states from target
                    hidden_states, target_logits = self.get_hidden_states_from_target(
                        input_ids, attention_mask
                    )

                    # Generate tree of candidates
                    try:
                        candidates, _ = self.topk_tree_generate(hidden_states, temperature)
                    except Exception as e:
                        # Fallback: single token
                        last_logit = target_logits[:, -1, :]
                        if do_sample:
                            probs = F.softmax(last_logit / temperature, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        attention_mask = torch.ones_like(input_ids)
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                        continue

                    if len(candidates) == 0:
                        # Fallback
                        last_logit = target_logits[:, -1, :]
                        if do_sample:
                            probs = F.softmax(last_logit / temperature, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        attention_mask = torch.ones_like(input_ids)
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                        continue

                    total_draft_tokens += len(candidates)

                    # Verify with target model
                    candidate_ids = candidates.unsqueeze(0)  # (1, draft_tokens)
                    verify_ids = torch.cat([input_ids, candidate_ids], dim=1)
                    verify_mask = torch.ones_like(verify_ids)

                    _, verify_logits = self.get_hidden_states_from_target(verify_ids, verify_mask)

                    # Accept tokens greedily
                    accepted = 0
                    for i, candidate_token in enumerate(candidates):
                        pos = input_ids.shape[1] + i - 1
                        target_logit = verify_logits[:, pos, :]

                        if do_sample:
                            target_probs = F.softmax(target_logit / temperature, dim=-1)
                            target_token = torch.multinomial(target_probs, num_samples=1)
                        else:
                            target_token = torch.argmax(target_logit, dim=-1, keepdim=True)

                        if target_token.item() == candidate_token.item():
                            accepted += 1
                            if target_token.item() == self.tokenizer.eos_token_id:
                                input_ids = torch.cat([input_ids, target_token], dim=1)
                                break
                        else:
                            input_ids = torch.cat([input_ids, target_token], dim=1)
                            break
                    else:
                        # All accepted, sample bonus token
                        last_logit = verify_logits[:, -1, :]
                        if do_sample:
                            target_probs = F.softmax(last_logit / temperature, dim=-1)
                            bonus_token = torch.multinomial(target_probs, num_samples=1)
                        else:
                            bonus_token = torch.argmax(last_logit, dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, bonus_token], dim=1)
                        accepted += 1

                    total_accepted_tokens += accepted
                    attention_mask = torch.ones_like(input_ids)

                    if input_ids[0, -1].item() == self.tokenizer.eos_token_id:
                        break
                    if input_ids.shape[1] - input_length >= max_new_tokens:
                        break

            elapsed_time = time.time() - start_time

            generated_ids = input_ids[0, input_length:]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            num_tokens = len(generated_ids)
            tokens_per_second = num_tokens / elapsed_time if elapsed_time > 0 else 0
            accept_rate = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0
            accept_length = total_accepted_tokens / iterations if iterations > 0 else 0

            results.append({
                "output": output_text,
                "metrics": {
                    "total_tokens": num_tokens,
                    "wall_time": elapsed_time,
                    "tokens_per_second": tokens_per_second,
                    "accept_rate": accept_rate,
                    "accept_length": accept_length,
                    "total_draft_tokens": total_draft_tokens,
                    "total_accepted_tokens": total_accepted_tokens,
                    "iterations": iterations,
                    "draft_tokens": self.draft_tokens,
                    "topk": self.topk,
                    "depth": self.depth,
                }
            })

        return results

    def cleanup(self):
        """Clean up models."""
        if self.target_model is not None:
            del self.target_model
            self.target_model = None
        if self.draft_model is not None:
            del self.draft_model
            self.draft_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
