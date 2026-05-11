"""Base class for speculative decoding algorithms."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Iterator, Any
import torch

from utils.prompt import detect_model_family, prepare_conversation, get_system_prompt


class BaseAlgorithm(ABC):
    """Base class for all speculative decoding algorithms."""

    def __init__(
        self,
        model_path: str,
        draft_model_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs
    ):
        self.model_path = model_path
        self.draft_model_path = draft_model_path
        self.device = device
        self.kwargs = kwargs
        self.model_family = detect_model_family(model_path)
        self.tokenizer = None

    def prepare_conversation(self, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Add system prompt if not present."""
        return prepare_conversation(conversation, self.model_family)

    def compute_accept_length(self, accept_lengths: List[int]) -> float:
        """Compute avg accept length (tokens per iteration).

        accept_lengths should contain accepted draft tokens (without bonus).
        +1 is added to match official EAGLE calculation: new_token/iterations.
        """
        if not accept_lengths:
            return 0.0
        return sum(accept_lengths) / len(accept_lengths) + 1

    def compute_cumulative_position_accuracy(
        self, accept_lengths: List[int], max_depth: int = None
    ) -> Dict[str, Any]:
        """Compute cumulative position accuracy.

        Shows the probability that at least N consecutive draft tokens are accepted.
        - pos1: P(accept_length >= 1) = probability first draft token is accepted
        - pos2: P(accept_length >= 2) = probability first two draft tokens are both accepted
        - etc.

        Args:
            accept_lengths: List of accept lengths per iteration
            max_depth: Maximum depth to compute (default: max observed + 1)

        Returns:
            Dict with:
                - 'accuracies': List[float] - accuracy for each position (0-1 scale)
                - 'accuracies_pct': List[float] - accuracy for each position (0-100 scale)
                - 'positions': List[int] - position indices [1, 2, 3, ...]
                - 'total_iterations': int - number of iterations
                - 'formatted': str - formatted string for logging (e.g., "0.85 0.61 0.56")
        """
        if not accept_lengths:
            return {
                'accuracies': [],
                'accuracies_pct': [],
                'positions': [],
                'total_iterations': 0,
                'formatted': '',
            }

        total_iters = len(accept_lengths)
        max_observed = max(accept_lengths)

        # Determine max depth to compute
        if max_depth is None:
            max_depth = max_observed + 1
        else:
            max_depth = max(max_depth, max_observed + 1)

        # Calculate cumulative accuracy for each position
        positions = list(range(1, max_depth + 1))
        accuracies = []
        accuracies_pct = []

        for pos in positions:
            count = sum(1 for al in accept_lengths if al >= pos)
            acc = count / total_iters
            accuracies.append(round(acc, 4))
            accuracies_pct.append(round(acc * 100, 2))

        # Create formatted string for easy logging
        formatted = ' '.join(f'{acc:.2f}' for acc in accuracies)

        return {
            'accuracies': accuracies,
            'accuracies_pct': accuracies_pct,
            'positions': positions,
            'total_iterations': total_iters,
            'formatted': formatted,
        }

    @abstractmethod
    def load_model(self):
        """Load model(s). Must set self.tokenizer."""
        pass

    @abstractmethod
    def _generate_once(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict:
        """Generate single response. Subclasses implement this.

        Args:
            conversation: Already prepared conversation with system prompt

        Returns:
            Dict with "output" (str) and "metrics" (dict with total_tokens, wall_time, etc.)
            Include "accept_lengths_raw" in metrics for speculative decoding algorithms.
        """
        pass

    def _generate_once_streaming(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Generate single response with streaming output. Override for streaming support.

        Yields iteration-by-iteration results for UI visualization:
        - Each yield contains: iteration info, new tokens, acceptance details
        - Final yield contains complete metrics

        Default implementation: calls _generate_once and yields final result only.
        """
        result = self._generate_once(conversation, max_new_tokens, temperature, **kwargs)
        yield {
            "final": True,
            "output": result["output"],
            "metrics": result["metrics"],
        }

    @torch.inference_mode()
    def generate(
        self,
        samples: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> List[Dict]:
        """Generate responses for samples. Handles both single/multi-turn.

        Args:
            samples: List of dicts with either:
                - "turns": [user_msg1, user_msg2, ...] for multi-turn
                - "conversation": [{role, content}, ...] for single-turn
        """
        if self.tokenizer is None:
            self.load_model()

        results = []
        for sample in samples:
            result = self._generate_sample(sample, max_new_tokens, temperature, **kwargs)
            results.append(result)
        return results

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Generate response with streaming output for UI visualization.

        Args:
            prompt: User input text (will be wrapped in conversation format)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)

        Yields:
            Dict with iteration details:
            - iteration: int, current iteration number
            - new_tokens: List[int], token IDs accepted this iteration
            - new_text: str, decoded text for new tokens
            - draft_tokens: List[int], all draft candidate tokens
            - draft_text: List[str], decoded draft tokens
            - accepted_count: int, number of accepted draft tokens
            - rejected_token: Optional[int], the rejected token (if any)
            - rejected_text: Optional[str], decoded rejected token
            - bonus_token: Optional[int], bonus token (if all draft accepted)
            - bonus_text: Optional[str], decoded bonus token
            - current_metrics: dict with running totals (accept_length, tokens_so_far, elapsed_time)

            Final yield includes:
            - final: True
            - output: str, complete generated text
            - metrics: dict, complete metrics
        """
        if self.tokenizer is None:
            self.load_model()

        conversation = [{"role": "user", "content": prompt}]
        conversation = self.prepare_conversation(conversation)

        # Generator-function gotcha: @torch.inference_mode() decorator only
        # holds while the function returns the generator object, NOT while
        # `next()` advances it. Must use the context manager inside the body
        # to actually wrap the yielded work — otherwise autograd tracking
        # stays on through every iteration (2-5× slowdown).
        with torch.inference_mode():
            yield from self._generate_once_streaming(conversation, max_new_tokens, temperature, **kwargs)

    def _generate_sample(
        self,
        sample: Dict,
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict:
        """Generate for one sample (single or multi-turn)."""
        turns = sample.get("turns")

        if turns:
            # Multi-turn: iterate through turns
            return self._generate_multi_turn(turns, max_new_tokens, temperature, **kwargs)
        else:
            # Single-turn: use conversation directly
            conversation = sample.get("conversation", [])
            conversation = self.prepare_conversation(conversation)
            return self._generate_once(conversation, max_new_tokens, temperature, **kwargs)

    def _generate_multi_turn(
        self,
        turns: List[str],
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict:
        """Generate multi-turn conversation (matching official EAGLE)."""
        messages = self.prepare_conversation([])  # Get system prompt

        all_outputs = []
        total_tokens = 0
        total_time = 0.0
        all_accept_lengths = []
        total_iterations = 0
        total_prefill_time = 0.0
        total_draft_time = 0.0
        total_target_time = 0.0
        total_verify_time = 0.0

        for user_content in turns:
            messages.append({"role": "user", "content": user_content})

            result = self._generate_once(messages.copy(), max_new_tokens, temperature, **kwargs)

            output_text = result["output"]
            metrics = result["metrics"]

            # Add to history for next turn
            messages.append({"role": "assistant", "content": output_text})

            # Accumulate
            all_outputs.append(output_text)
            total_tokens += metrics.get("total_tokens", 0)
            total_time += metrics.get("wall_time", 0)
            total_iterations += metrics.get("iterations", 0)

            if "accept_lengths_raw" in metrics:
                all_accept_lengths.extend(metrics["accept_lengths_raw"])
            if "prefill_time" in metrics:
                total_prefill_time += metrics["prefill_time"]
            if "draft_time" in metrics:
                total_draft_time += metrics["draft_time"]
            if "target_time" in metrics:
                total_target_time += metrics["target_time"]
            if "verify_time" in metrics:
                total_verify_time += metrics["verify_time"]

        total_gpu_time = total_draft_time + total_target_time + total_verify_time

        return {
            "output": all_outputs,
            "metrics": {
                "total_tokens": total_tokens,
                "wall_time": total_time,
                "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
                "accept_length": self.compute_accept_length(all_accept_lengths) if all_accept_lengths else 0,
                "iterations": total_iterations,
                "num_turns": len(turns),
                "accept_lengths_raw": all_accept_lengths,
                "prefill_time": total_prefill_time,
                "draft_time": total_draft_time,
                "target_time": total_target_time,
                "verify_time": total_verify_time,
                "draft_pct": total_draft_time / total_gpu_time * 100 if total_gpu_time > 0 else 0,
                "target_pct": total_target_time / total_gpu_time * 100 if total_gpu_time > 0 else 0,
                "verify_pct": total_verify_time / total_gpu_time * 100 if total_gpu_time > 0 else 0,
            }
        }

    def cleanup(self):
        """Cleanup resources and free GPU memory."""
        # Delete all model attributes that may hold GPU tensors
        for attr in ('model', 'target_model', 'draft_model', 'assistant_model', 'medusa_heads'):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
