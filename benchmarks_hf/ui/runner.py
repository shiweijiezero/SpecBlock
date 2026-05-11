"""Runner for algorithm invocation in UI."""

import gc
import sys
import os
import torch
from typing import Dict, Any, Iterator, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import get_algorithm


# Mapping from kwarg names to algorithm attribute names
_PARAM_ATTR_MAP = {
    "draft_tokens": ["draft_tokens", "K"],
    "depth": ["depth", "max_blocks"],
    "topk": ["topk"],
    "diff_steps": ["diff_steps", "num_refine_steps"],
    "ngram_size": ["ngram_size"],
    "total_tokens": ["total_tokens"],
    "beam_width": ["beam_width"],
    "max_blocks": ["max_blocks"],
    "draft_quantize": ["draft_quantize"],
}


class SpecDecodeRunner:
    """Manages algorithm loading and generation for UI."""

    def __init__(self):
        self.algorithm = None
        self.algorithm_name = None
        self.model_path = None
        self.draft_model_path = None
        self.algorithm_kwargs = {}  # Store kwargs to detect config changes

    def _needs_reload(
        self,
        algorithm_name: str,
        model_path: str,
        draft_model_path: Optional[str],
    ) -> bool:
        """Check if model reload is needed (algorithm or model paths changed)."""
        return (
            self.algorithm is None or
            self.algorithm_name != algorithm_name or
            self.model_path != model_path or
            self.draft_model_path != draft_model_path
        )

    def _update_params(self, **kwargs) -> str:
        """Update runtime parameters on existing algorithm without reloading.

        Returns description of what changed.
        """
        changed = []
        # Collect eagle3 tree config updates for a single batch call
        tree_config_keys = ("draft_tokens", "depth", "topk")
        tree_updates = {}

        for key, value in kwargs.items():
            if self.algorithm_kwargs.get(key) == value:
                continue
            # Batch eagle3 tree config updates
            if hasattr(self.algorithm, 'update_tree_config') and key in tree_config_keys:
                tree_updates[key] = value
                changed.append(f"{key}={value}")
                continue
            # Generic: update matching attribute on the algorithm
            attr_names = _PARAM_ATTR_MAP.get(key, [key])
            for attr in attr_names:
                if hasattr(self.algorithm, attr):
                    setattr(self.algorithm, attr, value)
                    changed.append(f"{key}={value}")
                    break

        if tree_updates:
            self.algorithm.update_tree_config(**tree_updates)

        self.algorithm_kwargs = kwargs.copy()
        return ", ".join(changed) if changed else ""

    def load_algorithm(
        self,
        algorithm_name: str,
        model_path: str,
        draft_model_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Load an algorithm with specified models.

        If only runtime params changed (same algorithm & models), updates params
        without reloading. Full reload only when algorithm or model paths change.

        Args:
            algorithm_name: Name of algorithm
            model_path: Path to target model
            draft_model_path: Path to draft model (required for most algorithms)
            **kwargs: Additional algorithm parameters (draft_tokens, depth, topk, etc.)

        Returns:
            Status message
        """
        # Same config: nothing to do
        if (self.algorithm is not None and
            self.algorithm_name == algorithm_name and
            self.model_path == model_path and
            self.draft_model_path == draft_model_path and
            self.algorithm_kwargs == kwargs):
            return f"Algorithm {algorithm_name} already loaded with same config."

        # Only params changed: update without reload
        if not self._needs_reload(algorithm_name, model_path, draft_model_path):
            changed = self._update_params(**kwargs)
            return f"Updated params: {changed} (no model reload)."

        # Full reload needed: cleanup previous and free GPU memory
        if self.algorithm is not None:
            self.algorithm.cleanup()
            self.algorithm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Load new algorithm
        try:
            self.algorithm = get_algorithm(
                algorithm_name,
                model_path=model_path,
                draft_model_path=draft_model_path,
                **kwargs
            )
            self.algorithm.load_model()
            self.algorithm_name = algorithm_name
            self.model_path = model_path
            self.draft_model_path = draft_model_path
            self.algorithm_kwargs = kwargs.copy()
            return f"Loaded {algorithm_name} with {kwargs} successfully."
        except Exception as e:
            self.algorithm = None
            raise RuntimeError(f"Failed to load {algorithm_name}: {e}")

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Generate response with streaming output.

        Args:
            prompt: User input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Iteration data dicts for UI visualization
        """
        if self.algorithm is None:
            raise RuntimeError("No algorithm loaded. Call load_algorithm first.")

        yield from self.algorithm.generate_streaming(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )

    def cleanup(self):
        """Release resources."""
        if self.algorithm is not None:
            self.algorithm.cleanup()
            self.algorithm = None
            self.algorithm_name = None


# Global runner instance for UI
_runner = None


def get_runner() -> SpecDecodeRunner:
    """Get or create the global runner instance."""
    global _runner
    if _runner is None:
        _runner = SpecDecodeRunner()
    return _runner
