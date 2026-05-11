"""Algorithm implementations for speculative decoding benchmarking."""

from .base import BaseAlgorithm
from .baseline import BaselineAlgorithm

__all__ = [
    "BaseAlgorithm",
    "BaselineAlgorithm",
    "get_algorithm",
]


def get_algorithm(
    algorithm_name: str,
    model_path: str,
    draft_model_path: str = None,
    device: str = "cuda",
    **kwargs
):
    """Factory function to get algorithm instance by name.

    Args:
        algorithm_name: Name of algorithm (baseline, eagle3, specblock)
        model_path: Path to target model
        draft_model_path: Path to draft model (required for speculative algorithms)
        device: Device to run on
        **kwargs: Additional algorithm-specific arguments

    Returns:
        Algorithm instance
    """
    algorithm_name = algorithm_name.lower()

    if algorithm_name == "baseline":
        return BaselineAlgorithm(model_path, device=device, **kwargs)

    elif algorithm_name == "eagle3":
        from .eagle3 import EAGLE3Algorithm
        if draft_model_path is None:
            raise ValueError("draft_model_path is required for EAGLE3")
        return EAGLE3Algorithm(model_path, draft_model_path, device=device, **kwargs)

    elif algorithm_name == "specblock":
        from .specblock import SpecBlockAlgorithm
        if draft_model_path is None:
            raise ValueError("draft_model_path is required for SpecBlock")
        return SpecBlockAlgorithm(model_path, draft_model_path, device=device, **kwargs)

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available: baseline, eagle3, specblock"
        )
