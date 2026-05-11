"""Dataset loaders for benchmarking."""

from .mtbench import load_mtbench
from .gsm8k import load_gsm8k
from .humaneval import load_humaneval
from .math500 import load_math500
from .ceval import load_ceval
from .cmmlu import load_cmmlu
from .cnn_dm import load_cnn_dm
from .alpaca import load_alpaca
from .wmt23 import load_wmt23
from .spec_bench_qa import load_spec_bench_qa
from .spec_bench_rag import load_spec_bench_rag
from .nq_open import load_nq_open
from .mixed_mct import load_mixed_mct
from .mixed_v2 import load_mixed_v2

__all__ = [
    "load_mtbench",
    "load_gsm8k",
    "load_humaneval",
    "load_math500",
    "load_ceval",
    "load_cmmlu",
    "load_cnn_dm",
    "load_alpaca",
    "load_wmt23",
    "load_spec_bench_qa",
    "load_spec_bench_rag",
    "load_nq_open",
    "load_mixed_mct",
    "load_mixed_v2",
    "load_benchmark_dataset",
]


def load_benchmark_dataset(name: str, num_samples: int = None, **kwargs):
    """Load dataset by name.

    Args:
        name: Dataset name
        num_samples: Number of samples to load
        **kwargs: Additional dataset-specific arguments

    Returns:
        List of conversation dictionaries
    """
    loaders = {
        "mtbench": load_mtbench,
        "gsm8k": load_gsm8k,
        "humaneval": load_humaneval,
        "math500": load_math500,
        "ceval": load_ceval,
        "cmmlu": load_cmmlu,
        "cnn_dm": load_cnn_dm,
        "alpaca": load_alpaca,
        "wmt23": load_wmt23,
        "nq_qa": load_spec_bench_qa,
        "nq_rag": load_spec_bench_rag,
        "nq_open": load_nq_open,
        "mixed_mct": load_mixed_mct,
        "mixed_v2": load_mixed_v2,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    if num_samples is not None:
        kwargs["num_samples"] = num_samples

    return loaders[name](**kwargs)
