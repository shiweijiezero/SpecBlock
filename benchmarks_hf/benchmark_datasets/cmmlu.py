"""CMMLU dataset loader."""

from typing import List, Dict
from datasets import load_dataset


def load_cmmlu(num_samples: int = 200) -> List[Dict]:
    """Load CMMLU Chinese multi-task dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        List of conversation dictionaries
    """
    dataset = load_dataset("zhaode/cmmlu")["train"]
    prompts = [q["instruction"] for q in dataset][:num_samples]

    result = []
    for i, prompt in enumerate(prompts):
        result.append({
            "id": i,
            "category": "chinese",
            "conversation": [{"role": "user", "content": prompt}],
        })

    return result
