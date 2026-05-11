"""Alpaca dataset loader."""

from typing import List, Dict
from datasets import load_dataset


def load_alpaca(num_samples: int = 200) -> List[Dict]:
    """Load Alpaca instruction-following dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        List of conversation dictionaries
    """
    dataset = load_dataset("tatsu-lab/alpaca")["train"]

    result = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        # Format: instruction + optional input
        content = sample["instruction"]
        if sample.get("input", "").strip():
            content += f"\n\nInput: {sample['input']}"

        result.append({
            "id": i,
            "category": "instruction",
            "conversation": [{"role": "user", "content": content}],
        })

    return result
