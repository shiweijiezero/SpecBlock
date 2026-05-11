"""WMT23 DE-EN translation dataset loader.

Uses `haoranxu/WMT23-Test` which contains WMT23 General Machine Translation
test set. 549 de-en pairs.

SpecBench 用的是 WMT14 de-en（10 年前数据）；WMT23 更新、句式更自然。
"""

from typing import List, Dict
from datasets import load_dataset


def load_wmt23(num_samples: int = 200) -> List[Dict]:
    """Load WMT23 German→English translation test set.

    Args:
        num_samples: Number of samples to load (max 549)

    Returns:
        List of conversation dictionaries
    """
    dataset = load_dataset("haoranxu/WMT23-Test", "de-en")["test"]

    result = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        pair = sample.get("de-en", sample)
        de_text = pair["de"]
        prompt = f"Translate the following German text to English:\n\n{de_text}"
        result.append({
            "id": i,
            "category": "translation",
            "conversation": [{"role": "user", "content": prompt}],
            "reference": pair.get("en", ""),
        })

    return result
