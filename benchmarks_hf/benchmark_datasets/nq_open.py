"""Natural Questions Open loader.

Plain open-domain QA from `nq_open` (no retrieval/passage context).
3610 dev questions / 87925 train. This is the `NQ` benchmark used in
the paper. Distinct from spec_bench_rag (80 questions with DPR top-5
passages, exposed as `nq_rag` for users who want the RAG variant).
"""

from typing import List, Dict
from datasets import load_dataset


def load_nq_open(num_samples: int = 3610, split: str = "validation") -> List[Dict]:
    """Load NQ-Open question-only QA.

    Args:
        num_samples: max 3610 on validation, 87925 on train.
        split: "validation" (default, 3610) or "train" (87925).

    Returns:
        List of conversation dicts. `reference` holds the list of accepted
        short-answer strings for accuracy eval; not used by adapt itself.
    """
    dataset = load_dataset("google-research-datasets/nq_open", split=split)

    result = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        result.append({
            "id": i,
            "category": "qa",
            "conversation": [{"role": "user", "content": sample["question"]}],
            "reference": sample.get("answer", []),
        })

    return result
