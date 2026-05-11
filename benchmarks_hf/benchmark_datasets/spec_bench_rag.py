"""Retrieval-Augmented QA loader from Spec-Bench.

Uses Spec-Bench's pre-curated 80-question RAG subset (NQ questions with DPR
top-5 retrieved passages concatenated), same data as EAGLE-3 paper.
"""

from typing import List, Dict

from .spec_bench_qa import _load_spec_bench_by_category


def load_spec_bench_rag(num_samples: int = 80) -> List[Dict]:
    """Load NQ + top-5 DPR context RAG subset from Spec-Bench.

    80 long-prompt questions. Each turn contains retrieved context +
    the question.
    """
    entries = _load_spec_bench_by_category("rag")

    result = []
    for i, entry in enumerate(entries):
        if i >= num_samples:
            break

        turn_text = entry["turns"][0]
        result.append({
            "id": entry.get("question_id", i),
            "category": "rag",
            "conversation": [{"role": "user", "content": turn_text}],
        })

    return result
