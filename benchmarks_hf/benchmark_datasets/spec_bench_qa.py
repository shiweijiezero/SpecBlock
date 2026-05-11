"""Natural Questions short-answer QA loader from Spec-Bench.

Uses Spec-Bench's pre-curated 80-question QA subset (derived from Natural
Questions), same data as EAGLE-3 paper comparison.

URL: https://github.com/hemingkx/Spec-Bench/blob/main/data/spec_bench/question.jsonl
"""

import json
import os
import urllib.request
from typing import List, Dict

SPEC_BENCH_URL = (
    "https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/"
    "data/spec_bench/question.jsonl"
)


def _get_spec_bench_cache_path():
    """Cache Spec-Bench question.jsonl locally."""
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache"))
    cache_path = os.path.join(cache_dir, "spec_bench", "question.jsonl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not os.path.exists(cache_path):
        print(f"Downloading Spec-Bench question set → {cache_path}")
        urllib.request.urlretrieve(SPEC_BENCH_URL, cache_path)

    return cache_path


def _load_spec_bench_by_category(category: str) -> List[Dict]:
    """Load all entries of given category from Spec-Bench question.jsonl."""
    path = _get_spec_bench_cache_path()
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("category") == category:
                entries.append(r)
    return entries


def load_spec_bench_qa(num_samples: int = 80) -> List[Dict]:
    """Load NQ-derived short-answer QA subset from Spec-Bench.

    80 questions total. Simple single-turn factual questions.
    """
    entries = _load_spec_bench_by_category("qa")

    result = []
    for i, entry in enumerate(entries):
        if i >= num_samples:
            break

        turn_text = entry["turns"][0]
        result.append({
            "id": entry.get("question_id", i),
            "category": "qa",
            "conversation": [{"role": "user", "content": turn_text}],
        })

    return result
