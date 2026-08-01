"""Enable deterministic batch-invariant CUDA operators for HF target models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ENABLED = False


def enable_hf_batch_invariant_ops() -> None:
    """Install the bundled fixed-tile CUDA operators once per process.

    Batched BF16 GEMMs can choose a different reduction path from B=1 and change
    greedy argmax decisions.  Speculative decoding must reproduce target-only
    token IDs, so the HF batch path uses the same deterministic operators as the
    SGLang exactness gate.
    """
    global _ENABLED
    if _ENABLED:
        return

    repo_root = Path(__file__).resolve().parents[2]
    sglang_python = repo_root / "sglang-main" / "python"
    if not sglang_python.is_dir():
        raise RuntimeError(
            "HF request batching requires the bundled sglang-main batch-invariant "
            f"operators, but {sglang_python} does not exist"
        )

    # Strict HF token parity relies on the fixed-reduction Triton path.  Do not
    # permit hardware-dependent DeepGEMM selection or a variant GEMM fallback.
    os.environ["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"] = "0"
    os.environ["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT"] = "0"
    sys.path.insert(0, str(sglang_python))
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        enable_batch_invariant_mode,
    )

    enable_batch_invariant_mode()
    _ENABLED = True
