"""
Shared metrics calculation functions for speculative decoding algorithms.

Aligned with SGLang's metrics calculation methodology.
See: sglang/srt/managers/scheduler_metrics_mixin.py
"""


def calculate_speculative_metrics(
    total_draft_tokens: int,
    iterations: int,
    total_accepted_from_drafts: int = None,
    total_accepted_tokens: int = None,
    include_verified_in_metrics: bool = True
) -> dict:
    """
    Calculate accept_rate and accept_length for speculative decoding algorithms.

    Args:
        total_draft_tokens: Total number of draft tokens proposed
        iterations: Number of speculative decoding iterations
        total_accepted_from_drafts: Number of draft tokens accepted (excluding verified_id and bonus)
            Use this OR total_accepted_tokens, not both.
        total_accepted_tokens: Total accepted tokens (verified_id + draft + bonus per iteration)
            Use this OR total_accepted_from_drafts, not both.
        include_verified_in_metrics: If True, include verified_id in metrics (SGLang-aligned)

    Returns:
        Dictionary with 'accept_rate' and 'accept_length' keys

    SGLang Alignment:
        SGLang includes verified_id in both metrics (but NOT bonus token):
        - spec_num_accepted_tokens = accepted_draft_tokens + verified_ids (bs)
        - accept_rate = spec_num_accepted_tokens / total_draft_tokens
        - accept_length = spec_num_accepted_tokens / iterations (minimum 1.0)

        See sglang/srt/managers/scheduler_metrics_mixin.py:80:
            self.spec_num_accepted_tokens += num_accepted_tokens + bs

    Note on bonus token:
        SGLang does NOT include bonus token in accept metrics.
        If you pass total_accepted_tokens (which includes bonus), we subtract iterations.
    """
    # Determine which input format is being used
    if total_accepted_from_drafts is not None:
        # Input: Only draft tokens accepted (no verified_id, no bonus)
        num_draft_only = total_accepted_from_drafts
    elif total_accepted_tokens is not None:
        # Input: verified_id (iterations) + draft + bonus (iterations)
        # To get draft only: total - verified_ids - bonus_tokens
        num_draft_only = total_accepted_tokens - (2 * iterations)
    else:
        raise ValueError("Must provide either total_accepted_from_drafts or total_accepted_tokens")

    if include_verified_in_metrics:
        # SGLang-aligned: include verified_id in metrics (but not bonus)
        # Each iteration produces 1 verified_id, so add iterations
        total_accepted_with_verified = num_draft_only + iterations

        # Accept rate: (accepted_draft_tokens + verified_ids) / total_draft_tokens
        accept_rate = total_accepted_with_verified / total_draft_tokens if total_draft_tokens > 0 else 0

        # Accept length: (accepted_draft_tokens + verified_ids) / iterations
        # Minimum is 1.0 (each iteration accepts at least 1 verified_id)
        accept_length = total_accepted_with_verified / iterations if iterations > 0 else 0
    else:
        # Only count draft tokens (not SGLang-aligned)
        accept_rate = num_draft_only / total_draft_tokens if total_draft_tokens > 0 else 0
        accept_length = num_draft_only / iterations if iterations > 0 else 0

    return {
        "accept_rate": accept_rate,
        "accept_length": accept_length,
    }
