from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _verify_tree_greedy_kernel(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    NUM_DRAFT_TOKENS: tl.constexpr,
    NUM_SPEC_STEPS: tl.constexpr,
):
    batch = tl.program_id(0)
    row = batch * NUM_DRAFT_TOKENS
    accept_row = batch * NUM_SPEC_STEPS

    last = tl.load(retrive_index + row)
    tl.store(accept_index + accept_row, last)
    accepted = tl.zeros((), tl.int32)
    current = tl.zeros((), tl.int64)

    for _ in range(1, NUM_SPEC_STEPS):
        safe_current = tl.maximum(current, 0)
        current = tl.load(
            retrive_next_token + row + safe_current,
            mask=current >= 0,
            other=-1,
        )
        matched_node = tl.full((), -1, tl.int64)
        while current >= 0:
            draft_index = tl.load(retrive_index + row + current)
            draft_token = tl.load(candidates + row + current)
            target_token = tl.load(target_predict + last)
            matched = draft_token == target_token
            tl.store(
                predicts + last,
                target_token.to(tl.int32),
                mask=matched,
            )
            next_accepted = accepted + matched.to(tl.int32)
            tl.store(
                accept_index + accept_row + next_accepted,
                draft_index.to(tl.int32),
                mask=matched,
            )
            accepted = next_accepted
            last = tl.where(matched, draft_index, last)
            matched_node = tl.where(matched, current, matched_node)
            current = tl.load(
                retrive_next_sibling + row + current,
                mask=~matched,
                other=-1,
            )
        current = matched_node

    tl.store(accept_token_num + batch, accepted)
    tl.store(predicts + last, tl.load(target_predict + last).to(tl.int32))


def verify_tree_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    batch_size, num_draft_tokens = candidates.shape
    _verify_tree_greedy_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        NUM_DRAFT_TOKENS=num_draft_tokens,
        NUM_SPEC_STEPS=accept_index.shape[1],
        num_warps=1,
        num_stages=1,
    )
