# coding=utf-8
# Copyright 2024 SpecForge Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SpecBlock: Block Causal Speculative Decoding Training Logic

Training logic for SpecBlock — adds Rank Head training and block-level TTT:
- Draft Loss: KL divergence with chain mask
- Rank Loss: soft CE with rank chain mask + freq reweight + ordinal smoothing + distance weight
- TTT: 多 block 链式 forward，模拟推理时的 block 自回归

Rank Label 计算（4 分类 bucket）：
  对每个位置，找 ground truth token 在 draft 分布中的排名，映射到 bucket：
  rank=1 → class 0 (正确), rank 2-4 → class 1 (小分叉),
  rank 5-10 → class 2 (大分叉), rank>10 → class 3 (放弃)

Rank Head 输入：
  detached normed hidden (H) + top-10 logit values (10) + entropy (1) = H+11 维

Rank Loss 四个组件：
  1. Rank chain mask: 课程学习，只在 slot 0..M 上算（M = 第一个 rank 预测错误的位置，含自身）
  2. Per-batch frequency reweighting: w(c) = min(1, budget / count_c)
     每个 slot 内统计各 class 数量，按 budget_per_class 做软缩放。高频 class 权重低，低频权重高。
  3. Asymmetric distance: w_dist(p, t) = 1 + |t-p| + max(t-p, 0)
     低估 2x slope（危险），高估 1x slope（安全）
  4. Ordinal label smoothing: kernel(c, t) = 1 / (1 + |c-t| + max(t-c, 0))^2
     非对称：高估方向扩散更多（安全），低估方向衰减更快（危险）

TTT 训练流程：
  Phase 1 (< rank_start_step): 单 block，只训 draft loss + rank head
  Phase 2 (>= rank_start_step): 多 block TTT
    - Block 1: condition = condition_proj(target_3H), input = GT token[i]
    - Block 2+: condition = draft_hidden from prev block, input = GT token[i+M₁+...]
    - M 随机采样 uniform(1, K)
    - Block 内仍保留 chain mask（slot k 的 draft loss 只在前 k slot 全对位置算）
    - Block 间不做 filter：所有 valid 位置都进入下个 block 训练（对齐推理分布：推理
      触发 Block 2 时 slot M-1 恰好 top-1 不对，filter "前 M 全对" 排除了推理的核心场景）
"""

import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft import Eagle3DraftModel


# ============================================================
#              Rank Label Computation
# ============================================================

def _compute_rank_labels(
    logits: torch.Tensor,        # [B, valid_len, V_draft]
    target_tokens: torch.Tensor, # [B, valid_len] - argmax of target distribution
    rank_classes: int = 4,
) -> torch.Tensor:
    """
    Compute rank labels: bucketed rank of target token in draft distribution.

    Uses topk(10) instead of full sort for efficiency: O(V) vs O(V log V).

    Bucket scheme (4 classes):
      - Class 0: rank=1 (top-1 correct, continue)
      - Class 1: rank 2-4 (small branch, factor=4)
      - Class 2: rank 5-10 (large branch, factor=10)
      - Class 3: rank>10 (give up, terminate)

    Args:
        logits: [B, valid_len, V_draft] - draft model logits (raw, not softmaxed)
        target_tokens: [B, valid_len] - ground truth token indices (in draft vocab)
        rank_classes: number of rank classes (default 4)

    Returns:
        rank_labels: [B, valid_len] - rank labels (0 to rank_classes-1)
    """
    # topk(10) is sufficient: we only need to distinguish rank 1, 2-4, 5-10, >10
    _, top10_indices = logits.topk(10, dim=-1, sorted=True)  # [B, valid_len, 10]

    # Check if target token is in top-10
    target_expanded = target_tokens.unsqueeze(-1)  # [B, valid_len, 1]
    match = (top10_indices == target_expanded)  # [B, valid_len, 10]

    # Find position in top-10 (0-indexed), or 10 if not found
    # match.any(-1) tells us if target is in top-10
    in_top10 = match.any(dim=-1)  # [B, valid_len]
    # argmax on match gives first True position (0-9)
    rank_positions = match.float().argmax(dim=-1)  # [B, valid_len], 0 if no match but masked by in_top10

    # Bucket: rank 0 → class 0, 1-3 → class 1, 4-9 → class 2
    rank_labels = torch.where(
        ~in_top10, 3,  # not in top-10 → class 3
        torch.where(
            rank_positions == 0, 0,  # top-1 → class 0
            torch.where(rank_positions <= 3, 1, 2)  # 2-4 → class 1, 5-10 → class 2
        )
    ).long()

    return rank_labels


def _asymmetric_ordinal_smooth(
    labels: torch.Tensor,   # [N] long
    num_classes: int = 4,
) -> torch.Tensor:
    """
    Asymmetric ordinal label smoothing.

    Kernel: 1 / (1 + |c-t| + max(t-c, 0))^2
    Consistent with w_dist asymmetry:
      - Over-estimation side (c > t): gentler decay (safe)
      - Under-estimation side (c < t): steeper decay (dangerous)

    Returns: [N, C] soft target distribution (sums to 1 per sample)
    """
    C = num_classes
    classes = torch.arange(C, device=labels.device).float()          # [C]
    t = labels.unsqueeze(-1).float()                                  # [N, 1]
    diff = t - classes                                                # [N, C]
    raw = 1.0 / (1.0 + diff.abs() + diff.clamp(min=0.0)).square()   # [N, C]
    return raw / raw.sum(dim=-1, keepdim=True)                        # [N, C]


# ============================================================
#        Vectorized Loss/Accuracy/Rank Computation
# ============================================================

@torch.compile(dynamic=True)
def _compute_batched_loss_acc_and_rank(
    logits: torch.Tensor,           # [B, S, K, V]
    rank_logits: torch.Tensor,      # [B, S, K, rank_classes]
    target_p_full: torch.Tensor,    # [B, S_full, V_draft] (padded)
    position_mask_full: torch.Tensor,  # [B, S_full, 1] (padded)
    loss_mask: torch.Tensor,        # [B, S, 1]
    K: int,
    rank_classes: int = 4,
    target_offset: int = 0,
    budget_per_class: int = 10,
    chain_snapshot_slot: int = -1,
    draft_loss_topk: int = -1,      # -1 = full vocab KL, >0 = top-K target tokens only
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Vectorized computation of draft loss, accuracy, and rank loss.

    Draft loss: KL with chain mask
    Rank loss: soft CE with four components:

    1. Rank chain mask: curriculum learning, train on slots 0..M (first rank prediction error, inclusive).
       Slots after the truncation point are skipped — rank head learns from its own error boundary.
    2. Per-batch frequency reweighting: w(c) = min(1, target_count / actual_count).
       Equalizes effective contribution per class within each slot.
       target_count = budget_per_class. budget_per_class <= 0 disables reweighting.
    3. Asymmetric distance: w_dist(p, t) = 1 + |t-p| + max(t-p, 0).
       Under-estimation (pred < true): 2x slope — risks missing correct token.
       Over-estimation (pred > true): 1x slope — safe but wastes branch budget.
    4. Ordinal label smoothing: kernel(c, t) = 1 / (1 + |c-t| + max(t-c, 0))^2.
       Asymmetric: more mass on over-estimation side (safe), less on under-estimation.

    Args:
        logits: [B, S, K, V] - draft model output
        rank_logits: [B, S, K, rank_classes] - rank head output
        target_p_full: [B, S_full, V_draft] - target probabilities (padded for offset)
        position_mask_full: [B, S_full, 1] - valid position mask (padded)
        loss_mask: [B, S, 1] - loss mask for current block's positions
        K: number of draft positions
        rank_classes: number of rank classes
        target_offset: offset into target_p_full for this block
        budget_per_class: target effective count per class for reweighting. <=0 disables.
        draft_loss_topk: if > 0, compute draft KL only over top-K target tokens per
            position (HASS-style). Draft model focuses capacity on high-probability
            region; long-tail target tokens are ignored. -1 = full vocab KL (default).
            No renormalization: loss magnitude scales with top-K probability mass,
            so low-entropy positions (high top-K mass) contribute more strongly.

    Returns:
        draft_losses: [K] - draft loss for each position
        accs: [K] - accuracy for each position
        rank_loss: scalar - rank CE loss
        rank_acc_per_class: [rank_classes] - per-class rank accuracy
        rank_count_per_class: [rank_classes] - per-class sample counts
        rank_chain_all_correct: scalar bool - whether all rank preds within chain are correct
        rank0_prf: [3] - rank0 precision, recall, F1
    """
    B, S, _, V = logits.shape
    device = logits.device

    # Pre-allocate output tensors
    draft_losses = torch.zeros(K, device=device)
    accs = torch.zeros(K, device=device)
    rank_loss_sum = torch.zeros(1, device=device)
    rank_loss_count = torch.zeros(1, device=device)
    rank_correct_per_class = torch.zeros(rank_classes, device=device)
    rank_count_per_class = torch.zeros(rank_classes, device=device)

    # Chain mask for draft loss (flows across blocks)
    # Intersect loss_mask with position_mask:
    # - Block 1: loss_mask is data mask, position_mask adds vocab filtering
    # - Block 2+: loss_mask is chain_snapshot from prev block, position_mask is redundant
    #   (chain_snapshot already zero where position_mask was zero) but intersection is safe.
    init_mask = loss_mask.squeeze(-1).float() * position_mask_full[:, :S, :].squeeze(-1).float()  # [B, S]
    chain_mask = init_mask  # [B, S]
    chain_snapshot = None  # snapshot at chain_snapshot_slot for next block's loss_mask

    # Rank chain mask: same initialization, breaks at first rank_label > 0
    rank_chain = init_mask.clone()  # [B, S]

    # Track rank correctness within chain (monitoring metric)
    rank_chain_correct_count = torch.zeros(1, device=device)
    rank_chain_total_count = torch.zeros(1, device=device)

    # Rank0 precision/recall counters
    rank0_pred_count = torch.zeros(1, device=device)      # predicted as class 0
    rank0_pred_correct = torch.zeros(1, device=device)     # predicted 0 and truly 0
    rank0_truth_count = torch.zeros(1, device=device)      # truly class 0
    rank0_truth_predicted = torch.zeros(1, device=device)   # truly 0 and predicted 0

    for k in range(K):
        valid_len = S - k
        if valid_len <= 0:
            continue

        # Aligned slicing: slot k at position i predicts target at position i + target_offset + k + 1
        # But target_p_full is already offset, so we slice from (target_offset + k)
        t_start = target_offset + k
        t_end = t_start + valid_len

        logits_k = logits[:, :valid_len, k, :]          # [B, valid_len, V]
        rank_logits_k = rank_logits[:, :valid_len, k, :]  # [B, valid_len, rank_classes]
        target_p_k = target_p_full[:, t_start:t_end, :]   # [B, valid_len, V]
        position_mask_k = position_mask_full[:, t_start:t_end, :]  # [B, valid_len, 1]

        chain_mask_k = chain_mask[:, :valid_len].unsqueeze(-1)
        combined_mask_k = position_mask_k * chain_mask_k

        # === Draft accuracy ===
        pred_tokens = logits_k.argmax(dim=-1)
        target_tokens = target_p_k.argmax(dim=-1)
        correct = (pred_tokens == target_tokens).float() * position_mask_k.squeeze(-1).float()
        mask_sum = position_mask_k.sum().float()
        accs[k] = correct.sum() / mask_sum.clamp(min=1.0)

        # === Draft loss (KL with chain mask, optional top-K restriction) ===
        logits_float = logits_k.float()
        log_probs = F.log_softmax(logits_float, dim=-1)
        combined_sum = combined_mask_k.sum()
        if draft_loss_topk > 0:
            # Top-K target tokens only: gather top-K target probs + aligned log_probs,
            # keep original target prob scale (no renormalize) so entropy shapes loss weight.
            top_vals, top_idx = target_p_k.topk(draft_loss_topk, dim=-1)             # [B, valid_len, topk]
            top_log_probs = torch.gather(log_probs, -1, top_idx)                     # [B, valid_len, topk]
            loss_per_pos = -(top_vals * top_log_probs).sum(dim=-1, keepdim=True)     # [B, valid_len, 1]
            loss = torch.where(combined_sum > 0,
                               (combined_mask_k * loss_per_pos).sum() / combined_sum.clamp_min(1.0),
                               combined_sum)
        else:
            loss = torch.where(combined_sum > 0, -(combined_mask_k * target_p_k * log_probs).sum() / combined_sum.clamp_min(1.0), combined_sum)
        draft_losses[k] = loss

        # === Rank loss: freq reweight + ordinal smoothing + distance weight ===
        rank_labels = _compute_rank_labels(logits_k.detach(), target_tokens, rank_classes)

        # Rank chain mask for this slot (includes this slot, breaks after)
        rank_mask_k = rank_chain[:, :valid_len] * position_mask_k.squeeze(-1)  # [B, valid_len]

        rank_logits_flat = rank_logits_k.reshape(-1, rank_classes)
        rank_labels_flat = rank_labels.reshape(-1)
        rank_mask_flat = rank_mask_k.reshape(-1).float()

        # --- Per-batch frequency reweighting: equalize class contributions ---
        labels_one_hot = F.one_hot(rank_labels_flat, rank_classes).float()  # [N, C]
        slot_class_counts = (labels_one_hot * rank_mask_flat.unsqueeze(-1)).sum(dim=0)  # [C]
        if budget_per_class > 0:
            freq_w = (float(budget_per_class) / slot_class_counts.clamp_min(1)).clamp(max=1.0)  # [C]
            effective_mask = freq_w[rank_labels_flat] * rank_mask_flat  # [N]
        else:
            effective_mask = rank_mask_flat

        # --- Ordinal label smoothing: asymmetric soft targets ---
        soft_targets = _asymmetric_ordinal_smooth(rank_labels_flat, rank_classes)  # [N, C]
        per_sample_ce = -(soft_targets * F.log_softmax(rank_logits_flat, dim=-1)).sum(dim=-1)  # [N]

        # --- Asymmetric distance weight: w(p,t) = 1 + |t-p| + max(t-p, 0) ---
        rank_pred_flat = rank_logits_flat.argmax(dim=-1)  # [N]
        diff = (rank_labels_flat - rank_pred_flat).float()  # positive = under-estimation
        dist_w = 1.0 + diff.abs() + diff.clamp(min=0.0)  # [N]

        # --- Weighted rank loss (freq-reweighted + distance) ---
        rank_loss_k = (per_sample_ce * dist_w * effective_mask).sum()
        rank_loss_sum = rank_loss_sum + rank_loss_k
        rank_loss_count = rank_loss_count + effective_mask.sum()

        # Per-class rank accuracy (vectorized over classes)
        rank_pred = rank_logits_k.argmax(dim=-1)  # [B, valid_len]
        rank_correct = (rank_pred == rank_labels) * rank_mask_k  # [B, valid_len]
        labels_oh = F.one_hot(rank_labels, rank_classes).float()  # [B, valid_len, C]
        mask_expanded = rank_mask_k.unsqueeze(-1)  # [B, valid_len, 1]
        rank_count_per_class += (labels_oh * mask_expanded).sum(dim=(0, 1))  # [C]
        rank_correct_per_class += (labels_oh * mask_expanded * rank_correct.unsqueeze(-1)).sum(dim=(0, 1))  # [C]

        # Rank0 precision/recall: pred=0 vs truth=0
        pred_is_0 = (rank_pred == 0) & rank_mask_k.bool()
        truth_is_0 = (rank_labels == 0) & rank_mask_k.bool()
        rank0_pred_count += pred_is_0.sum()
        rank0_pred_correct += (pred_is_0 & truth_is_0).sum()
        rank0_truth_count += truth_is_0.sum()
        rank0_truth_predicted += (truth_is_0 & pred_is_0).sum()

        # Track rank correctness within chain (monitoring metric)
        rank_chain_correct_count = rank_chain_correct_count + rank_correct.sum()
        rank_chain_total_count = rank_chain_total_count + rank_mask_k.sum()

        # Update rank chain: break after first rank prediction error (curriculum learning)
        rank_chain[:, :valid_len] = rank_chain[:, :valid_len] * (rank_pred == rank_labels).float()

        # Update draft chain mask
        pred_correct = (pred_tokens == target_tokens).float()
        chain_mask[:, :valid_len] = chain_mask[:, :valid_len] * pred_correct

        # Snapshot chain state at specified slot for cross-block continuation
        if k == chain_snapshot_slot:
            chain_snapshot = chain_mask.clone()

    # Average rank loss
    rank_loss = torch.where(rank_loss_count > 0, rank_loss_sum / rank_loss_count.clamp_min(1.0), rank_loss_count)

    # Per-class rank accuracy: [rank_classes]
    rank_acc_per_class = torch.where(rank_count_per_class > 0, rank_correct_per_class / rank_count_per_class.clamp_min(1.0), torch.zeros_like(rank_correct_per_class))

    # Rank0 precision / recall / F1 as a stacked tensor [prec, rec, f1]
    r0_prec = torch.where(rank0_pred_count > 0, rank0_pred_correct / rank0_pred_count.clamp_min(1.0), rank0_pred_count).squeeze(0)
    r0_rec = torch.where(rank0_truth_count > 0, rank0_truth_predicted / rank0_truth_count.clamp_min(1.0), rank0_truth_count).squeeze(0)
    r0_f1_denom = r0_prec + r0_rec
    r0_f1 = torch.where(r0_f1_denom > 0, 2 * r0_prec * r0_rec / r0_f1_denom.clamp_min(1.0), r0_f1_denom)
    rank0_prf = torch.stack([r0_prec, r0_rec, r0_f1])  # [3]

    # Rank chain all-correct (monitoring metric, no longer gates training)
    rank_chain_all_correct = (rank_chain_correct_count >= rank_chain_total_count - 0.5)  # scalar bool

    return draft_losses, accs, rank_loss.squeeze(0), rank_acc_per_class, rank_count_per_class, rank_chain_all_correct, rank0_prf, chain_snapshot




# ============================================================
#                    SpecBlock Model
# ============================================================

class SpecBlockModel(nn.Module):
    """Base class for SpecBlock models."""
    pass


class OnlineSpecBlockModel(SpecBlockModel):
    """
    Online SpecBlock training wrapper with Block-level TTT.

    训练流程：
    Phase 1 (< rank_start_step): 单 block，只训 draft loss
    Phase 2 (>= rank_start_step): 多 block TTT
      - Block 1: condition = target_hidden (3H → H), input = GT tokens，无前置约束
      - Block 2+: condition = draft_hidden from previous block, input = GT tokens (teacher forcing)
      - M 随机采样 uniform(1, K)
      - Block 间不做 filter：下个 block 的 loss_mask 保持原始数据 mask，所有 valid 位置都训练
        （对齐推理分布——推理触发 Block 2 时 slot M-1 正是 top-1 不对，filter "前 M 全对"
        恰好排除了推理会发生的场景）
      - Block 内保留 chain mask（slot k 的 draft loss 只在前 k slot 全对位置算）
      - ttt_m_valid_ratio 保留为监控指标（原 filter 模式下会保留的数据比例）
      - 所有 block 的 loss 累加

    Args:
        draft_model: SpecBlock draft model (LlamaForCausalLMSpecBlock)
        length: Number of draft tokens to predict (K)
        rank_start_step: Step to start training rank head
        num_ttt_blocks: Number of TTT blocks (default 4)
        budget_per_class: Target effective count per class for frequency reweighting (default 10).
            Within each slot, weight(c) = min(1, budget / count_c). <=0 disables.
        draft_loss_topk: HASS-style top-K restriction of draft KL loss. If >0, only the
            top-K target-probability tokens per position contribute to the loss; long-
            tail tokens (that the small draft can rarely learn) are ignored. -1 = full
            vocab KL (default, backward compatible).
    """

    def __init__(
        self,
        draft_model: Eagle3DraftModel,
        length: int = 7,
        rank_start_step: int = 2000,
        num_ttt_blocks: int = 4,
        budget_per_class: int = 10,
        draft_loss_topk: int = -1,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.rank_start_step = rank_start_step
        self.num_ttt_blocks = num_ttt_blocks
        self.budget_per_class = budget_per_class
        self.draft_loss_topk = draft_loss_topk

        config = draft_model.config
        self.rank_classes = getattr(config, 'rank_classes', 4)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        global_step: int = 0,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with multi-block TTT.

        Returns:
            plosses: [draft_loss_0, ..., draft_loss_{K-1}, rank_loss]
                     (accumulated over all blocks)
            vlosses: [] (empty, for compatibility)
            acces: [draft_acc_0, ..., draft_acc_{K-1}, rank_acc]
                   (from last block only)
        """
        batch_size, seq_len, _ = hidden_states.shape
        K = self.length
        device = hidden_states.device
        use_rank_truncation = global_step >= self.rank_start_step

        # Step 1: Compute target probabilities ONCE
        with torch.no_grad():
            target_p_full, position_mask_full = _compute_target_p(
                target=target,
                t2d=self.draft_model.t2d,
                loss_mask=loss_mask,
            )
        del target

        # Step 2: Pad target_p and position_mask for multi-block offset
        # Maximum offset = (num_ttt_blocks - 1) * K + K = num_ttt_blocks * K
        max_offset = self.num_ttt_blocks * K
        target_p_padded = F.pad(target_p_full, (0, 0, 0, max_offset), value=0.0)
        position_mask_padded = F.pad(position_mask_full, (0, 0, 0, max_offset), value=0)

        # Step 3: Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=device,
            )

        # Step 4: Accumulators
        total_draft_losses = torch.zeros(K, device=device)
        total_rank_loss = torch.zeros(1, device=device)
        block1_accs = None  # Block 1 accuracy 
        block1_rank_acc_per_class = None
        block1_rank_count_per_class = None
        block1_rank0_prf = torch.zeros(3, device=device)
        final_rank0_prf = torch.zeros(3, device=device)  # [precision, recall, f1]
        per_block_accs = {}  # block_idx -> ([K] accuracy tensor, target_offset)

        # TTT metrics tracking
        ttt_M_values = []           # M values per block transition
        blocks_trained = 0          # actual number of blocks trained
        curriculum_rank_passed = 0  # how many blocks chained (monitoring)
        ttt_m_valid_ratio = []      # per-transition chain pass ratio

        # Step 5: Multi-block TTT loop
        current_condition = hidden_states      # [B, S, 3H] for block 1
        current_input_ids = input_ids          # [B, S]
        target_offset = 0                      # cumulative offset into target
        use_draft_condition = False             # block 1 uses target hidden
        accumulated_kv_cache = None            # per-layer cached (K, V) from previous blocks
        ttt_valid_counts_list = []             # list of [S] M tensors, one per block
        block1_cross_kv = None                 # Block 1's all-K-slot KV for cross-position attention

        # Phase 1: single block only (draft + rank head foundation)
        # Phase 2: multi-block TTT with chain flow
        max_blocks = self.num_ttt_blocks if use_rank_truncation else 1

        for block_idx in range(max_blocks):
            # Check if target offset is still within valid range
            if target_offset >= seq_len:
                break

            # Build ttt_valid_counts from previous blocks
            ttt_vc = None
            if block_idx > 0 and ttt_valid_counts_list:
                ttt_vc = torch.stack(ttt_valid_counts_list, dim=0)  # [num_prev_blocks, S]

            # Forward pass (with KV cache from previous blocks)
            logits, rank_logits, draft_hidden, new_cross_kv, new_ttt_kv = self.draft_model.forward_parallel(
                hidden_states=current_condition,
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                use_draft_condition=use_draft_condition,
                prev_kv_cache=accumulated_kv_cache,
                ttt_valid_counts=ttt_vc,
                cross_kv_cache=block1_cross_kv,  # None for Block 1, Block 1's all-K-slot for Block 2+
            )  # logits: [B, S, K, V], rank_logits: [B, S, K, 11], draft_hidden: [B, S, K, H]

            # Determine M for next block (needed for chain_snapshot_slot)
            next_M = None
            if block_idx < max_blocks - 1:
                next_M = random.randint(1, K)

            # Compute loss for this block
            draft_losses_b, accs_b, rank_loss_b, rank_acc_per_class_b, rank_count_per_class_b, rank_chain_all_correct_b, rank0_prf_b, chain_snapshot = \
                _compute_batched_loss_acc_and_rank(
                    logits=logits,
                    rank_logits=rank_logits,
                    target_p_full=target_p_padded,
                    position_mask_full=position_mask_padded,
                    loss_mask=loss_mask,
                    K=K,
                    rank_classes=self.rank_classes,
                    target_offset=target_offset,
                    budget_per_class=self.budget_per_class,
                    chain_snapshot_slot=next_M - 1 if next_M is not None else -1,
                    draft_loss_topk=self.draft_loss_topk,
                )

            # Accumulate losses
            total_draft_losses = total_draft_losses + draft_losses_b
            total_rank_loss = total_rank_loss + rank_loss_b

            # Record per-block per-position accuracy (with offset for absolute pos)
            per_block_accs[block_idx] = (accs_b.detach(), target_offset)

            # Save Block 1 accuracy as the main metric
            if block_idx == 0:
                block1_accs = accs_b
                block1_rank_acc_per_class = rank_acc_per_class_b
                block1_rank_count_per_class = rank_count_per_class_b
                block1_rank0_prf = rank0_prf_b
            final_rank0_prf = rank0_prf_b

            blocks_trained = block_idx + 1

            # Prepare next block
            if block_idx < max_blocks - 1:
                M = next_M
                ttt_M_values.append(M)

                # Save Block 1's cross_kv (only on first block)
                if block1_cross_kv is None:
                    block1_cross_kv = new_cross_kv  # Block 1's all-K-slot KV per layer

                # Block 间不做 filter：loss_mask 保持原样（上一个 block 传入的），
                # 所有 valid 位置都进入下一个 block 的训练。
                # 原本这里用 chain_snapshot 替换 loss_mask 做 "前 M slot 全对才训练下一个 block"
                # 的过滤，但这与推理触发 Block 2 的条件冲突——推理触发需要 slot M-1 rank!=0
                # （即 slot M-1 top-1 不对），而 filter 要求 slot M-1 top-1 对。filter 掉的
                # 恰好是推理实际分布下的场景。故去掉 filter，让训练分布覆盖推理分布。
                # 保留 chain_snapshot 的通过率作为监控指标 ttt_m_valid_ratio
                # （= 原 filter 模式下会保留的数据比例）。
                with torch.no_grad():
                    prev_valid_count = loss_mask.sum().clamp_min(1)
                    m_passed_count = chain_snapshot.sum()
                    ttt_m_valid_ratio.append(
                        m_passed_count / prev_valid_count
                    )

                curriculum_rank_passed += 1

                # Store valid counts for ttt_cache mask
                ttt_valid_counts_list.append(
                    torch.full((seq_len,), M, dtype=torch.long, device=device)
                )

                # Update target offset
                target_offset += M

                # Accumulate TTT KV cache for next block (no detach, following Eagle3)
                if accumulated_kv_cache is None:
                    accumulated_kv_cache = new_ttt_kv
                else:
                    accumulated_kv_cache = [
                        (torch.cat([old_k, new_k], dim=2),
                         torch.cat([old_v, new_v], dim=2))
                        for (old_k, old_v), (new_k, new_v)
                        in zip(accumulated_kv_cache, new_ttt_kv)
                    ]

                # Extract draft hidden at truncation slot for next block condition
                current_condition = draft_hidden[:, :, M - 1, :]

                # Teacher forcing: use GT token at offset position as input
                if target_offset < seq_len:
                    offset_ids = torch.arange(seq_len, device=device)
                    shifted_positions = (offset_ids + target_offset).clamp(max=seq_len - 1)
                    current_input_ids = input_ids[:, shifted_positions]  # [B, S] teacher forcing

                use_draft_condition = True  # block 2+ uses draft hidden

        # Convert to lists: [draft_loss_0, ..., draft_loss_{K-1}, rank_loss]
        plosses = [total_draft_losses[k] for k in range(K)]
        plosses.append(total_rank_loss.squeeze(0))

        # Use Block 1 accuracy as main metrics ((acc_0, acc_1, ...))
        acces = [block1_accs[k] for k in range(K)]
        # Append per-class rank acc + count as a dict (for logging)
        # Use Block 1's rank metrics for main reporting
        rank_acc_dict = {}
        for c in range(self.rank_classes):
            rank_acc_dict[f"rank{c}"] = block1_rank_acc_per_class[c]
            rank_acc_dict[f"rank_count{c}"] = block1_rank_count_per_class[c]
        # Add rank0 precision/recall/F1
        rank_acc_dict["rank0_precision"] = block1_rank0_prf[0]
        rank_acc_dict["rank0_recall"] = block1_rank0_prf[1]
        rank_acc_dict["rank0_f1"] = block1_rank0_prf[2]

        # TTT metrics
        rank_acc_dict["ttt_blocks_trained"] = torch.tensor(float(blocks_trained), device=device)
        rank_acc_dict["ttt_curriculum_rank_passed"] = torch.tensor(float(curriculum_rank_passed), device=device)
        if ttt_M_values:
            rank_acc_dict["ttt_avg_M"] = torch.tensor(sum(ttt_M_values) / len(ttt_M_values), device=device)
            # M distribution: count per value 1..K
            for m_val in range(1, K + 1):
                rank_acc_dict[f"ttt_M_{m_val}"] = torch.tensor(
                    float(ttt_M_values.count(m_val)), device=device,
                )
        else:
            rank_acc_dict["ttt_avg_M"] = torch.tensor(0.0, device=device)
        if ttt_m_valid_ratio:
            rank_acc_dict["ttt_m_valid_ratio"] = torch.stack(ttt_m_valid_ratio).mean()
        else:
            rank_acc_dict["ttt_m_valid_ratio"] = torch.tensor(0.0, device=device)

        # Two views of per-position accuracy:
        # 1. block{i}_acc{k}: per-block internal slot accuracy (for TTT debugging)
        # 2. posacc_{N}: absolute position accuracy (for real position semantics)
        for bidx, (block_accs, offset) in per_block_accs.items():
            for k in range(K):
                rank_acc_dict[f"block{bidx}_acc{k+1}"] = block_accs[k]
                rank_acc_dict[f"posacc_{offset+k+1}"] = block_accs[k]

        acces.append(rank_acc_dict)

        return plosses, [], acces


@torch.compile(dynamic=None)
def _compute_target_p(target, t2d, loss_mask):
    """
    Compute target probabilities with vocab mapping.
    
    """
    target_head = target
    target_max_token = target_head.argmax(-1)
    target_mask = t2d[target_max_token]
    target_mask = target_mask[..., None].int()
    position_mask = target_mask * loss_mask
    target_head = target_head[..., t2d]
    target_head = target_head.float()
    target_p = nn.Softmax(dim=2)(target_head)
    target_p = target_p.detach()
    return target_p, position_mask
