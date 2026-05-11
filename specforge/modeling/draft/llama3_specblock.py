"""
SpecBlock: 在 decoder layers 之间注入 slot 间信息传递

核心改动：在 Layer 1 和 Layer 2 之间，将 slot k-1 的 Layer 1 输出 hidden
注入到 slot k 的 Layer 2 输入中。通过 concat + Linear(2H, H) 融合。

训推完全一致（纯架构改动），每层仍然完全并行。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers import LlamaConfig

from ._llama3_specblock_base import _LlamaForCausalLMSpecBlockBase


class LlamaForCausalLMSpecBlock(_LlamaForCausalLMSpecBlockBase):
    """
    SpecBlock: 在 decoder layers 间注入 slot 间信息传递。

    在 layer_idx > 0 时，将 slot k-1 的前一层输出注入到 slot k，
    通过 concat(hidden, shifted) + Linear(2H, H) 融合。
    """

    config_class = LlamaConfig

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        num_layers = len(self.layers)
        H = config.hidden_size

        # Shift projection: one per layer transition (num_layers - 1)
        if num_layers > 1:
            self.shift_proj = nn.ModuleList([
                nn.Linear(2 * H, H, bias=False)
                for _ in range(num_layers - 1)
            ])
        else:
            self.shift_proj = nn.ModuleList()

    def forward_parallel(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_draft_condition: bool = False,
        prev_kv_cache: Optional[list] = None,
        ttt_valid_counts: Optional[torch.Tensor] = None,
        cross_kv_cache: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, list]:
        """
        并行前向传播，带 shift 注入。

        与父类完全一致，唯一区别：在 layer_idx > 0 时做 shift 注入。
        """
        from ._llama3_specblock_base import (
            get_cached_cross_position_causal_mask,
            build_block2plus_mask,
            build_ttt_block_mask,
        )
        import torch.nn.functional as F

        B, S, _ = hidden_states.shape
        K = self.K
        H = self.hidden_size
        total_len = S * K

        # ===== 输入层 =====
        inputs = self.input_layer(
            hidden_states, input_ids, attention_mask,
            use_draft_condition=use_draft_condition,
        )

        # Reshape for decoder: [B, S, K, H] -> [B, S*K, H]
        hidden = inputs.view(B, total_len, H)

        # ===== Position IDs =====
        # Sequential: position i 的 slot k 使用 position_id = i + 1 + k
        base_pos = torch.arange(1, S + 1, device=hidden.device)
        slot_offsets = torch.arange(K, device=hidden.device)
        position_ids = (base_pos.unsqueeze(1) + slot_offsets.unsqueeze(0)).reshape(-1)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)

        # ===== Pre-compute max position =====
        max_position = S + K

        # ===== Slot_0 indices (precompute once) =====
        slot0_indices = torch.arange(0, total_len, K, device=hidden.device)

        # ===== Attention Mask & Combined Cache =====
        if cross_kv_cache is not None:
            C_ttt = prev_kv_cache[0][0].shape[2] if prev_kv_cache is not None else 0
            block_mask = build_block2plus_mask(
                S, K, C_ttt, ttt_valid_counts, B, hidden_states.device,
            )
            combined_cache = []
            for l in range(len(self.layers)):
                cross_k, cross_v = cross_kv_cache[l]
                if prev_kv_cache is not None:
                    ttt_k, ttt_v = prev_kv_cache[l]
                    combined_cache.append((
                        torch.cat([cross_k, ttt_k], dim=2),
                        torch.cat([cross_v, ttt_v], dim=2),
                    ))
                else:
                    combined_cache.append((cross_k, cross_v))
        elif prev_kv_cache is not None:
            C = prev_kv_cache[0][0].shape[2]
            block_mask = build_ttt_block_mask(
                S, K, C, ttt_valid_counts, B, hidden_states.device,
            )
            combined_cache = prev_kv_cache
        else:
            block_mask = get_cached_cross_position_causal_mask(S, K, B, str(hidden_states.device))
            combined_cache = None

        # ===== Decoder Layers with Shift Injection =====
        new_all_kv = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx > 0:
                # Shift injection: slot k gets slot k-1's hidden from previous layer
                # hidden shape: [B, S*K, H]
                h = hidden.view(B, S, K, H)
                shifted = torch.empty_like(h)
                shifted[:, :, 0, :] = h[:, :, 0, :]      # slot 0 <- 自身
                shifted[:, :, 1:, :] = h[:, :, :-1, :]  # slot k <- slot k-1
                shifted = shifted.view(B, S * K, H)
                # Split projection: W @ [x; s] = W[:,:H] @ x + W[:,H:] @ s
                # Avoids allocating the [B, S*K, 2H] cat tensor
                W = self.shift_proj[layer_idx - 1].weight  # [H, 2H]
                hidden = F.linear(hidden, W[:, :H]) + F.linear(shifted, W[:, H:])

            layer_cache = combined_cache[layer_idx] if combined_cache is not None else None
            hidden, all_kv = layer(
                hidden, block_mask, position_ids, max_position,
                kv_cache=layer_cache, slot0_indices=slot0_indices,
            )
            new_all_kv.append(all_kv)

        # ===== 输出层 =====
        hidden = hidden.view(B, S, K, H)
        draft_hidden = hidden

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        # Rank Head: top-20 fast path (must stay aligned with parent class
        # and SpecBlockInferenceModel._rank_forward).
        with torch.no_grad():
            top20_logits, _ = logits.detach().topk(20, dim=-1)
            top20_logits = top20_logits.float()
            top20_lse = torch.logsumexp(top20_logits, dim=-1, keepdim=True)

            top10_logits = top20_logits[..., :10]
            top10_lp = top10_logits - top20_lse
            top10_p = top10_lp.exp()
            gap_12 = top10_logits[..., 0:1] - top10_logits[..., 1:2]
            gap_13 = top10_logits[..., 0:1] - top10_logits[..., 2:3]
            gap_15 = top10_logits[..., 0:1] - top10_logits[..., 4:5]
            cumprob_top1 = top10_p[..., 0:1]

            top20_lp = top20_logits - top20_lse
            top20_p = top20_lp.exp()
            entropy = -(top20_p * top20_lp).nan_to_num(0.0).sum(-1, keepdim=True)

            rank_features = torch.cat(
                [top10_lp, gap_12, gap_13, gap_15, cumprob_top1, entropy], dim=-1,
            ).to(hidden.dtype)
        rank_input = torch.cat([hidden.detach(), rank_features], dim=-1)
        rank_logits = self.rank_head(rank_input)

        # cross_kv and ttt_kv are now the same (all K slots)
        return logits, rank_logits, draft_hidden, new_all_kv, new_all_kv
