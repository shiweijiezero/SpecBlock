"""
SpecBlock Inference Model with KV Cache.

Includes shift mechanism — cross-slot hidden injection between decoder layers.
layer_idx > 0: slot k receives slot k-1's previous-layer hidden, fused via
concat(hidden, shifted) + Linear(2H, H).

Weight-compatible with training model at specforge/modeling/draft/llama3_specblock.py.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._specblock_inference_model_base import _SpecBlockInferenceModelBase


def _apply_shift(hidden, B, N, K, H):
    """Apply shift injection: slot k gets slot k-1's hidden.

    Args:
        hidden: [B, N*K, H] tensor
        B, N, K, H: dimensions

    Returns:
        shifted: [B, N*K, H] tensor where slot k has slot k-1's value,
                 slot 0 has its own value
    """
    h = hidden.view(B, N, K, H)
    shifted = torch.empty_like(h)
    shifted[:, :, 0, :] = h[:, :, 0, :]       # slot 0 <- self
    shifted[:, :, 1:, :] = h[:, :, :-1, :]    # slot k <- slot k-1
    return shifted.view(B, N * K, H)


def _shift_proj_forward(hidden, shifted, proj):
    """Fused projection: W @ [x; s] via single F.linear on cat([x, s]).

    Single matmul on [B, N, 2H] @ [H, 2H]^T outperforms 2 separate
    matmuls + add in eager mode (1 dispatch vs 3, plus Inductor-friendly
    fusion). Originally split into 2 matmul to avoid the cat alloc, but
    PyTorch eager dispatch overhead dominates the cat cost at our shapes.
    """
    combined = torch.cat([hidden, shifted], dim=-1)  # [B, N, 2H]
    return proj(combined)


class SpecBlockInferenceModel(_SpecBlockInferenceModelBase):
    """SpecBlock model for inference with KV cache.

    Extends the internal base with shift_proj layers that inject
    cross-slot hidden state information between consecutive decoder layers.
    """

    def __init__(self, config):
        super().__init__(config)

        H = config.hidden_size
        num_layers = self.num_layers

        # Shift projection: one per layer transition (num_layers - 1)
        if num_layers > 1:
            self.shift_proj = nn.ModuleList([
                nn.Linear(2 * H, H, bias=False)
                for _ in range(num_layers - 1)
            ])
        else:
            self.shift_proj = nn.ModuleList()

    def prepare_for_inference(self, quantize=None):
        """Compile decoder layers, optionally quantize, and pre-split shift_proj weights."""
        # Always pre-split shift_proj weights BEFORE quantization,
        # because quantization replaces nn.Linear with DynamicQuantizedLinear
        # which lacks .weight attribute.
        H = self.config.hidden_size
        for proj in self.shift_proj:
            W = proj.weight.data
            proj._w1 = W[:, :H].contiguous()
            proj._w2 = W[:, H:].contiguous()
        super().prepare_for_inference(quantize=quantize)

    def forward_with_cache(self, hidden, input_ids, cache, position_id,
                           use_draft_condition=False, ttt_cache=None,
                           ttt_mask=None, update_cross_cache=True,
                           full_kv_mask=None):
        """Single position forward for draft, with shift injection.

        Args:
            hidden: [B, 1, 3H] (block 1) or [B, 1, H] (block 2+)
            input_ids: [B, 1]
            cache: list of [k_buf, v_buf, count, max_len] per layer (pre-allocated cache)
            position_id: int, position index for RoPE
            use_draft_condition: True for block 2+ (hidden is H, skip condition_proj)
            ttt_cache: per-layer list of (ttt_k, ttt_v) from prior TTT blocks
            ttt_mask: [B, C_ttt] bool mask
            update_cross_cache: if True, append all K slots to cross_cache
            full_kv_mask: optional pre-built combined cross+ttt mask shared
                across layers (saves per-layer torch.cat allocations).

        Returns:
            logits: [B, K, draft_vocab_size]
            rank_logits: [B, K, rank_classes]
            draft_hidden: [B, K, H]
            new_ttt_kv: per-layer list of (all_k, all_v)
        """
        B = hidden.shape[0]
        K = self.K
        H = self.config.hidden_size

        from . import _specblock_inference_model_base as _sim
        _deep = os.environ.get("STATIC_DRAFT_PROFILE_DEEP", "0") == "1"
        if _deep:
            _ev_il_s = torch.cuda.Event(enable_timing=True); _ev_il_s.record()
        x = self.input_layer(hidden, input_ids, use_draft_condition=use_draft_condition)
        # x: [B, K, H]
        if _deep:
            _ev_il_e = torch.cuda.Event(enable_timing=True); _ev_il_e.record()
            _sim._PROFILE_EVENTS.append(("fwd_input_layer", _ev_il_s, _ev_il_e))
        # Sequential position IDs: slot_k uses position_id + 1 + k
        pos_ids = torch.arange(
            position_id + 1, position_id + 1 + K, device=x.device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)
        max_pos = position_id + K

        if _deep:
            _ev_sh_s = torch.cuda.Event(enable_timing=True); _ev_sh_s.record()
        new_ttt_kv = []
        for layer_idx, layer in enumerate(self.layers):
            # Shift injection before layer (except first layer)
            if layer_idx > 0:
                shifted = _apply_shift(x, B, 1, K, H)
                x = _shift_proj_forward(x, shifted, self.shift_proj[layer_idx - 1])

            layer_ttt = ttt_cache[layer_idx] if ttt_cache is not None else None
            x, all_kv = layer(
                x, cache[layer_idx], pos_ids, max_position=max_pos,
                ttt_cache=layer_ttt, ttt_mask=ttt_mask,
                update_cross_cache=update_cross_cache,
                full_kv_mask=full_kv_mask,
            )
            new_ttt_kv.append(all_kv)
        if _deep:
            _ev_sh_e = torch.cuda.Event(enable_timing=True); _ev_sh_e.record()
            _sim._PROFILE_EVENTS.append(("fwd_layers_shift", _ev_sh_s, _ev_sh_e))

        draft_hidden = x  # [B, K, H] pre-norm
        if _deep:
            _ev_fn_s = torch.cuda.Event(enable_timing=True); _ev_fn_s.record()
        normed = self.norm(x)
        if _deep:
            _ev_fn_e = torch.cuda.Event(enable_timing=True); _ev_fn_e.record()
            _sim._PROFILE_EVENTS.append(("fwd_final_norm", _ev_fn_s, _ev_fn_e))
            _ev_lm_s = torch.cuda.Event(enable_timing=True); _ev_lm_s.record()
        logits = self.lm_head(normed)
        if _deep:
            _ev_lm_e = torch.cuda.Event(enable_timing=True); _ev_lm_e.record()
            _sim._PROFILE_EVENTS.append(("fwd_lm_head", _ev_lm_s, _ev_lm_e))
            _ev_rh_s = torch.cuda.Event(enable_timing=True); _ev_rh_s.record()
        rank_logits = self._rank_forward(normed, logits)
        if _deep:
            _ev_rh_e = torch.cuda.Event(enable_timing=True); _ev_rh_e.record()
            _sim._PROFILE_EVENTS.append(("fwd_rank_head", _ev_rh_s, _ev_rh_e))

        return logits, rank_logits, draft_hidden, new_ttt_kv

    def forward_with_cache_graph(self, hidden, input_ids, pos_ids, cache,
                                 rope_max_position, ttt_cache, ttt_mask,
                                 full_kv_mask):
        """Graph-safe BFS forward (update_cross_cache=False, use_draft_condition=True).

        All inputs are pre-built tensors. No internal arange, no dynamic rope slice,
        no in-place cache mutation. Safe for torch.cuda.CUDAGraph capture.
        """
        B = hidden.shape[0]
        K = self.K
        H = self.config.hidden_size

        x = self.input_layer(hidden, input_ids, use_draft_condition=True)

        new_ttt_kv = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx > 0:
                shifted = _apply_shift(x, B, 1, K, H)
                x = _shift_proj_forward(x, shifted, self.shift_proj[layer_idx - 1])

            layer_ttt = ttt_cache[layer_idx]
            x, all_kv = layer(
                x, cache[layer_idx], pos_ids, max_position=rope_max_position,
                ttt_cache=layer_ttt, ttt_mask=ttt_mask,
                update_cross_cache=False,
                full_kv_mask=full_kv_mask,
            )
            new_ttt_kv.append(all_kv)

        draft_hidden = x
        normed = self.norm(x)
        logits = self.lm_head(normed)
        rank_logits = self._rank_forward(normed, logits)

        return logits, rank_logits, draft_hidden, new_ttt_kv

    def update_cache_and_draft_graph_safe(self, hidden_3h, input_ids, pos_ids, cache,
                                           rope_max_position, N, cross_mask):
        """Graph-safe variant of update_cache_and_draft.

        Differences:
          - pos_ids pre-built [1, N*K] tensor
          - rope_max_position fixed constant (full cos/sin slice)
          - cross_mask [1, cross_count] for padded cross cache
          - No in-place cache write (caller writes + bumps count externally)
          - Returns pre-GQA-repeat KV per layer so caller does repeat_kv once
            just before scatter + slices last-K for TTT seeding
        """
        B, _, _ = hidden_3h.shape
        K = self.K
        H = self.config.hidden_size

        x = self.input_layer.forward_batch(hidden_3h, input_ids)  # [B, N*K, H]

        new_kv_full = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx > 0:
                shifted = _apply_shift(x, B, N, K, H)
                x = _shift_proj_forward(x, shifted, self.shift_proj[layer_idx - 1])

            x, pre_kv = layer.forward_batch_graph_safe(
                x, cache[layer_idx], pos_ids, rope_max_position, N, cross_mask,
            )
            new_kv_full.append(pre_kv)

        last_x = x[:, -K:, :]
        draft_hidden = last_x
        normed = self.norm(last_x)
        logits = self.lm_head(normed)
        rank_logits = self._rank_forward(normed, logits)

        return logits, rank_logits, draft_hidden, new_kv_full

    def forward_with_cache_graph_block1(self, hidden_3h, input_ids, pos_ids, cache,
                                        rope_max_position, cross_mask):
        """Graph-safe block-1 forward.

        Differences from BFS-path graph method:
          - hidden is [B, 1, 3H] (target condition, goes through condition_proj)
          - no TTT cache
          - update_cross_cache stays False inside the graph — caller does the
            scatter write + count bump outside the replay so no Python-list
            assignment is captured.
          - cross_mask is a [B, cross_count] bool tensor covering the padded
            cross cache (True = valid, False = padded).
        """
        B = hidden_3h.shape[0]
        K = self.K
        H = self.config.hidden_size

        x = self.input_layer(hidden_3h, input_ids, use_draft_condition=False)

        new_ttt_kv = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx > 0:
                shifted = _apply_shift(x, B, 1, K, H)
                x = _shift_proj_forward(x, shifted, self.shift_proj[layer_idx - 1])

            x, all_kv = layer(
                x, cache[layer_idx], pos_ids, max_position=rope_max_position,
                ttt_cache=None, ttt_mask=None,
                update_cross_cache=False,
                full_kv_mask=cross_mask,
            )
            new_ttt_kv.append(all_kv)

        draft_hidden = x
        normed = self.norm(x)
        logits = self.lm_head(normed)
        rank_logits = self._rank_forward(normed, logits)

        return logits, rank_logits, draft_hidden, new_ttt_kv

    def _forward_batch_cache(self, hidden_3h, input_ids, cache, start_position):
        """Batch forward for N positions with shift injection. Only updates cache."""
        B, N, _ = hidden_3h.shape
        K = self.K
        H = self.config.hidden_size

        x = self.input_layer.forward_batch(hidden_3h, input_ids)  # [B, N*K, H]

        # Sequential position IDs: position i's slot_k uses (start + i + 1 + k)
        base_pos = torch.arange(
            start_position + 1, start_position + 1 + N, device=x.device
        )
        slot_offsets = torch.arange(K, device=x.device)
        pos_ids = (base_pos.unsqueeze(1) + slot_offsets.unsqueeze(0)).reshape(-1)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)

        max_position = start_position + N + K - 1

        for layer_idx, layer in enumerate(self.layers):
            # Shift injection before layer (except first layer)
            if layer_idx > 0:
                shifted = _apply_shift(x, B, N, K, H)
                x = _shift_proj_forward(x, shifted, self.shift_proj[layer_idx - 1])

            x = layer.forward_batch(x, cache[layer_idx], pos_ids, max_position, N)

    def update_cache_and_draft(self, hidden_3h, input_ids, cache, start_position):
        """Merged cache update + block 1 draft with shift injection.

        Args:
            hidden_3h: [B, N, 3H] where N = accept_length + 1
            input_ids: [B, N]
            cache: cross-position cache (updated in-place)
            start_position: position id of the first token

        Returns:
            logits: [B, K, draft_vocab_size]
            rank_logits: [B, K, rank_classes]
            draft_hidden: [B, K, H]
            new_ttt_kv: per-layer (k, v)
            new_position: start_position + N
        """
        B, N, _ = hidden_3h.shape
        K = self.K
        H = self.config.hidden_size

        x = self.input_layer.forward_batch(hidden_3h, input_ids)  # [B, N*K, H]

        # Sequential position IDs: position i's slot_k uses (start + i + 1 + k)
        base_pos = torch.arange(
            start_position + 1, start_position + 1 + N, device=x.device
        )
        slot_offsets = torch.arange(K, device=x.device)
        pos_ids = (base_pos.unsqueeze(1) + slot_offsets.unsqueeze(0)).reshape(-1)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)
        max_position = start_position + N + K - 1

        new_ttt_kv = []
        for layer_idx, layer in enumerate(self.layers):
            # Shift injection before layer (except first layer)
            if layer_idx > 0:
                shifted = _apply_shift(x, B, N, K, H)
                x = _shift_proj_forward(x, shifted, self.shift_proj[layer_idx - 1])

            x, last_kv = layer.forward_batch(
                x, cache[layer_idx], pos_ids, max_position, N,
                return_last_kv=True,
            )
            new_ttt_kv.append(last_kv)

        # lm_head + rank_head only on last position's K slots
        last_x = x[:, -K:, :]
        draft_hidden = last_x
        normed = self.norm(last_x)
        logits = self.lm_head(normed)
        rank_logits = self._rank_forward(normed, logits)

        return logits, rank_logits, draft_hidden, new_ttt_kv, start_position + N
