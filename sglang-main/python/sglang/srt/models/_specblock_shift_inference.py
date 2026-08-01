"""
SpecBlock Shift Inference Model with KV Cache

在 SpecBlockInferenceModel 基础上增加 shift 机制：
在 decoder layers 之间注入 slot 间信息传递。
layer_idx > 0 时，将 slot k-1 的前一层输出注入到 slot k，
通过 concat(hidden, shifted) + Linear(2H, H) 融合。

与训练模型 (specforge/modeling/draft/llama3_specblock_shift.py) 权重兼容。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._specblock_inference import SpecBlockInferenceModel, repeat_kv


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


class SpecBlockShiftInferenceModel(SpecBlockInferenceModel):
    """SpecBlock Shift model for inference with KV cache.

    Extends SpecBlockInferenceModel with shift_proj layers that inject
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

        from . import _specblock_inference as _sim
        _deep = os.environ.get("STATIC_DRAFT_PROFILE_DEEP", "0") == "1"
        if _deep:
            _ev_il_s = torch.cuda.Event(enable_timing=True); _ev_il_s.record()
        x = self.input_layer(hidden, input_ids, use_draft_condition=use_draft_condition)
        # x: [B, K, H]
        if _deep:
            _ev_il_e = torch.cuda.Event(enable_timing=True); _ev_il_e.record()
            _sim._PROFILE_EVENTS.append(("fwd_input_layer", _ev_il_s, _ev_il_e))
        # Sequential position IDs: row b's slot_k uses base[b] + 1 + k.
        # Accept int (legacy single-req) or [B] tensor (multi-req batch).
        if torch.is_tensor(position_id):
            base = position_id.to(x.device, dtype=torch.long)
            if base.numel() == 1:
                base_int = int(base.item())
                pos_ids = torch.arange(
                    base_int + 1, base_int + 1 + K,
                    device=x.device, dtype=torch.long,
                ).unsqueeze(0).expand(B, -1)
                max_pos = base_int + K
            else:
                slot_offsets = torch.arange(
                    K, device=x.device, dtype=torch.long,
                )
                pos_ids = (base.unsqueeze(1) + 1) + slot_offsets.unsqueeze(0)
                max_pos = int(base.max().item()) + K
        else:
            pos_ids = torch.arange(
                position_id + 1, position_id + 1 + K,
                device=x.device, dtype=torch.long,
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

    def update_cache_and_draft_graph_safe(
        self,
        hidden_3h,
        input_ids,
        pos_ids,
        cache,
        rope_max_position,
        N,
        cross_mask,
        n_per_req,
    ):
        """Graph-safe refresh forward for padded request and accept buckets.

        Persistent cross KV is read through paged cache tuples but never
        mutated in the graph.  ``n_per_req`` selects each request's real last
        accepted position from the padded ``N`` bucket.  The method returns
        both GQA-expanded full current KV for graph-external masked scatter and
        non-expanded real-last-position KV for next-iteration TTT seeding.
        """
        B, _, _ = hidden_3h.shape
        K = self.K
        H = self.config.hidden_size

        x = self.input_layer.forward_batch(hidden_3h, input_ids)
        n_dev = n_per_req.to(x.device, dtype=torch.long)
        slot_offsets = torch.arange(K, device=x.device, dtype=torch.long)
        gather_pos = (
            (n_dev - 1).unsqueeze(1) * K + slot_offsets.unsqueeze(0)
        )

        new_cross_kv = []
        new_ttt_kv = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx > 0:
                shifted = _apply_shift(x, B, N, K, H)
                x = _shift_proj_forward(x, shifted, self.shift_proj[layer_idx - 1])

            x, pre_kv = layer.forward_batch_graph_safe(
                x, cache[layer_idx], pos_ids, rope_max_position, N, cross_mask,
            )
            pre_k, pre_v = pre_kv
            new_cross_kv.append(
                (
                    repeat_kv(pre_k, layer.self_attn.n_kv_groups),
                    repeat_kv(pre_v, layer.self_attn.n_kv_groups),
                )
            )
            kv_gather_idx = (
                gather_pos.unsqueeze(1).unsqueeze(-1)
                .expand(B, pre_k.shape[1], K, pre_k.shape[-1])
            )
            new_ttt_kv.append(
                (
                    torch.gather(pre_k, dim=2, index=kv_gather_idx),
                    torch.gather(pre_v, dim=2, index=kv_gather_idx),
                )
            )

        hidden_gather_idx = gather_pos.unsqueeze(-1).expand(B, K, H)
        last_x = torch.gather(x, dim=1, index=hidden_gather_idx)
        draft_hidden = last_x
        normed = self.norm(last_x)
        logits = self.lm_head(normed)
        rank_logits = self._rank_forward(normed, logits)

        return (
            logits,
            rank_logits,
            draft_hidden,
            new_cross_kv,
            new_ttt_kv,
        )

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

    def update_cache_and_draft(self, hidden_3h, input_ids, cache, start_position,
                               cross_mask=None, n_per_req=None,
                               flashinfer_cross=None,
                               max_position_hint=None):
        """Merged cache update + block 1 draft with shift injection.

        Args:
            hidden_3h: [B, N, 3H] where N = accept_length + 1
            input_ids: [B, N]
            cache: cross-position cache (updated in-place).  Paged layout
                supports either 1D [count] cross_loc (single-req or shared
                start) or 2D [B, max_count] cross_loc (batched per-req).
            start_position: int (all reqs share same start) OR
                torch.Tensor [B] / List[int] (per-req start).  When per-req,
                each row's pos_ids is computed independently and
                rope_max_position uses the global max.
            cross_mask: Optional [B, max_cross] bool tensor.  Required when
                start_position is batched and per-req cross_count differs;
                ``cache[i][2]`` should be set to the *padded* max_cross and
                cross_mask flags valid slots (True) vs padded (False).
            n_per_req: Optional [B] long tensor giving the *real* N_i per row
                when ``hidden_3h`` is padded to max_N.  Used to:
                  (a) gather draft_hidden / logits / rank_logits at the
                      real ``(N_i - 1)`` position instead of padded tail.
                  (b) compute per-req new_position = start_position + N_i.
                  (c) gather TTT KV at the real last position via layer.

        Returns:
            logits: [B, K, draft_vocab_size]
            rank_logits: [B, K, rank_classes]
            draft_hidden: [B, K, H]
            new_ttt_kv: per-layer (k, v)
            new_position: int (scalar mode) OR torch.Tensor [B] (batched mode)
        """
        B, N, _ = hidden_3h.shape
        K = self.K
        H = self.config.hidden_size

        x = self.input_layer.forward_batch(hidden_3h, input_ids)  # [B, N*K, H]

        # Sequential position IDs.  Two paths:
        #   scalar start_position: all B reqs share start (V1 behavior),
        #     pos_ids broadcast from a single [N*K] vector.
        #   batched start_position [B]: per-req pos_ids; max_position is
        #     global max so the cos/sin lookup table covers every row.
        if isinstance(start_position, (list, tuple)):
            start_position = torch.tensor(
                start_position, dtype=torch.long, device=x.device,
            )

        if isinstance(start_position, torch.Tensor):
            assert start_position.numel() == B, (
                f"batched start_position must have B={B} entries, "
                f"got {start_position.numel()}"
            )
            start_position = start_position.to(x.device, dtype=torch.long)
            n_arange = torch.arange(N, device=x.device, dtype=torch.long)
            base_pos = (
                start_position.unsqueeze(1) + 1 + n_arange.unsqueeze(0)
            )  # [B, N]
            slot_offsets = torch.arange(K, device=x.device, dtype=torch.long)
            pos_ids = (
                base_pos.unsqueeze(2) + slot_offsets.view(1, 1, K)
            ).reshape(B, N * K)
            # max_position needs to be int for RoPE table sizing.  Caller can
            # supply max_position_hint (cheap host-side max) to avoid the
            # GPU->CPU sync from .max().item().
            if max_position_hint is not None:
                max_position = int(max_position_hint) + N + K - 1
            else:
                max_position = int(start_position.max().item()) + N + K - 1
            new_position = start_position + N  # [B]
        else:
            base_pos = torch.arange(
                start_position + 1, start_position + 1 + N, device=x.device,
            )
            slot_offsets = torch.arange(K, device=x.device)
            pos_ids = (
                base_pos.unsqueeze(1) + slot_offsets.unsqueeze(0)
            ).reshape(-1)
            pos_ids = pos_ids.unsqueeze(0).expand(B, -1)
            max_position = start_position + N + K - 1
            new_position = start_position + N  # int

        new_ttt_kv = []
        for layer_idx, layer in enumerate(self.layers):
            # Shift injection before layer (except first layer)
            if layer_idx > 0:
                shifted = _apply_shift(x, B, N, K, H)
                x = _shift_proj_forward(x, shifted, self.shift_proj[layer_idx - 1])

            x, last_kv = layer.forward_batch(
                x, cache[layer_idx], pos_ids, max_position, N,
                return_last_kv=True, cross_mask=cross_mask,
                n_per_req=n_per_req, flashinfer_cross=flashinfer_cross,
            )
            new_ttt_kv.append(last_kv)

        # lm_head + rank_head only on last position's K slots.
        # Padded batching: per-req gather at real (N_i - 1) position.
        if n_per_req is not None:
            n_dev = n_per_req.to(x.device, dtype=torch.long)
            K_idx = torch.arange(K, device=x.device, dtype=torch.long)
            offs = (n_dev - 1) * K  # [B]
            gather_idx = offs.unsqueeze(1) + K_idx.unsqueeze(0)  # [B, K]
            gather_idx = gather_idx.unsqueeze(-1).expand(B, K, H)
            last_x = torch.gather(x, dim=1, index=gather_idx)
            # new_position is per-req base + real N_i.
            if isinstance(new_position, torch.Tensor):
                # Recompute: was start_position + max_N (padded).
                new_position = start_position + n_dev
        else:
            last_x = x[:, -K:, :]
        draft_hidden = last_x
        normed = self.norm(last_x)
        logits = self.lm_head(normed)
        rank_logits = self._rank_forward(normed, logits)

        return logits, rank_logits, draft_hidden, new_ttt_kv, new_position
