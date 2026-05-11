"""Draft Forward CUDA Graph Cache for SpecBlock BFS forward.

Scope: BFS path only (update_cross_cache=False, use_draft_condition=True).
Block 1 (update_cross_cache=True) keeps eager — cache write is in-place and
not trivially graph-safe.

Buckets: (B_bucket, cc_bucket, ttt_count). On first call per key we capture;
subsequent calls copy inputs into static buffers and replay.

Correctness relies on `full_kv_mask` masking the cc_bucket padding region to
False so softmax ignores uninitialized KV slots.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch


# ============================================================
#   Bucket definitions
# ============================================================

CC_BUCKETS: Tuple[int, ...] = (
    256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096,
    4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192,
)
B_BUCKETS: Tuple[int, ...] = (2, 4, 8, 16, 32)

# Block-1 (B=1 fixed) uses a finer cc granularity because cross_count grows by
# K per iter and the padded-to-bucket attention FLOPs are what eat the dispatch
# savings. For typical prompts (~200-500 tokens) this caps padding ~25%.
B1_CC_BUCKETS: Tuple[int, ...] = (
    128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584,
    4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192,
)

# Single-bucket override: when set, every iter uses the single max cc bucket.
# Trades extra attention FLOPs for ONE static shape throughout, which
# eliminates inter-iter bf16 tile-shift drift seen with multi-bucket layouts.
# Bucket size tunable via DRAFT_GRAPH_MAX_CC (default 4096 — must cover longest
# mtbench sample's cc growth).
if os.environ.get("DRAFT_GRAPH_SINGLE_BUCKET", "0") == "1":
    _single_cc = int(os.environ.get("DRAFT_GRAPH_MAX_CC", "4096"))
    CC_BUCKETS = (_single_cc,)
    B1_CC_BUCKETS = (_single_cc,)

# Override B bucket: useful when TREE_FIXED_N fixes pending count
# (e.g., DRAFT_GRAPH_FIXED_B=10 with TREE_FIXED_N=10 avoids the 10→16 padding).
_fixed_b = int(os.environ.get("DRAFT_GRAPH_FIXED_B", "0"))
if _fixed_b > 0:
    B_BUCKETS = (_fixed_b,)


def _b1_cc_bucket_for(cc: int) -> int:
    for b in B1_CC_BUCKETS:
        if cc <= b:
            return b
    return B1_CC_BUCKETS[-1]


class OutOfBucketError(RuntimeError):
    """Signals caller to fallback to eager path."""


def _cc_bucket_for(cc: int) -> int:
    for b in CC_BUCKETS:
        if cc <= b:
            return b
    return CC_BUCKETS[-1]


def _b_bucket_for(b: int) -> int:
    for B in B_BUCKETS:
        if b <= B:
            return B
    return B_BUCKETS[-1]


class DraftForwardGraphCache:
    """CUDA graph cache keyed by (B_bucket, cc_bucket, ttt_count)."""

    def __init__(
        self,
        draft_model,
        max_batch: int = 10,
        max_cross_count: int = 2048,
        verbose: bool = False,
    ):
        self.model = draft_model
        self.verbose = verbose

        attn0 = draft_model.layers[0].self_attn
        self.K = int(attn0.K)
        self.H = int(attn0.H)
        self.n_heads = int(attn0.n_heads)
        self.n_kv_heads = int(attn0.n_kv_heads)
        self.head_dim = int(attn0.head_dim)
        self.num_layers = int(draft_model.num_layers)

        params_iter = draft_model.parameters()
        p0 = next(params_iter)
        self.device = p0.device
        self.dtype = p0.dtype

        rope = attn0.rope
        self.rope_max_pos = int(getattr(rope, "max_pos", 8192))
        # max_position param must satisfy (max_position + 1) <= rope_max_pos so that
        # self.rope(rope_len) returns the FULL cached cos/sin (no dynamic slice).
        self.rope_max_position = self.rope_max_pos - 1

        self.max_batch = max_batch
        self.max_cross_count = max_cross_count

        self.graphs: dict = {}
        self.static_inputs: dict = {}
        self.static_outputs: dict = {}

        # Separate dicts for block-1 path (B=1, no ttt, different signature).
        self.b1_graphs: dict = {}
        self.b1_static_inputs: dict = {}
        self.b1_static_outputs: dict = {}

        # Dicts for update_cache_and_draft path, keyed by (N, cc_bucket).
        self.upd_graphs: dict = {}
        self.upd_static_inputs: dict = {}
        self.upd_static_outputs: dict = {}

        # Cache the target_hidden_size (3H) once — used to size block-1 static
        # hidden buffers ([1, 1, target_H * 3]).
        self.target_H3 = int(draft_model.input_layer.condition_proj.in_features)

        # Allow disabling specific buckets via env (debug).
        self._disable = os.environ.get("DRAFT_CUDA_GRAPH_DISABLE", "") == "1"

        # View-based capture: use real cross_cache tensors (as expand views)
        # instead of fresh static buffers. Eliminates per-replay cross-cache
        # copy. Requires real cache pre-allocated at max_cache_len.
        self._view_based = os.environ.get("DRAFT_GRAPH_VIEW_BASED", "0") == "1"

    # --------------------------------------------------------
    #   Capture
    # --------------------------------------------------------

    def _capture(self, key, real_cross_slices=None):
        """Capture a graph for `key=(B_bucket, cc_bucket, ttt_count)`.

        If `real_cross_slices` is provided (DRAFT_GRAPH_VIEW_BASED=1 path),
        the cross cache used by the captured graph is an `expand()` view over
        the real draft cache tensors — eliminating the per-replay copy_ of
        ~B_bucket*n_heads*cc_bucket*head_dim*2 bytes (≈30-500 MB). Trade-off:
        the real cache tensors MUST remain at the same data_ptr for the life
        of the graph (no `_ensure_cache_buf` reallocation). Caller ensures
        this by pre-allocating at max_cache_len.
        """
        B_bucket, cc_bucket, ttt_count = key
        K = self.K
        H = self.H
        device = self.device
        dtype = self.dtype

        hidden = torch.zeros(B_bucket, 1, H, device=device, dtype=dtype)
        input_ids = torch.zeros(B_bucket, 1, device=device, dtype=torch.long)
        # Dummy pos_ids within rope cache range (safe indexing).
        pos_ids = torch.zeros(B_bucket, K, device=device, dtype=torch.long)

        cross_cache = []
        use_view = real_cross_slices is not None and all(
            s is not None for s in real_cross_slices
        )
        for l in range(self.num_layers):
            if use_view:
                # View into real cache: [:, :, :cc_bucket, :] expanded to [B_bucket, ...]
                src_k, src_v = real_cross_slices[l]
                # src shape: [1, n_heads, max_cache_len, D] (pre-allocated at max)
                assert src_k.shape[2] >= cc_bucket, (
                    f"real cache cc={src_k.shape[2]} < bucket={cc_bucket}"
                )
                k = src_k[:, :, :cc_bucket, :].expand(B_bucket, -1, -1, -1)
                v = src_v[:, :, :cc_bucket, :].expand(B_bucket, -1, -1, -1)
            else:
                # Fresh static buffer (default path — requires copy at replay).
                k = torch.zeros(
                    B_bucket, self.n_heads, cc_bucket, self.head_dim,
                    device=device, dtype=dtype,
                )
                v = torch.zeros_like(k)
            cross_cache.append([k, v, cc_bucket, cc_bucket])

        # TTT cache: pre-GQA, n_kv_heads wide.
        ttt_cache = []
        for _ in range(self.num_layers):
            tk = torch.zeros(
                B_bucket, self.n_kv_heads, ttt_count, self.head_dim,
                device=device, dtype=dtype,
            )
            tv = torch.zeros_like(tk)
            ttt_cache.append((tk, tv))

        ttt_mask = torch.zeros(B_bucket, ttt_count, device=device, dtype=torch.bool)
        full_kv_mask = torch.zeros(
            B_bucket, cc_bucket + ttt_count, device=device, dtype=torch.bool
        )

        def _run():
            return self.model.forward_with_cache_graph(
                hidden=hidden,
                input_ids=input_ids,
                pos_ids=pos_ids,
                cache=cross_cache,
                rope_max_position=self.rope_max_position,
                ttt_cache=ttt_cache,
                ttt_mask=ttt_mask,
                full_kv_mask=full_kv_mask,
            )

        # Warmup to populate any lazy caches (causal_mask, rope bf16 cache, etc).
        for _ in range(3):
            _run()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits, rank_logits, draft_hidden, new_ttt_kv = _run()

        self.graphs[key] = graph
        self.static_inputs[key] = {
            "hidden": hidden,
            "input_ids": input_ids,
            "pos_ids": pos_ids,
            "cross_k": [c[0] for c in cross_cache],
            "cross_v": [c[1] for c in cross_cache],
            "ttt_k": [t[0] for t in ttt_cache],
            "ttt_v": [t[1] for t in ttt_cache],
            "ttt_mask": ttt_mask,
            "full_kv_mask": full_kv_mask,
        }
        self.static_outputs[key] = {
            "logits": logits,
            "rank_logits": rank_logits,
            "draft_hidden": draft_hidden,
            "new_ttt_k": [kv[0] for kv in new_ttt_kv],
            "new_ttt_v": [kv[1] for kv in new_ttt_kv],
        }

        if self.verbose:
            print(f"  [graph] captured key={key}")

    # --------------------------------------------------------
    #   Replay
    # --------------------------------------------------------

    def run(
        self,
        hidden: torch.Tensor,             # [N, 1, H]
        input_ids: torch.Tensor,          # [N, 1]
        cross_cache_slices: List,         # per-layer (k_view, v_view) or None
        effective_cross_count: int,       # real cc value
        ttt_cache: List,                  # per-layer (k, v) each [N, n_kv, ttt_count, D]
        ttt_mask: torch.Tensor,           # [N, ttt_count]
        position_id: int,                 # abs position for pos_ids arange base
    ):
        """Run BFS forward via graph replay."""
        if self._disable:
            raise RuntimeError("DRAFT_CUDA_GRAPH_DISABLE=1, graph runner disabled")

        N = hidden.shape[0]
        ttt_count = int(ttt_mask.shape[1])
        cc = int(effective_cross_count)
        if cc > CC_BUCKETS[-1] or N > B_BUCKETS[-1]:
            # Out of bucket range — signal caller to fallback to eager path.
            raise OutOfBucketError(
                f"N={N} or cc={cc} exceeds max bucket (B={B_BUCKETS[-1]}, cc={CC_BUCKETS[-1]})"
            )

        key = (_b_bucket_for(N), _cc_bucket_for(cc), ttt_count)
        B_bucket, cc_bucket, _ = key

        if key not in self.graphs:
            # View-based: pass real cache tensors so capture uses expand views
            # (avoids the per-replay copy). Otherwise fresh static buffer.
            real_for_capture = cross_cache_slices if self._view_based else None
            self._capture(key, real_cross_slices=real_for_capture)

        static = self.static_inputs[key]
        K = self.K
        device = self.device

        # --- Pos_ids: arange(position_id+1, position_id+1+K) repeated B_bucket times ---
        pos_range = torch.arange(
            position_id + 1, position_id + 1 + K, device=device, dtype=torch.long
        )
        static["pos_ids"].copy_(pos_range.unsqueeze(0).expand(B_bucket, -1))

        # --- Hidden + input_ids ---
        static["hidden"][:N].copy_(hidden)
        static["input_ids"][:N].copy_(input_ids)
        if N < B_bucket:
            static["hidden"][N:].zero_()
            static["input_ids"][N:].zero_()

        # --- Cross cache ---
        # In view-based mode the static cross_k/v IS an expand view of the real
        # cache; the data is already there, no copy needed.
        if not self._view_based:
            for l in range(self.num_layers):
                sk = static["cross_k"][l]
                sv = static["cross_v"][l]
                if cross_cache_slices[l] is not None:
                    src_k, src_v = cross_cache_slices[l]
                    # src shape: [1, n_heads, cc, D] → expand to [B_bucket, ...]
                    if cc > 0:
                        expanded_k = src_k[:, :, :cc, :].expand(B_bucket, -1, -1, -1)
                        expanded_v = src_v[:, :, :cc, :].expand(B_bucket, -1, -1, -1)
                        sk[:, :, :cc, :].copy_(expanded_k)
                        sv[:, :, :cc, :].copy_(expanded_v)
                if cc < cc_bucket:
                    sk[:, :, cc:, :].zero_()
                    sv[:, :, cc:, :].zero_()

        # --- TTT cache ---
        for l in range(self.num_layers):
            src_k, src_v = ttt_cache[l]
            static["ttt_k"][l][:N].copy_(src_k)
            static["ttt_v"][l][:N].copy_(src_v)
            if N < B_bucket:
                static["ttt_k"][l][N:].zero_()
                static["ttt_v"][l][N:].zero_()

        # --- TTT mask ---
        static["ttt_mask"][:N].copy_(ttt_mask)
        if N < B_bucket:
            static["ttt_mask"][N:].zero_()

        # --- Full KV mask: cross [:cc]=True, cross [cc:cc_bucket]=False, ttt=ttt_mask ---
        fm = static["full_kv_mask"]
        # First N rows: real mask
        if cc > 0:
            fm[:N, :cc].fill_(True)
        fm[:N, cc:cc_bucket].fill_(False)
        fm[:N, cc_bucket : cc_bucket + ttt_count].copy_(ttt_mask)
        if N < B_bucket:
            fm[N:].fill_(False)

        # --- Replay ---
        self.graphs[key].replay()

        # --- Clone outputs (truncate to raw N) ---
        out = self.static_outputs[key]
        logits = out["logits"][:N].clone()
        rank_logits = out["rank_logits"][:N].clone()
        draft_hidden = out["draft_hidden"][:N].clone()
        new_ttt_kv = [
            (
                out["new_ttt_k"][l][:N].clone(),
                out["new_ttt_v"][l][:N].clone(),
            )
            for l in range(self.num_layers)
        ]
        return logits, rank_logits, draft_hidden, new_ttt_kv

    # ============================================================
    #   Block-1 path: B=1, no TTT, variable cc, cache-write-outside
    # ============================================================

    def _capture_block1(self, cc_bucket: int):
        """Capture block-1 forward for a specific cross_count bucket.

        B=1 fixed. No TTT. `update_cross_cache` stays False inside the graph;
        caller writes the returned all_k/all_v into the real cache outside.
        """
        K = self.K
        device = self.device
        dtype = self.dtype
        H3 = self.target_H3

        hidden = torch.zeros(1, 1, H3, device=device, dtype=dtype)
        input_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
        pos_ids = torch.zeros(1, K, device=device, dtype=torch.long)

        cross_cache = []
        for _ in range(self.num_layers):
            k = torch.zeros(
                1, self.n_heads, cc_bucket, self.head_dim,
                device=device, dtype=dtype,
            )
            v = torch.zeros_like(k)
            cross_cache.append([k, v, cc_bucket, cc_bucket])

        cross_mask = torch.zeros(1, cc_bucket, device=device, dtype=torch.bool)

        def _run():
            return self.model.forward_with_cache_graph_block1(
                hidden_3h=hidden,
                input_ids=input_ids,
                pos_ids=pos_ids,
                cache=cross_cache,
                rope_max_position=self.rope_max_position,
                cross_mask=cross_mask,
            )

        for _ in range(3):
            _run()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits, rank_logits, draft_hidden, new_ttt_kv = _run()

        self.b1_graphs[cc_bucket] = graph
        self.b1_static_inputs[cc_bucket] = {
            "hidden": hidden,
            "input_ids": input_ids,
            "pos_ids": pos_ids,
            "cross_k": [c[0] for c in cross_cache],
            "cross_v": [c[1] for c in cross_cache],
            "cross_mask": cross_mask,
        }
        self.b1_static_outputs[cc_bucket] = {
            "logits": logits,
            "rank_logits": rank_logits,
            "draft_hidden": draft_hidden,
            "new_ttt_k": [kv[0] for kv in new_ttt_kv],
            "new_ttt_v": [kv[1] for kv in new_ttt_kv],
        }

        if self.verbose:
            print(f"  [graph] captured block1 cc_bucket={cc_bucket}")

    def run_block1(
        self,
        hidden_3h: torch.Tensor,       # [1, 1, 3H]
        input_ids: torch.Tensor,       # [1, 1]
        draft_cache: list,             # real cache (per-layer [k, v, count, max_len])
        position_id: int,
    ):
        """Replay block-1 forward, scatter new_k/new_v to real cache outside.

        Returns: (logits, rank_logits, draft_hidden, new_ttt_kv)
        new_ttt_kv is per-layer (all_k, all_v) pre-GQA, for downstream use.
        """
        if self._disable:
            raise RuntimeError("DRAFT_CUDA_GRAPH_DISABLE=1, graph runner disabled")

        # cross_count = cache count at entry (before this forward writes)
        cc = int(draft_cache[0][2]) if draft_cache[0][0] is not None else 0
        if cc > B1_CC_BUCKETS[-1]:
            raise OutOfBucketError(
                f"block1 cc={cc} exceeds max bucket {B1_CC_BUCKETS[-1]}"
            )

        cc_bucket = _b1_cc_bucket_for(cc)
        if cc_bucket not in self.b1_graphs:
            self._capture_block1(cc_bucket)

        static = self.b1_static_inputs[cc_bucket]
        K = self.K
        device = self.device

        # pos_ids = arange(position_id+1, position_id+1+K)
        pos_range = torch.arange(
            position_id + 1, position_id + 1 + K, device=device, dtype=torch.long
        )
        static["pos_ids"].copy_(pos_range.unsqueeze(0))

        # hidden, input_ids
        static["hidden"].copy_(hidden_3h)
        static["input_ids"].copy_(input_ids)

        # Cross cache: copy real [0:cc] from draft_cache into static buffer
        for l in range(self.num_layers):
            sk = static["cross_k"][l]
            sv = static["cross_v"][l]
            if cc > 0 and draft_cache[l][0] is not None:
                sk[:, :, :cc, :].copy_(draft_cache[l][0][:, :, :cc, :])
                sv[:, :, :cc, :].copy_(draft_cache[l][1][:, :, :cc, :])
            if cc < cc_bucket:
                sk[:, :, cc:, :].zero_()
                sv[:, :, cc:, :].zero_()

        # Cross mask: True for [:cc], False for [cc:cc_bucket]
        cm = static["cross_mask"]
        if cc > 0:
            cm[:, :cc].fill_(True)
        cm[:, cc:cc_bucket].fill_(False)

        # Replay
        self.b1_graphs[cc_bucket].replay()

        # Clone outputs (guard against being overwritten by subsequent work)
        out = self.b1_static_outputs[cc_bucket]
        logits = out["logits"].clone()
        rank_logits = out["rank_logits"].clone()
        draft_hidden = out["draft_hidden"].clone()
        new_ttt_kv = [
            (
                out["new_ttt_k"][l].clone(),
                out["new_ttt_v"][l].clone(),
            )
            for l in range(self.num_layers)
        ]

        # --- Outside-graph cache scatter write ---
        # new_k/new_v returned are pre-GQA ([1, n_kv_heads, K, D]); the real
        # cross cache stores post-GQA ([1, n_heads, K, D]). Do the repeat_kv
        # here and write.
        from ._specblock_inference_model_base import repeat_kv as _repeat_kv
        n_kv_groups = self.n_heads // self.n_kv_heads
        for l in range(self.num_layers):
            all_k, all_v = new_ttt_kv[l]
            new_k_full = _repeat_kv(all_k, n_kv_groups)  # [1, n_heads, K, D]
            new_v_full = _repeat_kv(all_v, n_kv_groups)
            lc = draft_cache[l]
            # Lazy-alloc cache buffer if needed (mirrors _ensure_cache_buf behaviour).
            from ._specblock_inference_model_base import SpecBlockAttentionWithCache as _Attn
            _Attn._ensure_cache_buf(lc, cc + K, new_k_full)
            lc[0][:, :, cc:cc + K, :] = new_k_full
            lc[1][:, :, cc:cc + K, :] = new_v_full
            lc[2] = cc + K

        return logits, rank_logits, draft_hidden, new_ttt_kv

    # ============================================================
    #   update_cache_and_draft path: B=1, batch N positions (accept_length+1)
    # ============================================================

    def _capture_upd(self, n_val: int, cc_bucket: int):
        """Capture update_cache_and_draft for given (N, cc_bucket)."""
        K = self.K
        device = self.device
        dtype = self.dtype
        H3 = self.target_H3
        NK = n_val * K

        hidden = torch.zeros(1, n_val, H3, device=device, dtype=dtype)
        input_ids = torch.zeros(1, n_val, device=device, dtype=torch.long)
        pos_ids = torch.zeros(1, NK, device=device, dtype=torch.long)

        cross_cache = []
        for _ in range(self.num_layers):
            k = torch.zeros(
                1, self.n_heads, cc_bucket, self.head_dim,
                device=device, dtype=dtype,
            )
            v = torch.zeros_like(k)
            cross_cache.append([k, v, cc_bucket, cc_bucket])

        cross_mask = torch.zeros(1, cc_bucket, device=device, dtype=torch.bool)

        def _run():
            return self.model.update_cache_and_draft_graph_safe(
                hidden_3h=hidden,
                input_ids=input_ids,
                pos_ids=pos_ids,
                cache=cross_cache,
                rope_max_position=self.rope_max_position,
                N=n_val,
                cross_mask=cross_mask,
            )

        for _ in range(3):
            _run()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits, rank_logits, draft_hidden, new_kv_full = _run()

        key = (n_val, cc_bucket)
        self.upd_graphs[key] = graph
        self.upd_static_inputs[key] = {
            "hidden": hidden,
            "input_ids": input_ids,
            "pos_ids": pos_ids,
            "cross_k": [c[0] for c in cross_cache],
            "cross_v": [c[1] for c in cross_cache],
            "cross_mask": cross_mask,
        }
        self.upd_static_outputs[key] = {
            "logits": logits,
            "rank_logits": rank_logits,
            "draft_hidden": draft_hidden,
            "new_k": [kv[0] for kv in new_kv_full],
            "new_v": [kv[1] for kv in new_kv_full],
        }

        if self.verbose:
            print(f"  [graph] captured upd N={n_val} cc_bucket={cc_bucket}")

    def run_update_cache(
        self,
        hidden_3h: torch.Tensor,      # [1, N, 3H]
        input_ids: torch.Tensor,      # [1, N]
        draft_cache: list,
        start_position: int,
    ):
        """Replay update_cache_and_draft; do cache scatter + count bump outside.

        Returns: (logits, rank_logits, draft_hidden, new_ttt_kv, new_position)
        where new_ttt_kv is per-layer (pre-GQA k, v) sliced to last K for TTT seeding.
        """
        if self._disable:
            raise RuntimeError("DRAFT_CUDA_GRAPH_DISABLE=1")

        B, N, _ = hidden_3h.shape
        assert B == 1
        K = self.K
        NK = N * K
        cc = int(draft_cache[0][2]) if draft_cache[0][0] is not None else 0

        if cc > B1_CC_BUCKETS[-1]:
            raise OutOfBucketError(f"upd cc={cc} exceeds max {B1_CC_BUCKETS[-1]}")

        cc_bucket = _b1_cc_bucket_for(cc)
        key = (N, cc_bucket)
        if key not in self.upd_graphs:
            self._capture_upd(N, cc_bucket)

        static = self.upd_static_inputs[key]
        device = self.device

        # pos_ids: for i in 0..N-1, slot k uses start+1+i+k
        base_pos = torch.arange(start_position + 1, start_position + 1 + N,
                                device=device, dtype=torch.long)
        slot_offsets = torch.arange(K, device=device, dtype=torch.long)
        pos_ids_flat = (base_pos.unsqueeze(1) + slot_offsets.unsqueeze(0)).reshape(-1)
        static["pos_ids"].copy_(pos_ids_flat.unsqueeze(0))

        # hidden, input_ids
        static["hidden"].copy_(hidden_3h)
        static["input_ids"].copy_(input_ids)

        # Cross cache: copy [0:cc] real, zero [cc:cc_bucket]
        for l in range(self.num_layers):
            sk = static["cross_k"][l]
            sv = static["cross_v"][l]
            if cc > 0 and draft_cache[l][0] is not None:
                sk[:, :, :cc, :].copy_(draft_cache[l][0][:, :, :cc, :])
                sv[:, :, :cc, :].copy_(draft_cache[l][1][:, :, :cc, :])
            if cc < cc_bucket:
                sk[:, :, cc:, :].zero_()
                sv[:, :, cc:, :].zero_()

        # Cross mask
        cm = static["cross_mask"]
        if cc > 0:
            cm[:, :cc].fill_(True)
        cm[:, cc:cc_bucket].fill_(False)

        # Replay
        self.upd_graphs[key].replay()

        out = self.upd_static_outputs[key]
        logits = out["logits"].clone()
        rank_logits = out["rank_logits"].clone()
        draft_hidden = out["draft_hidden"].clone()
        new_kv_full = [
            (out["new_k"][l].clone(), out["new_v"][l].clone())
            for l in range(self.num_layers)
        ]

        # External cache write: repeat_kv + scatter N*K positions + bump count
        from ._specblock_inference_model_base import repeat_kv as _repeat_kv
        from ._specblock_inference_model_base import SpecBlockAttentionWithCache as _Attn
        n_kv_groups = self.n_heads // self.n_kv_heads
        for l in range(self.num_layers):
            k_pre, v_pre = new_kv_full[l]  # [1, n_kv_heads, NK, D]
            k_post = _repeat_kv(k_pre, n_kv_groups)  # [1, n_heads, NK, D]
            v_post = _repeat_kv(v_pre, n_kv_groups)
            lc = draft_cache[l]
            _Attn._ensure_cache_buf(lc, cc + NK, k_post)
            lc[0][:, :, cc:cc + NK, :] = k_post
            lc[1][:, :, cc:cc + NK, :] = v_post
            lc[2] = cc + NK

        # last-K slots (pre-GQA) per layer for TTT seeding
        new_ttt_kv = [
            (kv[0][:, :, -K:, :].contiguous(), kv[1][:, :, -K:, :].contiguous())
            for kv in new_kv_full
        ]

        return logits, rank_logits, draft_hidden, new_ttt_kv, start_position + N
