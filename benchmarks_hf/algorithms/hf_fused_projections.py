"""Packed target projections for Llama HF inference."""

from __future__ import annotations

import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PackedProjection(nn.Module):
    def __init__(
        self,
        weights: tuple[torch.Tensor, ...],
        fuse_min_rows: int,
        fuse_max_rows: int | None = None,
    ):
        super().__init__()
        if not weights:
            raise ValueError("packed projection requires at least one weight")
        in_features = weights[0].shape[1]
        if any(weight.dim() != 2 or weight.shape[1] != in_features for weight in weights):
            raise ValueError("packed projection weights must share input width")
        packed = torch.cat(weights, dim=0).contiguous()
        self.weight = nn.Parameter(packed, requires_grad=False)
        self.output_sizes = tuple(int(weight.shape[0]) for weight in weights)
        self.offsets = [0]
        for size in self.output_sizes:
            self.offsets.append(self.offsets[-1] + size)
        self.in_features = int(in_features)
        self.fuse_min_rows = int(fuse_min_rows)
        self.fuse_max_rows = None if fuse_max_rows is None else int(fuse_max_rows)
        self._pending: torch.Tensor | None = None
        self._pending_input: tuple[int, tuple[int, ...]] | None = None

    def _slice_weight(self, index: int) -> torch.Tensor:
        return self.weight[self.offsets[index]:self.offsets[index + 1]]

    def project(self, index: int, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"projection input width {inputs.shape[-1]} != {self.in_features}"
            )
        rows = inputs.numel() // self.in_features
        use_fused = rows >= self.fuse_min_rows and (
            self.fuse_max_rows is None or rows <= self.fuse_max_rows
        )
        input_key = (inputs.data_ptr(), tuple(inputs.shape))

        if index == 0:
            if self._pending is not None:
                raise RuntimeError("previous fused projection was not fully consumed")
            if not use_fused:
                return F.linear(inputs, self._slice_weight(0))
            self._pending = F.linear(inputs, self.weight)
            self._pending_input = input_key
        elif self._pending is None:
            if use_fused:
                raise RuntimeError("fused projection consumers were called out of order")
            return F.linear(inputs, self._slice_weight(index))
        elif self._pending_input != input_key:
            raise RuntimeError("fused projection consumers received different inputs")

        output = self._pending[..., self.offsets[index]:self.offsets[index + 1]]
        if index + 1 == len(self.output_sizes):
            self._pending = None
            self._pending_input = None
        return output


class _ProjectionView(nn.Module):
    def __init__(self, packed: _PackedProjection, index: int):
        super().__init__()
        object.__setattr__(self, "_packed_ref", weakref.ref(packed))
        self.index = int(index)
        self.in_features = packed.in_features
        self.out_features = packed.output_sizes[index]

    @property
    def weight(self) -> torch.Tensor:
        packed = self._packed_ref()
        if packed is None:
            raise RuntimeError("packed projection owner was released")
        return packed._slice_weight(self.index)

    @property
    def bias(self):
        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        packed = self._packed_ref()
        if packed is None:
            raise RuntimeError("packed projection owner was released")
        return packed.project(self.index, inputs)


def _require_biasless(linears: tuple[nn.Linear, ...], name: str) -> None:
    if not all(isinstance(linear, nn.Linear) for linear in linears):
        raise TypeError(f"{name} fusion requires nn.Linear projections")
    if any(linear.bias is not None for linear in linears):
        raise ValueError(f"{name} fusion requires bias-free projections")


def fuse_llama_target_projections(
    model: nn.Module,
    *,
    gate_up_fuse_min_rows: int = 129,
) -> None:
    """Use native packed GEMMs for small-row QKV and large-row gate/up."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("target model does not expose model.layers")

    for layer in layers:
        attention = layer.self_attn
        qkv = (attention.q_proj, attention.k_proj, attention.v_proj)
        _require_biasless(qkv, "QKV")
        packed_qkv = _PackedProjection(
            tuple(projection.weight.detach() for projection in qkv),
            fuse_min_rows=1,
            fuse_max_rows=128,
        )
        attention.fused_qkv_proj = packed_qkv
        attention.q_proj = _ProjectionView(packed_qkv, 0)
        attention.k_proj = _ProjectionView(packed_qkv, 1)
        attention.v_proj = _ProjectionView(packed_qkv, 2)

        mlp = layer.mlp
        gate_up = (mlp.gate_proj, mlp.up_proj)
        _require_biasless(gate_up, "gate/up")
        packed_gate_up = _PackedProjection(
            tuple(projection.weight.detach() for projection in gate_up),
            fuse_min_rows=gate_up_fuse_min_rows,
        )
        mlp.fused_gate_up_proj = packed_gate_up
        mlp.gate_proj = _ProjectionView(packed_gate_up, 0)
        mlp.up_proj = _ProjectionView(packed_gate_up, 1)
