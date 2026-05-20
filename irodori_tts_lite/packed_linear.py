"""Int4-packed Linear and Embedding modules for RTN extra layers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_utils import dequant_extra_u8_to_weight


class PackedRTNLinear(nn.Module):
    """nn.Linear-compatible RTN int4 packed layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight_u8: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer("qweight_u8", qweight_u8.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("zeros", zeros.contiguous())
        if bias is not None:
            self.register_parameter(
                "bias", nn.Parameter(bias.detach(), requires_grad=False)
            )
        else:
            self.bias = None

    def _materialize(self, dtype: torch.dtype) -> torch.Tensor:
        return dequant_extra_u8_to_weight(
            self.qweight_u8,
            self.scales,
            self.zeros,
            self.in_features,
            self.out_features,
            dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._materialize(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, packed=int4"
        )


class PackedEmbedding(nn.Module):
    """nn.Embedding-compatible RTN int4 packed embedding table."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        qweight_u8: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx
        self.register_buffer("qweight_u8", qweight_u8.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("zeros", zeros.contiguous())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1).long()
        rows_q = self.qweight_u8[flat]
        rows_s = self.scales[flat]
        rows_z = self.zeros[flat]
        dtype = (
            self.scales.dtype
            if self.scales.dtype.is_floating_point
            else torch.float16
        )
        rows = dequant_extra_u8_to_weight(
            rows_q,
            rows_s,
            rows_z,
            self.embedding_dim,
            flat.shape[0],
            dtype,
        )
        return rows.reshape(*input_ids.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, packed=int4"
        )
