"""Observation attention with multi-axis RoPE and relation biases."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from server.training.config import MIN_ATTENTION_HEAD_DIMENSION


class StructuredObservationEncoder(nn.Module):
    """Apply repeated structure-aware self-attention blocks."""

    def __init__(
        self, *, d_model: int, layers: int, heads: int
    ) -> None:
        super().__init__()
        assert d_model % heads == 0
        assert d_model // heads >= MIN_ATTENTION_HEAD_DIMENSION
        self._layers = nn.ModuleList(
            _StructuredAttentionBlock(d_model=d_model, heads=heads)
            for _ in range(layers)
        )

    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor,
        coordinates: Tensor,
        coordinate_masks: Tensor,
    ) -> Tensor:
        """Encode a batch using semantic structure coordinates."""
        result = values
        for layer in self._layers:
            result = layer(
                result,
                padding_mask=padding_mask,
                coordinates=coordinates,
                coordinate_masks=coordinate_masks,
            )
        return result


class _StructuredAttentionBlock(nn.Module):
    def __init__(self, *, d_model: int, heads: int) -> None:
        super().__init__()
        self._heads = heads
        self._head_dim = d_model // heads
        axis_budget = self._head_dim // 4
        self._axis_dim = axis_budget - axis_budget % 2
        assert self._axis_dim >= 2
        self._qkv = nn.Linear(d_model, d_model * 3)
        self._output = nn.Linear(d_model, d_model)
        self._same_axis_bias = nn.Parameter(torch.zeros(3, heads))
        self._norm1 = nn.LayerNorm(d_model)
        self._norm2 = nn.LayerNorm(d_model)
        self._feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor,
        coordinates: Tensor,
        coordinate_masks: Tensor,
    ) -> Tensor:
        batch, tokens, d_model = values.shape
        projected = self._qkv(values).view(
            batch, tokens, 3, self._heads, self._head_dim
        )
        query = projected[:, :, 0].transpose(1, 2)
        key = projected[:, :, 1].transpose(1, 2)
        value = projected[:, :, 2].transpose(1, 2)
        for axis in range(3):
            start = axis * self._axis_dim
            query = _apply_axis_rope(
                query,
                coordinate=coordinates[:, :, axis],
                active=coordinate_masks[:, :, axis],
                start=start,
                width=self._axis_dim,
            )
            key = _apply_axis_rope(
                key,
                coordinate=coordinates[:, :, axis],
                active=coordinate_masks[:, :, axis],
                start=start,
                width=self._axis_dim,
            )
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(self._head_dim))
        scores = scores + self._relation_bias(
            coordinates=coordinates,
            coordinate_masks=coordinate_masks,
        )
        scores = scores.masked_fill(
            padding_mask[:, None, None, :], -torch.inf
        )
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, value)
        merged = attended.transpose(1, 2).reshape(
            batch, tokens, d_model
        )
        merged = self._output(merged)
        merged = merged.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden = self._norm1(values + merged)
        return self._norm2(hidden + self._feed_forward(hidden))

    def _relation_bias(
        self, *, coordinates: Tensor, coordinate_masks: Tensor
    ) -> Tensor:
        batch, tokens, _axes = coordinates.shape
        result = torch.zeros(
            (batch, self._heads, tokens, tokens),
            dtype=self._same_axis_bias.dtype,
            device=coordinates.device,
        )
        for axis in range(3):
            active = coordinate_masks[:, :, axis]
            same = (
                active.unsqueeze(2)
                & active.unsqueeze(1)
                & coordinates[:, :, axis]
                .unsqueeze(2)
                .eq(coordinates[:, :, axis].unsqueeze(1))
            )
            result = result + (
                same.unsqueeze(1)
                * self._same_axis_bias[axis].view(1, self._heads, 1, 1)
            )
        return result


def _apply_axis_rope(
    values: Tensor,
    *,
    coordinate: Tensor,
    active: Tensor,
    start: int,
    width: int,
) -> Tensor:
    segment = values[..., start : start + width]
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=values.dtype, device=values.device)
        / float(half)
    )
    positions = coordinate.to(dtype=values.dtype) * active.to(
        dtype=values.dtype
    )
    angles = positions[:, None, :, None] * frequencies
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    first = segment[..., :half]
    second = segment[..., half:]
    rotated = torch.cat(
        (
            first * cosine - second * sine,
            first * sine + second * cosine,
        ),
        dim=-1,
    )
    return torch.cat(
        (
            values[..., :start],
            rotated,
            values[..., start + width :],
        ),
        dim=-1,
    )


__all__ = ("StructuredObservationEncoder",)
