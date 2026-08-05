"""Observation attention with gated multi-axis RoPE and relations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from server.policy_model.observation.structure import (
    STRUCTURE_AXIS_COUNT,
    StructureAxis,
)

from ._module_call import call_tensor
from .config import MIN_ATTENTION_HEAD_DIMENSION

STRUCTURE_AXIS_DIMENSION = 4
_STRUCTURE_SCORE_DIMENSION = (
    STRUCTURE_AXIS_COUNT * STRUCTURE_AXIS_DIMENSION
)


@dataclass(frozen=True, slots=True)
class _AxisRotation:
    """One batch's reusable RoPE factors for a structure axis."""

    active: Tensor
    cosine: Tensor
    sine: Tensor


@dataclass(frozen=True, slots=True)
class _StructureContext:
    """Reusable structure relations for one observation batch."""

    rotations: tuple[_AxisRotation, ...]
    same_round_event: Tensor
    same_trick: Tensor
    same_play: Tensor


@final
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
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(STRUCTURE_AXIS_DIMENSION // 2)
            / float(STRUCTURE_AXIS_DIMENSION // 2)
        )
        self.register_buffer(
            "_rope_frequencies", frequencies, persistent=False
        )
        self._rope_frequencies: Tensor = frequencies

    @override
    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor,
        encoded_structure_coordinates: Tensor,
    ) -> Tensor:
        """Encode a batch using one-based semantic coordinates."""
        result = values
        structure_context = _structure_context(
            encoded_structure_coordinates,
            frequencies=self._rope_frequencies,
            dtype=values.dtype,
        )
        for layer in self._layers:
            next_result = call_tensor(
                layer,
                result,
                padding_mask=padding_mask,
                structure_context=structure_context,
            )
            result = next_result
        return result


@final
class _StructuredAttentionBlock(nn.Module):
    def __init__(self, *, d_model: int, heads: int) -> None:
        super().__init__()
        self._heads = heads
        self._head_dim = d_model // heads
        self._score_dim = self._head_dim + _STRUCTURE_SCORE_DIMENSION
        self._qkv = nn.Linear(d_model, d_model * 3)
        structure_projection_dimension = (
            heads * _STRUCTURE_SCORE_DIMENSION
        )
        self._structure_query = nn.Linear(
            d_model, structure_projection_dimension, bias=False
        )
        self._structure_key = nn.Linear(
            d_model, structure_projection_dimension, bias=False
        )
        self._output = nn.Linear(d_model, d_model)
        self._same_round_event_bias = nn.Parameter(torch.zeros(heads))
        self._same_trick_bias = nn.Parameter(torch.zeros(heads))
        self._same_play_bias = nn.Parameter(torch.zeros(heads))
        self._norm1 = nn.LayerNorm(d_model)
        self._norm2 = nn.LayerNorm(d_model)
        self._feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    @override
    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor,
        structure_context: _StructureContext,
    ) -> Tensor:
        batch, tokens, d_model = values.shape
        qkv = call_tensor(self._qkv, values)
        projected = qkv.view(
            batch, tokens, 3, self._heads, self._head_dim
        )
        content_query: Tensor = projected[:, :, 0].transpose(1, 2)
        content_key: Tensor = projected[:, :, 1].transpose(1, 2)
        value: Tensor = projected[:, :, 2].transpose(1, 2)
        structure_query_projected = call_tensor(
            self._structure_query, values
        )
        structure_query = self._structure_projection(
            structure_query_projected,
            structure_context=structure_context,
        )
        structure_key_projected = call_tensor(
            self._structure_key, values
        )
        structure_key = self._structure_projection(
            structure_key_projected,
            structure_context=structure_context,
        )
        query = torch.cat((content_query, structure_query), dim=-1)
        key = torch.cat((content_key, structure_key), dim=-1)
        attention_bias = self._relation_bias(structure_context)
        attention_bias = attention_bias.masked_fill(
            padding_mask[:, None, None, :], -torch.inf
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=0.0,
            scale=1.0 / math.sqrt(float(self._score_dim)),
        )
        merged = attended.transpose(1, 2).reshape(
            batch, tokens, d_model
        )
        merged = call_tensor(self._output, merged)
        merged = merged.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden = call_tensor(self._norm1, values + merged)
        feed_forward = call_tensor(self._feed_forward, hidden)
        result = call_tensor(self._norm2, hidden + feed_forward)
        return result

    def _structure_projection(
        self,
        projected: Tensor,
        *,
        structure_context: _StructureContext,
    ) -> Tensor:
        batch, tokens, _dimension = projected.shape
        axes = projected.view(
            batch,
            tokens,
            self._heads,
            STRUCTURE_AXIS_COUNT,
            STRUCTURE_AXIS_DIMENSION,
        ).permute(0, 2, 1, 3, 4)
        rotated: list[Tensor] = []
        for axis in StructureAxis:
            rotated.append(
                _apply_axis_rope(
                    axes[:, :, :, int(axis), :],
                    rotation=structure_context.rotations[int(axis)],
                )
            )
        return torch.cat(rotated, dim=-1)

    def _relation_bias(
        self, structure_context: _StructureContext
    ) -> Tensor:
        return (
            structure_context.same_round_event.unsqueeze(1)
            * self._same_round_event_bias.view(1, self._heads, 1, 1)
            + structure_context.same_trick.unsqueeze(1)
            * self._same_trick_bias.view(1, self._heads, 1, 1)
            + structure_context.same_play.unsqueeze(1)
            * self._same_play_bias.view(1, self._heads, 1, 1)
        )


def _structure_context(
    encoded_structure_coordinates: Tensor,
    *,
    frequencies: Tensor,
    dtype: torch.dtype,
) -> _StructureContext:
    rotations = tuple(
        _axis_rotation(
            encoded_structure_coordinates[:, :, int(axis)],
            frequencies=frequencies,
            dtype=dtype,
        )
        for axis in StructureAxis
    )
    round_event = encoded_structure_coordinates[
        :, :, int(StructureAxis.ROUND_EVENT)
    ]
    trick = encoded_structure_coordinates[
        :, :, int(StructureAxis.TRICK)
    ]
    play = encoded_structure_coordinates[
        :, :, int(StructureAxis.PLAY_POSITION)
    ]
    same_round_event = _same_active_coordinate(round_event)
    same_trick = _same_active_coordinate(trick)
    same_play = same_trick & _same_active_coordinate(play)
    distinct = ~torch.eye(
        int(round_event.shape[1]),
        dtype=torch.bool,
        device=round_event.device,
    ).unsqueeze(0)
    return _StructureContext(
        rotations=rotations,
        same_round_event=same_round_event & distinct,
        same_trick=same_trick & distinct,
        same_play=same_play & distinct,
    )


def _same_active_coordinate(encoded_coordinate: Tensor) -> Tensor:
    active = encoded_coordinate.gt(0)
    return (
        active.unsqueeze(2)
        & active.unsqueeze(1)
        & encoded_coordinate.unsqueeze(2).eq(
            encoded_coordinate.unsqueeze(1)
        )
    )


def _axis_rotation(
    encoded_coordinate: Tensor,
    *,
    frequencies: Tensor,
    dtype: torch.dtype,
) -> _AxisRotation:
    active = encoded_coordinate.gt(0)
    positions = (encoded_coordinate - 1).to(dtype=dtype)
    angles = positions[:, None, :, None] * frequencies.to(dtype=dtype)
    return _AxisRotation(
        active=active[:, None, :, None].to(dtype=dtype),
        cosine=torch.cos(angles),
        sine=torch.sin(angles),
    )


def _apply_axis_rope(
    values: Tensor,
    *,
    rotation: _AxisRotation,
) -> Tensor:
    half = STRUCTURE_AXIS_DIMENSION // 2
    first = values[..., :half]
    second = values[..., half:]
    rotated = torch.cat(
        (
            first * rotation.cosine - second * rotation.sine,
            first * rotation.sine + second * rotation.cosine,
        ),
        dim=-1,
    )
    return rotated * rotation.active


__all__ = ("StructuredObservationEncoder",)
