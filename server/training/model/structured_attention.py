"""Observation attention with gated multi-axis RoPE and relations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from server.training.observation_structure import (
    STRUCTURE_AXIS_COUNT,
    StructureAxis,
)

from .config import MIN_ATTENTION_HEAD_DIMENSION

STRUCTURE_AXIS_DIMENSION = 4
_STRUCTURE_SCORE_DIMENSION = (
    STRUCTURE_AXIS_COUNT * STRUCTURE_AXIS_DIMENSION
)


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
        encoded_structure_coordinates: Tensor,
    ) -> Tensor:
        """Encode a batch using one-based semantic coordinates."""
        result = values
        for layer in self._layers:
            result = layer(
                result,
                padding_mask=padding_mask,
                encoded_structure_coordinates=(
                    encoded_structure_coordinates
                ),
            )
        return result


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
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(STRUCTURE_AXIS_DIMENSION // 2)
            / float(STRUCTURE_AXIS_DIMENSION // 2)
        )
        self.register_buffer(
            "_rope_frequencies", frequencies, persistent=False
        )
        self._rope_frequencies: Tensor = frequencies

    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor,
        encoded_structure_coordinates: Tensor,
    ) -> Tensor:
        batch, tokens, d_model = values.shape
        projected = self._qkv(values).view(
            batch, tokens, 3, self._heads, self._head_dim
        )
        content_query = projected[:, :, 0].transpose(1, 2)
        content_key = projected[:, :, 1].transpose(1, 2)
        value = projected[:, :, 2].transpose(1, 2)
        structure_query = self._structure_projection(
            self._structure_query(values),
            encoded_structure_coordinates=encoded_structure_coordinates,
        )
        structure_key = self._structure_projection(
            self._structure_key(values),
            encoded_structure_coordinates=encoded_structure_coordinates,
        )
        query = torch.cat((content_query, structure_query), dim=-1)
        key = torch.cat((content_key, structure_key), dim=-1)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(self._score_dim))
        scores = scores + self._relation_bias(
            encoded_structure_coordinates
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

    def _structure_projection(
        self,
        projected: Tensor,
        *,
        encoded_structure_coordinates: Tensor,
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
                    encoded_coordinate=(
                        encoded_structure_coordinates[:, :, int(axis)]
                    ),
                    frequencies=self._rope_frequencies,
                )
            )
        return torch.cat(rotated, dim=-1)

    def _relation_bias(
        self, encoded_structure_coordinates: Tensor
    ) -> Tensor:
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
        same_round_event = same_round_event & distinct
        same_trick = same_trick & distinct
        same_play = same_play & distinct
        return (
            same_round_event.unsqueeze(1)
            * self._same_round_event_bias.view(1, self._heads, 1, 1)
            + same_trick.unsqueeze(1)
            * self._same_trick_bias.view(1, self._heads, 1, 1)
            + same_play.unsqueeze(1)
            * self._same_play_bias.view(1, self._heads, 1, 1)
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


def _apply_axis_rope(
    values: Tensor,
    *,
    encoded_coordinate: Tensor,
    frequencies: Tensor,
) -> Tensor:
    half = STRUCTURE_AXIS_DIMENSION // 2
    active = encoded_coordinate.gt(0)
    positions = (encoded_coordinate - 1).to(dtype=values.dtype)
    angles = positions[:, None, :, None] * frequencies.to(
        dtype=values.dtype
    )
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    first = values[..., :half]
    second = values[..., half:]
    rotated = torch.cat(
        (
            first * cosine - second * sine,
            first * sine + second * cosine,
        ),
        dim=-1,
    )
    return rotated * active[:, None, :, None].to(dtype=values.dtype)


__all__ = ("StructuredObservationEncoder",)
