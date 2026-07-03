"""Tensorization helpers for torch training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.numeric_features import (
    PAD_NUMERIC_FEATURES,
    numeric_feature_values,
)
from server.training.observation import Observation
from server.training.semantic_actions import (
    ARGUMENT_BOS_ID,
    MAX_ARGUMENT_TOKENS,
    SemanticArgumentPrefix,
    semantic_argument_id,
)
from server.training.vocab import PAD_COMPONENT_IDS, component_ids


@dataclass(frozen=True, slots=True)
class ObservationTensorBatch:
    """Batch-size-one observation component tensors."""

    token_type_ids: Tensor
    segment_ids: Tensor
    field_ids: Tensor
    value_ids: Tensor
    suit_ids: Tensor
    rank_ids: Tensor
    points_ids: Tensor
    color_ids: Tensor
    role_ids: Tensor
    trick_age_ids: Tensor
    trick_state_ids: Tensor
    play_order_ids: Tensor
    count_ids: Tensor
    play_width_ids: Tensor
    event_age_ids: Tensor
    numeric_values: Tensor
    numeric_masks: Tensor


@dataclass(frozen=True, slots=True)
class ArgumentPrefixTensorBatch:
    """Batch-size-one semantic argument prefix tensors."""

    argument_ids: Tensor
    argument_masks: Tensor


def tensorize_observation(
    *,
    observation: Observation,
    max_observation_tokens: int,
    device: torch.device,
) -> ObservationTensorBatch:
    """Tensorize one observation as batch size 1."""
    assert max_observation_tokens > 0
    assert len(observation.tokens) <= max_observation_tokens
    ids = [component_ids(token) for token in observation.tokens]
    numeric_features = [
        numeric_feature_values(token) for token in observation.tokens
    ]
    if not ids:
        ids.append(PAD_COMPONENT_IDS)
        numeric_features.append(PAD_NUMERIC_FEATURES)
    while len(ids) < max_observation_tokens:
        ids.append(PAD_COMPONENT_IDS)
        numeric_features.append(PAD_NUMERIC_FEATURES)
    return ObservationTensorBatch(
        token_type_ids=_long_tensor(
            [item.token_type for item in ids], device
        ),
        segment_ids=_long_tensor(
            [item.segment for item in ids], device
        ),
        field_ids=_long_tensor([item.field for item in ids], device),
        value_ids=_long_tensor([item.value for item in ids], device),
        suit_ids=_long_tensor([item.suit for item in ids], device),
        rank_ids=_long_tensor([item.rank for item in ids], device),
        points_ids=_long_tensor([item.points for item in ids], device),
        color_ids=_long_tensor([item.color for item in ids], device),
        role_ids=_long_tensor([item.role for item in ids], device),
        trick_age_ids=_long_tensor(
            [item.trick_age for item in ids], device
        ),
        trick_state_ids=_long_tensor(
            [item.trick_state for item in ids], device
        ),
        play_order_ids=_long_tensor(
            [item.play_order for item in ids], device
        ),
        count_ids=_long_tensor([item.count for item in ids], device),
        play_width_ids=_long_tensor(
            [item.play_width for item in ids], device
        ),
        event_age_ids=_long_tensor(
            [item.event_age for item in ids], device
        ),
        numeric_values=_numeric_tensor(
            [item.values for item in numeric_features], device
        ),
        numeric_masks=_numeric_tensor(
            [item.masks for item in numeric_features], device
        ),
    )


def tensorize_argument_prefix(
    *,
    prefix: SemanticArgumentPrefix,
    device: torch.device,
) -> ArgumentPrefixTensorBatch:
    """Tensorize one semantic argument prefix as batch size 1."""
    argument_ids = [ARGUMENT_BOS_ID]
    argument_ids.extend(
        semantic_argument_id(argument) for argument in prefix.arguments
    )
    assert len(argument_ids) <= MAX_ARGUMENT_TOKENS
    masks = [True for _ in argument_ids]
    while len(argument_ids) < MAX_ARGUMENT_TOKENS:
        argument_ids.append(0)
        masks.append(False)
    return ArgumentPrefixTensorBatch(
        argument_ids=_long_tensor(argument_ids, device),
        argument_masks=_bool_tensor(masks, device),
    )


def _long_tensor(values: list[int], device: torch.device) -> Tensor:
    return torch.tensor([values], dtype=torch.long, device=device)


def _numeric_tensor(
    values: list[tuple[float, ...]],
    device: torch.device,
) -> Tensor:
    return torch.tensor([values], dtype=torch.float32, device=device)


def _bool_tensor(values: list[bool], device: torch.device) -> Tensor:
    return torch.tensor([values], dtype=torch.bool, device=device)
