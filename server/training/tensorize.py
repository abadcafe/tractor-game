"""Tensorization helpers for torch training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.numeric_features import (
    PAD_NUMERIC_FEATURES,
    NumericFeatureValues,
    numeric_feature_values,
)
from server.training.observation import Observation
from server.training.semantic_actions import (
    ARGUMENT_BOS_ID,
    MAX_ARGUMENT_TOKENS,
    SemanticArgumentPrefix,
    semantic_argument_id,
)
from server.training.vocab import (
    PAD_COMPONENT_IDS,
    TokenComponentIds,
    component_ids,
)


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
    return tensorize_observations(
        observations=(observation,),
        max_observation_tokens=max_observation_tokens,
        device=device,
    )


def tensorize_observations(
    *,
    observations: tuple[Observation, ...],
    max_observation_tokens: int,
    device: torch.device,
) -> ObservationTensorBatch:
    """Tensorize observations as one batch."""
    assert max_observation_tokens > 0
    assert observations
    rows = tuple(
        _observation_rows(
            observation=observation,
            max_observation_tokens=max_observation_tokens,
        )
        for observation in observations
    )
    component_rows = tuple(row.components for row in rows)
    numeric_rows = tuple(row.numeric_features for row in rows)
    return ObservationTensorBatch(
        token_type_ids=_long_tensor_rows(
            tuple(
                tuple(item.token_type for item in row)
                for row in component_rows
            ),
            device,
        ),
        segment_ids=_long_tensor_rows(
            tuple(
                tuple(item.segment for item in row)
                for row in component_rows
            ),
            device,
        ),
        field_ids=_long_tensor_rows(
            tuple(
                tuple(item.field for item in row)
                for row in component_rows
            ),
            device,
        ),
        value_ids=_long_tensor_rows(
            tuple(
                tuple(item.value for item in row)
                for row in component_rows
            ),
            device,
        ),
        suit_ids=_long_tensor_rows(
            tuple(
                tuple(item.suit for item in row)
                for row in component_rows
            ),
            device,
        ),
        rank_ids=_long_tensor_rows(
            tuple(
                tuple(item.rank for item in row)
                for row in component_rows
            ),
            device,
        ),
        points_ids=_long_tensor_rows(
            tuple(
                tuple(item.points for item in row)
                for row in component_rows
            ),
            device,
        ),
        color_ids=_long_tensor_rows(
            tuple(
                tuple(item.color for item in row)
                for row in component_rows
            ),
            device,
        ),
        role_ids=_long_tensor_rows(
            tuple(
                tuple(item.role for item in row)
                for row in component_rows
            ),
            device,
        ),
        trick_age_ids=_long_tensor_rows(
            tuple(
                tuple(item.trick_age for item in row)
                for row in component_rows
            ),
            device,
        ),
        trick_state_ids=_long_tensor_rows(
            tuple(
                tuple(item.trick_state for item in row)
                for row in component_rows
            ),
            device,
        ),
        play_order_ids=_long_tensor_rows(
            tuple(
                tuple(item.play_order for item in row)
                for row in component_rows
            ),
            device,
        ),
        count_ids=_long_tensor_rows(
            tuple(
                tuple(item.count for item in row)
                for row in component_rows
            ),
            device,
        ),
        play_width_ids=_long_tensor_rows(
            tuple(
                tuple(item.play_width for item in row)
                for row in component_rows
            ),
            device,
        ),
        event_age_ids=_long_tensor_rows(
            tuple(
                tuple(item.event_age for item in row)
                for row in component_rows
            ),
            device,
        ),
        numeric_values=_numeric_tensor_rows(
            tuple(
                tuple(item.values for item in row)
                for row in numeric_rows
            ),
            device,
        ),
        numeric_masks=_numeric_tensor_rows(
            tuple(
                tuple(item.masks for item in row)
                for row in numeric_rows
            ),
            device,
        ),
    )


def tensorize_argument_prefix(
    *,
    prefix: SemanticArgumentPrefix,
    device: torch.device,
) -> ArgumentPrefixTensorBatch:
    """Tensorize one semantic argument prefix as batch size 1."""
    return tensorize_argument_prefixes(
        prefixes=(prefix,), device=device
    )


def tensorize_argument_prefixes(
    *,
    prefixes: tuple[SemanticArgumentPrefix, ...],
    device: torch.device,
) -> ArgumentPrefixTensorBatch:
    """Tensorize semantic argument prefixes as one batch."""
    assert prefixes
    rows = tuple(_argument_prefix_row(prefix) for prefix in prefixes)
    return ArgumentPrefixTensorBatch(
        argument_ids=_long_tensor_rows(
            tuple(row.argument_ids for row in rows), device
        ),
        argument_masks=_bool_tensor_rows(
            tuple(row.masks for row in rows), device
        ),
    )


@dataclass(frozen=True, slots=True)
class _ObservationRows:
    components: tuple[TokenComponentIds, ...]
    numeric_features: tuple[NumericFeatureValues, ...]


@dataclass(frozen=True, slots=True)
class _ArgumentPrefixRow:
    argument_ids: tuple[int, ...]
    masks: tuple[bool, ...]


def _observation_rows(
    *,
    observation: Observation,
    max_observation_tokens: int,
) -> _ObservationRows:
    assert len(observation.tokens) <= max_observation_tokens
    component_items = [
        component_ids(token) for token in observation.tokens
    ]
    numeric_items = [
        numeric_feature_values(token) for token in observation.tokens
    ]
    if not component_items:
        component_items.append(PAD_COMPONENT_IDS)
        numeric_items.append(PAD_NUMERIC_FEATURES)
    while len(component_items) < max_observation_tokens:
        component_items.append(PAD_COMPONENT_IDS)
        numeric_items.append(PAD_NUMERIC_FEATURES)
    return _ObservationRows(
        components=tuple(component_items),
        numeric_features=tuple(numeric_items),
    )


def _argument_prefix_row(
    prefix: SemanticArgumentPrefix,
) -> _ArgumentPrefixRow:
    argument_ids = [ARGUMENT_BOS_ID]
    argument_ids.extend(
        semantic_argument_id(argument) for argument in prefix.arguments
    )
    assert len(argument_ids) <= MAX_ARGUMENT_TOKENS
    masks = [True for _ in argument_ids]
    while len(argument_ids) < MAX_ARGUMENT_TOKENS:
        argument_ids.append(0)
        masks.append(False)
    return _ArgumentPrefixRow(
        argument_ids=tuple(argument_ids),
        masks=tuple(masks),
    )


def _long_tensor_rows(
    values: tuple[tuple[int, ...], ...], device: torch.device
) -> Tensor:
    return torch.tensor(values, dtype=torch.long, device=device)


def _numeric_tensor_rows(
    values: tuple[tuple[tuple[float, ...], ...], ...],
    device: torch.device,
) -> Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def _bool_tensor_rows(
    values: tuple[tuple[bool, ...], ...], device: torch.device
) -> Tensor:
    return torch.tensor(values, dtype=torch.bool, device=device)
