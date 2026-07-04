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
from server.training.semantic_actions.arguments import (
    SemanticArgumentPrefix,
)
from server.training.semantic_actions.codec import (
    SEMANTIC_CODEC,
    semantic_argument_id,
)
from server.training.vocab import component_ids
from server.training.vocab_schema import (
    PAD_COMPONENT_IDS,
    TokenComponentIds,
)

OBSERVATION_COMPONENT_COUNT: int = 15


@dataclass(frozen=True, slots=True)
class ObservationTensorBatch:
    """Packed model input tensors for a batch of observations."""

    component_ids: Tensor
    numeric_values: Tensor
    numeric_masks: Tensor


@dataclass(frozen=True, slots=True)
class ObservationComponentTensorBatch:
    """Named tensor views over packed observation component ids."""

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
    batch_token_count = max(
        max(len(observation.tokens), 1) for observation in observations
    )
    assert batch_token_count <= max_observation_tokens
    rows = tuple(
        _observation_rows(
            observation=observation,
            max_observation_tokens=batch_token_count,
        )
        for observation in observations
    )
    numeric_rows = tuple(row.numeric_features for row in rows)
    return ObservationTensorBatch(
        component_ids=_component_tensor_rows(
            tuple(row.components for row in rows), device
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


def stack_observation_batches(
    *,
    batches: tuple[ObservationTensorBatch, ...],
    device: torch.device,
) -> ObservationTensorBatch:
    """Pad and stack already tensorized observation batches."""
    assert batches
    max_observation_tokens = max(
        int(batch.component_ids.shape[1]) for batch in batches
    )
    component_rows: list[Tensor] = []
    numeric_value_rows: list[Tensor] = []
    numeric_mask_rows: list[Tensor] = []
    for batch in batches:
        _validate_observation_tensor_batch(batch)
        for row_index in range(int(batch.component_ids.shape[0])):
            component_rows.append(
                _pad_component_tensor_row(
                    batch.component_ids[row_index].to(device),
                    max_observation_tokens=max_observation_tokens,
                )
            )
            numeric_value_rows.append(
                _pad_numeric_tensor_row(
                    batch.numeric_values[row_index].to(device),
                    max_observation_tokens=max_observation_tokens,
                )
            )
            numeric_mask_rows.append(
                _pad_numeric_tensor_row(
                    batch.numeric_masks[row_index].to(device),
                    max_observation_tokens=max_observation_tokens,
                )
            )
    return ObservationTensorBatch(
        component_ids=torch.stack(tuple(component_rows), dim=0),
        numeric_values=torch.stack(tuple(numeric_value_rows), dim=0),
        numeric_masks=torch.stack(tuple(numeric_mask_rows), dim=0),
    )


def observation_component_tensors(
    batch: ObservationTensorBatch,
) -> ObservationComponentTensorBatch:
    """Return named component-id tensor views for a packed batch."""
    component_ids = batch.component_ids
    assert int(component_ids.shape[2]) == OBSERVATION_COMPONENT_COUNT
    return ObservationComponentTensorBatch(
        token_type_ids=component_ids[:, :, 0],
        segment_ids=component_ids[:, :, 1],
        field_ids=component_ids[:, :, 2],
        value_ids=component_ids[:, :, 3],
        suit_ids=component_ids[:, :, 4],
        rank_ids=component_ids[:, :, 5],
        points_ids=component_ids[:, :, 6],
        color_ids=component_ids[:, :, 7],
        role_ids=component_ids[:, :, 8],
        trick_age_ids=component_ids[:, :, 9],
        trick_state_ids=component_ids[:, :, 10],
        play_order_ids=component_ids[:, :, 11],
        count_ids=component_ids[:, :, 12],
        play_width_ids=component_ids[:, :, 13],
        event_age_ids=component_ids[:, :, 14],
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
    max_argument_tokens = max(
        len(prefix.arguments) + 1 for prefix in prefixes
    )
    assert max_argument_tokens <= SEMANTIC_CODEC.max_argument_tokens
    rows = tuple(
        _argument_prefix_row(
            prefix, max_argument_tokens=max_argument_tokens
        )
        for prefix in prefixes
    )
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
    components: tuple[tuple[int, ...], ...]
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
        _component_values(component_ids(token))
        for token in observation.tokens
    ]
    numeric_items = [
        numeric_feature_values(token) for token in observation.tokens
    ]
    if not component_items:
        component_items.append(_component_values(PAD_COMPONENT_IDS))
        numeric_items.append(PAD_NUMERIC_FEATURES)
    while len(component_items) < max_observation_tokens:
        component_items.append(_component_values(PAD_COMPONENT_IDS))
        numeric_items.append(PAD_NUMERIC_FEATURES)
    return _ObservationRows(
        components=tuple(component_items),
        numeric_features=tuple(numeric_items),
    )


def _argument_prefix_row(
    prefix: SemanticArgumentPrefix,
    *,
    max_argument_tokens: int,
) -> _ArgumentPrefixRow:
    argument_ids = [SEMANTIC_CODEC.argument_bos_id]
    argument_ids.extend(
        semantic_argument_id(argument) for argument in prefix.arguments
    )
    assert len(argument_ids) <= max_argument_tokens
    masks = [True for _ in argument_ids]
    while len(argument_ids) < max_argument_tokens:
        argument_ids.append(0)
        masks.append(False)
    return _ArgumentPrefixRow(
        argument_ids=tuple(argument_ids),
        masks=tuple(masks),
    )


def _component_values(
    values: TokenComponentIds,
) -> tuple[int, ...]:
    return (
        values.token_type,
        values.segment,
        values.field,
        values.value,
        values.suit,
        values.rank,
        values.points,
        values.color,
        values.role,
        values.trick_age,
        values.trick_state,
        values.play_order,
        values.count,
        values.play_width,
        values.event_age,
    )


def _component_tensor_rows(
    values: tuple[tuple[tuple[int, ...], ...], ...],
    device: torch.device,
) -> Tensor:
    return torch.tensor(values, dtype=torch.long, device=device)


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


def _validate_observation_tensor_batch(
    batch: ObservationTensorBatch,
) -> None:
    assert batch.component_ids.ndim == 3
    assert batch.numeric_values.ndim == 3
    assert batch.numeric_masks.ndim == 3
    assert int(batch.component_ids.shape[0]) == int(
        batch.numeric_values.shape[0]
    )
    assert int(batch.component_ids.shape[0]) == int(
        batch.numeric_masks.shape[0]
    )
    assert int(batch.component_ids.shape[1]) == int(
        batch.numeric_values.shape[1]
    )
    assert int(batch.component_ids.shape[1]) == int(
        batch.numeric_masks.shape[1]
    )
    assert int(batch.component_ids.shape[2]) == int(
        OBSERVATION_COMPONENT_COUNT
    )
    assert int(batch.numeric_values.shape[2]) == int(
        batch.numeric_masks.shape[2]
    )


def _pad_component_tensor_row(
    row: Tensor,
    *,
    max_observation_tokens: int,
) -> Tensor:
    assert row.ndim == 2
    assert int(row.shape[0]) <= max_observation_tokens
    assert int(row.shape[1]) == OBSERVATION_COMPONENT_COUNT
    pad_count = max_observation_tokens - int(row.shape[0])
    if pad_count == 0:
        return row
    padding = row.new_full(
        (pad_count, OBSERVATION_COMPONENT_COUNT),
        fill_value=PAD_COMPONENT_IDS.token_type,
    )
    return torch.cat((row, padding), dim=0)


def _pad_numeric_tensor_row(
    row: Tensor,
    *,
    max_observation_tokens: int,
) -> Tensor:
    assert row.ndim == 2
    assert int(row.shape[0]) <= max_observation_tokens
    pad_count = max_observation_tokens - int(row.shape[0])
    if pad_count == 0:
        return row
    padding = row.new_zeros((pad_count, int(row.shape[1])))
    return torch.cat((row, padding), dim=0)
