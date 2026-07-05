"""Tensorization helpers for torch training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.observation import Observation
from server.training.packed_observation import (
    OBSERVATION_COMPONENT_COUNT,
    PackedObservation,
    pack_observation,
    padded_packed_observation,
)
from server.training.semantic_actions.arguments import (
    SemanticArgumentPrefix,
)
from server.training.semantic_actions.codec import (
    SEMANTIC_CODEC,
    semantic_argument_id,
)


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
    assert observations
    return tensorize_packed_observations(
        observations=tuple(
            pack_observation(observation)
            for observation in observations
        ),
        max_observation_tokens=max_observation_tokens,
        device=device,
    )


def tensorize_packed_observations(
    *,
    observations: tuple[PackedObservation, ...],
    max_observation_tokens: int,
    device: torch.device,
) -> ObservationTensorBatch:
    """Tensorize packed observations as one batch."""
    assert max_observation_tokens > 0
    assert observations
    batch_token_count = max(
        observation.token_count() for observation in observations
    )
    assert batch_token_count <= max_observation_tokens
    rows = tuple(
        padded_packed_observation(
            observation,
            max_observation_tokens=batch_token_count,
        )
        for observation in observations
    )
    return ObservationTensorBatch(
        component_ids=_component_tensor_rows(
            tuple(row.component_rows for row in rows), device
        ),
        numeric_values=_numeric_tensor_rows(
            tuple(row.numeric_value_rows for row in rows), device
        ),
        numeric_masks=_numeric_tensor_rows(
            tuple(row.numeric_mask_rows for row in rows), device
        ),
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
            tuple(
                semantic_argument_id(argument)
                for argument in prefix.arguments
            ),
            max_argument_tokens=max_argument_tokens,
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


def tensorize_argument_id_prefixes(
    *,
    prefix_argument_ids: tuple[tuple[int, ...], ...],
    device: torch.device,
) -> ArgumentPrefixTensorBatch:
    """Tensorize semantic argument id prefixes as one batch."""
    assert prefix_argument_ids
    max_argument_tokens = max(
        len(argument_ids) + 1 for argument_ids in prefix_argument_ids
    )
    assert max_argument_tokens <= SEMANTIC_CODEC.max_argument_tokens
    rows = tuple(
        _argument_prefix_row(
            argument_ids,
            max_argument_tokens=max_argument_tokens,
        )
        for argument_ids in prefix_argument_ids
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
class _ArgumentPrefixRow:
    argument_ids: tuple[int, ...]
    masks: tuple[bool, ...]


def _argument_prefix_row(
    prefix_argument_ids: tuple[int, ...],
    *,
    max_argument_tokens: int,
) -> _ArgumentPrefixRow:
    argument_ids = [SEMANTIC_CODEC.argument_bos_id]
    argument_ids.extend(prefix_argument_ids)
    assert len(argument_ids) <= max_argument_tokens
    masks = [True for _ in argument_ids]
    while len(argument_ids) < max_argument_tokens:
        argument_ids.append(0)
        masks.append(False)
    return _ArgumentPrefixRow(
        argument_ids=tuple(argument_ids),
        masks=tuple(masks),
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
