"""Tensorization of compact typed observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.observation import Observation
from server.training.packed_observation import (
    MAX_LOSSLESS_OBSERVATION_TOKENS,
    PackedObservation,
    pack_observation,
    padded_packed_observation,
)
from server.training.semantic_actions.choices import CARD_CHOICE_COUNT
from server.training.tensor_staging import staged_tensor
from server.training.tokenization.encoding_schema import CATEGORY_COUNT


@dataclass(frozen=True, slots=True)
class ObservationTensorBatch:
    """Device tensors for one batch of typed token sequences."""

    category_ids: Tensor
    scalar_values: Tensor
    card_rule_values: Tensor
    coordinate_values: Tensor
    coordinate_masks: Tensor
    candidate_category_ids: Tensor
    candidate_counts: Tensor
    candidate_card_rule_values: Tensor
    query_indices: Tensor

    def __post_init__(self) -> None:
        assert self.category_ids.ndim == 3
        assert int(self.category_ids.shape[2]) == CATEGORY_COUNT
        batch, tokens = self.category_ids.shape[:2]
        assert self.scalar_values.shape == (batch, tokens)
        assert self.card_rule_values.shape == (batch, tokens, 2)
        assert self.coordinate_values.shape == (batch, tokens, 3)
        assert self.coordinate_masks.shape == (batch, tokens, 3)
        assert self.candidate_category_ids.shape == (
            batch,
            CARD_CHOICE_COUNT,
            3,
        )
        assert self.candidate_counts.shape == (
            batch,
            CARD_CHOICE_COUNT,
        )
        assert self.candidate_card_rule_values.shape == (
            batch,
            CARD_CHOICE_COUNT,
            2,
        )
        assert self.query_indices.shape == (batch,)


def tensorize_observation(
    *, observation: Observation, device: torch.device
) -> ObservationTensorBatch:
    """Tensorize one observation as batch size one."""
    return tensorize_observations(
        observations=(observation,), device=device
    )


def tensorize_observations(
    *, observations: tuple[Observation, ...], device: torch.device
) -> ObservationTensorBatch:
    """Pack and tensorize one non-empty observation batch."""
    assert observations
    return tensorize_packed_observations(
        observations=tuple(
            pack_observation(item) for item in observations
        ),
        device=device,
    )


def tensorize_packed_observations(
    *, observations: tuple[PackedObservation, ...], device: torch.device
) -> ObservationTensorBatch:
    """Tensorize prepacked observations with batch-local padding."""
    assert observations
    token_count = max(item.token_count() for item in observations)
    assert token_count <= MAX_LOSSLESS_OBSERVATION_TOKENS
    rows = tuple(
        padded_packed_observation(item, token_count=token_count)
        for item in observations
    )
    return ObservationTensorBatch(
        category_ids=staged_tensor(
            tuple(item.category_rows for item in rows),
            dtype=torch.long,
            device=device,
        ),
        scalar_values=staged_tensor(
            tuple(item.scalar_values for item in rows),
            dtype=torch.float32,
            device=device,
        ),
        card_rule_values=staged_tensor(
            tuple(item.card_rule_rows for item in rows),
            dtype=torch.float32,
            device=device,
        ),
        coordinate_values=staged_tensor(
            tuple(item.coordinate_rows for item in rows),
            dtype=torch.long,
            device=device,
        ),
        coordinate_masks=staged_tensor(
            tuple(item.coordinate_mask_rows for item in rows),
            dtype=torch.bool,
            device=device,
        ),
        candidate_category_ids=staged_tensor(
            tuple(item.candidate_category_rows for item in rows),
            dtype=torch.long,
            device=device,
        ),
        candidate_counts=staged_tensor(
            tuple(item.candidate_counts for item in rows),
            dtype=torch.float32,
            device=device,
        ),
        candidate_card_rule_values=staged_tensor(
            tuple(item.candidate_card_rule_rows for item in rows),
            dtype=torch.float32,
            device=device,
        ),
        query_indices=staged_tensor(
            tuple(item.query_index for item in rows),
            dtype=torch.long,
            device=device,
        ),
    )


__all__ = (
    "ObservationTensorBatch",
    "tensorize_observation",
    "tensorize_observations",
    "tensorize_packed_observations",
)
