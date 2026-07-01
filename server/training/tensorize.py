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
from server.training.selection_actions import (
    MAX_HAND_CARD_SLOTS,
    ActionQuery,
    SelectionState,
)
from server.training.tokens import CardToken
from server.training.vocab import PAD_COMPONENT_IDS, component_ids

SELECTION_FEATURE_COUNT: int = 6


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
    card_order_ids: Tensor
    play_width_ids: Tensor
    event_age_ids: Tensor
    numeric_values: Tensor
    numeric_masks: Tensor
    hand_token_indices: Tensor
    hand_card_masks: Tensor


@dataclass(frozen=True, slots=True)
class SelectionStateTensorBatch:
    """Batch-size-one selection state tensors."""

    selected_slot_masks: Tensor
    feature_values: Tensor


def tensorize_observation(
    *,
    observation: Observation,
    max_observation_tokens: int,
    device: torch.device,
) -> ObservationTensorBatch:
    """Tensorize one observation as batch size 1."""
    assert max_observation_tokens > 0
    assert len(observation.tokens) <= max_observation_tokens
    tokens = list(observation.tokens)
    hand_token_positions = _hand_token_positions(observation)
    ids = [component_ids(token) for token in tokens]
    numeric_features = [
        numeric_feature_values(token) for token in tokens
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
        card_order_ids=_long_tensor(
            [item.card_order for item in ids], device
        ),
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
        hand_token_indices=_long_tensor(
            _padded_hand_token_indices(hand_token_positions), device
        ),
        hand_card_masks=_bool_tensor(
            _padded_hand_card_masks(len(hand_token_positions)), device
        ),
    )


def tensorize_selection_state(
    *,
    query: ActionQuery,
    state: SelectionState,
    device: torch.device,
) -> SelectionStateTensorBatch:
    """Tensorize one incremental selection state as batch size 1."""
    assert len(state.selected_slots) <= MAX_HAND_CARD_SLOTS
    assert not _has_duplicate_slots(state.selected_slots)
    selected_slot_masks = [0.0 for _ in range(MAX_HAND_CARD_SLOTS)]
    for slot in state.selected_slots:
        assert 0 <= slot < MAX_HAND_CARD_SLOTS
        assert slot < len(query.hand_card_ids)
        selected_slot_masks[slot] = 1.0
    hand_size = min(len(query.hand_card_ids), MAX_HAND_CARD_SLOTS)
    exact_select = (
        0 if query.exact_select is None else query.exact_select
    )
    return SelectionStateTensorBatch(
        selected_slot_masks=torch.tensor(
            [selected_slot_masks],
            dtype=torch.float32,
            device=device,
        ),
        feature_values=torch.tensor(
            [
                [
                    len(state.selected_slots) / MAX_HAND_CARD_SLOTS,
                    query.min_select / MAX_HAND_CARD_SLOTS,
                    query.max_select / MAX_HAND_CARD_SLOTS,
                    exact_select / MAX_HAND_CARD_SLOTS,
                    hand_size / MAX_HAND_CARD_SLOTS,
                    1.0 if query.pass_allowed else 0.0,
                ]
            ],
            dtype=torch.float32,
            device=device,
        ),
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


def _hand_token_positions(observation: Observation) -> list[int]:
    positions = [
        index
        for index, token in enumerate(observation.tokens)
        if isinstance(token, CardToken) and token.segment == "self_hand"
    ]
    assert len(positions) <= MAX_HAND_CARD_SLOTS
    assert len(positions) == len(observation.hand_card_ids)
    return positions


def _padded_hand_token_indices(positions: list[int]) -> list[int]:
    values = list(positions)
    while len(values) < MAX_HAND_CARD_SLOTS:
        values.append(0)
    return values


def _padded_hand_card_masks(count: int) -> list[bool]:
    values = [index < count for index in range(MAX_HAND_CARD_SLOTS)]
    return values


def _has_duplicate_slots(slots: tuple[int, ...]) -> bool:
    return len(slots) != len(set(slots))
