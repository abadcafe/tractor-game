"""Queue-safe packed observation rows for training trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.numeric_features import (
    PAD_NUMERIC_FEATURES,
    numeric_feature_values,
)
from server.training.observation import Observation
from server.training.vocab import component_ids
from server.training.vocab_schema import (
    PAD_COMPONENT_IDS,
    TokenComponentIds,
)

OBSERVATION_COMPONENT_COUNT: int = 15

type ComponentRow = tuple[int, ...]
type NumericRow = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PackedObservation:
    """Torch-free model input rows for one observation."""

    component_rows: tuple[ComponentRow, ...]
    numeric_value_rows: tuple[NumericRow, ...]
    numeric_mask_rows: tuple[NumericRow, ...]

    def __post_init__(self) -> None:
        assert self.component_rows
        assert len(self.component_rows) == len(self.numeric_value_rows)
        assert len(self.component_rows) == len(self.numeric_mask_rows)
        assert all(
            len(row) == OBSERVATION_COMPONENT_COUNT
            for row in self.component_rows
        )

    def token_count(self) -> int:
        """Return packed observation token count."""
        return len(self.component_rows)


def pack_observation(observation: Observation) -> PackedObservation:
    """Pack one model observation into queue-safe CPU rows."""
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
    return PackedObservation(
        component_rows=tuple(component_items),
        numeric_value_rows=tuple(item.values for item in numeric_items),
        numeric_mask_rows=tuple(item.masks for item in numeric_items),
    )


def padded_packed_observation(
    packed: PackedObservation,
    *,
    max_observation_tokens: int,
) -> PackedObservation:
    """Pad one packed observation to the requested token count."""
    assert max_observation_tokens > 0
    assert packed.token_count() <= max_observation_tokens
    pad_count = max_observation_tokens - packed.token_count()
    if pad_count == 0:
        return packed
    pad_components = _component_values(PAD_COMPONENT_IDS)
    return PackedObservation(
        component_rows=(
            *packed.component_rows,
            *(pad_components for _ in range(pad_count)),
        ),
        numeric_value_rows=(
            *packed.numeric_value_rows,
            *(PAD_NUMERIC_FEATURES.values for _ in range(pad_count)),
        ),
        numeric_mask_rows=(
            *packed.numeric_mask_rows,
            *(PAD_NUMERIC_FEATURES.masks for _ in range(pad_count)),
        ),
    )


def _component_values(
    values: TokenComponentIds,
) -> ComponentRow:
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
