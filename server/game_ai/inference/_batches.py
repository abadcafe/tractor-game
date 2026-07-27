"""Exact-length observation batching and stable row restoration."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.observation import Observation
from server.training.packed_observation import (
    PackedObservation,
    pack_observation,
)


@dataclass(frozen=True, slots=True)
class PackedRows:
    """Unique packed observations and one source row per input."""

    unique: tuple[PackedObservation, ...]
    source_indices: tuple[int, ...]


def pack_unique(
    observations: tuple[Observation, ...],
) -> PackedRows:
    """Pack each observation once and deduplicate exact values."""
    assert observations
    return deduplicate_packed(
        tuple(pack_observation(item) for item in observations)
    )


def deduplicate_packed(
    packed_rows: tuple[PackedObservation, ...],
) -> PackedRows:
    """Deduplicate already packed observation rows."""
    assert packed_rows
    unique: list[PackedObservation] = []
    indices: dict[PackedObservation, int] = {}
    source_indices: list[int] = []
    for packed in packed_rows:
        source_index = indices.get(packed)
        if source_index is None:
            source_index = len(unique)
            indices[packed] = source_index
            unique.append(packed)
        source_indices.append(source_index)
    return PackedRows(
        unique=tuple(unique),
        source_indices=tuple(source_indices),
    )


def exact_length_groups(
    packed_rows: tuple[PackedObservation, ...],
) -> tuple[tuple[int, ...], ...]:
    """Group source indices by exact token count in stable order."""
    groups: dict[int, list[int]] = {}
    for index, packed in enumerate(packed_rows):
        groups.setdefault(packed.token_count(), []).append(index)
    return tuple(tuple(indices) for indices in groups.values())


__all__ = (
    "PackedRows",
    "deduplicate_packed",
    "exact_length_groups",
    "pack_unique",
)
