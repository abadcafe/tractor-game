"""Padding-aware observation batching and stable row restoration."""

from __future__ import annotations

from dataclasses import dataclass

from server.policy_model.observation.packing import (
    PackedObservation,
)


@dataclass(frozen=True, slots=True)
class PackedRows:
    """Unique packed observations and one source row per input."""

    unique: tuple[PackedObservation, ...]
    source_indices: tuple[int, ...]


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


def attention_batch_groups(
    *,
    packed_rows: tuple[PackedObservation, ...],
    indices: tuple[int, ...],
    max_rows: int,
) -> tuple[tuple[int, ...], ...]:
    """Coalesce nearby lengths within a quadratic work bound."""
    assert indices
    assert max_rows > 0
    ordered = tuple(
        sorted(
            indices, key=lambda index: packed_rows[index].token_count()
        )
    )
    groups: list[tuple[int, ...]] = []
    pending: list[int] = []
    useful_cells = 0
    for index in ordered:
        token_count = packed_rows[index].token_count()
        next_useful_cells = useful_cells + token_count * token_count
        next_rows = len(pending) + 1
        next_padded_cells = next_rows * token_count * token_count
        exceeds_memory_budget = (
            next_padded_cells > _MAX_PADDED_ATTENTION_CELLS
        )
        wastes_too_much_padding = (
            len(pending) >= _MIN_ROWS_BEFORE_PADDING_SPLIT
            and next_padded_cells
            > next_useful_cells * _MAX_PADDING_PERCENT // 100
        )
        if pending and (
            next_rows > max_rows
            or exceeds_memory_budget
            or wastes_too_much_padding
        ):
            groups.append(tuple(pending))
            pending.clear()
            useful_cells = 0
        pending.append(index)
        useful_cells += token_count * token_count
    if pending:
        groups.append(tuple(pending))
    return tuple(groups)


_MAX_PADDED_ATTENTION_CELLS = 16 * 1024 * 1024
_MAX_PADDING_PERCENT = 125
_MIN_ROWS_BEFORE_PADDING_SPLIT = 4


def inference_batch_row_limit(device_type: str) -> int:
    """Return each backend's measured throughput knee."""
    if device_type == "cpu":
        return 32
    if device_type == "mps":
        return 64
    assert device_type == "cuda"
    return 256


__all__ = (
    "PackedRows",
    "attention_batch_groups",
    "deduplicate_packed",
    "inference_batch_row_limit",
)
