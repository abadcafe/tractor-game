"""Policy sampling records owned by model ranks."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    MAX_ACTION_STEPS,
)
from server.training.tensorize import ObservationTensorBatch

_CHOICE_ID = struct.Struct("<q")


@dataclass(frozen=True, slots=True)
class CompactActionChoiceIds:
    """One compact little-endian action choice trace."""

    encoded_i64: bytes
    count: int

    def __post_init__(self) -> None:
        assert 0 < self.count <= MAX_ACTION_STEPS
        assert len(self.encoded_i64) == self.count * _CHOICE_ID.size

    @classmethod
    def from_tuple(
        cls, choice_ids: tuple[int, ...]
    ) -> CompactActionChoiceIds:
        assert choice_ids
        assert all(
            0 <= value < ACTION_CHOICE_COUNT for value in choice_ids
        )
        return cls(
            encoded_i64=struct.pack(
                f"<{len(choice_ids)}q", *choice_ids
            ),
            count=len(choice_ids),
        )

    @classmethod
    def from_i64_bytes(
        cls, *, data: bytes, count: int
    ) -> CompactActionChoiceIds:
        return cls(encoded_i64=data, count=count)

    def __len__(self) -> int:
        return self.count

    def to_tuple(self) -> tuple[int, ...]:
        values = struct.unpack(f"<{self.count}q", self.encoded_i64)
        return tuple(int(value) for value in values)


@dataclass(frozen=True, slots=True)
class DecisionHandle:
    """Stable reference to one model-rank-owned replay row."""

    model_rank_index: int
    policy_version: int
    row_index: int

    def __post_init__(self) -> None:
        assert self.model_rank_index >= 0
        assert self.policy_version >= 0
        assert self.row_index >= 0


@dataclass(frozen=True, slots=True)
class PolicySampleColumns:
    """Device policy samples before storage in a model-rank arena."""

    policy_versions: tuple[int, ...]
    observation_batch: ObservationTensorBatch
    choice_ids_padded: Tensor
    active_sample_indices: Tensor
    active_step_indices: Tensor
    legal_choice_masks: Tensor
    step_counts: Tensor
    choice_counts: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor

    def __post_init__(self) -> None:
        batch_size, max_steps = self.choice_ids_padded.shape
        active_count = int(self.active_sample_indices.shape[0])
        assert batch_size == len(self.policy_versions)
        assert 0 < max_steps <= MAX_ACTION_STEPS
        assert self.active_step_indices.shape == (active_count,)
        assert self.legal_choice_masks.shape == (
            active_count,
            ACTION_CHOICE_COUNT,
        )
        assert self.step_counts.shape == (batch_size,)
        assert self.choice_counts.shape == (batch_size,)
        assert self.old_log_probabilities.shape == (batch_size,)
        assert self.old_values.shape == (batch_size,)
        assert (
            int(self.observation_batch.category_ids.shape[0])
            == batch_size
        )


@dataclass(frozen=True, slots=True)
class CompactActionChoiceBatch:
    """Padded CPU choice traces used in response transport."""

    encoded_i64_rows: bytes
    row_count: int
    max_choice_count: int
    choice_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        assert self.row_count > 0
        assert 0 < self.max_choice_count <= MAX_ACTION_STEPS
        assert len(self.choice_counts) == self.row_count
        assert all(
            0 < count <= self.max_choice_count
            for count in self.choice_counts
        )
        assert len(self.encoded_i64_rows) == (
            self.row_count * self.max_choice_count * _CHOICE_ID.size
        )

    @classmethod
    def from_cpu_tensor(
        cls, *, choice_ids: Tensor, choice_counts: tuple[int, ...]
    ) -> CompactActionChoiceBatch:
        assert choice_ids.device.type == "cpu"
        assert choice_ids.dtype == torch.long
        max_count = max(choice_counts)
        compact = choice_ids[:, :max_count].contiguous()
        return cls(
            encoded_i64_rows=compact.numpy().tobytes(),
            row_count=len(choice_counts),
            max_choice_count=max_count,
            choice_counts=choice_counts,
        )

    def row_bytes(self, row_index: int) -> bytes:
        assert 0 <= row_index < self.row_count
        row_bytes = self.max_choice_count * _CHOICE_ID.size
        start = row_index * row_bytes
        return self.encoded_i64_rows[start : start + row_bytes]

    def compact_row(self, row_index: int) -> CompactActionChoiceIds:
        count = self.choice_counts[row_index]
        return CompactActionChoiceIds.from_i64_bytes(
            data=self.row_bytes(row_index)[: count * _CHOICE_ID.size],
            count=count,
        )

    def select_rows(
        self, rows: tuple[int, ...]
    ) -> CompactActionChoiceBatch:
        assert rows
        return CompactActionChoiceBatch(
            encoded_i64_rows=b"".join(
                self.row_bytes(row) for row in rows
            ),
            row_count=len(rows),
            max_choice_count=self.max_choice_count,
            choice_counts=tuple(
                self.choice_counts[row] for row in rows
            ),
        )


@dataclass(frozen=True, slots=True)
class CompactPolicyDecisionBatch:
    """Model-rank policy decisions encoded as response columns."""

    model_rank_index: int
    policy_versions: tuple[int, ...]
    row_indices: tuple[int, ...]
    choice_counts: tuple[int, ...]
    action_choice_batch: CompactActionChoiceBatch

    def __post_init__(self) -> None:
        rows = self.action_choice_batch.row_count
        assert self.model_rank_index >= 0
        assert len(self.policy_versions) == rows
        assert len(self.row_indices) == rows
        assert len(self.choice_counts) == rows
        assert all(version >= 0 for version in self.policy_versions)
        assert all(row >= 0 for row in self.row_indices)
        assert all(count > 0 for count in self.choice_counts)

    def row_count(self) -> int:
        return self.action_choice_batch.row_count


@dataclass(frozen=True, slots=True)
class RankReturnTargets:
    """Rank-local return targets materialized on one compute device."""

    policy_version: int
    model_rank_index: int
    row_indices: Tensor
    step_counts: Tensor
    return_values: Tensor
    round_count: int
    total_step_count: int
    max_step_count: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.model_rank_index >= 0
        assert self.row_indices.ndim == 1
        assert self.step_counts.shape == self.row_indices.shape
        assert self.return_values.shape == self.row_indices.shape
        assert self.row_indices.dtype == torch.long
        assert self.step_counts.dtype == torch.long
        assert self.return_values.dtype == torch.float32
        assert self.round_count >= 0
        assert self.total_step_count >= 0
        assert self.max_step_count >= 0
        if self.is_empty():
            assert self.total_step_count == 0
            assert self.max_step_count == 0
        else:
            assert self.total_step_count > 0
            assert self.max_step_count > 0

    def is_empty(self) -> bool:
        return int(self.row_indices.shape[0]) == 0


__all__ = (
    "CompactActionChoiceBatch",
    "CompactActionChoiceIds",
    "CompactPolicyDecisionBatch",
    "DecisionHandle",
    "PolicySampleColumns",
    "RankReturnTargets",
)
