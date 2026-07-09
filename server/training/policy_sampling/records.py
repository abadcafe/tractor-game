"""Device-local policy sampling records."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import ObservationTensorBatch

_TRACE_TOKEN = struct.Struct("<q")


@dataclass(frozen=True, slots=True)
class CompactTraceTokenIds:
    """Compact little-endian int64 semantic token trace."""

    encoded_i64: bytes
    count: int

    def __post_init__(self) -> None:
        assert self.count > 0
        assert len(self.encoded_i64) == self.count * _TRACE_TOKEN.size

    @classmethod
    def from_tuple(
        cls, token_ids: tuple[int, ...]
    ) -> CompactTraceTokenIds:
        """Encode explicit token ids."""
        assert token_ids
        encoded = struct.pack(f"<{len(token_ids)}q", *token_ids)
        return cls(encoded_i64=encoded, count=len(token_ids))

    @classmethod
    def from_cpu_tensor(
        cls, *, tokens: Tensor, count: int
    ) -> CompactTraceTokenIds:
        """Copy one CPU int64 token row."""
        assert tokens.device.type == "cpu"
        assert tokens.dtype == torch.long
        assert count > 0
        assert int(tokens.shape[0]) >= count
        return cls(
            encoded_i64=tokens[:count].contiguous().numpy().tobytes(),
            count=count,
        )

    @classmethod
    def from_i64_bytes(
        cls, *, data: bytes, count: int
    ) -> CompactTraceTokenIds:
        """Copy encoded int64 token bytes from a response frame."""
        assert count > 0
        expected_bytes = count * _TRACE_TOKEN.size
        assert len(data) == expected_bytes
        return cls(encoded_i64=data, count=count)

    def __len__(self) -> int:
        """Return token count."""
        return self.count

    def to_tuple(self) -> tuple[int, ...]:
        """Decode the compact trace for the worker-side rules layer."""
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
    """Device policy sample columns before optional test inspection."""

    policy_versions: tuple[int, ...]
    observation_batch: ObservationTensorBatch
    selected_token_ids_padded: Tensor
    active_sample_indices: Tensor
    active_step_indices: Tensor
    choice_token_ids: Tensor
    choice_masks: Tensor
    selected_choice_offsets: Tensor
    step_counts: Tensor
    choice_counts: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor

    def __post_init__(self) -> None:
        batch_size = len(self.policy_versions)
        max_generation_steps = int(
            self.selected_token_ids_padded.shape[1]
        )
        assert batch_size > 0
        assert max_generation_steps > 0
        assert (
            max_generation_steps <= SEMANTIC_CODEC.max_argument_tokens
        )
        assert all(version >= 0 for version in self.policy_versions)
        assert self.selected_token_ids_padded.shape == (
            batch_size,
            max_generation_steps,
        )
        assert self.selected_token_ids_padded.dtype == torch.long
        assert self.active_sample_indices.ndim == 1
        assert self.active_step_indices.shape == (
            int(self.active_sample_indices.shape[0]),
        )
        assert self.active_sample_indices.dtype == torch.long
        assert self.active_step_indices.dtype == torch.long
        assert self.choice_token_ids.ndim == 2
        assert int(self.choice_token_ids.shape[0]) == int(
            self.active_sample_indices.shape[0]
        )
        assert self.choice_token_ids.dtype == torch.int16
        assert self.choice_masks.shape == self.choice_token_ids.shape
        assert self.choice_masks.dtype == torch.bool
        assert self.selected_choice_offsets.shape == (
            int(self.active_sample_indices.shape[0]),
        )
        assert self.selected_choice_offsets.dtype == torch.long
        assert self.step_counts.shape == (batch_size,)
        assert self.step_counts.dtype == torch.long
        assert self.choice_counts.shape == (batch_size,)
        assert self.choice_counts.dtype == torch.long
        assert self.old_log_probabilities.shape == (batch_size,)
        assert self.old_values.shape == (batch_size,)
        assert int(self.observation_batch.component_ids.shape[0]) == (
            batch_size
        )
        device = self.observation_batch.component_ids.device
        assert self.selected_token_ids_padded.device == device
        assert self.active_sample_indices.device == device
        assert self.active_step_indices.device == device
        assert self.choice_token_ids.device == device
        assert self.choice_masks.device == device
        assert self.selected_choice_offsets.device == device
        assert self.step_counts.device == device
        assert self.choice_counts.device == device
        assert self.old_log_probabilities.device == device
        assert self.old_values.device == device


@dataclass(frozen=True, slots=True)
class CompactTraceTokenBatch:
    """Padded CPU int64 traces for one compact response batch."""

    encoded_i64_rows: bytes
    row_count: int
    max_trace_count: int
    trace_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        assert self.row_count > 0
        assert self.max_trace_count > 0
        assert len(self.trace_counts) == self.row_count
        assert all(
            0 < count <= self.max_trace_count
            for count in self.trace_counts
        )
        expected_bytes = (
            self.row_count * self.max_trace_count * _TRACE_TOKEN.size
        )
        assert len(self.encoded_i64_rows) == expected_bytes

    @classmethod
    def from_cpu_tensor(
        cls, *, tokens: Tensor, trace_counts: tuple[int, ...]
    ) -> CompactTraceTokenBatch:
        """Copy padded CPU token rows into response-ready bytes."""
        assert tokens.device.type == "cpu"
        assert tokens.dtype == torch.long
        row_count = len(trace_counts)
        assert row_count > 0
        max_trace_count = max(trace_counts)
        assert max_trace_count > 0
        assert tokens.shape[0] == row_count
        assert int(tokens.shape[1]) >= max_trace_count
        compact = tokens[:, :max_trace_count].contiguous()
        return cls(
            encoded_i64_rows=compact.numpy().tobytes(),
            row_count=row_count,
            max_trace_count=max_trace_count,
            trace_counts=trace_counts,
        )

    def row_bytes(self, row_index: int) -> bytes:
        """Return one padded token row."""
        assert 0 <= row_index < self.row_count
        row_byte_count = self.max_trace_count * _TRACE_TOKEN.size
        start = row_index * row_byte_count
        return self.encoded_i64_rows[start : start + row_byte_count]

    def compact_row(self, row_index: int) -> CompactTraceTokenIds:
        """Return one compact trace for worker-side rule decoding."""
        trace_count = self.trace_counts[row_index]
        return CompactTraceTokenIds.from_i64_bytes(
            data=self.row_bytes(row_index)[
                : trace_count * _TRACE_TOKEN.size
            ],
            count=trace_count,
        )

    def select_rows(
        self, rows: tuple[int, ...]
    ) -> CompactTraceTokenBatch:
        """Return a compact batch containing selected rows."""
        assert rows
        row_bytes = b"".join(self.row_bytes(row) for row in rows)
        return CompactTraceTokenBatch(
            encoded_i64_rows=row_bytes,
            row_count=len(rows),
            max_trace_count=self.max_trace_count,
            trace_counts=tuple(self.trace_counts[row] for row in rows),
        )


@dataclass(frozen=True, slots=True)
class CompactPolicyDecisionBatch:
    """Model-rank policy decisions encoded as response columns."""

    model_rank_index: int
    policy_versions: tuple[int, ...]
    row_indices: tuple[int, ...]
    choice_counts: tuple[int, ...]
    trace_token_batch: CompactTraceTokenBatch

    def __post_init__(self) -> None:
        row_count = self.trace_token_batch.row_count
        assert self.model_rank_index >= 0
        assert len(self.policy_versions) == row_count
        assert len(self.row_indices) == row_count
        assert len(self.choice_counts) == row_count
        assert all(version >= 0 for version in self.policy_versions)
        assert all(row_index >= 0 for row_index in self.row_indices)
        assert all(
            choice_count > 0 for choice_count in self.choice_counts
        )

    def row_count(self) -> int:
        """Return decision row count."""
        return self.trace_token_batch.row_count


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
        assert self.step_counts.device == self.row_indices.device
        assert self.return_values.device == self.row_indices.device
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
        """Return whether this rank has no committed samples."""
        return int(self.row_indices.shape[0]) == 0
