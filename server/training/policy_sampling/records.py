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
class SampledPolicyBatch:
    """Device policy samples before model-rank slot assignment."""

    policy_versions: tuple[int, ...]
    status_codes: Tensor
    observation_batch: ObservationTensorBatch
    selected_token_ids_padded: Tensor
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
        assert self.status_codes.shape == (batch_size,)
        assert self.status_codes.dtype == torch.long
        assert self.selected_token_ids_padded.shape == (
            batch_size,
            max_generation_steps,
        )
        assert self.selected_token_ids_padded.dtype == torch.long
        assert self.choice_token_ids.ndim == 3
        assert self.choice_token_ids.shape[:2] == (
            batch_size,
            max_generation_steps,
        )
        assert self.choice_token_ids.dtype == torch.int16
        assert self.choice_masks.shape == self.choice_token_ids.shape
        assert self.choice_masks.dtype == torch.bool
        assert self.selected_choice_offsets.shape == (
            batch_size,
            max_generation_steps,
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
        assert self.status_codes.device == device
        assert self.selected_token_ids_padded.device == device
        assert self.choice_token_ids.device == device
        assert self.choice_masks.device == device
        assert self.selected_choice_offsets.device == device
        assert self.step_counts.device == device
        assert self.choice_counts.device == device
        assert self.old_log_probabilities.device == device
        assert self.old_values.device == device


@dataclass(frozen=True, slots=True)
class ModelRankPolicyDecision:
    """Model-rank response before worker-side action decoding."""

    trace_token_ids: CompactTraceTokenIds
    decision_handle: DecisionHandle
    choice_count: int

    def __post_init__(self) -> None:
        assert len(self.trace_token_ids) > 0
        assert self.choice_count > 0


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
