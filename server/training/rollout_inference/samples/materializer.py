"""Materialize compact policy decisions across the device boundary."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import torch
from torch import Tensor

from server.training.rollout_inference.samples.records import (
    CompactActionChoiceBatch,
    CompactPolicyDecisionBatch,
)

_INITIAL_VALUE_CAPACITY = 256


@dataclass(slots=True)
class PolicyDecisionMaterializer:
    """Pack one device batch and cross the host boundary once."""

    device: torch.device
    _device_values: Tensor = field(init=False)
    _host_values: Tensor = field(init=False)
    _cuda_event: torch.cuda.Event | None = field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        assert self.device.type in ("cpu", "cuda", "mps")
        self._device_values = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )
        self.device = self._device_values.device
        self._host_values = self._device_values
        if self.device.type == "cuda":
            self._cuda_event = torch.cuda.Event()

    def materialize(
        self,
        *,
        model_rank_index: int,
        policy_versions: tuple[int, ...],
        row_start: int,
        choice_ids_padded: Tensor,
        step_counts: Tensor,
        choice_counts: Tensor,
    ) -> CompactPolicyDecisionBatch:
        row_count, padded_step_count = choice_ids_padded.shape
        assert model_rank_index >= 0
        assert row_start >= 0
        assert row_count > 0
        assert len(policy_versions) == row_count
        assert choice_ids_padded.dtype == torch.long
        assert step_counts.dtype == torch.long
        assert choice_counts.dtype == torch.long
        assert step_counts.shape == (row_count,)
        assert choice_counts.shape == (row_count,)
        assert choice_ids_padded.device == self.device
        assert step_counts.device == self.device
        assert choice_counts.device == self.device

        choice_value_count = row_count * padded_step_count
        required_value_count = choice_value_count + 2 * row_count
        self._ensure_capacity(required_value_count)
        device_values = self._device_values[:required_value_count]
        device_values[:choice_value_count].view(
            row_count, padded_step_count
        ).copy_(choice_ids_padded)
        step_start = choice_value_count
        choice_count_start = step_start + row_count
        device_values[step_start:choice_count_start].copy_(step_counts)
        device_values[choice_count_start:].copy_(choice_counts)

        host_values = self._copy_to_host(required_value_count)
        host_counts = _cpu_int_tuple(host_values[step_start:])
        host_step_counts = host_counts[:row_count]
        return CompactPolicyDecisionBatch(
            model_rank_index=model_rank_index,
            policy_versions=policy_versions,
            row_indices=tuple(range(row_start, row_start + row_count)),
            choice_counts=host_counts[row_count:],
            action_choice_batch=CompactActionChoiceBatch.from_cpu_tensor(
                choice_ids=host_values[:choice_value_count].view(
                    row_count, padded_step_count
                ),
                choice_counts=host_step_counts,
            ),
        )

    def _ensure_capacity(self, required_value_count: int) -> None:
        assert required_value_count > 0
        current_capacity = int(self._device_values.shape[0])
        if current_capacity >= required_value_count:
            return
        capacity = max(
            _INITIAL_VALUE_CAPACITY,
            required_value_count,
            current_capacity * 2,
        )
        self._device_values = torch.empty(
            (capacity,), dtype=torch.long, device=self.device
        )
        if self.device.type == "cpu":
            self._host_values = self._device_values
            return
        self._host_values = torch.empty(
            (capacity,),
            dtype=torch.long,
            device=torch.device("cpu"),
            pin_memory=self.device.type == "cuda",
        )

    def _copy_to_host(self, value_count: int) -> Tensor:
        device_values = self._device_values[:value_count]
        if self.device.type == "cpu":
            return device_values
        host_values = self._host_values[:value_count]
        host_values.copy_(
            device_values, non_blocking=self.device.type == "cuda"
        )
        event = self._cuda_event
        if event is not None:
            event.record(torch.cuda.current_stream(self.device))
            event.synchronize()
        return host_values


def _cpu_int_tuple(values: Tensor) -> tuple[int, ...]:
    assert values.device.type == "cpu"
    assert values.dtype == torch.long
    assert values.ndim == 1
    count = int(values.shape[0])
    decoded = struct.unpack(
        f"<{count}q", values.contiguous().numpy().tobytes()
    )
    return tuple(int(value) for value in decoded)


__all__ = ("PolicyDecisionMaterializer",)
