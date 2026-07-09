"""PPO update profiling records and accumulators."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from server.training.runtime.config import PPOProfileMode

type _ProfileField = Literal[
    "minibatch_loss_seconds",
    "observation_batch_seconds",
    "observation_encode_seconds",
    "value_head_seconds",
    "argument_select_seconds",
    "argument_decode_seconds",
    "argument_distribution_seconds",
    "backward_seconds",
    "optimizer_step_seconds",
]


@dataclass(frozen=True, slots=True)
class PPOUpdateProfile:
    """Timing and shape profile for one PPO update."""

    update_seconds: float
    minibatch_loss_seconds: float
    observation_batch_seconds: float
    observation_encode_seconds: float
    value_head_seconds: float
    argument_select_seconds: float
    argument_decode_seconds: float
    argument_distribution_seconds: float
    backward_seconds: float
    optimizer_step_seconds: float
    argument_decode_fraction: float
    argument_trace_batch_count: int
    argument_trace_row_count: int
    argument_trace_token_count: int
    argument_trace_valid_token_count: int
    argument_trace_padding_token_count: int


def ppo_update_profile_is_finite(profile: PPOUpdateProfile) -> bool:
    """Return whether all profile counters are finite and valid."""
    finite_nonnegative_values = (
        profile.update_seconds,
        profile.minibatch_loss_seconds,
        profile.observation_batch_seconds,
        profile.observation_encode_seconds,
        profile.value_head_seconds,
        profile.argument_select_seconds,
        profile.argument_decode_seconds,
        profile.argument_distribution_seconds,
        profile.backward_seconds,
        profile.optimizer_step_seconds,
    )
    return (
        all(
            math.isfinite(value) and value >= 0.0
            for value in finite_nonnegative_values
        )
        and math.isfinite(profile.argument_decode_fraction)
        and 0.0 <= profile.argument_decode_fraction <= 1.0
        and profile.argument_trace_batch_count >= 0
        and profile.argument_trace_row_count >= 0
        and profile.argument_trace_token_count >= 0
        and profile.argument_trace_valid_token_count >= 0
        and profile.argument_trace_padding_token_count >= 0
    )


def blank_update_profile(*, update_seconds: float) -> PPOUpdateProfile:
    """Build a zero-detail profile with an optional total time."""
    assert math.isfinite(update_seconds)
    assert update_seconds >= 0.0
    return PPOUpdateProfile(
        update_seconds=update_seconds,
        minibatch_loss_seconds=0.0,
        observation_batch_seconds=0.0,
        observation_encode_seconds=0.0,
        value_head_seconds=0.0,
        argument_select_seconds=0.0,
        argument_decode_seconds=0.0,
        argument_distribution_seconds=0.0,
        backward_seconds=0.0,
        optimizer_step_seconds=0.0,
        argument_decode_fraction=0.0,
        argument_trace_batch_count=0,
        argument_trace_row_count=0,
        argument_trace_token_count=0,
        argument_trace_valid_token_count=0,
        argument_trace_padding_token_count=0,
    )


@dataclass(frozen=True, slots=True)
class _ProfileMark:
    """CPU wall-time mark plus an optional CUDA stream event."""

    wall_seconds: float
    cuda_event: torch.cuda.Event | None


_NO_PROFILE_MARK = _ProfileMark(wall_seconds=0.0, cuda_event=None)


@dataclass(frozen=True, slots=True)
class _CudaProfileSegment:
    """Deferred CUDA event timing segment."""

    field_name: _ProfileField
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


def _cuda_profile_segment_list() -> list[_CudaProfileSegment]:
    return []


@dataclass(slots=True)
class PPOProfileAccumulator:
    """Mutable profiler for one PPO update."""

    device: torch.device
    mode: PPOProfileMode
    update_start: _ProfileMark
    minibatch_loss_seconds: float = 0.0
    observation_batch_seconds: float = 0.0
    observation_encode_seconds: float = 0.0
    value_head_seconds: float = 0.0
    argument_select_seconds: float = 0.0
    argument_decode_seconds: float = 0.0
    argument_distribution_seconds: float = 0.0
    backward_seconds: float = 0.0
    optimizer_step_seconds: float = 0.0
    argument_trace_batch_count: int = 0
    argument_trace_row_count: int = 0
    argument_trace_token_count: int = 0
    argument_trace_valid_token_count: int = 0
    argument_trace_padding_token_count: int = 0
    _cuda_segments: list[_CudaProfileSegment] = field(
        default_factory=_cuda_profile_segment_list
    )

    @classmethod
    def start(
        cls,
        *,
        device: torch.device,
        mode: PPOProfileMode,
    ) -> PPOProfileAccumulator:
        if mode == "off":
            update_start = _NO_PROFILE_MARK
        else:
            _synchronize_cuda_device(device)
            if mode == "basic":
                update_start = _ProfileMark(
                    wall_seconds=time.perf_counter(),
                    cuda_event=None,
                )
            else:
                update_start = _profile_mark(device)
        return cls(device=device, mode=mode, update_start=update_start)

    def mark(self) -> _ProfileMark:
        if self.mode != "detailed":
            return _NO_PROFILE_MARK
        return _profile_mark(self.device)

    def record_elapsed(
        self,
        field_name: _ProfileField,
        start: _ProfileMark,
    ) -> None:
        if self.mode != "detailed":
            return
        if start.cuda_event is None:
            self._add_seconds(
                field_name,
                max(time.perf_counter() - start.wall_seconds, 0.0),
            )
            return
        end_event = _cuda_profile_event()
        end_event.record()
        self._cuda_segments.append(
            _CudaProfileSegment(
                field_name=field_name,
                start_event=start.cuda_event,
                end_event=end_event,
            )
        )

    def record_argument_trace_lengths(
        self, trace_lengths: Tensor
    ) -> None:
        if self.mode != "detailed":
            return
        assert trace_lengths.ndim == 1
        cpu_lengths = trace_lengths.detach().cpu()
        lengths = tuple(
            int(cpu_lengths[index].item())
            for index in range(int(cpu_lengths.shape[0]))
        )
        self._record_argument_trace_length_values(lengths)

    def _record_argument_trace_length_values(
        self, trace_lengths: tuple[int, ...]
    ) -> None:
        assert trace_lengths
        assert all(length >= 0 for length in trace_lengths)
        row_count = len(trace_lengths)
        valid_token_count = sum(trace_lengths)
        max_token_count = max(trace_lengths)
        token_count = row_count * max_token_count
        self.argument_trace_batch_count += 1
        self.argument_trace_row_count += row_count
        self.argument_trace_token_count += token_count
        self.argument_trace_valid_token_count += valid_token_count
        self.argument_trace_padding_token_count += (
            token_count - valid_token_count
        )

    def finish(self) -> PPOUpdateProfile:
        if self.mode == "off":
            return blank_update_profile(update_seconds=0.0)
        if self.mode == "basic":
            _synchronize_cuda_device(self.device)
            update_seconds = max(
                time.perf_counter() - self.update_start.wall_seconds,
                0.0,
            )
            return blank_update_profile(update_seconds=update_seconds)
        if self.device.type == "cuda":
            _synchronize_cuda_device(self.device)
            self._apply_cuda_segments()
        update_seconds = max(
            time.perf_counter() - self.update_start.wall_seconds, 0.0
        )
        decode_fraction = (
            0.0
            if update_seconds <= 0.0
            else self.argument_decode_seconds / update_seconds
        )
        return PPOUpdateProfile(
            update_seconds=update_seconds,
            minibatch_loss_seconds=self.minibatch_loss_seconds,
            observation_batch_seconds=self.observation_batch_seconds,
            observation_encode_seconds=self.observation_encode_seconds,
            value_head_seconds=self.value_head_seconds,
            argument_select_seconds=self.argument_select_seconds,
            argument_decode_seconds=self.argument_decode_seconds,
            argument_distribution_seconds=(
                self.argument_distribution_seconds
            ),
            backward_seconds=self.backward_seconds,
            optimizer_step_seconds=self.optimizer_step_seconds,
            argument_decode_fraction=decode_fraction,
            argument_trace_batch_count=(
                self.argument_trace_batch_count
            ),
            argument_trace_row_count=self.argument_trace_row_count,
            argument_trace_token_count=self.argument_trace_token_count,
            argument_trace_valid_token_count=(
                self.argument_trace_valid_token_count
            ),
            argument_trace_padding_token_count=(
                self.argument_trace_padding_token_count
            ),
        )

    def _apply_cuda_segments(self) -> None:
        for segment in self._cuda_segments:
            self._add_seconds(
                segment.field_name,
                max(
                    segment.start_event.elapsed_time(segment.end_event)
                    / 1000.0,
                    0.0,
                ),
            )
        self._cuda_segments.clear()

    def _add_seconds(
        self,
        field_name: _ProfileField,
        seconds: float,
    ) -> None:
        if field_name == "minibatch_loss_seconds":
            self.minibatch_loss_seconds += seconds
            return
        if field_name == "observation_batch_seconds":
            self.observation_batch_seconds += seconds
            return
        if field_name == "observation_encode_seconds":
            self.observation_encode_seconds += seconds
            return
        if field_name == "value_head_seconds":
            self.value_head_seconds += seconds
            return
        if field_name == "argument_select_seconds":
            self.argument_select_seconds += seconds
            return
        if field_name == "argument_decode_seconds":
            self.argument_decode_seconds += seconds
            return
        if field_name == "argument_distribution_seconds":
            self.argument_distribution_seconds += seconds
            return
        if field_name == "backward_seconds":
            self.backward_seconds += seconds
            return
        if field_name == "optimizer_step_seconds":
            self.optimizer_step_seconds += seconds


def _profile_mark(device: torch.device) -> _ProfileMark:
    cuda_event: torch.cuda.Event | None = None
    if device.type == "cuda":
        cuda_event = _cuda_profile_event()
        cuda_event.record()
    return _ProfileMark(
        wall_seconds=time.perf_counter(),
        cuda_event=cuda_event,
    )


def _synchronize_cuda_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_profile_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)
