"""Execution-only training configuration.

This module owns rollout-worker CPU placement, model-rank placement,
profiling, and watchdog settings.  These values are deliberately
excluded from checkpoints so a saved model can resume on different
machines and devices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from server import result as _result

type ModelRankKind = Literal["none", "cuda", "mps"]
type ModelRankDevice = str
type PPOProfileMode = Literal["off", "basic", "detailed"]
type CpuSet = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ModelRankPlacement:
    """Placement of trainable model ranks."""

    kind: ModelRankKind
    devices: tuple[ModelRankDevice, ...]

    def __post_init__(self) -> None:
        assert self.kind in ("none", "cuda", "mps")
        if self.kind == "none":
            assert self.devices == ()
            return
        if self.kind == "mps":
            assert self.devices == ("mps",)
            return
        assert self.devices
        assert all(_is_cuda_device(device) for device in self.devices)
        indices = tuple(
            _cuda_device_index(device) for device in self.devices
        )
        assert len(indices) == len(set(indices))


@dataclass(frozen=True, slots=True)
class ExecutionTimeouts:
    """Watchdog limits for distinct runtime stages."""

    round_seconds: float = 120.0
    sampling_start_seconds: float = 240.0
    rollout_sample_seconds: float = 240.0
    sampling_stop_seconds: float = 240.0
    state_sync_seconds: float = 300.0
    update_seconds: float = 3600.0

    def __post_init__(self) -> None:
        assert _positive_finite(self.round_seconds)
        assert _positive_finite(self.sampling_start_seconds)
        assert _positive_finite(self.rollout_sample_seconds)
        assert _positive_finite(self.sampling_stop_seconds)
        assert _positive_finite(self.state_sync_seconds)
        assert _positive_finite(self.update_seconds)


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Runtime settings that may change freely across resume."""

    worker_cpus: CpuSet = ()
    model_ranks: ModelRankPlacement = field(
        default_factory=lambda: ModelRankPlacement(
            kind="none", devices=()
        )
    )
    ppo_profile: PPOProfileMode = "off"
    timeouts: ExecutionTimeouts = field(
        default_factory=ExecutionTimeouts
    )
    telemetry_interval_seconds: float = 1.0
    model_inference_batch_size: int = 64
    game_envs_per_worker: int = 1
    samples_per_update: int = 1024

    def __post_init__(self) -> None:
        assert _cpu_set_is_valid(self.worker_cpus)
        assert self.ppo_profile in ("off", "basic", "detailed")
        assert _is_finite(self.telemetry_interval_seconds)
        assert self.telemetry_interval_seconds > 0.0
        assert self.model_inference_batch_size > 0
        assert self.game_envs_per_worker > 0
        assert self.samples_per_update > 0

    def worker_process_count(self) -> int:
        """Return OS worker process count."""
        if not self.worker_cpus:
            return 1
        return len(self.worker_cpus)

    def worker_cpu_set(self, worker_index: int) -> CpuSet:
        """Return the CPU affinity for one worker process."""
        assert worker_index >= 0
        assert worker_index < self.worker_process_count()
        if not self.worker_cpus:
            return ()
        return (self.worker_cpus[worker_index],)

    def model_rank_process_count(self) -> int:
        """Return standalone model-rank process count."""
        return len(self.model_ranks.devices)

    def uses_model_rank_processes(self) -> bool:
        """Return whether model compute is delegated out of workers."""
        return self.model_ranks.kind != "none"

    def model_rank_index_for_worker(self, worker_index: int) -> int:
        """Return the stable model-rank index for one worker."""
        assert worker_index >= 0
        assert self.model_ranks.devices
        return worker_index % len(self.model_ranks.devices)


def parse_cpu_set(text: str) -> _result.Ok[CpuSet] | _result.Rejected:
    """Parse a Linux-style CPU list such as ``0-3,6``."""
    if not text:
        return _result.Rejected(reason="CPU set must not be empty")
    cpus: list[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        if not part:
            return _result.Rejected(reason=f"invalid CPU set: {text}")
        parsed_part = _parse_cpu_set_part(part)
        if isinstance(parsed_part, _result.Rejected):
            return parsed_part
        for cpu in parsed_part.value:
            if cpu in seen:
                return _result.Rejected(
                    reason=f"duplicate CPU in set: {cpu}"
                )
            seen.add(cpu)
            cpus.append(cpu)
    return _result.Ok(value=tuple(cpus))


def parse_model_rank_placement(
    text: str,
) -> _result.Ok[ModelRankPlacement] | _result.Rejected:
    """Parse the user-facing model-rank placement string."""
    if text == "none":
        return _result.Ok(
            value=ModelRankPlacement(kind="none", devices=())
        )
    if text == "mps":
        return _result.Ok(
            value=ModelRankPlacement(kind="mps", devices=("mps",))
        )
    if not text.startswith("cuda:"):
        return _result.Rejected(
            reason=(
                "--model-ranks must be none, mps, or "
                "cuda:<index>[,<index>...]"
            )
        )
    raw_indices = text.removeprefix("cuda:").split(",")
    if not raw_indices or any(not index for index in raw_indices):
        return _result.Rejected(reason="invalid CUDA model rank list")
    indices: list[int] = []
    for raw_index in raw_indices:
        if not raw_index.isdecimal():
            return _result.Rejected(
                reason=f"invalid CUDA model rank index: {raw_index}"
            )
        index = int(raw_index)
        if index in indices:
            return _result.Rejected(
                reason=f"duplicate CUDA model rank index: {index}"
            )
        indices.append(index)
    return _result.Ok(
        value=ModelRankPlacement(
            kind="cuda",
            devices=tuple(f"cuda:{index}" for index in indices),
        )
    )


def _parse_cpu_set_part(
    part: str,
) -> _result.Ok[tuple[int, ...]] | _result.Rejected:
    if "-" not in part:
        cpu_result = _parse_cpu_number(part, part)
        if isinstance(cpu_result, _result.Rejected):
            return cpu_result
        return _result.Ok(value=(cpu_result.value,))
    bounds = part.split("-")
    if len(bounds) != 2:
        return _result.Rejected(reason=f"invalid CPU range: {part}")
    start_result = _parse_cpu_number(bounds[0], part)
    if isinstance(start_result, _result.Rejected):
        return start_result
    end_result = _parse_cpu_number(bounds[1], part)
    if isinstance(end_result, _result.Rejected):
        return end_result
    if end_result.value < start_result.value:
        return _result.Rejected(
            reason=f"CPU range is descending: {part}"
        )
    return _result.Ok(
        value=tuple(range(start_result.value, end_result.value + 1))
    )


def _parse_cpu_number(
    text: str, label: str
) -> _result.Ok[int] | _result.Rejected:
    if not text.isdecimal():
        return _result.Rejected(reason=f"invalid CPU number: {label}")
    value = int(text)
    if value < 0:
        return _result.Rejected(reason=f"invalid CPU number: {label}")
    return _result.Ok(value=value)


def _cpu_set_is_valid(cpus: CpuSet) -> bool:
    return len(cpus) == len(set(cpus)) and all(cpu >= 0 for cpu in cpus)


def _is_cuda_device(device: str) -> bool:
    if not device.startswith("cuda:"):
        return False
    index = device.removeprefix("cuda:")
    return index.isdecimal()


def _cuda_device_index(device: str) -> int:
    assert device.startswith("cuda:")
    index = device.removeprefix("cuda:")
    assert index.isdecimal()
    return int(index)


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0.0
