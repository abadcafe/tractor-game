"""Prepared PPO update batches resident on the learner device."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.update_input import (
    PPOBatchSource,
)


@dataclass(frozen=True, slots=True)
class PreparedPPOBatch:
    """One PPO rollout source with normalized device advantages."""

    source: PPOBatchSource
    advantages: Tensor
    sample_count: int

    def __post_init__(self) -> None:
        assert self.sample_count > 0
        assert self.source.sample_count() == self.sample_count
        assert self.advantages.ndim == 1
        assert int(self.advantages.shape[0]) == self.sample_count
        assert self.source.observation_token_counts.shape == (
            self.sample_count,
        )


@dataclass(frozen=True, slots=True)
class PPOEpochSchedule:
    """Deterministic sample order for one PPO epoch."""

    indices: Tensor
    observation_token_counts: Tensor
    sample_count: int

    def __post_init__(self) -> None:
        assert self.sample_count > 0
        assert self.indices.ndim == 1
        assert int(self.indices.shape[0]) == self.sample_count
        assert self.indices.dtype == torch.long
        assert self.observation_token_counts.device.type == "cpu"
        assert self.observation_token_counts.dtype == torch.long
        assert self.observation_token_counts.shape == (
            self.sample_count,
        )


def prepare_ppo_batch(
    *,
    source: PPOBatchSource,
    advantages: Tensor,
) -> PreparedPPOBatch:
    """Tensorize a full PPO update batch once on the learner device."""
    sample_count = source.sample_count()
    assert sample_count > 0
    return PreparedPPOBatch(
        source=source,
        advantages=advantages,
        sample_count=sample_count,
    )


def prepare_ppo_epoch_schedule(
    *,
    batch: PreparedPPOBatch,
    indices: Tensor,
    cpu_indices: Tensor,
) -> PPOEpochSchedule:
    """Create one shuffled epoch schedule without rollout copies."""
    assert indices.ndim == 1
    assert cpu_indices.device.type == "cpu"
    assert cpu_indices.dtype == torch.long
    assert cpu_indices.shape == indices.shape
    sample_count = int(indices.shape[0])
    assert sample_count == batch.sample_count
    return PPOEpochSchedule(
        indices=indices.to(
            dtype=torch.long,
            device=batch.advantages.device,
        ),
        observation_token_counts=(
            batch.source.observation_token_counts.index_select(
                0, cpu_indices
            )
        ),
        sample_count=sample_count,
    )


def prepared_ppo_epoch_minibatch(
    *,
    batch: PreparedPPOBatch,
    schedule: PPOEpochSchedule,
    start: int,
    end: int,
    global_count: Tensor,
) -> TensorizedPPOMinibatch:
    """Return one scheduled minibatch from the prepared batch."""
    assert global_count.shape == ()
    assert schedule.sample_count == batch.sample_count
    assert 0 <= start <= end <= schedule.sample_count
    if start == end:
        return empty_ppo_minibatch(
            device=batch.advantages.device,
            global_count=global_count,
        )
    return batch.source.select_minibatch(
        indices=schedule.indices[start:end],
        advantages=batch.advantages,
        global_count=global_count,
        observation_token_count=int(
            schedule.observation_token_counts[start:end].max().item()
        ),
    )


def empty_ppo_minibatch(
    *, device: torch.device, global_count: Tensor
) -> TensorizedPPOMinibatch:
    """Return an empty rank-local minibatch for synchronized DDP."""
    assert global_count.shape == ()
    empty_float = torch.empty((0,), dtype=torch.float32, device=device)
    empty_long = torch.empty((0,), dtype=torch.long, device=device)
    return TensorizedPPOMinibatch(
        observation_batch=None,
        replay=None,
        sample_indices=empty_long,
        old_log_probabilities=empty_float,
        old_values=empty_float,
        advantages=empty_float,
        value_targets=empty_float,
        local_count=0,
        global_count=global_count,
    )
