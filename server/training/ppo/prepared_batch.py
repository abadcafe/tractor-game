"""Prepared PPO update batches resident on the learner device."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.replay_tensors import ReadyPPOBatch
from server.training.tensorize import (
    ObservationTensorBatch,
)


@dataclass(frozen=True, slots=True)
class PreparedPPOBatch:
    """One PPO rollout batch tensorized once on a learner device."""

    batch: ReadyPPOBatch
    old_log_probabilities: Tensor
    old_values: Tensor
    advantages: Tensor
    return_values: Tensor
    sample_count: int

    def __post_init__(self) -> None:
        assert self.sample_count > 0
        assert self.batch.sample_count() == self.sample_count
        assert (
            int(self.old_log_probabilities.shape[0])
            == self.sample_count
        )
        assert int(self.old_values.shape[0]) == self.sample_count
        assert int(self.advantages.shape[0]) == self.sample_count
        assert int(self.return_values.shape[0]) == self.sample_count


def prepare_ppo_batch(
    *,
    batch: ReadyPPOBatch,
) -> PreparedPPOBatch:
    """Tensorize a full PPO update batch once on the learner device."""
    assert not batch.is_empty()
    sample_count = batch.sample_count()
    return PreparedPPOBatch(
        batch=batch,
        old_log_probabilities=batch.old_log_probabilities,
        old_values=batch.old_values,
        advantages=batch.raw_advantages,
        return_values=batch.return_values,
        sample_count=sample_count,
    )


def prepared_ppo_minibatch(
    *,
    batch: PreparedPPOBatch,
    indices: Tensor,
    global_count: Tensor,
) -> TensorizedPPOMinibatch:
    """Select a rank-local minibatch view from a prepared PPO batch."""
    assert global_count.shape == ()
    assert indices.ndim == 1
    local_count = int(indices.shape[0])
    device = batch.old_log_probabilities.device
    index_tensor = indices.to(device=device, dtype=torch.long)
    if local_count == 0:
        return empty_ppo_minibatch(
            device=device,
            global_count=global_count,
        )
    return TensorizedPPOMinibatch(
        observation_batch=_select_observation_rows(
            batch.batch.observation_batch, index_tensor=index_tensor
        ),
        replay=batch.batch.replay,
        sample_indices=index_tensor,
        old_log_probabilities=batch.old_log_probabilities.index_select(
            dim=0, index=index_tensor
        ),
        old_values=batch.old_values.index_select(
            dim=0, index=index_tensor
        ),
        advantages=batch.advantages.index_select(
            dim=0, index=index_tensor
        ),
        return_values=batch.return_values.index_select(
            dim=0, index=index_tensor
        ),
        local_count=local_count,
        global_count=global_count,
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
        return_values=empty_float,
        local_count=0,
        global_count=global_count,
    )


def _select_observation_rows(
    batch: ObservationTensorBatch, *, index_tensor: Tensor
) -> ObservationTensorBatch:
    return ObservationTensorBatch(
        component_ids=batch.component_ids.index_select(
            dim=0, index=index_tensor
        ),
        numeric_values=batch.numeric_values.index_select(
            dim=0, index=index_tensor
        ),
        numeric_masks=batch.numeric_masks.index_select(
            dim=0, index=index_tensor
        ),
    )
