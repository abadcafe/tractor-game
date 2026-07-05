"""Tensorized PPO minibatches."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from server.training.ppo.replay_tensors import PPOReplayTensorBatch
from server.training.tensorize import ObservationTensorBatch


@dataclass(frozen=True, slots=True)
class TensorizedPPOMinibatch:
    """Rank-local PPO minibatch tensors plus recorded trace metadata."""

    observation_batch: ObservationTensorBatch | None
    replay: PPOReplayTensorBatch | None
    sample_indices: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    advantages: Tensor
    return_values: Tensor
    local_count: int
    global_count: Tensor

    def __post_init__(self) -> None:
        assert self.local_count >= 0
        assert self.global_count.shape == ()
        assert self.global_count.dtype == self.sample_indices.dtype
        assert self.sample_indices.ndim == 1
        assert int(self.sample_indices.shape[0]) == self.local_count
        assert (
            int(self.old_log_probabilities.shape[0]) == self.local_count
        )
        assert int(self.old_values.shape[0]) == self.local_count
        assert int(self.advantages.shape[0]) == self.local_count
        assert int(self.return_values.shape[0]) == self.local_count
        if self.local_count == 0:
            assert self.observation_batch is None
            assert self.replay is None
        else:
            assert self.observation_batch is not None
            assert self.replay is not None

    def is_empty(self) -> bool:
        """Return whether this rank owns no samples."""
        return self.local_count == 0
