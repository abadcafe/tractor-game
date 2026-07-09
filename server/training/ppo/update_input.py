"""PPO update input contract for synchronized ranks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from server.training.ppo.minibatch import TensorizedPPOMinibatch


class PPOBatchSource(Protocol):
    """Rank-local PPO samples hidden behind minibatch selection."""

    @property
    def policy_version(self) -> int: ...

    @property
    def raw_advantages(self) -> Tensor: ...

    def sample_count(self) -> int:
        """Return the number of trainable samples."""
        ...

    def select_minibatch(
        self,
        *,
        indices: Tensor,
        advantages: Tensor,
        global_count: Tensor,
    ) -> TensorizedPPOMinibatch:
        """Return one rank-local minibatch view."""
        ...


@dataclass(frozen=True, slots=True)
class PPOUpdateInput:
    """Rank-local rollout payload for one synchronized PPO update."""

    policy_version: int
    local_batch: PPOBatchSource | None

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        if self.local_batch is not None:
            assert self.local_batch.sample_count() > 0
            assert (
                self.local_batch.policy_version == self.policy_version
            )

    def local_transition_count(self) -> int:
        """Return this rank's local trainable transition count."""
        if self.local_batch is None:
            return 0
        return self.local_batch.sample_count()

    def is_empty_rank(self) -> bool:
        """Return whether this rank owns no samples."""
        return self.local_batch is None
