"""PPO update input contract for synchronized ranks."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.ppo.replay_tensors import RolloutTensorBatch


@dataclass(frozen=True, slots=True)
class PPOUpdateInput:
    """Rank-local rollout payload for one synchronized PPO update."""

    policy_version: int
    local_rollout: RolloutTensorBatch | None

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        if self.local_rollout is not None:
            assert not self.local_rollout.is_empty()
            assert (
                self.local_rollout.policy_version == self.policy_version
            )

    def local_transition_count(self) -> int:
        """Return this rank's local trainable transition count."""
        if self.local_rollout is None:
            return 0
        return self.local_rollout.transition_count()

    def is_empty_rank(self) -> bool:
        """Return whether this rank owns no samples."""
        return self.local_rollout is None
