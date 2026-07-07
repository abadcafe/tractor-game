"""PPO update input contract for synchronized ranks."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.ppo.replay_tensors import ReadyPPOBatch


@dataclass(frozen=True, slots=True)
class PPOUpdateInput:
    """Rank-local rollout payload for one synchronized PPO update."""

    policy_version: int
    local_batch: ReadyPPOBatch | None

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        if self.local_batch is not None:
            assert not self.local_batch.is_empty()
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
