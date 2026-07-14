"""Portable deterministic samplers for training.

The sampler derives randomness from explicit keys instead of process,
device, or torch global RNG state.  This makes checkpoint resume
independent of CPU/GPU backend and independent of worker process count.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

_U53_SCALE = 1.0 / float(1 << 53)


@dataclass(frozen=True, slots=True)
class PolicyDecisionKey:
    """Stable key for every stochastic policy decision."""

    base_seed: int
    policy_version: int
    rollout_id: str
    episode_id: int
    player_index: int
    decision_index: int

    def __post_init__(self) -> None:
        assert self.base_seed >= 0
        assert self.policy_version >= 0
        assert self.rollout_id
        assert self.episode_id >= 0
        assert self.player_index in (0, 1, 2, 3)
        assert self.decision_index >= 0


@dataclass(frozen=True, slots=True)
class ShuffleKey:
    """Stable key for PPO sample shuffling."""

    base_seed: int
    policy_version: int
    epoch: int

    def __post_init__(self) -> None:
        assert self.base_seed >= 0
        assert self.policy_version >= 0
        assert self.epoch >= 0


def policy_choice_threshold(
    *,
    key: PolicyDecisionKey,
    argument_index: int,
) -> float:
    """Return the stable threshold for one policy choice."""
    assert argument_index >= 0
    return _unit_float(
        (
            "policy",
            key.base_seed,
            key.policy_version,
            key.episode_id,
            key.player_index,
            key.decision_index,
            argument_index,
        )
    )


def uniform_choice_offset(
    *,
    key: PolicyDecisionKey,
    argument_index: int,
    choice_count: int,
) -> int:
    """Return a stable uniform offset for a legal choice count."""
    assert choice_count > 0
    threshold = policy_choice_threshold(
        key=key,
        argument_index=argument_index,
    )
    return min(int(threshold * choice_count), choice_count - 1)


def shuffled_indices(
    *, key: ShuffleKey, length: int
) -> tuple[int, ...]:
    """Return a deterministic Fisher-Yates permutation."""
    assert length >= 0
    order = list(range(length))
    for index in range(length - 1):
        remaining = length - index
        offset = int(
            _unit_float(
                (
                    "shuffle",
                    key.base_seed,
                    key.policy_version,
                    key.epoch,
                    index,
                )
            )
            * remaining
        )
        swap_index = index + min(offset, remaining - 1)
        order[index], order[swap_index] = (
            order[swap_index],
            order[index],
        )
    return tuple(order)


def _unit_float(parts: tuple[object, ...]) -> float:
    digest = hashlib.blake2b(
        _key_bytes(parts), digest_size=8, person=b"tractor"
    ).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return float(value >> 11) * _U53_SCALE


def _key_bytes(parts: tuple[object, ...]) -> bytes:
    return "|".join(str(part) for part in parts).encode("utf-8")
