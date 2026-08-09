"""Deterministic random streams for rollout policy decisions.

The sampler derives randomness from explicit keys instead of process,
device, or torch global RNG state.  This makes checkpoint resume
independent of CPU/GPU backend and independent of worker process count.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from server.game import Seat, seat_id

_U53_SCALE = 1.0 / float(1 << 53)


@dataclass(frozen=True, slots=True)
class TrainingDecisionKey:
    """Stable key for every stochastic policy decision."""

    base_seed: int
    policy_version: int
    rollout_id: str
    round_id: int
    seat: Seat
    decision_index: int

    def __post_init__(self) -> None:
        assert self.base_seed >= 0
        assert self.policy_version >= 0
        assert self.rollout_id
        assert self.round_id >= 0
        assert self.decision_index >= 0


def policy_choice_threshold(
    *,
    key: TrainingDecisionKey,
    step_index: int,
) -> float:
    """Return the stable threshold for one policy choice."""
    assert step_index >= 0
    return _unit_float(
        (
            "policy",
            key.base_seed,
            key.policy_version,
            key.round_id,
            seat_id(key.seat),
            key.decision_index,
            step_index,
        )
    )


def uniform_choice_offset(
    *,
    key: TrainingDecisionKey,
    step_index: int,
    choice_count: int,
) -> int:
    """Return a stable uniform offset for a legal choice count."""
    assert choice_count > 0
    threshold = policy_choice_threshold(
        key=key,
        step_index=step_index,
    )
    return min(int(threshold * choice_count), choice_count - 1)


def _unit_float(parts: tuple[object, ...]) -> float:
    digest = hashlib.blake2b(
        _key_bytes(parts), digest_size=8, person=b"tractor"
    ).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return float(value >> 11) * _U53_SCALE


def _key_bytes(parts: tuple[object, ...]) -> bytes:
    return "|".join(str(part) for part in parts).encode("utf-8")
