"""Deterministic PPO sample shuffling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

_U53_SCALE = 1.0 / float(1 << 53)


@dataclass(frozen=True, slots=True)
class ShuffleKey:
    """Stable identity for one PPO epoch permutation."""

    base_seed: int
    policy_version: int
    epoch: int

    def __post_init__(self) -> None:
        assert self.base_seed >= 0
        assert self.policy_version >= 0
        assert self.epoch >= 0


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
        "|".join(str(part) for part in parts).encode("utf-8"),
        digest_size=8,
        person=b"tractor",
    ).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return float(value >> 11) * _U53_SCALE
