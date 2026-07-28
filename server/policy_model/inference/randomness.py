"""Deterministic random streams for live policy inference."""

from __future__ import annotations

import hashlib

from server.game import Seat, seat_id
from server.policy_model.inference.contracts import SamplingSeed

_U53_SCALE = 1.0 / float(1 << 53)


def choice_threshold(
    *,
    sampling_seed: SamplingSeed,
    seat: Seat,
    step_index: int,
) -> float:
    """Return one batching-independent action sampling threshold."""
    assert step_index >= 0
    encoded = "|".join(
        (
            "inference",
            str(sampling_seed.seed),
            str(sampling_seed.ordinal),
            str(seat_id(seat)),
            str(step_index),
        )
    ).encode("utf-8")
    digest = hashlib.blake2b(
        encoded,
        digest_size=8,
        person=b"tractor",
    ).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return float(value >> 11) * _U53_SCALE
