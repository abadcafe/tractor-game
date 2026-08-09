"""Deterministic identities and random streams for rollout games."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from server.game import Partnership, Seat, seat_id


@dataclass(frozen=True, slots=True)
class GameIdentity:
    """Globally stable identity for one long-lived rollout game."""

    base_seed: int
    worker_index: int
    game_env_index: int
    game_ordinal: int

    def __post_init__(self) -> None:
        assert self.base_seed >= 0
        assert self.worker_index >= 0
        assert self.game_env_index >= 0
        assert self.game_ordinal >= 0

    def game_seed(self) -> int:
        """Return the deterministic pure-game seed."""
        return _stable_u63(
            (
                "game",
                self.base_seed,
                self.worker_index,
                self.game_env_index,
                self.game_ordinal,
            )
        )

    def model_partnership(self) -> Partnership:
        """Alternate the model side across adjacent games and envs."""
        parity = (
            self.worker_index + self.game_env_index + self.game_ordinal
        ) % 2
        if parity == 0:
            return Partnership.FIRST
        return Partnership.SECOND

    def auto_seed(self, seat: Seat) -> int:
        """Return one independent deterministic Auto-policy seed."""
        return _stable_u63(
            (
                "auto",
                self.base_seed,
                self.worker_index,
                self.game_env_index,
                self.game_ordinal,
                seat_id(seat),
            )
        )

    def next_game(self) -> GameIdentity:
        """Return the identity of the next game in this environment."""
        return GameIdentity(
            base_seed=self.base_seed,
            worker_index=self.worker_index,
            game_env_index=self.game_env_index,
            game_ordinal=self.game_ordinal + 1,
        )


def _stable_u63(parts: tuple[object, ...]) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"tractgame",
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & (
        (1 << 63) - 1
    )


__all__ = ("GameIdentity",)
