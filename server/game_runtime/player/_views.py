"""Strongly typed lobby descriptions for players."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

type BotPolicyName = Literal["auto", "ai"]


@dataclass(frozen=True, slots=True)
class UserId:
    """Validated external identity for one human player."""

    value: str

    @classmethod
    def parse(cls, value: str | None) -> UserId | None:
        """Return a validated identity or ``None`` for missing input."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        return cls(normalized)


class HumanPlayerDescription(TypedDict):
    """Lobby projection for a human-controlled seat."""

    kind: Literal["human"]
    connected: bool
    mine: bool


class BotPlayerDescription(TypedDict):
    """Lobby projection for an automatically controlled seat."""

    kind: Literal["bot"]
    policy: BotPolicyName


type PlayerDescription = HumanPlayerDescription | BotPlayerDescription


__all__ = (
    "BotPolicyName",
    "BotPlayerDescription",
    "HumanPlayerDescription",
    "PlayerDescription",
    "UserId",
)
