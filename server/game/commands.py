"""Commands accepted by the game aggregate."""

from __future__ import annotations

from dataclasses import dataclass

from server.game.rules.cards import CardId


@dataclass(frozen=True, slots=True)
class ConfirmRound:
    """Confirm game start or the next round."""


@dataclass(frozen=True, slots=True)
class RevealBid:
    """Reveal cards while cards are being dealt."""

    card_ids: tuple[CardId, ...]


@dataclass(frozen=True, slots=True)
class PassBid:
    """Decline to reveal cards for the current deal step."""


@dataclass(frozen=True, slots=True)
class Bury:
    """Put cards back into the bottom pile."""

    card_ids: tuple[CardId, ...]


@dataclass(frozen=True, slots=True)
class Stir:
    """Reveal a stronger pair and change trump."""

    card_ids: tuple[CardId, ...]


@dataclass(frozen=True, slots=True)
class PassStir:
    """Decline to stir."""


@dataclass(frozen=True, slots=True)
class Play:
    """Play physical cards from the current hand."""

    card_ids: tuple[CardId, ...]


type Command = (
    ConfirmRound | RevealBid | PassBid | Bury | Stir | PassStir | Play
)
