"""Tagged game phases with no invalid phase combinations."""

from __future__ import annotations

from dataclasses import dataclass

from server.game.config import Seat
from server.game.rules.cards import Card

from .round_values import (
    BottomExchangeCause,
    BottomExchangeEvent,
    CompletedRound,
    Contract,
    Declaration,
    Hands,
    InternalCompletedTrick,
    OpenTrick,
    StirEvent,
    TeamLevels,
)


@dataclass(frozen=True, slots=True)
class AwaitingStart:
    """Players are confirming the first round."""

    levels: TeamLevels
    confirmed: frozenset[Seat]


@dataclass(frozen=True, slots=True)
class DealBid:
    """Exactly one dealt-card recipient must bid or pass."""

    round_number: int
    levels: TeamLevels
    fixed_declarer: Seat | None
    start_player: Seat
    deck: tuple[Card, ...]
    bottom_cards: tuple[Card, ...]
    hands: Hands
    deal_cursor: int
    current_actor: Seat
    declaration: Declaration | None
    bid_reveals: tuple[Declaration, ...]


@dataclass(frozen=True, slots=True)
class BottomExchange:
    """One player has picked up the bottom and must bury cards."""

    round_number: int
    levels: TeamLevels
    contract: Contract
    declaration: Declaration | None
    hands: Hands
    bottom_cards: tuple[Card, ...]
    current_actor: Seat
    hand_after_pickup: tuple[Card, ...]
    cause: BottomExchangeCause
    bid_reveals: tuple[Declaration, ...]
    stir_events: tuple[StirEvent, ...]
    exchange_events: tuple[BottomExchangeEvent, ...]


@dataclass(frozen=True, slots=True)
class StirDecision:
    """Exactly one player may stir or pass."""

    round_number: int
    levels: TeamLevels
    contract: Contract
    declaration: Declaration | None
    hands: Hands
    bottom_cards: tuple[Card, ...]
    bottom_owner: Seat
    current_actor: Seat
    passed: frozenset[Seat]
    bid_reveals: tuple[Declaration, ...]
    stir_events: tuple[StirEvent, ...]
    exchange_events: tuple[BottomExchangeEvent, ...]


@dataclass(frozen=True, slots=True)
class Playing:
    """A round is actively playing tricks."""

    round_number: int
    levels: TeamLevels
    contract: Contract
    declaration: Declaration | None
    hands: Hands
    bottom_cards: tuple[Card, ...]
    bottom_owner: Seat
    defender_point_cards: tuple[Card, ...]
    current_trick: OpenTrick
    previous_trick: InternalCompletedTrick | None
    bid_reveals: tuple[Declaration, ...]
    stir_events: tuple[StirEvent, ...]
    exchange_events: tuple[BottomExchangeEvent, ...]


@dataclass(frozen=True, slots=True)
class RoundReview:
    """Players are reviewing and confirming a completed round."""

    completed: CompletedRound
    confirmed: frozenset[Seat]


@dataclass(frozen=True, slots=True)
class Finished:
    """The game has a winner and accepts no further commands."""

    completed: CompletedRound
    winning_team: int


type GamePhase = (
    AwaitingStart
    | DealBid
    | BottomExchange
    | StirDecision
    | Playing
    | RoundReview
    | Finished
)
