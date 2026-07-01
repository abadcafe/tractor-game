"""Shared test helpers for player tests."""

from __future__ import annotations

from typing import Literal, TypeGuard
from unittest.mock import AsyncMock, MagicMock

from server.protocol import (
    AwaitingAction,
    BidEventSnapshot,
    BottomExchangeEventSnapshot,
    CompletedTrickSnapshot,
    FailedThrowSnapshot,
    RoundPhase,
    ScoringSnapshot,
    StateMessage,
    StateSnapshot,
    StirDeclarationEventSnapshot,
    StirringStateSnapshot,
    TrickSnapshot,
)
from server.rules.cards import POINTS_MAP, Card, Rank, Suit

type TestSuit = Literal[
    "hearts", "spades", "diamonds", "clubs", "joker"
]
type TestRank = Literal[
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
    "A",
    "SJ",
    "BJ",
]


def is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def card(
    suit: TestSuit,
    rank: TestRank,
    deck: Literal[1, 2] = 1,
) -> Card:
    """Create a protocol card snapshot for player tests."""
    card_rank = Rank(rank)
    return Card(
        id=f"D{deck}-{suit}-{rank}",
        suit=Suit(suit),
        rank=card_rank,
        points=POINTS_MAP[card_rank],
    )


def make_snapshot(
    *,
    phase: RoundPhase = "PLAYING",
    awaiting_action: AwaitingAction | None = "play",
    action_hints: list[list[Card]] | None = None,
    trump_rank: TestRank = "2",
    trump_suit: TestSuit | None = None,
    player_hand: list[Card] | None = None,
    player_hand_counts: list[int] | None = None,
    bottom_cards: list[Card] | None = None,
    declarer_team: int | None = None,
    declarer_player: int | None = None,
    defender_points: int = 0,
    trick: TrickSnapshot | None = None,
    last_completed_trick: CompletedTrickSnapshot | None = None,
    defender_point_cards: list[Card] | None = None,
    failed_throw: FailedThrowSnapshot | None = None,
    bid_events: list[BidEventSnapshot] | None = None,
    bid_winner: BidEventSnapshot | None = None,
    stir_events: list[StirDeclarationEventSnapshot] | None = None,
    own_bottom_exchange_events: (
        list[BottomExchangeEventSnapshot] | None
    ) = None,
    stirring_state: StirringStateSnapshot | None = None,
    scoring: ScoringSnapshot | None = None,
    winning_team: int | None = None,
    team0_level: TestRank = "2",
    team1_level: TestRank = "2",
    next_round_confirmed: list[int] | None = None,
) -> StateSnapshot:
    """Create a real StateSnapshot with sensible defaults."""
    return StateSnapshot(
        phase=phase,
        awaiting_action=awaiting_action,
        action_hints=action_hints if action_hints is not None else [],
        trump_rank=Rank(trump_rank),
        trump_suit=Suit(trump_suit) if trump_suit is not None else None,
        player_hand=player_hand if player_hand is not None else [],
        player_hand_counts=player_hand_counts
        if player_hand_counts is not None
        else [0, 0, 0, 0],
        bottom_cards=bottom_cards if bottom_cards is not None else [],
        declarer_team=declarer_team,
        declarer_player=declarer_player,
        defender_points=defender_points,
        trick=trick,
        last_completed_trick=last_completed_trick,
        defender_point_cards=defender_point_cards
        if defender_point_cards is not None
        else [],
        failed_throw=failed_throw,
        bid_events=bid_events if bid_events is not None else [],
        bid_winner=bid_winner,
        stir_events=stir_events if stir_events is not None else [],
        own_bottom_exchange_events=own_bottom_exchange_events
        if own_bottom_exchange_events is not None
        else [],
        stirring_state=stirring_state,
        scoring=scoring,
        winning_team=winning_team,
        team0_level=Rank(team0_level),
        team1_level=Rank(team1_level),
        next_round_confirmed=next_round_confirmed
        if next_round_confirmed is not None
        else [],
    )


def make_game(snapshot: StateSnapshot | None = None) -> MagicMock:
    """Create a mock Game that returns the given snapshot."""
    game = MagicMock()
    game.snapshot = MagicMock(return_value=snapshot or make_snapshot())
    game.receive = AsyncMock()
    return game


def make_state_message(
    snapshot: StateSnapshot | None = None,
    *,
    seq: int = 1,
    error: str | None = None,
) -> StateMessage:
    state = snapshot or make_snapshot()
    return StateMessage(
        seq=seq,
        state=state,
        error=error,
    )
