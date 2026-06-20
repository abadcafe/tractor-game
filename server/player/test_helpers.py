"""Shared test helpers for player tests."""

from __future__ import annotations

from typing import Literal, TypeGuard
from unittest.mock import AsyncMock, MagicMock

from server.messages import StateMessage
from server.snapshot import (
    ScoringSnapshot,
    StateSnapshot,
    StirringStateSnapshot,
    TrickSnapshot,
)
from server.sm.card_model import Card, Rank, Suit
from server.sm.types import BidEvent, CompletedTrick, FailedThrow


def is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1, suffix: str = "") -> Card:
    """Create a real Card for testing."""
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}{suffix}",
        suit=suit,
        rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0,
        deck=deck,
    )


def make_snapshot(
    *,
    phase: str = "PLAYING",
    awaiting_action: str | None = "play",
    action_hints: list[list[Card]] | None = None,
    trump_rank: Rank = Rank.TWO,
    trump_suit: Suit | None = None,
    player_hand: list[Card] | None = None,
    player_hand_counts: list[int] | None = None,
    bottom_cards: list[Card] | None = None,
    declarer_team: int | None = None,
    declarer_player: int | None = None,
    defender_points: int = 0,
    trick: TrickSnapshot | None = None,
    trick_history: list[CompletedTrick] | None = None,
    failed_throw: FailedThrow | None = None,
    bid_events: list[BidEvent] | None = None,
    bid_winner: BidEvent | None = None,
    stirring_state: StirringStateSnapshot | None = None,
    scoring: ScoringSnapshot | None = None,
    winning_team: int | None = None,
    team0_level: Rank = Rank.TWO,
    team1_level: Rank = Rank.TWO,
    next_round_confirmed: list[int] | None = None,
) -> StateSnapshot:
    """Create a real StateSnapshot with sensible defaults."""
    return StateSnapshot(
        phase=phase,
        awaiting_action=awaiting_action,
        action_hints=action_hints if action_hints is not None else [],
        trump_rank=trump_rank,
        trump_suit=trump_suit,
        player_hand=player_hand if player_hand is not None else [],
        player_hand_counts=player_hand_counts if player_hand_counts is not None else [0, 0, 0, 0],
        bottom_cards=bottom_cards if bottom_cards is not None else [],
        declarer_team=declarer_team,
        declarer_player=declarer_player,
        defender_points=defender_points,
        trick=trick,
        trick_history=trick_history if trick_history is not None else [],
        failed_throw=failed_throw,
        bid_events=bid_events if bid_events is not None else [],
        bid_winner=bid_winner,
        stirring_state=stirring_state,
        scoring=scoring,
        winning_team=winning_team,
        team0_level=team0_level,
        team1_level=team1_level,
        next_round_confirmed=next_round_confirmed if next_round_confirmed is not None else [],
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
