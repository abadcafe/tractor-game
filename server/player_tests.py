"""Tests for server/player.py -- Player, AutoPlayer, HumanPlayer, PlayerAction types."""

import asyncio
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.actions import (
    BidAction, StirAction, SkipStirAction,
    DiscardAction, PlayAction, NextRoundAction,
)
from server.player import AutoPlayer, HumanPlayer
from server.snapshot import (
    ScoringSnapshot,
    StateSnapshot, StirringStateSnapshot, TrickSnapshot,
)
from server.sm.card_model import Card, Suit, Rank
from server.sm.types import BidEvent, CompletedTrick


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1, suffix: str = "") -> Card:
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


def _make_snapshot(
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
        bid_events=bid_events if bid_events is not None else [],
        bid_winner=bid_winner,
        stirring_state=stirring_state,
        scoring=scoring,
        winning_team=winning_team,
        team0_level=team0_level,
        team1_level=team1_level,
        next_round_confirmed=next_round_confirmed if next_round_confirmed is not None else [],
    )


def _make_game(snapshot: StateSnapshot | None = None) -> MagicMock:
    """Create a mock Game that returns the given snapshot."""
    game = MagicMock()
    game.snapshot = MagicMock(return_value=snapshot or _make_snapshot())
    game.act = AsyncMock()
    return game


# ---- PlayerAction types ----山水


def test_bid_action_fields():
    c1, c2 = _card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2)
    action = BidAction(cards=[c1, c2], count=2)
    assert action.cards == [c1, c2]
    assert action.count == 2


def test_play_action_fields():
    c1 = _card(Suit.SPADES, Rank.ACE, 1)
    action = PlayAction(cards=[c1])
    assert action.cards == [c1]


def test_stir_action_fields():
    c1, c2 = _card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2)
    action = StirAction(cards=[c1, c2])
    assert action.cards == [c1, c2]


def test_skip_stir_action_fields():
    action = SkipStirAction()
    assert isinstance(action, SkipStirAction)


def test_discard_action_fields():
    c1, c2, c3 = _card(Suit.DIAMONDS, Rank.THREE, 1), _card(Suit.CLUBS, Rank.FOUR, 1), _card(Suit.SPADES, Rank.FIVE, 1)
    action = DiscardAction(cards=[c1, c2, c3])
    assert action.cards == [c1, c2, c3]


def test_next_round_action_fields():
    action = NextRoundAction()
    assert isinstance(action, NextRoundAction)


# ---- AutoPlayer ----


@pytest.mark.asyncio
async def test_auto_player_play_when_current():
    """AutoPlayer submits a play action when it's their turn in PLAYING phase."""
    card = _card(Suit.SPADES, Rank.ACE, 1)
    snap = _make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[card],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=1)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()


@pytest.mark.asyncio
async def test_auto_player_play_from_action_hints():
    """AutoPlayer picks from the same action_hints visible to human players."""
    card1 = _card(Suit.SPADES, Rank.ACE, 1)
    snap = _make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[card1],
        action_hints=[[card1]],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    assert call_args[0][0] == 0  # player_index


@pytest.mark.asyncio
async def test_auto_player_error_skips_failed_hint_candidate():
    """A rejected card action is not repeated for the same player-facing state."""
    card1 = _card(Suit.SPADES, Rank.ACE, 1)
    card2 = _card(Suit.HEARTS, Rank.ACE, 1)
    snap = _make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[card1, card2],
        action_hints=[[card1], [card2]],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)

    def choose_first(seq: list[list[Card]]) -> list[Card]:
        return seq[0]

    with patch("server.player.random.choice", side_effect=choose_first):
        await player.on_state(game)
        await asyncio.sleep(0.05)
        first_action = game.act.call_args[0][2]
        assert isinstance(first_action, PlayAction)
        assert first_action.cards == [card1]

        game.act.reset_mock()
        await player.on_state(game, error="rejected")
        await asyncio.sleep(0.05)
        second_action = game.act.call_args[0][2]
        assert isinstance(second_action, PlayAction)
        assert second_action.cards == [card2]


@pytest.mark.asyncio
async def test_auto_player_stale_error_does_not_skip_hint_candidate():
    """A stale seq rejection does not prove the selected cards were illegal."""
    card1 = _card(Suit.SPADES, Rank.ACE, 1)
    card2 = _card(Suit.HEARTS, Rank.ACE, 1)
    snap = _make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[card1, card2],
        action_hints=[[card1], [card2]],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)

    def choose_first(seq: list[list[Card]]) -> list[Card]:
        return seq[0]

    with patch("server.player.random.choice", side_effect=choose_first):
        await player.on_state(game)
        await asyncio.sleep(0.05)
        first_action = game.act.call_args[0][2]
        assert isinstance(first_action, PlayAction)
        assert first_action.cards == [card1]

        game.act.reset_mock()
        await player.on_state(game, error="stale action: expected 3, got 2")
        await asyncio.sleep(0.05)
        second_action = game.act.call_args[0][2]
        assert isinstance(second_action, PlayAction)
        assert second_action.cards == [card1]


@pytest.mark.asyncio
async def test_auto_player_ignores_when_not_awaiting():
    """AutoPlayer does not act when awaiting_action is None (not their turn)."""
    snap = _make_snapshot(
        phase="PLAYING",
        awaiting_action=None,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_ignores_stirring_when_not_awaiting():
    """AutoPlayer does not stir when awaiting_action is None in STIRRING."""
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action=None,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_ignores_discard_when_not_awaiting():
    """AutoPlayer does not discard when awaiting_action is None in STIRRING."""
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action=None,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_next_round():
    """AutoPlayer submits NextRoundAction when awaiting next_round."""
    snap = _make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    assert isinstance(call_args[0][2], NextRoundAction)


@pytest.mark.asyncio
async def test_auto_player_submits_next_round_even_when_other_player():
    """AutoPlayer submits NextRoundAction whenever awaiting_action is next_round."""
    snap = _make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()


@pytest.mark.asyncio
async def test_auto_player_discard_when_current():
    """AutoPlayer submits DiscardAction when awaiting discard and it's their turn."""
    card1 = _card(Suit.DIAMONDS, Rank.THREE, 1)
    card2 = _card(Suit.CLUBS, Rank.FOUR, 1)
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="discard",
        stirring_state=StirringStateSnapshot(
            phase="EXCHANGING",
            trump_suit=None,
            current_player=0,
            declarer_player=0,
            exchanging_player=0,
            exchange_count=2,
        ),
        player_hand=[card1, card2],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    assert isinstance(call_args[0][2], DiscardAction)


@pytest.mark.asyncio
async def test_auto_player_stir_when_current():
    """AutoPlayer can stir from the same action_hints visible to human players."""
    card1 = _card(Suit.HEARTS, Rank.TWO, 1)
    card2 = _card(Suit.HEARTS, Rank.TWO, 2)
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        action_hints=[[card1, card2]],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    with patch("server.player.random.random", return_value=0.4):
        await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    action = game.act.call_args[0][2]
    assert isinstance(action, StirAction)
    assert action.cards == [card1, card2]


@pytest.mark.asyncio
async def test_auto_player_stir_pass():
    """AutoPlayer passes during STIRRING when action_hints is empty."""
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        player_hand=[],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    action = game.act.call_args[0][2]
    assert isinstance(action, SkipStirAction)


@pytest.mark.asyncio
async def test_auto_player_stir_randomly_skips_hint():
    """AutoPlayer keeps the old optional-stir behavior by skipping half the time."""
    card1 = _card(Suit.HEARTS, Rank.TWO, 1)
    card2 = _card(Suit.HEARTS, Rank.TWO, 2)
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        action_hints=[[card1, card2]],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    with patch("server.player.random.random", return_value=0.6):
        await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    action = game.act.call_args[0][2]
    assert isinstance(action, SkipStirAction)


@pytest.mark.asyncio
async def test_auto_player_bid_during_dealing():
    """AutoPlayer can bid during DEAL_BID phase if hand has trump rank cards
    and awaiting_action is 'bid'."""
    trump_card = _card(Suit.HEARTS, Rank.TWO, 1)
    snap = _make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[trump_card],
        trump_rank=Rank.TWO,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    # May bid or skip (random), but should not error


@pytest.mark.asyncio
async def test_auto_player_ignores_dealing_if_no_trump_rank():
    """AutoPlayer sends SkipBidAction during DEAL_BID if hand has no trump rank cards."""
    non_trump = _card(Suit.SPADES, Rank.THREE, 1)
    snap = _make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[non_trump],
        trump_rank=Rank.TWO,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    # Should send SkipBidAction (not no action)
    game.act.assert_awaited()
    from server.actions import SkipBidAction
    call_args = game.act.call_args
    assert isinstance(call_args[0][2], SkipBidAction)


# ---- HumanPlayer ----


@pytest.mark.asyncio
async def test_human_player_handle_connection_sends_state():
    """HumanPlayer.handle_connection accepts WS, binds it, and processes messages."""
    from fastapi import WebSocketDisconnect
    ws = AsyncMock()
    # receive_json raises after one iteration to end the loop
    ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
    snap = _make_snapshot()
    game = _make_game(snap)
    game.is_over = MagicMock(return_value=False)
    game.current_seq = 1
    player = HumanPlayer(index=0)

    await player.handle_connection(ws, game)
    ws.accept.assert_awaited_once()


@pytest.mark.asyncio
async def test_human_player_connection_takeover():
    """HumanPlayer.handle_connection closes old WS and binds new one."""
    from fastapi import WebSocketDisconnect
    old_ws = AsyncMock()
    new_ws = AsyncMock()
    snap = _make_snapshot()
    game = _make_game(snap)
    game.is_over = MagicMock(return_value=False)
    game.current_seq = 1
    # Set up old connection via handle_connection with a WS that disconnects
    old_ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
    player = HumanPlayer(index=0)
    await player.handle_connection(old_ws, game)
    assert player.is_connected() is False  # cleaned up after disconnect

    # New connection should be accepted
    new_ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
    await player.handle_connection(new_ws, game)
    new_ws.accept.assert_awaited_once()


@pytest.mark.asyncio
async def test_human_player_does_not_send_when_no_ws():
    """HumanPlayer.on_state does nothing when no WS is bound (not connected)."""
    snap = _make_snapshot()
    game = _make_game(snap)
    player = HumanPlayer(index=0)
    # Should not raise
    await player.on_state(game)


def test_human_player_is_connected_false():
    """HumanPlayer.is_connected() returns False when no WS is bound."""
    player = HumanPlayer(index=0)
    assert player.is_connected() is False


# ---- Bug 4 regression: stir must use same-suit pairs ----


@pytest.mark.asyncio
async def test_auto_player_stir_only_uses_same_suit_pairs():
    """AutoPlayer stirs only from server-provided action_hints."""
    card_hearts_2_d1 = _card(Suit.HEARTS, Rank.TWO, 1)
    card_spades_2_d1 = _card(Suit.SPADES, Rank.TWO, 1)
    card_hearts_2_d2 = _card(Suit.HEARTS, Rank.TWO, 2)

    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        player_hand=[card_hearts_2_d1, card_spades_2_d1, card_hearts_2_d2],
        action_hints=[[card_hearts_2_d1, card_hearts_2_d2]],
        trump_rank=Rank.TWO,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)

    with patch("server.player.random.random", return_value=0.4):
        await player.on_state(game)

    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    action = call_args[0][2]

    assert isinstance(action, StirAction)
    assert action.cards == [card_hearts_2_d1, card_hearts_2_d2]


# ---- Stirring exchange count typed access ----


@pytest.mark.asyncio
async def test_auto_player_discard_with_stirring_exchange_count():
    """AutoPlayer._handle_discard uses StirringStateSnapshot.exchange_count.

    The snapshot's stirring_state.exchange_count provides the discard count
    during the STIRRING EXCHANGING sub-phase.
    """
    card1 = _card(Suit.DIAMONDS, Rank.THREE, 1)
    card2 = _card(Suit.CLUBS, Rank.FOUR, 1)
    card3 = _card(Suit.SPADES, Rank.FIVE, 1)

    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="discard",
        player_hand=[card1, card2, card3],
        stirring_state=StirringStateSnapshot(
            phase="EXCHANGING",
            trump_suit=None,
            current_player=0,
            declarer_player=0,
            exchanging_player=0,
            exchange_count=3,
        ),
    )

    game = _make_game(snap)
    player = AutoPlayer(index=0)

    # Must not raise AttributeError
    await player.on_state(game)
    await asyncio.sleep(0.05)

    game.act.assert_awaited()
    call_args = game.act.call_args
    action = call_args[0][2]
    assert isinstance(action, DiscardAction)
    assert len(action.cards) == 3
