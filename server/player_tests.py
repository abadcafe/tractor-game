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
    ExchangeStateSnapshot, ScoringSnapshot,
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
    current_player: int = 0,
    legal_actions: list[list[Card]] | None = None,
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
    exchange_state: ExchangeStateSnapshot | None = None,
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
        current_player=current_player,
        legal_actions=legal_actions if legal_actions is not None else [],
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
        exchange_state=exchange_state,
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
        current_player=1,
        legal_actions=[[card]],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=1)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()


@pytest.mark.asyncio
async def test_auto_player_play_from_legal_actions():
    """AutoPlayer picks from legal_actions when playing."""
    card1 = _card(Suit.SPADES, Rank.ACE, 1)
    snap = _make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        current_player=0,
        legal_actions=[[card1]],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    assert call_args[0][0] == 0  # player_index


@pytest.mark.asyncio
async def test_auto_player_ignores_wrong_player():
    """AutoPlayer does not act when current_player != self.index in non-dealing phase."""
    snap = _make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        current_player=2,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_ignores_wrong_player_stirring():
    """AutoPlayer does not stir when current_player != self.index in STIRRING."""
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        current_player=2,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_ignores_wrong_player_discard():
    """AutoPlayer does not discard when current_player != self.index in EXCHANGE."""
    snap = _make_snapshot(
        phase="EXCHANGE",
        awaiting_action="discard",
        current_player=2,
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
        phase="COMPLETE",
        awaiting_action="next_round",
        current_player=0,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    assert isinstance(call_args[0][1], NextRoundAction)


@pytest.mark.asyncio
async def test_auto_player_ignores_wrong_player_next_round():
    """AutoPlayer does not submit NextRoundAction when current_player != self.index."""
    snap = _make_snapshot(
        phase="COMPLETE",
        awaiting_action="next_round",
        current_player=2,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_player_discard_when_current():
    """AutoPlayer submits DiscardAction when awaiting discard and it's their turn."""
    card1 = _card(Suit.DIAMONDS, Rank.THREE, 1)
    card2 = _card(Suit.CLUBS, Rank.FOUR, 1)
    snap = _make_snapshot(
        phase="EXCHANGE",
        awaiting_action="discard",
        current_player=0,
        player_hand=[card1, card2],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    assert isinstance(call_args[0][1], DiscardAction)


@pytest.mark.asyncio
async def test_auto_player_stir_when_current():
    """AutoPlayer acts during STIRRING when it's their turn."""
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        current_player=0,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()


@pytest.mark.asyncio
async def test_auto_player_stir_pass():
    """AutoPlayer can pass during STIRRING if no valid stir cards."""
    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        current_player=0,
        player_hand=[],
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_awaited()


@pytest.mark.asyncio
async def test_auto_player_bid_during_dealing():
    """AutoPlayer can bid during DEAL_BID phase if hand has trump rank cards."""
    trump_card = _card(Suit.HEARTS, Rank.TWO, 1)
    snap = _make_snapshot(
        phase="DEAL_BID",
        awaiting_action=None,
        current_player=0,
        player_hand=[trump_card],
        trump_rank=Rank.TWO,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    # May or may not bid (random), but should not error


@pytest.mark.asyncio
async def test_auto_player_ignores_dealing_if_no_trump_rank():
    """AutoPlayer does not bid during DEAL_BID if hand has no trump rank cards."""
    non_trump = _card(Suit.SPADES, Rank.THREE, 1)
    snap = _make_snapshot(
        phase="DEAL_BID",
        awaiting_action=None,
        current_player=0,
        player_hand=[non_trump],
        trump_rank=Rank.TWO,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)
    await player.on_state(game)
    await asyncio.sleep(0.05)
    game.act.assert_not_awaited()


# ---- HumanPlayer ----


@pytest.mark.asyncio
async def test_human_player_sends_state_on_push():
    """HumanPlayer sends state JSON via WebSocket on on_state."""
    ws = AsyncMock()
    snap = _make_snapshot()
    game = _make_game(snap)
    player = HumanPlayer(index=0, ws=ws)
    await player.on_state(game)
    ws.send_json.assert_awaited_once()
    sent_data = ws.send_json.call_args[0][0]
    assert sent_data["type"] == "state"
    assert "state" in sent_data
    assert sent_data["state"]["phase"] == "PLAYING"


@pytest.mark.asyncio
async def test_human_player_set_ws_replaces_reference():
    """HumanPlayer.set_ws replaces the WebSocket reference."""
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    snap = _make_snapshot()
    game = _make_game(snap)
    player = HumanPlayer(index=0, ws=ws1)
    await player.on_state(game)
    ws1.send_json.assert_awaited_once()

    player.set_ws(ws2)
    await player.on_state(game)
    ws2.send_json.assert_awaited_once()
    assert ws1.send_json.await_count == 1  # not called again


@pytest.mark.asyncio
async def test_human_player_does_not_send_when_no_ws():
    """HumanPlayer does nothing when ws is None."""
    snap = _make_snapshot()
    game = _make_game(snap)
    player = HumanPlayer(index=0, ws=None)
    # Should not raise
    await player.on_state(game)


def test_human_player_is_connected_true():
    """HumanPlayer.is_connected() returns True when ws is set."""
    ws = AsyncMock()
    player = HumanPlayer(index=0, ws=ws)
    assert player.is_connected() is True


def test_human_player_is_connected_false():
    """HumanPlayer.is_connected() returns False when ws is None."""
    player = HumanPlayer(index=0, ws=None)
    assert player.is_connected() is False


def test_human_player_is_connected_false_after_set_ws_none():
    """HumanPlayer.is_connected() returns False after set_ws(None)."""
    ws = AsyncMock()
    player = HumanPlayer(index=0, ws=ws)
    assert player.is_connected() is True
    player.set_ws(None)
    assert player.is_connected() is False


# ---- Bug 4 regression: stir must use same-suit pairs ----


@pytest.mark.asyncio
async def test_auto_player_stir_only_uses_same_suit_pairs():
    """AutoPlayer._handle_stir must only stir with same-suit pairs of trump rank.

    Regression test for Bug 4: when a player had 2+ trump-rank cards of
    different suits, the old code did `trump_cards[:2]` which could pick
    two cards of different suits — an invalid stir pair. The stirring SM
    would reject it, but the AutoPlayer would keep retrying with the same
    invalid pair in a tight loop.

    The fix groups trump-rank cards by suit and only picks a pair from
    a single suit group.
    """
    # Create 2 trump-rank cards of DIFFERENT suits and 2 of the same suit (forming a valid pair)
    card_hearts_2_d1 = _card(Suit.HEARTS, Rank.TWO, 1)
    card_spades_2_d1 = _card(Suit.SPADES, Rank.TWO, 1)
    card_hearts_2_d2 = _card(Suit.HEARTS, Rank.TWO, 2)

    snap = _make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        current_player=0,
        player_hand=[card_hearts_2_d1, card_spades_2_d1, card_hearts_2_d2],
        trump_rank=Rank.TWO,
    )
    game = _make_game(snap)
    player = AutoPlayer(index=0)

    # Force random.random() to 0.4 so the stir branch is taken
    with patch("server.player.random.random", return_value=0.4):
        await player.on_state(game)

    await asyncio.sleep(0.05)
    game.act.assert_awaited()
    call_args = game.act.call_args
    action = call_args[0][1]

    if isinstance(action, StirAction):
        # If stirring, both cards must be the same suit
        assert len(action.cards) == 2
        suits = {c.suit for c in action.cards}
        assert len(suits) == 1, (
            f"StirAction used cards of different suits: {suits}"
        )


# ---- Exchange state typed access ----


@pytest.mark.asyncio
async def test_auto_player_discard_with_exchange_state_snapshot():
    """AutoPlayer._handle_discard uses ExchangeStateSnapshot.count.

    The snapshot's exchange_state is now a structured ExchangeStateSnapshot
    instead of a dict, so attribute access (exc.count) works directly.
    """
    card1 = _card(Suit.DIAMONDS, Rank.THREE, 1)
    card2 = _card(Suit.CLUBS, Rank.FOUR, 1)
    card3 = _card(Suit.SPADES, Rank.FIVE, 1)

    snap = _make_snapshot(
        phase="EXCHANGE",
        awaiting_action="discard",
        current_player=0,
        player_hand=[card1, card2, card3],
        exchange_state=ExchangeStateSnapshot(
            phase="PICKED_UP",
            declarer_player=0,
            count=3,
        ),
    )

    game = _make_game(snap)
    player = AutoPlayer(index=0)

    # Must not raise AttributeError
    await player.on_state(game)
    await asyncio.sleep(0.05)

    game.act.assert_awaited()
    call_args = game.act.call_args
    action = call_args[0][1]
    assert isinstance(action, DiscardAction)
    assert len(action.cards) == 3
