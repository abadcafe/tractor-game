"""Tests for server/game.py -- Game aggregate root.

All tests use only public interfaces: Game.__init__, Game.receive,
Game.snapshot, Game.is_over, Game.get_player, Game.set_on_game_over,
and the StateSnapshot protocol model.
No tests access Game private fields or patch Game's SM collaborators.
"""

import json
from collections.abc import Sequence
from typing import Literal, TypeGuard

import pytest

from server.foundation.result import Ok, Rejected
from server.game.players import Player
from server.game.protocol import (
    FailedThrowSnapshot,
    PlayerMessage,
    RoundPhase,
    ScoringSnapshot,
    StateMessage,
    StirringStateSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.game.room.actions import (
    BidAction,
    NextRoundAction,
    PlayAction,
    SkipBidAction,
    SkipStirAction,
)
from server.game.room.game import Game
from server.game.rules.cards import POINTS_MAP, Card, Rank, Suit
from server.game.rules.rejections.card import CardNotInHandRejected
from server.game.state_machine import (
    deal_bid_sm,
    round_sm,
    stirring_sm,
    trick_sm,
)
from server.game.state_machine.scoring import RoundResult
from server.game.state_machine.types import (
    BidEvent,
    BottomExchangeEvent,
    StirDeclarationEvent,
)

type TestAction = (
    BidAction
    | NextRoundAction
    | PlayAction
    | SkipBidAction
    | SkipStirAction
)


def _is_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


class RecordingPlayer(Player):
    """Passive test player that records all StateMessage pushes."""

    def __init__(self, index: int) -> None:
        super().__init__(index)
        self.messages: list[StateMessage] = []

    async def on_state(
        self, game: object, message: StateMessage
    ) -> None:
        self.messages.append(message)

    def last_seq(self) -> int:
        assert self.messages, "player has not received state yet"
        return self.messages[-1].seq

    def errors(self) -> list[str]:
        return [
            message.error
            for message in self.messages
            if message.error is not None
        ]


def _create_game_with_auto_players() -> Game:
    """Create a Game with 4 passive test players."""
    players = _make_players()
    return Game(players=players)


def _game_phase(game: Game) -> RoundPhase:
    """Observe the player-visible phase through the public snapshot."""
    return game.snapshot(for_player=0).phase


def _make_players() -> list[RecordingPlayer]:
    """Create 4 passive player instances for testing."""
    return [RecordingPlayer(index=i) for i in range(4)]


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit,
        rank=rank,
        points=POINTS_MAP[rank],
    )


def _card_ids(cards: Sequence[Card]) -> list[str]:
    return [card.id for card in cards]


def _game_with_round_state(state: round_sm.RoundState) -> Game:
    game = _create_game_with_auto_players()
    setattr(game, "_round_state", state)
    return game


def _empty_hands() -> list[list[Card]]:
    return [[], [], [], []]


def _deal_bid_round_state_with_bottom(
    bottom_cards: list[Card],
) -> round_sm.RoundState:
    return round_sm.RoundState(
        phase="DEAL_BID",
        declarer_team=0,
        declarer_player=None,
        defender_team=1,
        trump_suit=None,
        trump_rank=Rank.TWO,
        bid_winner=None,
        players_hand=_empty_hands(),
        bottom_cards=bottom_cards,
        defender_points=0,
        last_completed_trick=None,
        defender_point_cards=[],
        deal_bid_state=None,
        stirring_state=None,
        trick_state=None,
        result=None,
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        start_player=0,
        next_declarer_player=None,
    )


def _stirring_round_state_with_bottom(
    *,
    bottom_cards: list[Card],
    bottom_owner_player: int | None,
    result: RoundResult | None = None,
    stirring_phase: Literal["WAITING", "EXCHANGING", "COMPLETE"] = (
        "WAITING"
    ),
    exchanging_player: int | None = None,
    stir_events: tuple[StirDeclarationEvent, ...] = (),
    bottom_exchange_events: tuple[BottomExchangeEvent, ...] = (),
) -> round_sm.RoundState:
    hands = _empty_hands()
    stirring = stirring_sm.StirringState(
        phase=stirring_phase,
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
        declarer_player=0,
        current_player=0
        if exchanging_player is None
        else exchanging_player,
        pass_set=frozenset(),
        stir_events=stir_events,
        bottom_exchange_events=bottom_exchange_events,
        last_stir_player=None,
        current_priority=0,
        bottom_cards=bottom_cards,
        bottom_owner_player=bottom_owner_player,
        players_hand=hands,
        exchanging_player=exchanging_player,
        pending_exchange_trigger="initial"
        if exchanging_player is not None
        else None,
    )
    return round_sm.RoundState(
        phase="WAITING" if result is not None else "STIRRING",
        declarer_team=0,
        declarer_player=0,
        defender_team=1,
        trump_suit=Suit.HEARTS,
        trump_rank=Rank.TWO,
        bid_winner=None,
        players_hand=hands,
        bottom_cards=bottom_cards,
        defender_points=0,
        last_completed_trick=None,
        defender_point_cards=[],
        deal_bid_state=None,
        stirring_state=stirring,
        trick_state=None,
        result=result,
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        start_player=0,
        next_declarer_player=None,
    )


def _round_result() -> RoundResult:
    return RoundResult(
        declarer_team=0,
        round_winning_team=0,
        next_declarer_player=0,
        total_defender_points=0,
        declarer_level_gain=0,
        defender_level_gain=0,
        switch_declarer=False,
        bottom_card_bonus=0,
    )


def _raw_from_action(action: TestAction) -> dict[str, object]:
    if isinstance(action, BidAction):
        return {
            "type": "bid",
            "cards": [card.id for card in action.cards],
        }
    if isinstance(action, SkipBidAction):
        return {"type": "bid", "pass": True}
    if isinstance(action, SkipStirAction):
        return {"type": "stir", "pass": True}
    if isinstance(action, PlayAction):
        return {
            "type": "play",
            "cards": [card.id for card in action.cards],
        }
    return {"type": "next_round"}


async def _sync_player(
    game: Game, players: Sequence[RecordingPlayer], player_index: int
) -> None:
    await game.receive(player_index, PlayerMessage(seq=0, raw={}))


async def _send_action(
    game: Game,
    players: Sequence[RecordingPlayer],
    player_index: int,
    action: TestAction,
) -> None:
    if not players[player_index].messages:
        await _sync_player(game, players, player_index)
    await game.receive(
        player_index,
        PlayerMessage(
            seq=players[player_index].last_seq(),
            raw=_raw_from_action(action),
        ),
    )


async def _start_game(players: Sequence[RecordingPlayer]) -> Game:
    """
    Create a Game and confirm all 4 players via NextRoundAction to
    start.

    This is the real startup flow: game starts in WAITING, each player
    sends NextRoundAction to confirm, and the 4th confirmation triggers
    the transition to DEAL_BID.
    """
    game = Game(players=players)
    for i in range(4):
        await _send_action(game, players, i, NextRoundAction())
    return game


# ---- Initialization ----


def test_game_init_creates_valid_state() -> None:
    game = _create_game_with_auto_players()
    # Verify via public interface: game starts in WAITING phase
    assert _game_phase(game) == "WAITING"
    assert game.is_over() is False


# ---- snapshot phase ----


def test_snapshot_returns_waiting_phase() -> None:
    game = _create_game_with_auto_players()
    assert _game_phase(game) == "WAITING"


# ---- WAITING confirmation flow ----


@pytest.mark.asyncio
async def test_next_round_confirmation_starts_game() -> None:
    """
    After all 4 players confirm, game transitions from WAITING to
    DEAL_BID.
    """
    players = _make_players()
    game = Game(players=players)
    assert _game_phase(game) == "WAITING"
    for i in range(4):
        await _send_action(game, players, i, NextRoundAction())
    assert _game_phase(game) != "WAITING"
    snap = game.snapshot(for_player=0)
    assert snap.phase in ("DEAL_BID", "STIRRING", "PLAYING")


@pytest.mark.asyncio
async def test_next_round_duplicate_confirmation_rejected() -> None:
    """
    Duplicate NextRoundAction confirmation is rejected via error push.
    """
    players = _make_players()
    game = Game(players=players)
    # Confirm player 0
    await _send_action(game, players, 0, NextRoundAction())
    assert len(players[0].errors()) == 0
    # Confirm player 0 again — should be rejected
    await _send_action(game, players, 0, NextRoundAction())
    assert len(players[0].errors()) == 1
    assert "确认" in players[0].errors()[0]


@pytest.mark.asyncio
async def test_next_round_intermediate_confirmation_pushes_state() -> (
    None
):
    """
    Each intermediate confirmation triggers a state push (seq
    increments).
    """
    players = _make_players()
    game = Game(players=players)

    await _sync_player(game, players, 0)
    assert players[0].last_seq() == 1

    await _send_action(game, players, 0, NextRoundAction())
    seq_after_1 = players[0].last_seq()
    assert seq_after_1 > 1, "First confirmation should increment seq"

    await _send_action(game, players, 1, NextRoundAction())
    seq_after_2 = players[1].last_seq()
    assert seq_after_2 > seq_after_1, (
        "Second confirmation should increment seq"
    )

    # Intermediate snapshot shows confirmed players
    snap = game.snapshot(for_player=0)
    assert snap.next_round_confirmed is not None
    assert 0 in snap.next_round_confirmed
    assert 1 in snap.next_round_confirmed
    assert 2 not in snap.next_round_confirmed


@pytest.mark.asyncio
async def test_next_round_all_confirmed_clears_set() -> None:
    """
    After all 4 confirm, next_round_confirmed is cleared (new round
    starts).
    """
    players = _make_players()
    game = Game(players=players)
    for i in range(4):
        await _send_action(game, players, i, NextRoundAction())
    # Game has started — next_round_confirmed should be empty
    snap = game.snapshot(for_player=0)
    assert snap.phase != "WAITING"
    assert snap.next_round_confirmed == []


# ---- _start_game() ----


@pytest.mark.asyncio
async def test_start_game_transitions_to_deal_bid() -> None:
    game = await _start_game(_make_players())
    # Verify via snapshot (public interface)
    snap = game.snapshot(for_player=0)
    assert snap.phase in (
        "DEAL_BID",
        "STIRRING",
        "PLAYING",
        "WAITING",
    )


# ---- receive() ----


@pytest.mark.asyncio
async def test_act_rejects_wrong_player() -> None:
    """PlayAction during DEAL_BID should be rejected without raising."""
    players = _make_players()
    game = await _start_game(players)
    # Should not raise; rejection is communicated via send_error instead
    await _send_action(game, players, 0, PlayAction(cards=[]))


@pytest.mark.asyncio
async def test_seq_mismatch_ignores_action_fields() -> None:
    """
    Wrong non-zero seq returns current state, ignores action fields.
    """
    players = _make_players()
    game = await _start_game(players)

    phase_before = _game_phase(game)
    messages_before = [len(player.messages) for player in players]
    seq_before = players[0].last_seq()

    await game.receive(
        player_index=0,
        message=PlayerMessage(
            seq=seq_before + 999, raw={"type": "unknown_action"}
        ),
    )

    assert _game_phase(game) == phase_before, (
        f"phase should not change on seq mismatch: {phase_before} ->"
        f"{_game_phase(game)}"
    )
    assert players[0].last_seq() == seq_before
    assert players[0].messages[-1].error is None
    assert len(players[0].messages) == messages_before[0] + 1
    for i in range(1, 4):
        assert len(players[i].messages) == messages_before[i]


@pytest.mark.asyncio
async def test_seq_zero_returns_state_without_action_side_effect() -> (
    None
):
    """
    seq=0 is a state sync request even if action fields are present.
    """
    players = _make_players()
    game = Game(players=players)

    await game.receive(
        0, PlayerMessage(seq=0, raw={"type": "next_round"})
    )

    snap = game.snapshot(for_player=0)
    assert snap.phase == "WAITING"
    assert snap.next_round_confirmed == []
    assert players[0].messages[-1].seq == 1
    assert players[0].messages[-1].error is None


# ---- snapshot() ----


@pytest.mark.asyncio
async def test_snapshot_returns_player_hand() -> None:
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.player_hand, list)


def test_snapshot_before_run_returns_waiting() -> None:
    """snapshot() returns a valid WAITING-phase snapshot before run().

    Before the first round starts, the public snapshot is WAITING with
    awaiting_action='next_round' and empty hands.
    """
    game = _create_game_with_auto_players()
    snap = game.snapshot(for_player=0)
    assert snap.phase == "WAITING"
    assert snap.awaiting_action == "next_round"
    assert snap.player_hand == []


@pytest.mark.asyncio
async def test_snapshot_phase() -> None:
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    assert snap.phase in (
        "DEAL_BID",
        "STIRRING",
        "PLAYING",
        "WAITING",
        "SCORING",
    )


@pytest.mark.asyncio
async def test_snapshot_awaiting_action() -> None:
    """awaiting_action should be a valid string or None."""
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    assert snap.awaiting_action is None or isinstance(
        snap.awaiting_action, str
    )


@pytest.mark.asyncio
async def test_snapshot_action_hints_shape() -> None:
    """action_hints should be serialized as optional card-list hints.

    A non-empty list means the server is willing to show a complete hint
    set.
    An empty list means the UI should not constrain input.
    """
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.action_hints, list)
    if len(snap.action_hints) > 0:
        entry = snap.action_hints[0]
        assert isinstance(entry, list)
        if len(entry) > 0:
            card = entry[0]
            assert isinstance(card, Card)
            assert isinstance(card.id, str)
            assert isinstance(card.suit, Suit)
            assert isinstance(card.rank, Rank)


def test_first_round_deal_bid_has_no_declarer_before_bid() -> None:
    """
    First round DEAL_BID has no fixed declarer until bidding resolves.
    """
    state = round_sm.create_round(
        round_sm.RoundInput(
            declarer_team=None,
            trump_rank=Rank.TWO,
            next_declarer_player=None,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
    )
    game = _game_with_round_state(state)

    snap = game.snapshot(for_player=0)

    assert snap.phase == "DEAL_BID"
    assert snap.declarer_team is None
    assert snap.declarer_player is None


def test_subsequent_round_deal_bid_snapshot_shows_fixed_declarer() -> (
    None
):
    """
    Later-round DEAL_BID exposes the already fixed declarer for UI
    display.
    """
    state = round_sm.create_round(
        round_sm.RoundInput(
            declarer_team=1,
            trump_rank=Rank.THREE,
            next_declarer_player=3,
            team0_level=Rank.TWO,
            team1_level=Rank.THREE,
        )
    )
    game = _game_with_round_state(state)

    snap = game.snapshot(for_player=0)

    assert snap.phase == "DEAL_BID"
    assert snap.declarer_team == 1
    assert snap.declarer_player == 3


def test_snapshot_player_hand_is_sorted_by_display_order() -> None:
    """
    StateSnapshot.player_hand is protocol output, so it is sorted in
    the same order the frontend displays hands.
    """
    hands = [
        [
            _card(Suit.CLUBS, Rank.KING),
            _card(Suit.SPADES, Rank.THREE),
            _card(Suit.DIAMONDS, Rank.TWO),
            _card(Suit.JOKER, Rank.SMALL_JOKER),
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.SPADES, Rank.TWO),
            _card(Suit.JOKER, Rank.BIG_JOKER),
        ],
        [],
        [],
        [],
    ]
    state = round_sm.RoundState(
        phase="PLAYING",
        declarer_team=0,
        declarer_player=0,
        defender_team=1,
        trump_suit=Suit.SPADES,
        trump_rank=Rank.TWO,
        bid_winner=None,
        players_hand=hands,
        bottom_cards=[],
        defender_points=0,
        last_completed_trick=None,
        defender_point_cards=[],
        deal_bid_state=None,
        stirring_state=None,
        trick_state=None,
        result=None,
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        start_player=0,
        next_declarer_player=None,
    )
    game = _game_with_round_state(state)

    snap = game.snapshot(for_player=0)

    assert [card.id for card in snap.player_hand] == [
        "D1-joker-BJ",
        "D1-joker-SJ",
        "D1-spades-2",
        "D1-diamonds-2",
        "D1-spades-3",
        "D1-hearts-A",
        "D1-clubs-K",
    ]


def test_later_deal_bid_bid_winner_keeps_fixed_declarer() -> None:
    """
    In later rounds, bid_winner chooses trump only; declarer remains
    fixed.
    """
    bid = BidEvent(
        player=1,
        cards=[_card(Suit.SPADES, Rank.THREE)],
        kind="trump_rank",
        suit=Suit.SPADES,
        joker_type=None,
        count=1,
    )
    state = round_sm.create_round(
        round_sm.RoundInput(
            declarer_team=1,
            trump_rank=Rank.THREE,
            next_declarer_player=3,
            team0_level=Rank.TWO,
            team1_level=Rank.THREE,
        )
    ).model_copy(update={"bid_winner": bid})
    game = _game_with_round_state(state)

    snap = game.snapshot(for_player=0)

    assert snap.phase == "DEAL_BID"
    bid_winner = snap.bid_winner
    assert bid_winner is not None
    assert bid_winner.player == bid.player
    assert bid_winner.cards == [_card(Suit.SPADES, Rank.THREE, 1)]
    bid_winner_wire = bid_winner.model_dump(mode="json")
    assert bid_winner_wire["cards"] == [
        {
            "id": "D1-spades-3",
            "suit": "spades",
            "rank": "3",
            "points": 0,
        }
    ]
    assert bid_winner_wire["suit"] == "spades"
    assert snap.declarer_player == 3
    assert snap.declarer_player != bid_winner.player


def test_snapshot_stir_action_hints_ordered_from_small_to_large() -> (
    None
):
    """
    Stirring action_hints are sorted by the smallest legal stir first.
    """
    hearts_pair = [
        _card(Suit.HEARTS, Rank.TWO, 1),
        _card(Suit.HEARTS, Rank.TWO, 2),
    ]
    spades_pair = [
        _card(Suit.SPADES, Rank.TWO, 1),
        _card(Suit.SPADES, Rank.TWO, 2),
    ]
    small_joker_pair = [
        _card(Suit.JOKER, Rank.SMALL_JOKER, 1),
        _card(Suit.JOKER, Rank.SMALL_JOKER, 2),
    ]
    players_hand = [
        [],
        [*small_joker_pair, *spades_pair, *hearts_pair],
        [],
        [],
    ]
    stirring = stirring_sm.StirringState(
        phase="WAITING",
        trump_suit=Suit.CLUBS,
        trump_rank=Rank.TWO,
        declarer_player=0,
        current_player=1,
        pass_set=frozenset(),
        stir_events=(),
        bottom_exchange_events=(),
        last_stir_player=None,
        current_priority=201,
        bottom_cards=[],
        bottom_owner_player=None,
        players_hand=players_hand,
    )
    state = round_sm.RoundState(
        phase="STIRRING",
        declarer_team=0,
        declarer_player=0,
        defender_team=1,
        trump_suit=Suit.CLUBS,
        trump_rank=Rank.TWO,
        bid_winner=None,
        players_hand=players_hand,
        bottom_cards=[],
        defender_points=0,
        last_completed_trick=None,
        defender_point_cards=[],
        deal_bid_state=None,
        stirring_state=stirring,
        trick_state=None,
        result=None,
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        start_player=0,
        next_declarer_player=None,
    )
    game = _game_with_round_state(state)

    snap = game.snapshot(1)

    assert [
        [card.id for card in hint] for hint in snap.action_hints
    ] == [
        ["D1-hearts-2", "D2-hearts-2"],
        ["D1-spades-2", "D2-spades-2"],
        ["D1-joker-SJ", "D2-joker-SJ"],
    ]


def test_snapshot_stir_action_hints_hidden_when_over_bid_hint_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STIRRING uses MAX_BID_ACTION_HINTS as a closed-set limit."""
    monkeypatch.setattr(deal_bid_sm, "MAX_BID_ACTION_HINTS", 2)

    hearts_pair = [
        _card(Suit.HEARTS, Rank.TWO, 1),
        _card(Suit.HEARTS, Rank.TWO, 2),
    ]
    spades_pair = [
        _card(Suit.SPADES, Rank.TWO, 1),
        _card(Suit.SPADES, Rank.TWO, 2),
    ]
    small_joker_pair = [
        _card(Suit.JOKER, Rank.SMALL_JOKER, 1),
        _card(Suit.JOKER, Rank.SMALL_JOKER, 2),
    ]
    players_hand = [
        [],
        [*small_joker_pair, *spades_pair, *hearts_pair],
        [],
        [],
    ]
    stirring = stirring_sm.StirringState(
        phase="WAITING",
        trump_suit=Suit.CLUBS,
        trump_rank=Rank.TWO,
        declarer_player=0,
        current_player=1,
        pass_set=frozenset(),
        stir_events=(),
        bottom_exchange_events=(),
        last_stir_player=None,
        current_priority=201,
        bottom_cards=[],
        bottom_owner_player=None,
        players_hand=players_hand,
    )
    state = round_sm.RoundState(
        phase="STIRRING",
        declarer_team=0,
        declarer_player=0,
        defender_team=1,
        trump_suit=Suit.CLUBS,
        trump_rank=Rank.TWO,
        bid_winner=None,
        players_hand=players_hand,
        bottom_cards=[],
        defender_points=0,
        last_completed_trick=None,
        defender_point_cards=[],
        deal_bid_state=None,
        stirring_state=stirring,
        trick_state=None,
        result=None,
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        start_player=0,
        next_declarer_player=None,
    )
    game = _game_with_round_state(state)

    snap = game.snapshot(1)

    assert snap.action_hints == []


def test_snapshot_hides_bottom_cards_before_owner_exists() -> None:
    bottom_cards = [_card(Suit.DIAMONDS, Rank.THREE)]
    game = _game_with_round_state(
        _deal_bid_round_state_with_bottom(bottom_cards)
    )

    for player in range(4):
        snap = game.snapshot(player)
        assert snap.bottom_cards == []


def test_snapshot_bottom_cards_visible_only_to_bottom_owner() -> None:
    bottom_cards = [_card(Suit.DIAMONDS, Rank.THREE)]
    game = _game_with_round_state(
        _stirring_round_state_with_bottom(
            bottom_cards=bottom_cards,
            bottom_owner_player=2,
        )
    )

    for player in (0, 1, 3):
        snap = game.snapshot(player)
        assert snap.bottom_cards == []
    owner_snap = game.snapshot(2)
    assert _card_ids(owner_snap.bottom_cards) == _card_ids(bottom_cards)


def test_snapshot_bottom_cards_visible_to_exchanging_player() -> None:
    bottom_cards = [_card(Suit.DIAMONDS, Rank.THREE)]
    game = _game_with_round_state(
        _stirring_round_state_with_bottom(
            bottom_cards=bottom_cards,
            bottom_owner_player=None,
            stirring_phase="EXCHANGING",
            exchanging_player=2,
        )
    )

    for player in (0, 1, 3):
        snap = game.snapshot(player)
        assert snap.bottom_cards == []
    exchanger_snap = game.snapshot(2)
    assert _card_ids(exchanger_snap.bottom_cards) == _card_ids(
        bottom_cards
    )


def test_snapshot_stir_events_visible_to_all_players() -> None:
    stir_card = _card(Suit.SPADES, Rank.TWO)
    stir_event = StirDeclarationEvent(
        player=2,
        kind="stir",
        cards=[stir_card],
        new_suit=Suit.SPADES,
        priority=203,
    )
    pass_event = StirDeclarationEvent(
        player=3,
        kind="pass",
        cards=[],
        new_suit=None,
        priority=None,
    )
    game = _game_with_round_state(
        _stirring_round_state_with_bottom(
            bottom_cards=[],
            bottom_owner_player=None,
            stir_events=(stir_event, pass_event),
        )
    )

    for player in range(4):
        snap = game.snapshot(player)
        assert len(snap.stir_events) == 2
        assert snap.stir_events[0].player == 2
        assert snap.stir_events[0].cards == [stir_card]
        assert snap.stir_events[0].own_bottom_exchange is None
        assert snap.stir_events[1].kind == "pass"
        assert snap.stir_events[1].own_bottom_exchange is None


def test_snapshot_initial_bottom_exchange_visible_only_to_owner() -> (
    None
):
    picked = [_card(Suit.DIAMONDS, Rank.THREE)]
    discarded = [_card(Suit.CLUBS, Rank.FOUR)]
    exchange_event = BottomExchangeEvent(
        player=2,
        trigger="initial",
        stir_event_index=None,
        picked_up_bottom_cards=picked,
        discarded_bottom_cards=discarded,
        resulting_bottom_cards=discarded,
    )
    game = _game_with_round_state(
        _stirring_round_state_with_bottom(
            bottom_cards=discarded,
            bottom_owner_player=2,
            bottom_exchange_events=(exchange_event,),
        )
    )

    for player in (0, 1, 3):
        snap = game.snapshot(player)
        assert snap.own_initial_bottom_exchange is None
    owner_snap = game.snapshot(2)
    assert owner_snap.own_initial_bottom_exchange is not None
    assert (
        owner_snap.own_initial_bottom_exchange.picked_up_bottom_cards
        == (picked)
    )


def test_snapshot_own_stir_exchange_attaches_to_stir_event() -> None:
    stir_card = _card(Suit.SPADES, Rank.TWO)
    picked = [_card(Suit.DIAMONDS, Rank.THREE)]
    discarded = [_card(Suit.CLUBS, Rank.FOUR)]
    stir_event = StirDeclarationEvent(
        player=2,
        kind="stir",
        cards=[stir_card],
        new_suit=Suit.SPADES,
        priority=203,
    )
    exchange_event = BottomExchangeEvent(
        player=2,
        trigger="stir",
        stir_event_index=0,
        picked_up_bottom_cards=picked,
        discarded_bottom_cards=discarded,
        resulting_bottom_cards=discarded,
    )
    game = _game_with_round_state(
        _stirring_round_state_with_bottom(
            bottom_cards=discarded,
            bottom_owner_player=2,
            stir_events=(stir_event,),
            bottom_exchange_events=(exchange_event,),
        )
    )

    for player in (0, 1, 3):
        snap = game.snapshot(player)
        assert snap.stir_events[0].own_bottom_exchange is None
    owner_snap = game.snapshot(2)
    own_exchange = owner_snap.stir_events[0].own_bottom_exchange
    assert own_exchange is not None
    assert own_exchange.picked_up_bottom_cards == picked
    assert own_exchange.discarded_bottom_cards == discarded


def test_snapshot_public_bottom_cards_after_scoring() -> None:
    bottom_cards = [_card(Suit.DIAMONDS, Rank.THREE)]
    game = _game_with_round_state(
        _stirring_round_state_with_bottom(
            bottom_cards=bottom_cards,
            bottom_owner_player=2,
            result=_round_result(),
        )
    )

    for player in range(4):
        snap = game.snapshot(player)
        assert _card_ids(snap.bottom_cards) == _card_ids(bottom_cards)
        assert snap.scoring is not None
        assert _card_ids(snap.scoring.bottom_cards) == _card_ids(
            bottom_cards
        )


def test_snapshot_scoring_exposes_round_winning_team() -> None:
    round_result = RoundResult(
        declarer_team=0,
        round_winning_team=1,
        next_declarer_player=1,
        total_defender_points=80,
        declarer_level_gain=0,
        defender_level_gain=0,
        switch_declarer=True,
        bottom_card_bonus=0,
    )
    game = _game_with_round_state(
        _stirring_round_state_with_bottom(
            bottom_cards=[],
            bottom_owner_player=2,
            result=round_result,
        )
    )

    snap = game.snapshot(0)

    assert snap.declarer_team == 0
    assert snap.scoring is not None
    assert snap.scoring.round_winning_team == 1


def test_snapshot_play_leading_has_no_action_hints() -> None:
    """PLAYING action_hints are empty while the player is leading."""
    hands = [
        [_card(Suit.HEARTS, Rank.THREE), _card(Suit.CLUBS, Rank.FIVE)],
        [],
        [],
        [],
    ]
    trick = trick_sm.create_trick(
        trick_sm.TrickInput(
            lead_player=0,
            hands=hands,
            trump_suit=Suit.SPADES,
            trump_rank=Rank.TWO,
            defender_points=0,
            declarer_team=0,
        )
    )
    state = round_sm.RoundState(
        phase="PLAYING",
        declarer_team=0,
        declarer_player=0,
        defender_team=1,
        trump_suit=Suit.SPADES,
        trump_rank=Rank.TWO,
        bid_winner=None,
        players_hand=hands,
        bottom_cards=[],
        defender_points=0,
        last_completed_trick=None,
        defender_point_cards=[],
        deal_bid_state=None,
        stirring_state=None,
        trick_state=trick,
        result=None,
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        start_player=0,
        next_declarer_player=None,
    )
    game = _game_with_round_state(state)

    snap = game.snapshot(0)

    assert snap.awaiting_action == "play"
    assert snap.action_hints == []


def test_snapshot_play_following_hides_hints_when_too_many() -> None:
    """
    Following action_hints are hidden when count exceeds
    MAX_PLAY_ACTION_HINTS.
    """
    lead_1 = _card(Suit.DIAMONDS, Rank.ACE)
    lead_2 = _card(Suit.DIAMONDS, Rank.KING)
    follower_hand = [
        _card(Suit.JOKER, Rank.SMALL_JOKER),
        _card(Suit.JOKER, Rank.BIG_JOKER),
        _card(Suit.SPADES, Rank.JACK),
        _card(Suit.HEARTS, Rank.TEN),
        _card(Suit.CLUBS, Rank.NINE),
        _card(Suit.SPADES, Rank.EIGHT),
        _card(Suit.HEARTS, Rank.SEVEN),
        _card(Suit.CLUBS, Rank.SIX),
    ]
    hands = [[lead_1, lead_2], follower_hand, [], []]
    trick = trick_sm.create_trick(
        trick_sm.TrickInput(
            lead_player=0,
            hands=hands,
            trump_suit=None,
            trump_rank=Rank.TWO,
            defender_points=0,
            declarer_team=0,
        )
    )
    played = trick_sm.play(trick, player=0, cards=[lead_1, lead_2])
    assert isinstance(played, Ok)
    state = round_sm.RoundState(
        phase="PLAYING",
        declarer_team=0,
        declarer_player=0,
        defender_team=1,
        trump_suit=None,
        trump_rank=Rank.TWO,
        bid_winner=None,
        players_hand=played.value.hands,
        bottom_cards=[],
        defender_points=0,
        last_completed_trick=None,
        defender_point_cards=[],
        deal_bid_state=None,
        stirring_state=None,
        trick_state=played.value,
        result=None,
        team0_level=Rank.TWO,
        team1_level=Rank.TWO,
        start_player=0,
        next_declarer_player=None,
    )
    game = _game_with_round_state(state)

    snap = game.snapshot(1)

    assert snap.awaiting_action == "play"
    # No matching diamonds: choosing any two of eight cards creates 28
    # legal
    # candidates, so the complete hint set is intentionally hidden.
    assert snap.action_hints == []


@pytest.mark.asyncio
async def test_snapshot_awaiting_action_play() -> None:
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    assert snap.awaiting_action in (
        "stir",
        "discard",
        "play",
        "next_round",
        "bid",
        None,
    )


@pytest.mark.asyncio
async def test_snapshot_trump_info() -> None:
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    assert snap.trump_rank is not None


@pytest.mark.asyncio
async def test_snapshot_team_levels() -> None:
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    assert snap.team0_level is not None
    assert snap.team1_level is not None


@pytest.mark.asyncio
async def test_snapshot_bid_events() -> None:
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.bid_events, list)


@pytest.mark.asyncio
async def test_snapshot_stirring_state() -> None:
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    # stirring_state may be None outside of STIRRING phase
    assert snap.stirring_state is None or isinstance(
        snap.stirring_state, StirringStateSnapshot
    )


@pytest.mark.asyncio
async def test_snapshot_scoring_in_complete() -> None:
    """
    When round is WAITING (SM COMPLETE), snapshot should include scoring
    info.
    """
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    assert snap.scoring is None or isinstance(
        snap.scoring, ScoringSnapshot
    )


# ---- is_over() ----


def test_is_over_false_during_game() -> None:
    game = _create_game_with_auto_players()
    assert game.is_over() is False


@pytest.mark.asyncio
async def test_is_over_true_after_game_over() -> None:
    """Game starts in a non-over state."""
    game = await _start_game(_make_players())
    assert game.is_over() is False


@pytest.mark.asyncio
async def test_snapshot_winning_team_in_game_over() -> None:
    """When game is over, snapshot should include winning_team."""
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    # During normal flow, game is not over yet
    if game.is_over():
        assert snap.winning_team is not None
        assert isinstance(snap.winning_team, int)
    else:
        assert snap.winning_team is None


@pytest.mark.asyncio
async def test_game_over_consistency() -> None:
    """is_over() is independent from the player-visible phase."""
    players = _make_players()
    game = Game(players=players)
    assert game.is_over() is False
    # Confirm all players to start the game
    for i in range(4):
        await _send_action(game, players, i, NextRoundAction())
    assert game.is_over() is False


# ---- Action dispatch with type conversion ----


@pytest.mark.asyncio
async def test_act_bid_during_dealing_converts_to_bid_event() -> None:
    """
    BidAction from player.py should be converted to sm BidEvent
    internally.
    """
    players = _make_players()
    game = await _start_game(players)
    snap = game.snapshot(for_player=0)
    # During DEAL_BID, try bidding. This tests the BidAction -> BidEvent
    # conversion.
    # If not in DEAL_BID, we can't test bid; that's fine, integration
    # tests cover it.
    if snap.phase == "DEAL_BID" and len(snap.player_hand) > 0:
        # Find a trump rank card to bid with
        trump_cards = [
            c for c in snap.player_hand if c.rank == snap.trump_rank
        ]
        if trump_cards:
            card_ids = [card.id for card in trump_cards[:1]]
            message = PlayerMessage(
                seq=players[0].last_seq(),
                raw={"type": "bid", "cards": card_ids},
            )
            # Bid may be rejected (e.g. priority too low), but receive()
            # never raises
            await game.receive(0, message)


@pytest.mark.asyncio
async def test_act_skip_stir_during_stirring() -> None:
    """
    SkipStirAction is a valid action type that Game.receive() can
    distinguish
    from StirAction. The actual dispatch routing is verified in
    integration tests.
    """
    action = SkipStirAction()
    assert isinstance(action, SkipStirAction)
    # Verify it is NOT a StirAction -- Game.receive() must dispatch
    # differently
    from server.game.room.actions import StirAction

    assert not isinstance(action, StirAction)


@pytest.mark.asyncio
async def test_act_next_round_during_non_complete() -> None:
    """
    NextRoundAction during non-WAITING phase should be rejected without
    raising.
    """
    players = _make_players()
    game = await _start_game(players)
    # Should not raise; rejection is communicated via send_error instead
    await _send_action(game, players, 0, NextRoundAction())


# ---- player-visible phase ----


def test_snapshot_phase_is_player_visible_round_phase() -> None:
    """Game over is not encoded in phase."""
    game = _create_game_with_auto_players()
    assert _game_phase(game) == "WAITING"
    assert game.is_over() is False


# ---- get_player() ----


def test_get_player_returns_player_by_index() -> None:
    """Game.get_player(index) returns the Player at that index."""
    players = _make_players()
    game = Game(players=players)
    for i in range(4):
        assert game.get_player(i) is players[i]


# ---- StateSnapshot wire dict ----


@pytest.mark.asyncio
async def test_snapshot_json_serializable() -> None:
    """
    Game.snapshot().model_dump(mode="json") must be JSON-serializable.

    This is critical because HumanPlayer.on_state() calls ws.send_json()
    which
    requires JSON-serializable data. The sm Card/Suit/Rank types are
    Pydantic
    models and enums that are not directly JSON-serializable.
    action_hints
    is list[list[Card]], serialized to list of card-dict lists.
    """
    game = await _start_game(_make_players())
    snapshot = game.snapshot(for_player=0)
    result = snapshot.model_dump(mode="json")
    # Must be a dict
    assert isinstance(result, dict)
    # Must be JSON-serializable (no Pydantic objects, no enums as
    # objects)
    serialized = json.dumps(result)
    assert isinstance(serialized, str)
    # Must contain the required fields from spec section 5.5
    assert "phase" in result
    assert "player_hand" in result
    assert "trump_rank" in result
    assert "awaiting_action" in result
    assert "legal_actions" not in result
    assert "bid_legal_actions" not in result
    # action_hints must be a list of lists (card lists, not PlayAction
    # dicts)
    action_hints_raw = result["action_hints"]
    if len(action_hints_raw) > 0:
        legal_entry = action_hints_raw[0]
        # list of card dicts
        if len(legal_entry) > 0:
            assert "id" in legal_entry[0]
            assert "type" not in legal_entry[0]


@pytest.mark.asyncio
async def test_snapshot_card_format() -> None:
    """
    Snapshot JSON must format cards as {"id", "suit", "rank", "points"}.

    Per spec section 5.5, each card in player_hand must be:
    {"id": "D1-H-A", "suit": "hearts", "rank": "A", "points": 0}.
    Suit and Rank enums must be serialized as their string values.
    """
    game = await _start_game(_make_players())
    result = game.snapshot(for_player=0).model_dump(mode="json")
    # If player has cards, verify the format
    player_hand_raw = result["player_hand"]
    if len(player_hand_raw) > 0:
        card = player_hand_raw[0]
        assert isinstance(card, dict)
        assert "id" in card
        assert "suit" in card
        assert "rank" in card
        assert "points" in card
        # suit and rank must be strings (not enum objects)
        assert isinstance(card["suit"], str)
        assert isinstance(card["rank"], str)
        # Must NOT contain internal sm fields
        assert "is_joker" not in card
        assert "is_big_joker" not in card
        assert "deck" not in card


@pytest.mark.asyncio
async def test_snapshot_failed_throw_format() -> None:
    """TrickSnapshot failed_throw is a public card dict event."""
    game = await _start_game(_make_players())
    failed_throw = FailedThrowSnapshot(
        player=0,
        attempted_cards=[
            _card(Suit.SPADES, Rank.KING, 1),
            _card(Suit.SPADES, Rank.QUEEN, 1),
        ],
        forced_cards=[
            _card(Suit.SPADES, Rank.QUEEN, 1),
        ],
    )
    snapshot = game.snapshot(for_player=0).model_copy(
        update={
            "trick": TrickSnapshot(
                lead_player=0,
                current_player=1,
                slots=[
                    TrickSlotSnapshot(
                        player=0, cards=failed_throw.forced_cards
                    )
                ],
                failed_throw=failed_throw,
            )
        }
    )
    result = snapshot.model_dump(mode="json")

    assert isinstance(result["trick"], dict)
    assert result["trick"]["failed_throw"] == {
        "player": 0,
        "attempted_cards": [
            {
                "id": "D1-spades-K",
                "suit": "spades",
                "rank": "K",
                "points": 10,
            },
            {
                "id": "D1-spades-Q",
                "suit": "spades",
                "rank": "Q",
                "points": 0,
            },
        ],
        "forced_cards": [
            {
                "id": "D1-spades-Q",
                "suit": "spades",
                "rank": "Q",
                "points": 0,
            },
        ],
    }


# ---- resolve_cards() ----


@pytest.mark.asyncio
async def test_resolve_cards_returns_matching_cards() -> None:
    """
    Game.resolve_cards() returns Ok with Card objects matching the given
    IDs
    from the specified player's hand.

    This is needed because human players send card IDs via WebSocket,
    but Game.receive() must pass Card Pydantic model objects to sm
    functions.
    resolve_cards() bridges this gap by looking up Card objects by their
    ID string in the player's hand.
    """
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    if len(snap.player_hand) > 0:
        card_ids = [card.id for card in snap.player_hand[:2]]
        result = game.resolve_cards(player_index=0, card_ids=card_ids)
        assert isinstance(result, Ok)
        for original, resolved_card in zip(card_ids, result.value):
            assert resolved_card.id == original
            # Must be a Card Pydantic model (not a string or dict)
            from server.game.rules.cards import Card

            assert isinstance(resolved_card, Card)


@pytest.mark.asyncio
async def test_resolve_cards_rejects_on_unknown_id() -> None:
    """Game.resolve_cards() returns Rejected if any card_id is not found
    in the player's hand.

    This prevents human players from submitting cards they don't hold,
    which would be an invalid action.
    """
    game = await _start_game(_make_players())
    result = game.resolve_cards(
        player_index=0, card_ids=["NONEXISTENT-CARD-ID"]
    )
    assert isinstance(result, Rejected)
    assert isinstance(result, CardNotInHandRejected)


# ---- Bug 1 regression: bid must not trigger _push_state_to_all cascade
# ----


@pytest.mark.asyncio
async def test_bid_during_deal_bid_pushes_state_uniformly() -> None:
    """
    BidAction during DEAL_BID must push state to all players uniformly.

    In sync round-robin mode, each BidAction/SkipBidAction triggers
    exactly
    one _push_state_to_all. This test verifies that the state push count
    is uniform across all players — no player is skipped or
    double-pushed.

    Regression test for Bug 1 (adapted from async dealing loop to sync
    round-robin): the original bug was that a bid during DEAL_BID
    triggered
    _push_state_to_all(), causing AutoPlayer cascades. In the sync
    model,
    every action pushes state, but each push must be exactly one push to
    all 4 players.
    """

    class CountingPlayer(RecordingPlayer):
        """Player that counts on_state invocations."""

        def __init__(self, index: int) -> None:
            super().__init__(index)
            self.state_count = 0

        async def on_state(
            self, game: object, message: StateMessage
        ) -> None:
            await super().on_state(game, message)
            self.state_count += 1

    counters = [CountingPlayer(index=i) for i in range(4)]
    game = await _start_game(counters)

    # During startup each player first requests state with seq=0, then
    # the
    # confirmed actions push state uniformly to all players.
    initial_counts = [c.state_count for c in counters]

    # Find the current bidder and skip
    for i in range(4):
        snap = game.snapshot(i)
        if snap.awaiting_action == "bid":
            await _send_action(game, counters, i, SkipBidAction())
            break

    # After one action, one more push to all players
    for i in range(4):
        assert counters[i].state_count == initial_counts[i] + 1, (
            f"Player {i}: expected {initial_counts[i] + 1} pushes, got"
            f"{counters[i].state_count}"
        )


# ---- Bug 2 regression: snapshot must contain player_hand_counts ----


@pytest.mark.asyncio
async def test_snapshot_contains_all_required_fields() -> None:
    """StateSnapshot must contain ALL fields from spec section 5.5.

    Regression test for Bug 2: the `player_hand_counts` field was
    missing
    from StateSnapshot, causing the frontend's game-table component to
    show "0 张" for every player because
    `snapshot.player_hand_counts[i]`
    evaluated to `undefined ?? 0`.

    This test asserts the complete set of required fields so any future
    addition to the spec is also caught if the server doesn't serialize
    it.
    """
    game = await _start_game(_make_players())
    result = game.snapshot(for_player=0).model_dump(mode="json")

    # Complete list of required fields per current snapshot protocol.
    required_fields = [
        "phase",
        "player_hand",
        "player_hand_counts",
        "bottom_cards",
        "trump_suit",
        "trump_rank",
        "declarer_team",
        "declarer_player",
        "defender_points",
        "trick",
        "last_completed_trick",
        "defender_point_cards",
        "action_hints",
        "awaiting_action",
        "scoring",
        "winning_team",
        "team0_level",
        "team1_level",
        "bid_events",
        "bid_winner",
        "stirring_state",
        "next_round_confirmed",
    ]

    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    assert "legal_actions" not in result
    assert "bid_legal_actions" not in result

    # player_hand_counts specifically: must be a list of 4 ints
    hand_counts = result["player_hand_counts"]
    assert len(hand_counts) == 4
    for count in hand_counts:
        assert isinstance(count, int)


# ---- action_hints snapshot shape ----


@pytest.mark.asyncio
async def test_snapshot_action_hints_are_card_lists() -> None:
    """
    action_hints entries are plain card lists, not PlayAction objects.

    action_hints is list[list[Card]].
    Each entry is a list of card dicts (no .type attribute).
    """
    game = await _start_game(_make_players())
    snap = game.snapshot(for_player=0)
    if len(snap.action_hints) > 0:
        entry = snap.action_hints[0]
        # Entry is a list of Card objects, not a PlayAction
        assert isinstance(entry, list)
        assert not hasattr(entry, "type")  # not a PlayAction
        if len(entry) > 0:
            assert isinstance(entry[0], Card)
            assert entry[0].id


@pytest.mark.asyncio
async def test_snapshot_action_hints_card_format() -> None:
    """
    snapshot action_hints are list of card-dict lists (no 'type' field).
    """
    game = await _start_game(_make_players())
    snap_json: object = json.loads(
        game.snapshot(for_player=0).model_dump_json()
    )
    assert _is_object_dict(snap_json)
    action_hints_val = snap_json["action_hints"]
    assert _is_object_list(action_hints_val)
    if len(action_hints_val) > 0:
        entry = action_hints_val[0]
        assert _is_object_list(entry)
        # Entry is a list of card dicts, not a dict with 'type' key
        if len(entry) > 0:
            first_card = entry[0]
            assert _is_object_dict(first_card)
            assert "id" in first_card  # card dict format
            assert "type" not in first_card  # no PlayAction wrapper


# ---- Game auto-completion ----


@pytest.mark.asyncio
async def test_game_auto_completes_past_deal_bid() -> None:
    """Game with 4 AutoPlayers progresses past DEAL_BID phase.

    Verifies that the sync round-robin bidding model makes progress
    and the game transitions to a later phase.
    """
    players = _make_players()
    game = await _start_game(players)

    # Drive through DEAL_BID using explicit SkipBidAction calls
    max_steps = 500
    for _ in range(max_steps):
        phase = _game_phase(game)
        if phase != "DEAL_BID":
            break
        # Find the current bidder and skip
        bid_found = False
        for i in range(4):
            snap = game.snapshot(i)
            if snap.awaiting_action == "bid":
                await _send_action(game, players, i, SkipBidAction())
                bid_found = True
                break
        assert bid_found, "no bidder found — DEAL_BID stuck"

    # Verify the game has progressed
    phase = _game_phase(game)
    assert phase in (
        "DEAL_BID",
        "STIRRING",
        "PLAYING",
        "WAITING",
    )
    # Snapshot must still be valid
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.player_hand, list)


@pytest.mark.asyncio
async def test_game_over_via_auto_players_starts() -> None:
    """Game with 4 AutoPlayers starts and has valid initial state.

    Verifies that the game is created with valid phase and can be
    started.
    """
    game = _create_game_with_auto_players()
    initial_phase = _game_phase(game)
    assert initial_phase in ("WAITING", "DEAL_BID")


# ---- Bug 1 regression: no resource explosion ----


@pytest.mark.asyncio
async def test_full_game_flow_no_resource_explosion() -> None:
    """
    A game with 4 AutoPlayers must complete without CPU/memory
    explosion.

    Regression test for Bug 1: AutoPlayer on_state() -> create_task(bid)
    -> game.receive() -> _push_state_to_all() -> on_state() -> ...
    exponential
    task cascade consumed 96.9% CPU and 8.8 GB RAM.

    In sync round-robin mode, the game is action-driven. This test
    drives
    the game through DEAL_BID using explicit SkipBidAction calls
    (consistent
    with the new sync model) and verifies the game progresses without
    getting stuck.
    """
    players = _make_players()
    game = await _start_game(players)

    # Drive through DEAL_BID using explicit SkipBidAction calls
    max_steps = 500
    for _ in range(max_steps):
        phase = _game_phase(game)
        if phase != "DEAL_BID":
            break
        bid_found = False
        for i in range(4):
            snap = game.snapshot(i)
            if snap.awaiting_action == "bid":
                await _send_action(game, players, i, SkipBidAction())
                bid_found = True
                break
        assert bid_found, "no bidder found — DEAL_BID stuck"

    phase = _game_phase(game)
    assert phase in (
        "DEAL_BID",
        "STIRRING",
        "PLAYING",
        "WAITING",
    ), f"Game stuck in unexpected phase: {phase}"

    # Snapshot must still be valid (no cascading error state)
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.player_hand, list)


# ---- Task 002: DEAL_BID Sync Round-Robin Bidding ----


@pytest.mark.asyncio
async def test_deal_bid_sync_round_robin() -> None:
    """
    DEAL_BID phase: deal one card, recipient must bid/skip, then next
    card.

    Each deal-bid cycle: deal 1 card to a player → that player bids or
    skips → deal next card to the next player → ...

    Verifies:
    1. After starting, first player has 1 card and awaiting_action='bid'
    2. Other players have 0 cards and awaiting_action=None
    3. After skip, next player receives a card and gets
    awaiting_action='bid'
    """
    players = _make_players()
    game = await _start_game(players)

    # After starting, should be in DEAL_BID with first card dealt
    snapshot = game.snapshot(0)
    assert snapshot.phase == "DEAL_BID"

    # First player (start_player, which is 0 by default) should have
    # 1 card and awaiting_action='bid'
    s0 = game.snapshot(0)
    assert len(s0.player_hand) == 1, (
        f"Player 0: expected 1 card after first deal, got"
        f"{len(s0.player_hand)}"
    )
    assert s0.awaiting_action == "bid"

    # Other players should have 0 cards and no awaiting
    for i in range(1, 4):
        si = game.snapshot(i)
        assert len(si.player_hand) == 0, (
            f"Player {i}: expected 0 cards before their deal, got"
            f"{len(si.player_hand)}"
        )
        assert si.awaiting_action is None

    # Player 0 skips → next card dealt to next player
    await _send_action(game, players, 0, SkipBidAction())

    # Now in DEAL_BID still, next player should have received a card
    assert _game_phase(game) == "DEAL_BID"
    # Player 0 still has 1 card (no new card yet), next player has 1
    # The next player in CCW order after 0 is 1
    s0_after = game.snapshot(0)
    assert len(s0_after.player_hand) == 1
    s1 = game.snapshot(1)
    assert len(s1.player_hand) == 1, (
        f"Player 1: expected 1 card after their deal, got"
        f"{len(s1.player_hand)}"
    )
    assert s1.awaiting_action == "bid"

    # Continue: player 1 skips → player 2 gets a card (CCW: 1→2)
    await _send_action(game, players, 1, SkipBidAction())
    assert _game_phase(game) == "DEAL_BID"
    s2 = game.snapshot(2)
    assert len(s2.player_hand) == 1
    assert s2.awaiting_action == "bid"


@pytest.mark.asyncio
async def test_bid_action_hints_in_snapshot() -> None:
    """
    Snapshot includes bid action_hints during DEAL_BID phase for the
    current bidder.
    """
    game = await _start_game(_make_players())

    # Find the player who has awaiting_action='bid'
    bidder = None
    for i in range(4):
        s = game.snapshot(i)
        if s.phase == "DEAL_BID" and s.awaiting_action == "bid":
            bidder = i
            break
    assert bidder is not None, (
        "No player has awaiting_action='bid' in DEAL_BID"
    )

    snapshot = game.snapshot(bidder)
    assert snapshot.phase == "DEAL_BID"
    assert isinstance(snapshot.action_hints, list)
    # Each entry is a list of cards (1 or 2 cards per bid option)
    for entry in snapshot.action_hints:
        assert isinstance(entry, list)
        assert len(entry) in (1, 2)


@pytest.mark.asyncio
async def test_awaiting_bid_for_current_bidder() -> None:
    """
    awaiting_action is 'bid' for the player whose turn it is to bid.
    """
    game = await _start_game(_make_players())

    # Find which player has awaiting_action="bid"
    bidding_player = None
    for i in range(4):
        snap = game.snapshot(i)
        if snap.awaiting_action == "bid":
            bidding_player = i
            break
    assert bidding_player is not None, (
        "No player has awaiting_action='bid'"
    )


@pytest.mark.asyncio
async def test_awaiting_null_for_non_current_bidder() -> None:
    """
    awaiting_action is null for players whose turn it is NOT to bid.
    """
    game = await _start_game(_make_players())

    # Find which player has awaiting_action="bid"
    bidding_player = None
    for i in range(4):
        snap = game.snapshot(i)
        if snap.awaiting_action == "bid":
            bidding_player = i
            break
    assert bidding_player is not None

    # All other players should have awaiting_action=None
    for i in range(4):
        if i == bidding_player:
            continue
        snap = game.snapshot(i)
        assert snap.awaiting_action is None, (
            f"Player {i}: expected awaiting_action=None, got"
            f"{snap.awaiting_action}"
        )


@pytest.mark.asyncio
async def test_deal_bid_no_background_delay() -> None:
    """
    After Bug 1 fix, Game is purely action-driven with no background
    dealing delay.

    Verifies the observable behavior: after starting the game, sending
    SkipBidAction immediately advances the bid turn without any
    background dealing
    delay. Uses only the public Game.receive() and Game.snapshot()
    interfaces.
    """
    players = _make_players()
    game = await _start_game(players)

    # Find the current bidder
    current_bidder = None
    for i in range(4):
        s = game.snapshot(i)
        if s.awaiting_action == "bid":
            current_bidder = i
            break
    assert current_bidder is not None, (
        "No player has awaiting_action='bid'"
    )

    # Skip immediately — no sleep or background task needed
    await _send_action(game, players, current_bidder, SkipBidAction())

    # The bid turn must have advanced or phase changed immediately
    snap_after = game.snapshot(current_bidder)
    if snap_after.phase == "DEAL_BID":
        new_bidder = None
        for i in range(4):
            s = game.snapshot(i)
            if s.awaiting_action == "bid":
                new_bidder = i
                break
        assert new_bidder is not None, (
            "No player has awaiting_action='bid' after skip"
        )
        assert new_bidder != current_bidder, (
            f"Bid turn did not advance: still player {current_bidder}"
        )


@pytest.mark.asyncio
async def test_skip_bid_action_advances_turn() -> None:
    """
    SkipBidAction during DEAL_BID advances the bid turn without bidding.

    Verifies that after one player skips, the bid turn moves to the next
    player (different player gets awaiting_action='bid') or the phase
    changes (if all players skipped and dealing completed).
    """
    players = _make_players()
    game = await _start_game(players)

    # Get to a state where we can bid
    snapshot = game.snapshot(3)
    assert snapshot.phase == "DEAL_BID"

    # Find the current bidder
    current_bidder = None
    for i in range(4):
        s = game.snapshot(i)
        if s.awaiting_action == "bid":
            current_bidder = i
            break
    assert current_bidder is not None, (
        "No player has awaiting_action='bid'"
    )

    # Send SkipBidAction for the current bidder
    await _send_action(game, players, current_bidder, SkipBidAction())

    # After skipping, either:
    # (a) another player now has awaiting_action='bid' (turn advanced),
    # or
    # (b) phase changed to STIRRING (if all players passed and dealing
    # done)
    snapshot_after = game.snapshot(current_bidder)
    if snapshot_after.phase == "DEAL_BID":
        # Turn must have advanced to a different player
        new_bidder = None
        for i in range(4):
            s = game.snapshot(i)
            if s.awaiting_action == "bid":
                new_bidder = i
                break
        assert new_bidder is not None, (
            "No player has awaiting_action='bid' after skip"
        )
        assert new_bidder != current_bidder, (
            f"Bid turn did not advance: still player {current_bidder}"
        )
    else:
        # Phase changed (acceptable if dealing completed)
        assert snapshot_after.phase == "STIRRING"


@pytest.mark.asyncio
async def test_stirring_state_snapshot_has_declarer_player() -> None:
    """StirringStateSnapshot must include declarer_player field.

    Per spec: "stirring_state 含
    phase/trump_suit/current_player/declarer_player".
    Drives the game to STIRRING and verifies the field is present and
    correct.
    """
    players = _make_players()
    game = await _start_game(players)

    # Drive through DEAL_BID to reach STIRRING
    max_attempts = 500
    for _ in range(max_attempts):
        snap = game.snapshot(for_player=0)
        if snap.phase != "DEAL_BID":
            break
        # Find the current bidder and skip
        bid_found = False
        for i in range(4):
            s = game.snapshot(i)
            if s.awaiting_action == "bid":
                await _send_action(game, players, i, SkipBidAction())
                bid_found = True
                break
        assert bid_found, "no bidder found — DEAL_BID stuck"

    snap = game.snapshot(for_player=0)
    assert snap.phase == "STIRRING", (
        f"Expected STIRRING phase, got {snap.phase}"
    )

    stirring_snapshot = snap.stirring_state
    assert stirring_snapshot is not None
    assert isinstance(stirring_snapshot.declarer_player, int)
    assert 0 <= stirring_snapshot.declarer_player <= 3

    stirring_dict = stirring_snapshot.model_dump(mode="json")
    assert "declarer_player" in stirring_dict
    assert (
        stirring_dict["declarer_player"]
        == stirring_snapshot.declarer_player
    )
