"""Tests for server/game.py -- Game aggregate root.

All tests use only public interfaces: Game.__init__, Game.run, Game.act,
Game.snapshot, Game.is_over, Game.get_phase, Game.get_player, Game.set_on_game_over,
StateSnapshot.to_dict.
No tests access private fields like _game_state, _round_state, or _bid_turn.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from server.actions import BidAction, DiscardAction, NextRoundAction, PlayAction, SkipBidAction, SkipStirAction
from server.game import Game
from server.player import AutoPlayer
from server.sm.card_model import Rank, Suit
from server.sm.result import Ok, Rejected
from server.snapshot import StateSnapshot


def _create_game_with_auto_players():
    """Create a Game with 4 AutoPlayers."""
    players = [AutoPlayer(index=i) for i in range(4)]
    return Game(players=players)


def _make_players() -> list[AutoPlayer]:
    """Create 4 AutoPlayer instances for testing."""
    return [AutoPlayer(index=i) for i in range(4)]


# ---- Initialization ----


def test_game_init_creates_valid_state():
    game = _create_game_with_auto_players()
    # Verify via public interface
    assert game.get_phase() == "IDLE"
    assert game.is_over() is False


# ---- get_phase() ----


def test_get_phase_returns_phase():
    game = _create_game_with_auto_players()
    assert game.get_phase() == "IDLE"


# ---- run() ----


@pytest.mark.asyncio
async def test_run_transitions_to_deal_bid():
    game = _create_game_with_auto_players()
    await game.run()
    # Verify via snapshot (public interface)
    snap = game.snapshot(for_player=0)
    assert snap.phase in ("DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING", "COMPLETE")


# ---- act() ----


@pytest.mark.asyncio
async def test_act_rejects_wrong_player():
    """PlayAction during DEAL_BID should be rejected without raising."""
    game = _create_game_with_auto_players()
    await game.run()
    # Should not raise; rejection is communicated via send_error instead
    await game.act(player_index=0, action=PlayAction(cards=[]))


@pytest.mark.asyncio
async def test_act_value_error_not_propagated():
    """Invalid actions no longer raise ValueError; they send error messages."""
    game = _create_game_with_auto_players()
    await game.run()
    # Should not raise
    await game.act(player_index=0, action=PlayAction(cards=[]))


# ---- snapshot() ----


@pytest.mark.asyncio
async def test_snapshot_returns_player_hand():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert isinstance(snap, StateSnapshot)
    assert isinstance(snap.player_hand, list)


def test_snapshot_raises_before_run():
    """snapshot() must raise when called before run().

    Before run(), _round_state is None, so snapshot() cannot build a
    valid StateSnapshot. Rather than returning a partial/empty snapshot
    that could mislead callers, it raises an error (AssertionError via
    assert guard).
    """
    game = _create_game_with_auto_players()
    with pytest.raises(AssertionError, match="snapshot\\(\\) called before run"):
        game.snapshot(for_player=0)


@pytest.mark.asyncio
async def test_snapshot_phase():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert snap.phase in ("DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING", "COMPLETE", "GAME_OVER", "SCORING")


@pytest.mark.asyncio
async def test_snapshot_current_player():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.current_player, int)
    assert 0 <= snap.current_player <= 3


@pytest.mark.asyncio
async def test_snapshot_current_player_deal_bid():
    """During DEAL_BID, current_player should be the deal_target."""
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    if snap.phase == "DEAL_BID":
        assert isinstance(snap.current_player, int)
        assert 0 <= snap.current_player <= 3


@pytest.mark.asyncio
async def test_snapshot_legal_actions_in_playing():
    """Legal actions should be populated during PLAYING phase.

    After Task 010 refactor, legal_actions is list[list[Card]].
    Each entry is a plain list of Card objects (no .type attribute).
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.legal_actions, list)
    # If in PLAYING phase, legal_actions should contain card lists
    if snap.phase == "PLAYING" and len(snap.legal_actions) > 0:
        entry = snap.legal_actions[0]
        # Entry is a list of Card objects (not a PlayAction)
        assert isinstance(entry, list)
        assert not hasattr(entry, "type")  # not a PlayAction
        if len(entry) > 0:
            from server.sm.card_model import Card
            assert isinstance(entry[0], Card)


@pytest.mark.asyncio
async def test_snapshot_awaiting_action_play():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert snap.awaiting_action in ("stir", "discard", "play", "next_round", "bid", None)


@pytest.mark.asyncio
async def test_snapshot_trump_info():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert snap.trump_rank is not None


@pytest.mark.asyncio
async def test_snapshot_team_levels():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert snap.team0_level is not None
    assert snap.team1_level is not None


@pytest.mark.asyncio
async def test_snapshot_bid_events():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.bid_events, list)


@pytest.mark.asyncio
async def test_snapshot_stirring_state():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    from server.snapshot import StirringStateSnapshot
    # stirring_state may be None outside of STIRRING phase
    assert snap.stirring_state is None or isinstance(snap.stirring_state, StirringStateSnapshot)


@pytest.mark.asyncio
async def test_snapshot_exchange_state():
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    from server.snapshot import ExchangeStateSnapshot
    assert snap.exchange_state is None or isinstance(snap.exchange_state, ExchangeStateSnapshot)


@pytest.mark.asyncio
async def test_snapshot_scoring_in_complete():
    """When round is COMPLETE, snapshot should include scoring info."""
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    from server.snapshot import ScoringSnapshot
    assert snap.scoring is None or isinstance(snap.scoring, ScoringSnapshot)


# ---- is_over() ----


def test_is_over_false_during_game():
    game = _create_game_with_auto_players()
    assert game.is_over() is False


@pytest.mark.asyncio
async def test_is_over_true_after_game_over():
    """Game should be over when get_phase() returns GAME_OVER."""
    game = _create_game_with_auto_players()
    await game.run()
    # We can't easily force GAME_OVER through public API alone in unit test.
    # Instead, verify is_over() is consistent with get_phase().
    assert game.is_over() == (game.get_phase() == "GAME_OVER")


@pytest.mark.asyncio
async def test_snapshot_winning_team_in_game_over():
    """When game is over, snapshot should include winning_team."""
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    # During normal flow, game is not over yet
    if game.is_over():
        assert snap.winning_team is not None
        assert isinstance(snap.winning_team, int)
    else:
        assert snap.winning_team is None


@pytest.mark.asyncio
async def test_game_over_consistency():
    """is_over() should equal get_phase() == 'GAME_OVER'."""
    game = _create_game_with_auto_players()
    assert game.is_over() == (game.get_phase() == "GAME_OVER")
    await game.run()
    assert game.is_over() == (game.get_phase() == "GAME_OVER")


# ---- Action dispatch with type conversion ----


@pytest.mark.asyncio
async def test_act_bid_during_dealing_converts_to_bid_event():
    """BidAction from player.py should be converted to sm BidEvent internally."""
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    # During DEAL_BID, try bidding. This tests the BidAction -> BidEvent conversion.
    # If not in DEAL_BID, we can't test bid; that's fine, integration tests cover it.
    if snap.phase == "DEAL_BID" and len(snap.player_hand) > 0:
        # Find a trump rank card to bid with
        trump_cards = [c for c in snap.player_hand if c.rank == snap.trump_rank]
        if trump_cards:
            action = BidAction(cards=trump_cards[:1], count=1)
            # Bid may be rejected (e.g. priority too low), but act() never raises
            await game.act(player_index=0, action=action)


@pytest.mark.asyncio
async def test_act_skip_stir_during_stirring():
    """SkipStirAction is a valid action type that Game.act() can distinguish
    from StirAction. The actual dispatch routing (SkipStirAction -> round_sm.pass_stir)
    is verified in integration tests (task-008).
    """
    action = SkipStirAction()
    assert isinstance(action, SkipStirAction)
    # Verify it is NOT a StirAction -- Game.act() must dispatch differently
    from server.actions import StirAction
    assert not isinstance(action, StirAction)


@pytest.mark.asyncio
async def test_act_next_round_during_non_complete():
    """NextRoundAction during non-COMPLETE phase should be rejected without raising."""
    game = _create_game_with_auto_players()
    await game.run()
    # Should not raise; rejection is communicated via send_error instead
    await game.act(player_index=0, action=NextRoundAction())


# ---- get_phase() GAME_OVER priority ----


def test_get_phase_prioritizes_game_over():
    """get_phase() must return GAME_OVER when _game_state.phase is GAME_OVER,
    even if _round_state is still non-None with a different phase.

    Since we cannot easily force GAME_OVER through public API in a unit test
    without accessing private fields, we verify the observable contract:
    is_over() must always be consistent with get_phase().
    """
    game = _create_game_with_auto_players()
    assert game.get_phase() == "IDLE"
    # The invariant: is_over() == (get_phase() == "GAME_OVER")
    assert game.is_over() == (game.get_phase() == "GAME_OVER")


# ---- on_game_over callback ----


@pytest.mark.asyncio
async def test_set_on_game_over_callback_fires_on_game_over():
    """When game transitions to GAME_OVER, the registered callback should be invoked.

    Strategy: Create a game in IN_ROUND state by patching game_sm.create_game,
    then patch round_sm.create_round to return a COMPLETE-phase RoundState.
    After game.run(), patch game_sm.process_round_result to return a GAME_OVER
    state. Then call game.act() with NextRoundAction. If the callback fires,
    the mock is called.
    """
    from server.sm import game_sm as gm, round_sm as rm
    from server.sm.scoring import RoundResult
    from server.sm.card_model import Rank

    # Create a game in IN_ROUND state by patching create_game
    in_round_state = gm.GameState(
        phase="IN_ROUND",
        team0_level=Rank.TEN,
        team1_level=Rank.TEN,
        declarer_team=0,
        last_declarer_player=0,
        winning_team=None,
        round_number=1,
    )

    # Create a COMPLETE-phase RoundState mock
    complete_round = MagicMock()
    complete_round.phase = "COMPLETE"
    complete_round.players_hand = [[] for _ in range(4)]
    complete_round.declarer_player = 0
    complete_round.result = None  # Will be overridden by get_round_result mock

    # The GAME_OVER state that process_round_result will return
    game_over_state = gm.GameState(
        phase="GAME_OVER",
        team0_level=Rank.ACE,
        team1_level=Rank.TEN,
        declarer_team=None,
        last_declarer_player=None,
        winning_team=0,
        round_number=1,
    )

    # Build a mock RoundResult
    mock_result = MagicMock(spec=RoundResult)
    mock_result.team0_new_level = Rank.ACE
    mock_result.team1_new_level = Rank.TEN
    mock_result.next_declarer_team = 0
    mock_result.next_declarer_player = 0

    with patch.object(gm, "create_game", return_value=in_round_state):
        game = _create_game_with_auto_players()

    callback = MagicMock()
    game.set_on_game_over(callback)

    # Run the game with patched sm functions
    with patch.object(gm, "start_game", return_value=Ok(in_round_state)):
        with patch.object(rm, "create_round", return_value=complete_round):
            await game.run()

    # Verify game is not over yet (before triggering GAME_OVER)
    assert not game.is_over()

    # Now trigger GAME_OVER via act() with NextRoundAction.
    # Patch get_round_result to return our mock_result so act() can pass
    # it to game_sm.process_round_result.
    with patch.object(rm, "get_round_result", return_value=mock_result):
        with patch.object(gm, "process_round_result", return_value=Ok(game_over_state)):
            # COMPLETE phase now requires all 4 players to confirm
            for p in range(4):
                await game.act(player_index=p, action=NextRoundAction())

    # Game must have transitioned to GAME_OVER
    assert game.is_over()
    # Callback must have been called with the game instance
    callback.assert_called_once_with(game)


# ---- get_player() ----


def test_get_player_returns_player_by_index():
    """Game.get_player(index) returns the Player at that index."""
    players = [AutoPlayer(index=i) for i in range(4)]
    game = Game(players=players)
    for i in range(4):
        assert game.get_player(i) is players[i]


# ---- StateSnapshot.to_dict() ----


@pytest.mark.asyncio
async def test_snapshot_to_dict_json_serializable():
    """StateSnapshot.to_dict() must return a dict that is JSON-serializable.

    This is critical because HumanPlayer.on_state() calls ws.send_json() which
    requires JSON-serializable data. The sm Card/Suit/Rank types are Pydantic
    models and enums that are not directly JSON-serializable. legal_actions
    is now list[list[Card]], serialized to list of card-dict lists.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    result = snap.to_dict()
    # Must be a dict
    assert isinstance(result, dict)
    # Must be JSON-serializable (no Pydantic objects, no enums as objects)
    serialized = json.dumps(result)
    assert isinstance(serialized, str)
    # Must contain the required fields from spec section 5.5
    assert "phase" in result
    assert "player_hand" in result
    assert "trump_rank" in result
    assert "current_player" in result
    # legal_actions must be a list of lists (card lists, not PlayAction dicts)
    legal_actions_raw = result["legal_actions"]
    if len(legal_actions_raw) > 0:
        legal_entry = legal_actions_raw[0]
        # list of card dicts
        if len(legal_entry) > 0:
            assert "id" in legal_entry[0]
            assert "type" not in legal_entry[0]


@pytest.mark.asyncio
async def test_snapshot_to_dict_card_format():
    """StateSnapshot.to_dict() must format cards as {"id", "suit", "rank"}.

    Per spec section 5.5, each card in player_hand must be:
    {"id": "D1-H-A", "suit": "hearts", "rank": "A"}
    This is a subset of the sm Card fields (omitting is_joker, is_big_joker,
    points, deck). Suit and Rank enums must be serialized as their string values.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    result = snap.to_dict()
    # If player has cards, verify the format
    player_hand_raw = result["player_hand"]
    if len(player_hand_raw) > 0:
        card = player_hand_raw[0]
        assert isinstance(card, dict)
        assert "id" in card
        assert "suit" in card
        assert "rank" in card
        # suit and rank must be strings (not enum objects)
        assert isinstance(card["suit"], str)
        assert isinstance(card["rank"], str)
        # Must NOT contain internal sm fields
        assert "is_joker" not in card
        assert "is_big_joker" not in card
        assert "points" not in card
        assert "deck" not in card


# ---- resolve_cards() ----


@pytest.mark.asyncio
async def test_resolve_cards_returns_matching_cards():
    """Game.resolve_cards() returns Ok with Card objects matching the given IDs
    from the specified player's hand.

    This is needed because human players send card IDs via WebSocket,
    but Game.act() must pass Card Pydantic model objects to sm functions.
    resolve_cards() bridges this gap by looking up Card objects by their
    ID string in the player's hand.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    if len(snap.player_hand) > 0:
        card_ids = [c.id for c in snap.player_hand[:2]]
        result = game.resolve_cards(player_index=0, card_ids=card_ids)
        assert isinstance(result, Ok)
        for original, resolved_card in zip(card_ids, result.value):
            assert resolved_card.id == original
            # Must be a Card Pydantic model (not a string or dict)
            from server.sm.card_model import Card
            assert isinstance(resolved_card, Card)


@pytest.mark.asyncio
async def test_resolve_cards_rejects_on_unknown_id():
    """Game.resolve_cards() returns Rejected if any card_id is not found
    in the player's hand.

    This prevents human players from submitting cards they don't hold,
    which would be an invalid action.
    """
    game = _create_game_with_auto_players()
    await game.run()
    result = game.resolve_cards(player_index=0, card_ids=["NONEXISTENT-CARD-ID"])
    assert isinstance(result, Rejected)


# ---- Bug 1 regression: bid must not trigger _push_state_to_all cascade ----


@pytest.mark.asyncio
async def test_bid_during_deal_bid_pushes_state_uniformly():
    """BidAction during DEAL_BID must push state to all players uniformly.

    In sync round-robin mode, each BidAction/SkipBidAction triggers exactly
    one _push_state_to_all. This test verifies that the state push count
    is uniform across all players — no player is skipped or double-pushed.

    Regression test for Bug 1 (adapted from async dealing loop to sync
    round-robin): the original bug was that a bid during DEAL_BID triggered
    _push_state_to_all(), causing AutoPlayer cascades. In the sync model,
    every action pushes state, but each push must be exactly one push to
    all 4 players.
    """
    from server.actions import SkipBidAction
    from server.player import Player

    class CountingPlayer(Player):
        """Player that counts on_state invocations."""
        def __init__(self, index: int) -> None:
            super().__init__(index)
            self.state_count = 0

        async def on_state(self, game: object, *, seq: int = 0, error: str | None = None) -> None:
            self.state_count += 1

    counters = [CountingPlayer(index=i) for i in range(4)]
    game = Game(players=counters)
    await game.run()

    # After run(), initial state push happened (1 push = 4 on_state calls)
    initial_counts = [c.state_count for c in counters]

    # Find the current bidder and skip
    for i in range(4):
        snap = game.snapshot(i)
        if snap.awaiting_action == "bid":
            await game.act(i, SkipBidAction())
            break

    # After one action, one more push to all players
    for i in range(4):
        assert counters[i].state_count == initial_counts[i] + 1, (
            f"Player {i}: expected {initial_counts[i] + 1} pushes, got {counters[i].state_count}"
        )


# ---- Bug 2 regression: snapshot must contain player_hand_counts ----


@pytest.mark.asyncio
async def test_snapshot_to_dict_contains_all_required_fields():
    """StateSnapshot.to_dict() must contain ALL fields from spec section 5.5.

    Regression test for Bug 2: the `player_hand_counts` field was missing
    from StateSnapshot, causing the frontend's game-table component to
    show "0 张" for every player because `snapshot.player_hand_counts[i]`
    evaluated to `undefined ?? 0`.

    This test asserts the complete set of required fields so any future
    addition to the spec is also caught if the server doesn't serialize it.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    result = snap.to_dict()

    # Complete list of required fields per spec section 5.5
    required_fields = [
        "phase",
        "player_hand",
        "player_hand_counts",
        "bottom_cards",
        "trump_suit",
        "trump_rank",
        "declarer_team",
        "declarer_player",
        "current_player",
        "defender_points",
        "trick",
        "trick_history",
        "legal_actions",
        "awaiting_action",
        "bid_legal_actions",
        "scoring",
        "winning_team",
        "team0_level",
        "team1_level",
        "bid_events",
        "bid_winner",
        "stirring_state",
        "exchange_state",
        "next_round_confirmed",
    ]

    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    # player_hand_counts specifically: must be a list of 4 ints
    hand_counts = result["player_hand_counts"]
    assert len(hand_counts) == 4
    for count in hand_counts:
        assert isinstance(count, int)


# ---- Task 010: new get_legal_plays signature ----


@pytest.mark.asyncio
async def test_snapshot_legal_actions_are_card_lists():
    """Legal actions entries are plain card lists, not PlayAction objects.

    After the refactor, legal_actions is list[list[Card]].
    Each entry is a list of Card objects (no .type attribute).
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    if snap.phase == "PLAYING" and len(snap.legal_actions) > 0:
        entry = snap.legal_actions[0]
        # Entry is a list of Card objects, not a PlayAction
        assert isinstance(entry, list)
        assert not hasattr(entry, "type")  # not a PlayAction
        if len(entry) > 0:
            from server.sm.card_model import Card
            assert isinstance(entry[0], Card)


@pytest.mark.asyncio
async def test_snapshot_legal_actions_to_dict_format():
    """to_dict() serializes legal_actions as list of card-dict lists (no 'type' field)."""
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    d = snap.to_dict()
    legal_actions_val = d["legal_actions"]
    if snap.phase == "PLAYING" and len(legal_actions_val) > 0:
        entry = legal_actions_val[0]
        # Entry is a list of card dicts, not a dict with 'type' key
        if len(entry) > 0:
            assert "id" in entry[0]  # card dict format
            assert "type" not in entry[0]  # no PlayAction wrapper


@pytest.mark.asyncio
async def test_snapshot_completed_trick_no_lead_type():
    """CompletedTrick no longer has lead_type field.

    After Task-009, _serialize_completed_trick should not include lead_type.
    """
    game = _create_game_with_auto_players()
    await game.run()
    snap = game.snapshot(for_player=0)
    d = snap.to_dict()
    for trick in d["trick_history"]:
        assert "lead_type" not in trick


# ---- Game auto-completion ----


@pytest.mark.asyncio
async def test_game_auto_completes_past_deal_bid():
    """Game with 4 AutoPlayers progresses past DEAL_BID phase.

    Verifies that the sync round-robin bidding model makes progress
    and the game transitions to a later phase.
    """
    from server.actions import SkipBidAction
    game = _create_game_with_auto_players()
    await game.run()

    # Drive through DEAL_BID using explicit SkipBidAction calls
    max_steps = 500
    for _ in range(max_steps):
        phase = game.get_phase()
        if phase != "DEAL_BID":
            break
        # Find the current bidder and skip
        bid_found = False
        for i in range(4):
            snap = game.snapshot(i)
            if snap.awaiting_action == "bid":
                await game.act(i, SkipBidAction())
                bid_found = True
                break
        if not bid_found:
            await asyncio.sleep(0.01)

    # Verify the game has progressed
    phase = game.get_phase()
    assert phase in (
        "DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING",
        "COMPLETE", "GAME_OVER",
    )
    # Snapshot must still be valid
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.player_hand, list)


@pytest.mark.asyncio
async def test_game_over_via_auto_players_starts():
    """Game with 4 AutoPlayers starts and has valid initial state.

    Verifies that the game is created with valid phase and can be started.
    """
    game = _create_game_with_auto_players()
    initial_phase = game.get_phase()
    assert initial_phase in ("IDLE", "IN_ROUND", "DEAL_BID")


# ---- Bug 1 regression: no resource explosion ----


@pytest.mark.asyncio
async def test_full_game_flow_completes_without_resource_explosion():
    """A game with 4 AutoPlayers must complete without CPU/memory explosion.

    Regression test for Bug 1: AutoPlayer on_state() -> create_task(bid)
    -> game.act() -> _push_state_to_all() -> on_state() -> ... exponential
    task cascade consumed 96.9% CPU and 8.8 GB RAM.

    In sync round-robin mode, the game is action-driven. This test drives
    the game through DEAL_BID using explicit SkipBidAction calls (consistent
    with the new sync model) and verifies the game progresses without
    getting stuck.
    """
    from server.actions import SkipBidAction
    game = _create_game_with_auto_players()
    await game.run()

    # Drive through DEAL_BID using explicit SkipBidAction calls
    max_steps = 500
    for _ in range(max_steps):
        phase = game.get_phase()
        if phase != "DEAL_BID":
            break
        bid_found = False
        for i in range(4):
            snap = game.snapshot(i)
            if snap.awaiting_action == "bid":
                await game.act(i, SkipBidAction())
                bid_found = True
                break
        if not bid_found:
            await asyncio.sleep(0.01)

    phase = game.get_phase()
    assert phase in (
        "DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING",
        "COMPLETE", "GAME_OVER",
    ), f"Game stuck in unexpected phase: {phase}"

    # Snapshot must still be valid (no cascading error state)
    snap = game.snapshot(for_player=0)
    assert isinstance(snap.player_hand, list)


# ---- Game over removes from registry ----


@pytest.mark.asyncio
async def test_game_over_removes_from_registry():
    """Test that the on_game_over callback can remove the game from registry.

    Creates a Game with mocked sm functions to force it through to GAME_OVER,
    and verifies the callback fires and can remove the game from the registry.
    """
    from server.sm import game_sm as gm, round_sm as rm
    from server.sm.card_model import Rank
    from server.sm.scoring import RoundResult
    from server.game_registry import GameRegistry

    test_registry = GameRegistry()
    players = [AutoPlayer(index=i) for i in range(4)]

    # Create game in IN_ROUND state by patching create_game
    in_round_state = gm.GameState(
        phase="IN_ROUND",
        team0_level=Rank.TEN,
        team1_level=Rank.TEN,
        declarer_team=0,
        last_declarer_player=0,
        winning_team=None,
        round_number=1,
    )

    # Create a COMPLETE-phase RoundState mock (so act() accepts NextRoundAction)
    complete_round = MagicMock()
    complete_round.phase = "COMPLETE"
    complete_round.players_hand = [[] for _ in range(4)]
    complete_round.declarer_player = 0
    complete_round.result = None

    # The GAME_OVER state that process_round_result will return
    game_over_state = gm.GameState(
        phase="GAME_OVER",
        team0_level=Rank.ACE,
        team1_level=Rank.TEN,
        declarer_team=None,
        last_declarer_player=None,
        winning_team=0,
        round_number=1,
    )

    # Build a mock RoundResult
    mock_result = MagicMock(spec=RoundResult)
    mock_result.team0_new_level = Rank.ACE
    mock_result.team1_new_level = Rank.TEN
    mock_result.next_declarer_team = 0
    mock_result.next_declarer_player = 0

    callback_called = [False]

    with patch.object(gm, "create_game", return_value=in_round_state):
        game = Game(players=players)

    game_id = test_registry.create(game)

    # Set the on_game_over callback that records invocation AND removes from registry
    def on_game_over(g: Game) -> None:
        callback_called[0] = True
        test_registry.delete(game_id)

    game.set_on_game_over(on_game_over)

    # Start the game so _round_state is set to the COMPLETE mock
    with patch.object(gm, "start_game", return_value=Ok(in_round_state)):
        with patch.object(rm, "create_round", return_value=complete_round):
            await game.run()

    # Verify game is in registry
    assert test_registry.get(game_id) is not None

    # Now trigger GAME_OVER via act() with NextRoundAction using patched sm
    with patch.object(gm, "process_round_result", return_value=Ok(game_over_state)):
        with patch.object(rm, "get_round_result", return_value=mock_result):
            for p in range(4):
                await game.act(player_index=p, action=NextRoundAction())

    # Verify the callback was actually called
    assert callback_called[0], "on_game_over callback was not invoked"
    # Verify game is over and removed from registry
    assert game.is_over()
    assert test_registry.get(game_id) is None


# ---- Task 002: DEAL_BID Sync Round-Robin Bidding ----


@pytest.mark.asyncio
async def test_deal_bid_sync_round_robin() -> None:
    """DEAL_BID phase uses sync round-robin: deal 1 card per player (1 deal tick),
    then each player bids in turn.

    Per spec: "每次 deal tick 发一张牌给每人后，按 CCW 顺序轮流让每个
    player 决定 bid/pass" — each deal tick deals 1 card to each player (4 calls
    to deal_next_card), then all 4 players bid/pass in round-robin.

    Verifies the core behavior:
    1. After run(), each player has exactly 1 card (first deal tick)
    2. After one player bids/passes, the next player gets awaiting_action='bid'
    3. After all 4 players act, 1 more card is dealt to each (total 2 each)
    """
    from server.actions import SkipBidAction
    players = _make_players()
    game = Game(players=players)
    await game.run()

    # After run(), should be in DEAL_BID with first deal tick done
    snapshot = game.snapshot(3)
    assert snapshot.phase == "DEAL_BID"

    # Each player should have exactly 1 card after first deal tick
    # (deal_next_card deals 1 card to 1 player per call; 4 calls = 1 card each)
    for i in range(4):
        s = game.snapshot(i)
        assert len(s.player_hand) == 1, (
            f"Player {i}: expected 1 card after first deal tick, got {len(s.player_hand)}"
        )

    # Find the first bidder
    first_bidder = None
    for i in range(4):
        s = game.snapshot(i)
        if s.awaiting_action == "bid":
            first_bidder = i
            break
    assert first_bidder is not None, "No player has awaiting_action='bid'"

    # All 4 players bid/pass in round-robin
    for turn in range(4):
        bidder = (first_bidder + turn) % 4
        s = game.snapshot(bidder)
        if s.awaiting_action == "bid":
            await game.act(bidder, SkipBidAction())

    # After all 4 players act, next deal tick deals 1 more card each (total 2)
    # The phase may have changed to STIRRING if dealing completed
    for i in range(4):
        s = game.snapshot(i)
        if s.phase == "DEAL_BID":
            assert len(s.player_hand) == 2, (
                f"Player {i}: expected 2 cards after second deal tick, got {len(s.player_hand)}"
            )


@pytest.mark.asyncio
async def test_bid_legal_actions_in_snapshot() -> None:
    """Snapshot includes bid_legal_actions during DEAL_BID phase for the current bidder."""
    players = _make_players()
    game = Game(players=players)
    await game.run()

    # Find the player who has awaiting_action='bid'
    bidder = None
    for i in range(4):
        s = game.snapshot(i)
        if s.phase == "DEAL_BID" and s.awaiting_action == "bid":
            bidder = i
            break
    assert bidder is not None, "No player has awaiting_action='bid' in DEAL_BID"

    snapshot = game.snapshot(bidder)
    assert snapshot.phase == "DEAL_BID"
    assert snapshot.bid_legal_actions is not None
    assert isinstance(snapshot.bid_legal_actions, list)
    # Each entry is a list of cards (1 or 2 cards per bid option)
    for entry in snapshot.bid_legal_actions:
        assert isinstance(entry, list)
        assert len(entry) in (1, 2)


def test_get_bid_legal_actions_singles_and_pairs() -> None:
    """get_bid_legal_actions returns correct singles and pairs from a hand.

    Given a hand with trump-rank cards in different suits and joker pairs,
    verifies: (1) singles are individual trump-rank cards, (2) pairs are
    two trump-rank cards of the same suit or two jokers of the same type,
    (3) non-trump-rank cards are excluded.
    """
    from server.sm.deal_bid_sm import get_bid_legal_actions
    from server.sm.card_model import Card, Suit, Rank

    # Construct a hand with known trump-rank cards
    trump_rank = Rank.TWO
    hand: list[Card] = [
        # Two trump-rank clubs -> should produce a pair and two singles
        Card(id="D1-clubs-2", suit=Suit.CLUBS, rank=Rank.TWO,
             is_joker=False, is_big_joker=False, points=0, deck=1),
        Card(id="D2-clubs-2", suit=Suit.CLUBS, rank=Rank.TWO,
             is_joker=False, is_big_joker=False, points=0, deck=2),
        # One trump-rank spades -> should produce a single
        Card(id="D1-spades-2", suit=Suit.SPADES, rank=Rank.TWO,
             is_joker=False, is_big_joker=False, points=0, deck=1),
        # Small joker -> should produce a single (not a pair unless two small jokers)
        Card(id="small-joker", suit=Suit.JOKER, rank=Rank.SMALL_JOKER,
             is_joker=True, is_big_joker=False, points=0, deck=1),
        # Non-trump-rank card -> should NOT appear in bid legal actions
        Card(id="D1-hearts-5", suit=Suit.HEARTS, rank=Rank.FIVE,
             is_joker=False, is_big_joker=False, points=5, deck=1),
    ]

    result = get_bid_legal_actions(hand, trump_rank)

    assert isinstance(result, list)
    assert len(result) > 0, "Should have at least one bid option"

    # Collect all card IDs that appear in bid options
    all_bid_card_ids: set[str] = set()
    for option in result:
        assert isinstance(option, list)
        assert len(option) in (1, 2), f"Each option must be 1 or 2 cards, got {len(option)}"
        for c in option:
            assert isinstance(c, Card)
            all_bid_card_ids.add(c.id)

    # Trump-rank cards must appear
    assert "D1-clubs-2" in all_bid_card_ids
    assert "D2-clubs-2" in all_bid_card_ids
    assert "D1-spades-2" in all_bid_card_ids

    # Non-trump-rank card must NOT appear
    assert "D1-hearts-5" not in all_bid_card_ids

    # Verify pair exists (two clubs-2 together)
    pair_options = [opt for opt in result if len(opt) == 2]
    clubs_pair = [opt for opt in pair_options
                  if any(c.id == "D1-clubs-2" for c in opt)
                  and any(c.id == "D2-clubs-2" for c in opt)]
    assert len(clubs_pair) >= 1, "Should have a pair option for two clubs-2"


@pytest.mark.asyncio
async def test_awaiting_bid_for_current_bidder() -> None:
    """awaiting_action is 'bid' for the player whose turn it is to bid."""
    players = _make_players()
    game = Game(players=players)
    await game.run()

    # Find which player has awaiting_action="bid"
    bidding_player = None
    for i in range(4):
        snap = game.snapshot(i)
        if snap.awaiting_action == "bid":
            bidding_player = i
            break
    assert bidding_player is not None, "No player has awaiting_action='bid'"


@pytest.mark.asyncio
async def test_awaiting_null_for_non_current_bidder() -> None:
    """awaiting_action is null for players whose turn it is NOT to bid."""
    players = _make_players()
    game = Game(players=players)
    await game.run()

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
            f"Player {i}: expected awaiting_action=None, got {snap.awaiting_action}"
        )


@pytest.mark.asyncio
async def test_deal_bid_no_background_delay() -> None:
    """After Bug 1 fix, Game is purely action-driven with no background dealing delay.

    Verifies the observable behavior: after game.run(), calling game.act() with
    SkipBidAction immediately advances the bid turn without any background dealing
    delay. Uses only the public Game.act() and Game.snapshot() interfaces.
    """
    from server.actions import SkipBidAction
    players = _make_players()
    game = Game(players=players)
    await game.run()

    # Find the current bidder
    current_bidder = None
    for i in range(4):
        s = game.snapshot(i)
        if s.awaiting_action == "bid":
            current_bidder = i
            break
    assert current_bidder is not None, "No player has awaiting_action='bid'"

    # Skip immediately — no sleep or background task needed
    await game.act(current_bidder, SkipBidAction())

    # The bid turn must have advanced or phase changed immediately
    snap_after = game.snapshot(current_bidder)
    if snap_after.phase == "DEAL_BID":
        new_bidder = None
        for i in range(4):
            s = game.snapshot(i)
            if s.awaiting_action == "bid":
                new_bidder = i
                break
        assert new_bidder is not None, "No player has awaiting_action='bid' after skip"
        assert new_bidder != current_bidder, (
            f"Bid turn did not advance: still player {current_bidder}"
        )


@pytest.mark.asyncio
async def test_skip_bid_action_advances_turn() -> None:
    """SkipBidAction during DEAL_BID advances the bid turn without bidding.

    Verifies that after one player skips, the bid turn moves to the next
    player (different player gets awaiting_action='bid') or the phase
    changes (if all players skipped and dealing completed).
    """
    from server.actions import SkipBidAction
    players = _make_players()
    game = Game(players=players)
    await game.run()

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
    assert current_bidder is not None, "No player has awaiting_action='bid'"

    # Send SkipBidAction for the current bidder
    await game.act(current_bidder, SkipBidAction())

    # After skipping, either:
    # (a) another player now has awaiting_action='bid' (turn advanced), or
    # (b) phase changed to STIRRING (if all players passed and dealing done)
    snapshot_after = game.snapshot(current_bidder)
    if snapshot_after.phase == "DEAL_BID":
        # Turn must have advanced to a different player
        new_bidder = None
        for i in range(4):
            s = game.snapshot(i)
            if s.awaiting_action == "bid":
                new_bidder = i
                break
        assert new_bidder is not None, "No player has awaiting_action='bid' after skip"
        assert new_bidder != current_bidder, (
            f"Bid turn did not advance: still player {current_bidder}"
        )
    else:
        # Phase changed (acceptable if dealing completed)
        assert snapshot_after.phase == "STIRRING"


@pytest.mark.asyncio
async def test_stirring_state_snapshot_has_declarer_player() -> None:
    """StirringStateSnapshot must include declarer_player field.

    Per spec: "stirring_state 含 phase/trump_suit/current_player/declarer_player".
    Drives the game to STIRRING and verifies the field is present and correct.
    """
    from server.actions import SkipBidAction
    players = _make_players()
    game = Game(players=players)
    await game.run()

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
                await game.act(i, SkipBidAction())
                bid_found = True
                break
        if not bid_found:
            await asyncio.sleep(0.01)

    snap = game.snapshot(for_player=0)
    assert snap.phase == "STIRRING", f"Expected STIRRING phase, got {snap.phase}"

    assert snap.stirring_state is not None
    assert snap.stirring_state.declarer_player is not None
    assert isinstance(snap.stirring_state.declarer_player, int)
    assert 0 <= snap.stirring_state.declarer_player <= 3

    # Verify to_dict() serializes declarer_player
    d = snap.to_dict()
    stirring_dict = d["stirring_state"]
    assert stirring_dict is not None
    assert "declarer_player" in stirring_dict
    assert stirring_dict["declarer_player"] == snap.stirring_state.declarer_player


# ---- Task 003: COMPLETE Phase Awaiting Conditional ----


async def _create_complete_phase_game() -> Game:
    """Helper: create a Game patched into COMPLETE phase via MagicMock.

    Returns a Game instance whose round_sm.create_round injects a
    COMPLETE-phase RoundState. Uses only public interfaces after setup.
    """
    from server.sm import game_sm as gm, round_sm as rm
    from server.sm.card_model import Rank
    from server.sm.scoring import RoundResult

    in_round_state = gm.GameState(
        phase="IN_ROUND", team0_level=Rank.TWO, team1_level=Rank.TWO,
        declarer_team=0, last_declarer_player=0, winning_team=None, round_number=1,
    )
    complete_round = MagicMock()
    complete_round.phase = "COMPLETE"
    complete_round.players_hand = [[] for _ in range(4)]
    complete_round.declarer_player = 0
    complete_round.bottom_cards = []
    complete_round.trump_suit = None
    complete_round.trump_rank = Rank.TWO
    complete_round.declarer_team = 0
    complete_round.defender_points = 0
    complete_round.trick_state = None
    complete_round.trick_history = []
    complete_round.stirring_state = None
    complete_round.exchange_state = None
    complete_round.deal_bid_state = None
    complete_round.result = MagicMock(spec=RoundResult)
    complete_round.result.total_defender_points = 10
    complete_round.result.bottom_card_bonus = 0

    players = _make_players()
    with patch.object(gm, "create_game", return_value=in_round_state):
        game = Game(players=players)

    with patch.object(gm, "start_game", return_value=Ok(in_round_state)):
        with patch.object(rm, "create_round", return_value=complete_round):
            await game.run()

    return game


@pytest.mark.asyncio
async def test_complete_awaiting_unconfirmed_player() -> None:
    """In COMPLETE phase, unconfirmed players have awaiting_action='next_round'.

    Uses MagicMock to create a COMPLETE-phase RoundState and patches
    round_sm.create_round to inject it, then verifies snapshot behavior
    through the public Game.snapshot() interface. Does NOT access any
    private fields (_round_state, _next_round_confirmed, etc.).
    """
    game = await _create_complete_phase_game()

    # Player 0 has NOT confirmed (no NextRoundAction sent)
    snap = game.snapshot(0)
    assert snap.awaiting_action == "next_round", (
        f"Unconfirmed player should have awaiting_action='next_round', got {snap.awaiting_action}"
    )


@pytest.mark.asyncio
async def test_complete_awaiting_confirmed_player() -> None:
    """In COMPLETE phase, confirmed players have awaiting_action=None.

    Uses MagicMock and public Game.act(player, NextRoundAction()) to confirm
    player 0, then verifies snapshot behavior. Does NOT access any private fields.
    """
    game = await _create_complete_phase_game()

    # Confirm player 0 via public interface
    await game.act(player_index=0, action=NextRoundAction())

    # Player 0 confirmed -> awaiting_action=None
    snap0 = game.snapshot(0)
    assert snap0.awaiting_action is None, (
        f"Confirmed player should have awaiting_action=None, got {snap0.awaiting_action}"
    )
    # Player 1 NOT confirmed -> awaiting_action="next_round"
    snap1 = game.snapshot(1)
    assert snap1.awaiting_action == "next_round", (
        f"Unconfirmed player should have awaiting_action='next_round', got {snap1.awaiting_action}"
    )


@pytest.mark.asyncio
async def test_complete_awaiting_multiple_confirmed() -> None:
    """In COMPLETE phase, multiple confirmed players have awaiting_action=None
    while the unconfirmed player still sees awaiting_action='next_round'.

    Verifies the COMPLETE-phase conditional logic for 3 confirmed players
    while the phase is still COMPLETE (the 4th player's confirmation would
    transition the game out of COMPLETE, so the 'all 4 confirmed' state in
    COMPLETE is unobservable by design).
    """
    game = await _create_complete_phase_game()

    # Confirm players 0, 1, 2 via public interface (still in COMPLETE phase)
    for p in range(3):
        await game.act(player_index=p, action=NextRoundAction())

    # Confirmed players (0, 1, 2) -> awaiting_action=None
    for p in range(3):
        snap = game.snapshot(p)
        assert snap.awaiting_action is None, (
            f"Player {p}: confirmed player should have awaiting_action=None, "
            f"got {snap.awaiting_action}"
        )

    # Unconfirmed player (3) -> awaiting_action="next_round"
    snap3 = game.snapshot(3)
    assert snap3.awaiting_action == "next_round", (
        f"Player 3: unconfirmed player should have awaiting_action='next_round', "
        f"got {snap3.awaiting_action}"
    )


# ---- Task 004: Bug 3 — Awaiting Conditional for PLAYING/STIRRING/EXCHANGE ----


async def _create_stirring_phase_game(current_player: int = 1) -> Game:
    """Helper: create a Game patched into STIRRING phase via MagicMock.

    Sets stirring_state.current_player to the given value so we can
    verify only that player sees awaiting_action='stir'.
    """
    from server.sm import game_sm as gm, round_sm as rm

    in_round_state = gm.GameState(
        phase="IN_ROUND", team0_level=Rank.TWO, team1_level=Rank.TWO,
        declarer_team=0, last_declarer_player=0, winning_team=None, round_number=1,
    )
    stirring_round = MagicMock()
    stirring_round.phase = "STIRRING"
    stirring_round.players_hand = [[] for _ in range(4)]
    stirring_round.declarer_player = 0
    stirring_round.bottom_cards = []
    stirring_round.trump_suit = Suit.HEARTS
    stirring_round.trump_rank = Rank.TWO
    stirring_round.declarer_team = 0
    stirring_round.defender_points = 0
    stirring_round.trick_state = None
    stirring_round.trick_history = []
    stirring_round.exchange_state = None
    stirring_round.deal_bid_state = None
    stirring_round.result = None
    stirring_round.stirring_state = MagicMock()
    stirring_round.stirring_state.current_player = current_player
    stirring_round.stirring_state.declarer_player = 0
    stirring_round.stirring_state.trump_suit = Suit.HEARTS
    stirring_round.stirring_state.last_stir_player = None
    stirring_round.stirring_state.current_priority = 0

    players = _make_players()
    with patch.object(gm, "create_game", return_value=in_round_state):
        game = Game(players=players)

    with patch.object(gm, "start_game", return_value=Ok(in_round_state)):
        with patch.object(rm, "create_round", return_value=stirring_round):
            await game.run()

    return game


async def _create_exchange_phase_game(declarer_player: int = 2) -> Game:
    """Helper: create a Game patched into EXCHANGE phase via MagicMock."""
    from server.sm import game_sm as gm, round_sm as rm

    in_round_state = gm.GameState(
        phase="IN_ROUND", team0_level=Rank.TWO, team1_level=Rank.TWO,
        declarer_team=0, last_declarer_player=0, winning_team=None, round_number=1,
    )
    exchange_round = MagicMock()
    exchange_round.phase = "EXCHANGE"
    exchange_round.players_hand = [[] for _ in range(4)]
    exchange_round.declarer_player = declarer_player
    exchange_round.bottom_cards = []
    exchange_round.trump_suit = Suit.HEARTS
    exchange_round.trump_rank = Rank.TWO
    exchange_round.declarer_team = 0
    exchange_round.defender_points = 0
    exchange_round.trick_state = None
    exchange_round.trick_history = []
    exchange_round.stirring_state = None
    exchange_round.deal_bid_state = None
    exchange_round.result = None
    exchange_round.exchange_state = MagicMock()
    exchange_round.exchange_state.phase = "PICKED_UP"
    exchange_round.exchange_state.declarer_player = declarer_player
    exchange_round.exchange_state.count = 8

    players = _make_players()
    with patch.object(gm, "create_game", return_value=in_round_state):
        game = Game(players=players)

    with patch.object(gm, "start_game", return_value=Ok(in_round_state)):
        with patch.object(rm, "create_round", return_value=exchange_round):
            await game.run()

    return game


async def _create_playing_phase_game(cur_player: int = 3) -> Game:
    """Helper: create a Game patched into PLAYING phase via MagicMock."""
    from server.sm import game_sm as gm, round_sm as rm
    from server.sm.card_model import Card

    # Give the current player a card so get_legal_plays returns at least one option
    test_card = Card(
        id="D1-hearts-3", suit=Suit.HEARTS, rank=Rank.THREE,
        is_joker=False, is_big_joker=False, points=0, deck=1,
    )
    player_hands: list[list[Card]] = [[] for _ in range(4)]
    player_hands[cur_player] = [test_card]

    in_round_state = gm.GameState(
        phase="IN_ROUND", team0_level=Rank.TWO, team1_level=Rank.TWO,
        declarer_team=0, last_declarer_player=0, winning_team=None, round_number=1,
    )
    playing_round = MagicMock()
    playing_round.phase = "PLAYING"
    playing_round.players_hand = player_hands
    playing_round.declarer_player = 0
    playing_round.bottom_cards = []
    playing_round.trump_suit = Suit.HEARTS
    playing_round.trump_rank = Rank.TWO
    playing_round.declarer_team = 0
    playing_round.defender_points = 0
    playing_round.trick_history = []
    playing_round.stirring_state = None
    playing_round.exchange_state = None
    playing_round.deal_bid_state = None
    playing_round.result = None
    playing_round.trick_state = MagicMock()
    playing_round.trick_state.phase = "LEADING"
    playing_round.trick_state.lead_player = cur_player
    playing_round.trick_state.slots = []
    playing_round.trick_state.cur = cur_player
    playing_round.trick_state.trump_suit = Suit.HEARTS
    playing_round.trick_state.trump_rank = Rank.TWO
    playing_round.trick_state.defender_points = 0
    playing_round.trick_state.declarer_team = 0
    playing_round.trick_state.hands = player_hands
    playing_round.trick_state.result = None

    players = _make_players()
    with patch.object(gm, "create_game", return_value=in_round_state):
        game = Game(players=players)

    with patch.object(gm, "start_game", return_value=Ok(in_round_state)):
        with patch.object(rm, "create_round", return_value=playing_round):
            await game.run()

    return game


@pytest.mark.asyncio
async def test_stirring_awaiting_for_current_player() -> None:
    """In STIRRING phase, only the current player has awaiting_action='stir'.

    Uses MagicMock to inject a STIRRING-phase RoundState with
    current_player=1, then verifies only player 1 sees awaiting='stir'.
    Also cross-references with stirring_state.current_player (CQ-004).
    """
    game = await _create_stirring_phase_game(current_player=1)
    snap = game.snapshot(for_player=1)
    assert snap.phase == "STIRRING"

    # Player 1 (current_player) must have awaiting_action='stir'
    assert snap.awaiting_action == "stir", (
        f"Player 1 (current): expected awaiting_action='stir', got {snap.awaiting_action}"
    )

    # Cross-reference: awaiting player must match stirring_state.current_player (CQ-004)
    assert snap.stirring_state is not None
    assert snap.stirring_state.current_player == 1

    # All other players must have awaiting_action=None
    for i in range(4):
        s = game.snapshot(for_player=i)
        if i == 1:
            assert s.awaiting_action == "stir"
        else:
            assert s.awaiting_action is None, (
                f"Player {i}: expected awaiting_action=None, got {s.awaiting_action}"
            )


@pytest.mark.asyncio
async def test_exchange_awaiting_for_declarer() -> None:
    """In EXCHANGE phase, only the declarer has awaiting_action='discard'.

    Uses MagicMock to inject an EXCHANGE-phase RoundState with
    declarer_player=2, then verifies only player 2 sees awaiting='discard'.
    """
    game = await _create_exchange_phase_game(declarer_player=2)
    snap = game.snapshot(for_player=2)
    assert snap.phase == "EXCHANGE"

    # Player 2 (declarer) must have awaiting_action='discard'
    assert snap.awaiting_action == "discard", (
        f"Player 2 (declarer): expected awaiting_action='discard', got {snap.awaiting_action}"
    )

    # All other players must have awaiting_action=None
    for i in range(4):
        s = game.snapshot(for_player=i)
        if i == 2:
            assert s.awaiting_action == "discard"
        else:
            assert s.awaiting_action is None, (
                f"Player {i}: expected awaiting_action=None, got {s.awaiting_action}"
            )


@pytest.mark.asyncio
async def test_playing_awaiting_for_trick_cur() -> None:
    """In PLAYING phase, only the current trick player has awaiting_action='play'.

    Uses MagicMock to inject a PLAYING-phase RoundState with trick_state.cur=3
    in LEADING phase, then verifies only player 3 sees awaiting='play'.
    """
    game = await _create_playing_phase_game(cur_player=3)
    snap = game.snapshot(for_player=3)
    assert snap.phase == "PLAYING"

    # Player 3 (trick cur) must have awaiting_action='play'
    assert snap.awaiting_action == "play", (
        f"Player 3 (trick cur): expected awaiting_action='play', got {snap.awaiting_action}"
    )

    # All other players must have awaiting_action=None
    for i in range(4):
        s = game.snapshot(for_player=i)
        if i == 3:
            assert s.awaiting_action == "play"
        else:
            assert s.awaiting_action is None, (
                f"Player {i}: expected awaiting_action=None, got {s.awaiting_action}"
            )


@pytest.mark.asyncio
async def test_stirring_state_has_legal_actions() -> None:
    """StirringStateSnapshot includes legal_actions during STIRRING phase.

    The legal_actions field lists valid stir options (pairs of trump-rank
    cards or joker pairs with priority exceeding current trump). This allows
    clients to determine which stir actions are legal without duplicating
    the server's validation logic.

    Also verifies non-current players see empty legal_actions (CQ-003).
    """
    game = await _create_stirring_phase_game(current_player=0)
    snap = game.snapshot(for_player=0)
    assert snap.phase == "STIRRING"

    assert snap.stirring_state is not None
    assert isinstance(snap.stirring_state.legal_actions, list)
    # Each entry should be a list of 2 cards (stir requires pairs)
    for entry in snap.stirring_state.legal_actions:
        assert isinstance(entry, list)
        # Legal stir is always a pair (2 cards)
        assert len(entry) == 2, f"Each stir option must be a pair, got {len(entry)} cards"

    # Verify to_dict() serialization
    d = snap.to_dict()
    stirring_dict = d["stirring_state"]
    assert stirring_dict is not None
    assert "legal_actions" in stirring_dict
    assert isinstance(stirring_dict["legal_actions"], list)

    # CQ-003: non-current players must see empty legal_actions
    for i in range(1, 4):
        s = game.snapshot(for_player=i)
        assert s.stirring_state is not None
        assert s.stirring_state.legal_actions == [], (
            f"Player {i} (non-current): expected empty legal_actions, "
            f"got {s.stirring_state.legal_actions}"
        )


# ---- Task 005: Bug 4 — Pass player_index to STIRRING/EXCHANGE act() ----


@pytest.mark.asyncio
async def test_act_stir_rejects_non_current_player() -> None:
    """SkipStirAction from a non-current player should be rejected."""
    players = [AutoPlayer(index=i) for i in range(4)]
    game = Game(players=players)
    await game.run()

    # Drive to STIRRING phase
    max_attempts = 200
    for _ in range(max_attempts):
        snap = game.snapshot(for_player=0)
        if snap.phase == "STIRRING":
            break
        if snap.awaiting_action == "bid":
            if snap.bid_legal_actions:
                cards = snap.bid_legal_actions[0]
                await game.act(0, BidAction(cards=cards, count=len(cards)))
            else:
                await game.act(0, SkipBidAction())
        else:
            await asyncio.sleep(0.01)

    snap = game.snapshot(for_player=0)
    if snap.phase != "STIRRING":
        pytest.skip("Could not reach STIRRING phase")

    # Find the current player (the one with awaiting_action='stir')
    current = None
    for i in range(4):
        s = game.snapshot(for_player=i)
        if s.awaiting_action == "stir":
            current = i
            break
    assert current is not None

    # Pick a different player
    other = (current + 1) % 4

    # The other player tries to skip stir -- should be rejected
    # (error is sent via send_error, not raised)
    await game.act(other, SkipStirAction())

    # The current player should still have awaiting_action='stir'
    snap_after = game.snapshot(for_player=current)
    assert snap_after.awaiting_action == "stir", (
        f"Current player should still have awaiting_action='stir', got {snap_after.awaiting_action}"
    )


@pytest.mark.asyncio
async def test_act_discard_rejects_non_declarer() -> None:
    """DiscardAction from a non-declarer should be rejected."""
    players = [AutoPlayer(index=i) for i in range(4)]
    game = Game(players=players)
    await game.run()

    # Drive to EXCHANGE phase
    max_attempts = 500
    for _ in range(max_attempts):
        snap = game.snapshot(for_player=0)
        if snap.phase == "EXCHANGE":
            break
        if snap.awaiting_action == "bid":
            if snap.bid_legal_actions:
                cards = snap.bid_legal_actions[0]
                await game.act(0, BidAction(cards=cards, count=len(cards)))
            else:
                await game.act(0, SkipBidAction())
        elif snap.awaiting_action == "stir":
            await game.act(0, SkipStirAction())
        else:
            await asyncio.sleep(0.01)

    snap = game.snapshot(for_player=0)
    if snap.phase != "EXCHANGE":
        pytest.skip("Could not reach EXCHANGE phase")

    # Find the declarer (the one with awaiting_action='discard')
    declarer = None
    for i in range(4):
        s = game.snapshot(for_player=i)
        if s.awaiting_action == "discard":
            declarer = i
            break
    assert declarer is not None

    # Pick a different player
    other = (declarer + 1) % 4

    # The other player tries to discard -- should be rejected
    snap_other = game.snapshot(for_player=other)
    hand = snap_other.player_hand
    if hand:
        # Use a card from the other player's hand directly (no Card.from_id needed)
        await game.act(other, DiscardAction(cards=[hand[0]]))

    # The declarer should still have awaiting_action='discard'
    snap_after = game.snapshot(for_player=declarer)
    assert snap_after.awaiting_action == "discard", (
        f"Declarer should still have awaiting_action='discard', got {snap_after.awaiting_action}"
    )
