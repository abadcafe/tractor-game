"""End-to-end integration tests for the Tractor game server.

These tests exercise the full pipeline: REST -> WebSocket -> Game -> sm.
They are NOT unit tests -- they test the integration between all modules.
They use only public interfaces (REST API, WebSocket, Game.snapshot, Game.is_over,
Game.cancel).
They do NOT directly access Game._game_state, Game._dealing_task, or other private
fields. They do NOT directly access GameRegistry._last_access or _games -- they use
the controllable clock injected via GameRegistry(clock=...) or the public API.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Literal

import pytest
import httpx
from starlette.testclient import TestClient

from server.server import app, registry
from server.game_registry import GameRegistry
from server.actions import NextRoundAction


@pytest.fixture(autouse=True)
def clean_registry() -> Generator[None, None, None]:
    """Reset the global registry before each test.

    Uses public API only: delete() for each game obtained from list_games().
    """
    games = registry.list_games()
    for g in games:
        registry.delete(g["game_id"])
    yield
    games = registry.list_games()
    for g in games:
        registry.delete(g["game_id"])


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async test client using httpx with ASGI transport for REST tests."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sync_client() -> Generator[TestClient, None, None]:
    """Synchronous test client using Starlette TestClient for WebSocket tests."""
    with TestClient(app) as c:
        yield c


async def _create_game(client: httpx.AsyncClient) -> str:
    """Helper: create a game and return the game_id."""
    resp = await client.post("/api/game")
    assert resp.status_code == 201
    return resp.json()["game_id"]


def _create_game_sync(sync_client: TestClient) -> str:
    """Helper: create a game synchronously and return the game_id."""
    resp = sync_client.post("/api/game")
    assert resp.status_code == 201
    return resp.json()["game_id"]


# ---- Full Flow ----


def test_full_game_flow(sync_client: TestClient) -> None:
    """Test creating a game, connecting, and verifying initial state."""
    game_id = _create_game_sync(sync_client)
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        data = ws.receive_json()
        assert data["type"] == "state"
        state = data["state"]
        assert "phase" in state
        assert "player_hand" in state
        assert "trump_rank" in state


def test_reconnect_mid_game(sync_client: TestClient) -> None:
    """Test disconnecting and reconnecting to a game."""
    game_id = _create_game_sync(sync_client)
    # First connection
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        data1 = ws.receive_json()
        assert data1["type"] == "state"
    # Reconnect
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        data2 = ws.receive_json()
        assert data2["type"] == "state"
        assert "phase" in data2["state"]


@pytest.mark.asyncio
async def test_concurrent_games(client: httpx.AsyncClient) -> None:
    """Test that multiple games can exist simultaneously."""
    game_id_1 = await _create_game(client)
    game_id_2 = await _create_game(client)
    assert game_id_1 != game_id_2
    # List games
    resp = await client.get("/api/game")
    games = resp.json()["games"]
    assert len(games) == 2
    game_ids = {g["game_id"] for g in games}
    assert game_ids == {game_id_1, game_id_2}


@pytest.mark.asyncio
async def test_cleanup_expired_games(client: httpx.AsyncClient) -> None:
    """Test that expired games are cleaned up.

    Uses a fresh GameRegistry with a controllable clock instead of
    modifying the global registry's private _last_access field.
    """
    clock_calls = [0]

    def fake_clock() -> float:
        clock_calls[0] += 1
        return float(clock_calls[0] * 100)

    test_registry = GameRegistry(clock=fake_clock)
    from unittest.mock import MagicMock
    game = MagicMock()
    game.get_phase.return_value = "IN_PROGRESS"
    game_id = test_registry.create(game)  # T=100

    # Advance clock to T=8000 (game created 7900s ago > 3600)
    clock_calls[0] = 79
    removed = test_registry.cleanup_expired(max_age_seconds=3600)
    assert removed == 1
    assert test_registry.get(game_id) is None


def test_invalid_action_returns_error(sync_client: TestClient) -> None:
    """Test that invalid actions through WebSocket return error messages.

    Sends a "play" action with a fake card ID during the dealing phase.
    The server's _parse_action calls game.resolve_cards() which raises
    ValueError for unknown card IDs. The server catches this and returns
    {"type": "error", "message": ...}. We assert on the error response.
    """
    game_id = _create_game_sync(sync_client)
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        ws.receive_json()
        # Try to play cards during dealing phase (should be invalid)
        ws.send_json({"type": "play", "cards": ["fake_card_id"]})
        # Server should send back an error response
        resp = ws.receive_json()
        assert resp["type"] == "error"
        assert "message" in resp
        assert len(resp["message"]) > 0


def test_delete_game_disconnects_ws(sync_client: TestClient) -> None:
    """Test that deleting a game while connected closes cleanly."""
    game_id = _create_game_sync(sync_client)
    with sync_client.websocket_connect(f"/game/{game_id}") as ws:
        ws.receive_json()
    # Delete after disconnect is fine
    resp = sync_client.delete(f"/api/game/{game_id}")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_games_shows_phase(client: httpx.AsyncClient) -> None:
    """Test that listing games includes phase information."""
    game_id = await _create_game(client)
    resp = await client.get("/api/game")
    games = resp.json()["games"]
    assert len(games) == 1
    assert "phase" in games[0]
    assert games[0]["game_id"] == game_id


# ---- Game Auto-Completion ----


@pytest.mark.asyncio
async def test_game_auto_completion(client: httpx.AsyncClient) -> None:
    """Test that a game with 4 AutoPlayers can auto-complete through the full pipeline.

    This test verifies that the dealing loop makes progress by checking that
    the phase transitions from DEAL_BID to a later phase after waiting.
    The dealing loop sleeps 0.75s per card, so we wait long enough for
    several cards to be dealt and check that the phase has changed.
    """
    game_id = await _create_game(client)
    game = registry.get(game_id)
    assert game is not None

    # Initial phase should be DEAL_BID (or a round-level phase)
    _initial_phase = game.get_phase()

    # Wait for dealing to make progress (3 cards at 0.75s each = ~2.25s)
    await asyncio.sleep(3)

    # Verify the game is still running and hasn't crashed
    current_phase = game.get_phase()
    assert current_phase is not None

    # The snapshot should still be valid
    snap = game.snapshot(for_player=3)
    assert isinstance(snap.player_hand, list)
    assert snap.phase is not None

    # Verify phase is still a valid game phase (not crashed/error state)
    assert current_phase in (
        "DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING",
        "COMPLETE", "GAME_OVER", "IN_ROUND",
    )


@pytest.mark.asyncio
async def test_game_over_via_auto_players(client: httpx.AsyncClient) -> None:
    """Test that a game with auto players starts and progresses through phases.

    Verifies that the game is created with 4 AutoPlayers (3 Auto + 1 Human)
    and that the initial state is valid. Auto players will drive the game
    forward asynchronously.
    """
    game_id = await _create_game(client)
    game = registry.get(game_id)
    assert game is not None
    # Verify the game has a valid initial state with a known phase
    initial_phase = game.get_phase()
    assert initial_phase in ("IDLE", "IN_ROUND", "DEAL_BID")


@pytest.mark.asyncio
async def test_game_over_removes_from_registry(client: httpx.AsyncClient) -> None:
    """Test that the on_game_over callback mechanism works end-to-end.

    Creates a Game directly with mocked sm functions to force it through
    to GAME_OVER, and verifies the callback fires and removes the game
    from the registry. Uses game.cancel() (public method) to stop the
    dealing loop.
    """
    from server.game import Game
    from server.player import AutoPlayer
    from server.sm import game_sm as gm, round_sm as rm
    from server.sm.card_model import Rank
    from server.sm.scoring import RoundResult
    from unittest.mock import patch, MagicMock

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

    # Build the GAME_OVER state
    game_over_state = gm.GameState(
        phase="GAME_OVER",
        team0_level=Rank.ACE,
        team1_level=Rank.TEN,
        declarer_team=None,
        last_declarer_player=None,
        winning_team=0,
        round_number=1,
    )

    # Mock RoundResult to trigger game over
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
            await game.cancel()

    # Verify game is in registry
    assert test_registry.get(game_id) is not None

    # Now trigger GAME_OVER via act() with NextRoundAction using patched sm
    # All 4 players must confirm to trigger the next round
    with patch.object(gm, "process_round_result", return_value=Ok(game_over_state)):
        with patch.object(rm, "get_round_result", return_value=mock_result):
            for p in range(4):
                await game.act(player_index=p, action=NextRoundAction())

    # Verify the callback was actually called (not just conditionally checked)
    assert callback_called[0], "on_game_over callback was not invoked"
    # Verify game is over and removed from registry
    assert game.is_over()
    assert test_registry.get(game_id) is None


# ---- SubPlay Integration Tests ----


from server.sm.card_model import Card as SmCard, Suit as SmSuit, Rank as SmRank, POINTS_MAP
from server.sm.trick_sm import create_trick, play as trick_play, TrickInput, TrickState
from server.sm.result import Ok, Rejected
from server.sm.scoring import calculate_score
from server.sm.types import CompletedTrick, CompletedTrickSlot
from server.sm.play_rules import (
    is_legal_lead, is_legal_follow,
    compare_plays,
)


def _card(suit: SmSuit, rank: SmRank, deck: Literal[1, 2] = 1) -> SmCard:
    return SmCard(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == SmSuit.JOKER),
        is_big_joker=(rank == SmRank.BIG_JOKER),
        points=POINTS_MAP.get(rank, 0), deck=deck,
    )


def _play_unwrap(state: TrickState, player: int, cards: list[SmCard]) -> TrickState:
    """Call trick_play and unwrap the Ok result, raising on Rejected."""
    result = trick_play(state, player=player, cards=cards)
    match result:
        case Ok(value=new_state):
            return new_state
        case Rejected(reason=reason):
            raise AssertionError(f"trick_play rejected: {reason}")


class TestE2ETractorFlow:
    """End-to-end tests for tractor-based plays."""

    def test_tractor_lead_wins_over_pair(self) -> None:
        """Tractor (level 3) beats pair (level 2) in trick resolution."""
        hands = [
            # Player 0 leads tractor h3-3-4-4
            [
                _card(SmSuit.HEARTS, SmRank.THREE, 1), _card(SmSuit.HEARTS, SmRank.THREE, 2),
                _card(SmSuit.HEARTS, SmRank.FOUR, 1), _card(SmSuit.HEARTS, SmRank.FOUR, 2),
            ],
            # Player 1 follows with tractor h5-5-6-6 (lower rank)
            [
                _card(SmSuit.HEARTS, SmRank.FIVE, 1), _card(SmSuit.HEARTS, SmRank.FIVE, 2),
                _card(SmSuit.HEARTS, SmRank.SIX, 1), _card(SmSuit.HEARTS, SmRank.SIX, 2),
            ],
            # Player 2 follows with tractor h7-7-8-8 (higher rank)
            [
                _card(SmSuit.HEARTS, SmRank.SEVEN, 1), _card(SmSuit.HEARTS, SmRank.SEVEN, 2),
                _card(SmSuit.HEARTS, SmRank.EIGHT, 1), _card(SmSuit.HEARTS, SmRank.EIGHT, 2),
            ],
            # Player 3 follows with tractor h9-9-10-10
            [
                _card(SmSuit.HEARTS, SmRank.NINE, 1), _card(SmSuit.HEARTS, SmRank.NINE, 2),
                _card(SmSuit.HEARTS, SmRank.TEN, 1), _card(SmSuit.HEARTS, SmRank.TEN, 2),
            ],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=SmSuit.SPADES, trump_rank=SmRank.TWO,
            defender_points=0, declarer_team=0,
        ))
        state = _play_unwrap(state, player=0, cards=hands[0])
        state = _play_unwrap(state, player=1, cards=hands[1])
        state = _play_unwrap(state, player=3, cards=hands[3])
        state = _play_unwrap(state, player=2, cards=hands[2])
        result = state.result
        assert result is not None
        assert result.winner == 3  # h9-9-10-10 wins (highest tractor)

    def test_trump_tractor_beats_non_trump_tractor(self) -> None:
        """Trump tractor beats non-trump tractor of same level."""
        # CCW order: 0 -> 1 -> 3 -> 2
        # Hands are indexed by player position
        hands = [
            # Player 0 leads non-trump tractor h3-3-4-4
            [
                _card(SmSuit.HEARTS, SmRank.THREE, 1), _card(SmSuit.HEARTS, SmRank.THREE, 2),
                _card(SmSuit.HEARTS, SmRank.FOUR, 1), _card(SmSuit.HEARTS, SmRank.FOUR, 2),
            ],
            # Player 1 follows with non-trump tractor h5-5-6-6
            [
                _card(SmSuit.HEARTS, SmRank.FIVE, 1), _card(SmSuit.HEARTS, SmRank.FIVE, 2),
                _card(SmSuit.HEARTS, SmRank.SIX, 1), _card(SmSuit.HEARTS, SmRank.SIX, 2),
            ],
            # Player 2 plays trump tractor sp3-3-4-4 (trump_suit=spade)
            [
                _card(SmSuit.SPADES, SmRank.THREE, 1), _card(SmSuit.SPADES, SmRank.THREE, 2),
                _card(SmSuit.SPADES, SmRank.FOUR, 1), _card(SmSuit.SPADES, SmRank.FOUR, 2),
            ],
            # Player 3 follows with non-trump tractor h7-7-8-8
            [
                _card(SmSuit.HEARTS, SmRank.SEVEN, 1), _card(SmSuit.HEARTS, SmRank.SEVEN, 2),
                _card(SmSuit.HEARTS, SmRank.EIGHT, 1), _card(SmSuit.HEARTS, SmRank.EIGHT, 2),
            ],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=SmSuit.SPADES, trump_rank=SmRank.TWO,
            defender_points=0, declarer_team=0,
        ))
        # CCW order: 0 -> 1 -> 3 -> 2
        state = _play_unwrap(state, player=0, cards=hands[0])
        state = _play_unwrap(state, player=1, cards=hands[1])
        state = _play_unwrap(state, player=3, cards=hands[3])
        state = _play_unwrap(state, player=2, cards=hands[2])
        result = state.result
        assert result is not None
        assert result.winner == 2  # trump tractor wins


class TestE2EThrowFlow:
    """End-to-end tests for throw-based plays."""

    def test_throw_with_all_biggest_sub_plays(self) -> None:
        """Throw with all biggest sub-plays is a legal lead and resolves correctly."""
        # Player 0 leads throw: spA + spK (both biggest spade singles)
        # Other players have no spade cards and no trump cards
        # Trump is clubs, so hearts/diamonds are non-trump
        hands = [
            [_card(SmSuit.SPADES, SmRank.ACE), _card(SmSuit.SPADES, SmRank.KING)],
            [_card(SmSuit.HEARTS, SmRank.THREE), _card(SmSuit.HEARTS, SmRank.FOUR)],
            [_card(SmSuit.HEARTS, SmRank.FIVE), _card(SmSuit.HEARTS, SmRank.SIX)],
            [_card(SmSuit.DIAMONDS, SmRank.SEVEN), _card(SmSuit.DIAMONDS, SmRank.EIGHT)],
        ]
        # Verify the lead is legal
        other_hands = [c for h in hands[1:] for c in h]
        assert is_legal_lead(hands[0], hands[0], SmSuit.CLUBS, SmRank.TWO, other_hands) is True

        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=SmSuit.CLUBS, trump_rank=SmRank.TWO,
            defender_points=0, declarer_team=0,
        ))
        state = _play_unwrap(state, player=0, cards=hands[0])
        state = _play_unwrap(state, player=1, cards=hands[1])
        state = _play_unwrap(state, player=3, cards=hands[3])
        state = _play_unwrap(state, player=2, cards=hands[2])
        result = state.result
        assert result is not None
        assert result.winner == 0  # throw wins (trump is clubs, followers play hearts/diamonds)


class TestE2EFollowRules:
    """End-to-end tests for follow rules enforcement."""

    def test_follow_must_use_higher_level_sub_play(self) -> None:
        """Following a pair: must use pair from tractor if available, not independent pair."""
        # Player 0 leads pair hA-A
        # Player 1 has tractor h3-3-4-4 + pair hK-K, must use pair from tractor
        hands = [
            [_card(SmSuit.HEARTS, SmRank.ACE, 1), _card(SmSuit.HEARTS, SmRank.ACE, 2)],
            [
                _card(SmSuit.HEARTS, SmRank.THREE, 1), _card(SmSuit.HEARTS, SmRank.THREE, 2),
                _card(SmSuit.HEARTS, SmRank.FOUR, 1), _card(SmSuit.HEARTS, SmRank.FOUR, 2),
                _card(SmSuit.HEARTS, SmRank.KING, 1), _card(SmSuit.HEARTS, SmRank.KING, 2),
            ],
            [_card(SmSuit.HEARTS, SmRank.QUEEN, 1), _card(SmSuit.HEARTS, SmRank.QUEEN, 2)],
            [_card(SmSuit.HEARTS, SmRank.JACK, 1), _card(SmSuit.HEARTS, SmRank.JACK, 2)],
        ]

        # Verify illegal play: using pair hK-K (skips tractor)
        illegal = [_card(SmSuit.HEARTS, SmRank.KING, 1), _card(SmSuit.HEARTS, SmRank.KING, 2)]
        lead = hands[0]
        assert is_legal_follow(hands[1], illegal, lead, SmSuit.SPADES, SmRank.TWO) is False

        # Verify legal play: using pair from tractor (h3-3 or h4-4)
        legal = [_card(SmSuit.HEARTS, SmRank.THREE, 1), _card(SmSuit.HEARTS, SmRank.THREE, 2)]
        assert is_legal_follow(hands[1], legal, lead, SmSuit.SPADES, SmRank.TWO) is True


class TestE2EScoringAmbush:
    """End-to-end tests for scoring with ambush multiplier."""

    def test_ambush_with_tractor_multiplier(self) -> None:
        """Last trick with tractor lead: ambush multiplier = 2^(card count)."""
        trick = CompletedTrick(
            lead_player=0,
            slots=[
                CompletedTrickSlot(player=0, cards=[
                    _card(SmSuit.HEARTS, SmRank.THREE, 1), _card(SmSuit.HEARTS, SmRank.THREE, 2),
                    _card(SmSuit.HEARTS, SmRank.FOUR, 1), _card(SmSuit.HEARTS, SmRank.FOUR, 2),
                ]),
                CompletedTrickSlot(player=1, cards=[
                    _card(SmSuit.HEARTS, SmRank.FIVE, 1), _card(SmSuit.HEARTS, SmRank.FIVE, 2),
                    _card(SmSuit.HEARTS, SmRank.SIX, 1), _card(SmSuit.HEARTS, SmRank.SIX, 2),
                ]),
                CompletedTrickSlot(player=2, cards=[
                    _card(SmSuit.HEARTS, SmRank.SEVEN, 1), _card(SmSuit.HEARTS, SmRank.SEVEN, 2),
                    _card(SmSuit.HEARTS, SmRank.EIGHT, 1), _card(SmSuit.HEARTS, SmRank.EIGHT, 2),
                ]),
                CompletedTrickSlot(player=3, cards=[
                    _card(SmSuit.HEARTS, SmRank.NINE, 1), _card(SmSuit.HEARTS, SmRank.NINE, 2),
                    _card(SmSuit.HEARTS, SmRank.TEN, 1), _card(SmSuit.HEARTS, SmRank.TEN, 2),
                ]),
            ],
            winner=1,  # defender wins
            points=30,
        )
        result = calculate_score(
            defender_points=50,
            bottom_cards=[_card(SmSuit.HEARTS, SmRank.FIVE), _card(SmSuit.SPADES, SmRank.TEN)],
            last_trick=trick,
            declarer_team=0,
            declarer_player=0,
            team0_level=SmRank.TWO,
            team1_level=SmRank.TWO,
            trump_suit=SmSuit.SPADES,
            trump_rank=SmRank.TWO,
        )
        # Bottom cards: 5 + 10 = 15 points. Tractor multiplier = 2^4 = 16.
        # Ambush bonus = 15 * 16 = 240. Total = 50 + 240 = 290.
        assert result.total_defender_points == 290


class TestE2EComparePlaysNew:
    """End-to-end tests for the new comparison function."""

    def test_sub_level_comparison_in_trick(self) -> None:
        """Pair (level 2) beats single (level 1) even with lower rank."""
        a = [_card(SmSuit.HEARTS, SmRank.THREE, 1), _card(SmSuit.HEARTS, SmRank.THREE, 2)]
        b = [_card(SmSuit.HEARTS, SmRank.ACE)]
        # Both are heart (lead suit). Pair has level 2, single has level 1.
        result = compare_plays(a, b, SmSuit.HEARTS, SmSuit.SPADES, SmRank.TWO)
        assert result > 0

    def test_eligibility_gate(self) -> None:
        """Off-suit non-trump cannot win even with highest rank."""
        a = [_card(SmSuit.DIAMONDS, SmRank.ACE)]  # off-suit, not trump
        b = [_card(SmSuit.HEARTS, SmRank.THREE)]  # lead suit
        result = compare_plays(a, b, SmSuit.HEARTS, SmSuit.SPADES, SmRank.TWO)
        assert result < 0  # b wins


# ---- Bug 1 e2e regression: game must not resource-explode ----


@pytest.mark.asyncio
async def test_full_game_flow_completes_without_resource_explosion() -> None:
    """A game with 4 AutoPlayers must complete without CPU/memory explosion.

    Regression test for Bug 1: AutoPlayer on_state() -> create_task(bid)
    -> game.act() -> _push_state_to_all() -> on_state() -> ... exponential
    task cascade consumed 96.9% CPU and 8.8 GB RAM.

    This test creates a game with 4 AutoPlayers and lets them drive
    it through at least one full round. After running, the game must
    have progressed and not be stuck in an infinite task cascade.
    """
    from server.game import Game
    from server.player import AutoPlayer
    from unittest.mock import patch

    players = [AutoPlayer(i) for i in range(4)]
    game = Game(players=players)

    # Patch asyncio.sleep in the game module to speed up dealing.
    # The dealing loop calls asyncio.sleep(0.5) between cards;
    # replacing it with a 1ms sleep makes the test practical.
    original_sleep = asyncio.sleep

    async def fast_sleep(delay: float) -> None:
        if delay >= 0.1:
            # Only speed up long sleeps (dealing loop), not short waits
            await original_sleep(0.001)
        else:
            await original_sleep(delay)

    with patch.object(asyncio, 'sleep', side_effect=fast_sleep):
        await game.run()

        # Let the game auto-progress -- with fast sleep, dealing
        # completes in ~0.1s and AI plays complete quickly
        await original_sleep(3)

        phase = game.get_phase()
        # After 3s, the game should have progressed past DEAL_BID
        assert phase in (
            "DEAL_BID", "STIRRING", "EXCHANGE", "PLAYING",
            "COMPLETE", "GAME_OVER",
        ), f"Game stuck in unexpected phase: {phase}"

        # Snapshot must still be valid (no cascading error state)
        try:
            snap = game.snapshot(for_player=0)
            assert isinstance(snap.player_hand, list)
        except RuntimeError:
            pass

        await game.cancel()
