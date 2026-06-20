"""Tests for server/game_registry.py -- in-memory game registry with timeout cleanup."""

from unittest.mock import MagicMock

from server.game_registry import GameRegistry


def _make_game(phase: str = "IN_PROGRESS") -> MagicMock:
    """Create a mock Game object for testing."""
    game = MagicMock()
    game.get_phase.return_value = phase
    return game


def test_create_stores_and_returns_id():
    registry = GameRegistry()
    game = _make_game()
    game_id = registry.create(game)
    assert isinstance(game_id, str)
    assert len(game_id) > 0
    retrieved = registry.get(game_id)
    assert retrieved is game


def test_create_generates_unique_ids():
    registry = GameRegistry()
    game1 = _make_game()
    game2 = _make_game()
    id1 = registry.create(game1)
    id2 = registry.create(game2)
    assert id1 != id2


def test_get_returns_game():
    registry = GameRegistry()
    game = _make_game()
    game_id = registry.create(game)
    result = registry.get(game_id)
    assert result is game


def test_get_missing_returns_none():
    registry = GameRegistry()
    assert registry.get("nonexistent") is None


def test_get_updates_last_access():
    """get() should update the last access timestamp for the game.

    Uses a controllable clock (lambda returning incrementing timestamps)
    instead of accessing private _last_access field. The test verifies
    that cleanup_expired does NOT remove a game that was recently accessed
    via get(), proving that get() updates the timestamp.
    """
    # Use a controllable clock: starts at T=100, increments by 100 each call
    clock_calls = [0]

    def fake_clock():
        clock_calls[0] += 1
        return float(clock_calls[0] * 100)

    registry = GameRegistry(clock=fake_clock)
    game = _make_game()
    game_id = registry.create(game)  # clock returns 100.0

    # Access the game at time T=200
    registry.get(game_id)  # clock returns 200.0

    # Now advance clock to T=500 and cleanup with max_age=250
    # Game was accessed at T=200, which is 300 seconds ago (500-200=300 > 250)
    # So it should be expired
    clock_calls[0] = 4  # next clock call will return 500.0
    removed = registry.cleanup_expired(max_age_seconds=250)
    assert removed == 1

    # Now test the positive case: get() updates timestamp so game is NOT expired
    registry2 = GameRegistry(clock=fake_clock)
    clock_calls[0] = 0
    game2 = _make_game()
    game_id2 = registry2.create(game2)  # clock returns 100.0

    # Access the game at time T=200
    registry2.get(game_id2)  # clock returns 200.0

    # Advance clock to T=300 and cleanup with max_age=250
    # Game was accessed at T=200, which is 100 seconds ago (300-200=100 < 250)
    # So it should NOT be expired
    clock_calls[0] = 2  # next clock call will return 300.0
    removed2 = registry2.cleanup_expired(max_age_seconds=250)
    assert removed2 == 0


def test_delete_removes_game():
    registry = GameRegistry()
    game = _make_game()
    game_id = registry.create(game)
    registry.delete(game_id)
    assert registry.get(game_id) is None


def test_delete_missing_is_noop():
    registry = GameRegistry()
    # Should not raise
    registry.delete("nonexistent")


def test_list_games_returns_phase_info():
    registry = GameRegistry()
    game1 = _make_game(phase="DEAL_BID")
    game2 = _make_game(phase="WAITING")
    id1 = registry.create(game1)
    id2 = registry.create(game2)
    result = registry.list_games()
    assert len(result) == 2
    ids_in_result = {r["game_id"] for r in result}
    assert ids_in_result == {id1, id2}
    phases = {r["game_id"]: r["phase"] for r in result}
    assert phases[id1] == "DEAL_BID"
    assert phases[id2] == "WAITING"


def test_list_games_empty():
    registry = GameRegistry()
    assert registry.list_games() == []


def test_cleanup_expired_removes_old():
    """cleanup_expired removes games older than max_age using the clock.

    Uses a controllable clock instead of modifying private _last_access.
    """
    clock_calls = [0]

    def fake_clock():
        clock_calls[0] += 1
        return float(clock_calls[0] * 100)

    registry = GameRegistry(clock=fake_clock)
    game = _make_game()
    game_id = registry.create(game)  # clock returns 100.0 (timestamp = 100)

    # Advance clock to T=8000 (8000 - 100 = 7900 seconds ago, > 3600)
    clock_calls[0] = 79
    removed = registry.cleanup_expired(max_age_seconds=3600)
    assert removed == 1
    assert registry.get(game_id) is None


def test_cleanup_expired_keeps_recent():
    """cleanup_expired keeps games within max_age using the clock.

    Uses a controllable clock instead of modifying private _last_access.
    """
    clock_calls = [0]

    def fake_clock():
        clock_calls[0] += 1
        return float(clock_calls[0] * 100)

    registry = GameRegistry(clock=fake_clock)
    game = _make_game()
    game_id = registry.create(game)  # clock returns 100.0

    # Advance clock only slightly: T=200 (200 - 100 = 100 seconds ago, < 3600)
    clock_calls[0] = 1
    removed = registry.cleanup_expired(max_age_seconds=3600)
    assert removed == 0
    assert registry.get(game_id) is game


def test_cleanup_expired_mixed():
    """cleanup_expired correctly handles a mix of old and recent games.

    Uses a controllable clock to simulate time passage without
    accessing private _last_access.
    """
    clock_calls = [0]

    def fake_clock():
        clock_calls[0] += 1
        return float(clock_calls[0] * 100)

    registry = GameRegistry(clock=fake_clock)
    old_game = _make_game()
    new_game = _make_game()
    old_id = registry.create(old_game)  # clock returns 100.0

    # Simulate old_game being created long ago by resetting clock
    # We need: old_game created at T=100, new_game created at T=7200
    # Then at T=8000, old_game is 7900s old (>3600), new_game is 800s old (<3600)

    # Reset clock so new_game gets a high timestamp
    clock_calls[0] = 71  # next clock call returns 7200.0
    new_id = registry.create(new_game)  # clock returns 7200.0

    # Advance clock to T=8000
    clock_calls[0] = 79  # next clock call returns 8000.0
    removed = registry.cleanup_expired(max_age_seconds=3600)
    assert removed == 1
    assert registry.get(old_id) is None
    assert registry.get(new_id) is new_game


def test_list_games_returns_real_phase():
    """list_games should use game.get_phase() for phase info, not a simple is_over() check."""
    from unittest.mock import MagicMock
    registry = GameRegistry()
    game1 = MagicMock()
    game1.get_phase.return_value = "DEAL_BID"
    game2 = MagicMock()
    game2.get_phase.return_value = "PLAYING"
    id1 = registry.create(game1)
    id2 = registry.create(game2)
    result = registry.list_games()
    phases = {r["game_id"]: r["phase"] for r in result}
    assert phases[id1] == "DEAL_BID"
    assert phases[id2] == "PLAYING"
