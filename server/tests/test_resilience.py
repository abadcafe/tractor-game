"""Tests for server.resilience module."""
import pytest
import time
from server.engine.game_state import GameState, GameSettings, Phase, PlayerState, TeamState, TrickSlot
from server.engine.card import Rank
from server.storage.game_store import GameStore
from server import resilience
from server.resilience import cleanup_expired_sessions, get_settings, update_settings


@pytest.fixture(autouse=True)
def _reset_settings():
    """Reset the module-level settings singleton after each test."""
    original = resilience._settings.model_copy(deep=True)
    yield
    resilience._settings = original


def _make_state(phase: Phase = Phase.DEALING) -> GameState:
    return GameState(
        phase=phase,
        current_level=Rank.TWO,
        players=[
            PlayerState(index=i, name=f"P{i}", hand=[], team_index=i % 2, is_human=i == 3, is_declarer=False)
            for i in range(4)
        ],
        teams=[
            TeamState(index=0, tricks=[], current_level=Rank.TWO),
            TeamState(index=1, tricks=[], current_level=Rank.TWO),
        ],
        current_player_index=0,
        trump_suit=None,
        trump_rank=Rank.TWO,
        declarer_team_index=0,
        current_trick=[TrickSlot(player_index=i, cards=None) for i in range(4)],
        lead_player_index=0,
        lead_play_type=None,
        bottom_cards=[],
        trick_history=[],
        last_completed_trick=None,
        bidding_history=[],
        stir_history=[],
        defender_points=0,
        settings=GameSettings(),
    )


class TestCleanupExpiredSessions:
    def test_cleanup_expired_sessions_removes_old(self):
        store = GameStore()
        game_id = store.create(_make_state())
        # Simulate old session by manually setting last_accessed
        store._last_accessed[game_id] = time.time() - 7200  # 2 hours ago
        removed = cleanup_expired_sessions(store, max_age_seconds=3600)
        assert removed == 1
        assert store.get(game_id) is None

    def test_cleanup_expired_sessions_keeps_recent(self):
        store = GameStore()
        game_id = store.create(_make_state())
        removed = cleanup_expired_sessions(store, max_age_seconds=3600)
        assert removed == 0
        assert store.get(game_id) is not None


class TestServerSettings:
    def test_get_settings_default(self):
        settings = get_settings()
        assert settings.model == "gpt-4o"
        assert settings.target_level == Rank.ACE

    def test_update_settings(self):
        update_settings(model="gpt-4o-mini")
        settings = get_settings()
        assert settings.model == "gpt-4o-mini"

    def test_update_settings_persists_across_multiple_updates(self):
        update_settings(model="gpt-4o-mini")
        update_settings(target_level=Rank.TWO)
        settings = get_settings()
        assert settings.model == "gpt-4o-mini"
        assert settings.target_level == Rank.TWO

    def test_update_settings_rejects_unknown_fields(self):
        with pytest.raises(ValueError, match="Unknown settings fields"):
            update_settings(modle="gpt-4o-mini")
