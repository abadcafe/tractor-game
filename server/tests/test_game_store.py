"""Tests for storage.game_store module."""
import pytest
from server.engine.game_state import GameState, GameSettings, Phase, PlayerState, TeamState, TrickSlot
from server.engine.card import Rank
from server.storage.game_store import GameStore


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


class TestGameStore:
    def test_game_store_create(self):
        store = GameStore()
        state = _make_state()
        game_id = store.create(state)
        assert game_id is not None
        assert len(game_id) > 0

    def test_game_store_get(self):
        store = GameStore()
        state = _make_state()
        game_id = store.create(state)
        retrieved = store.get(game_id)
        assert retrieved is not None
        assert retrieved.phase == Phase.DEALING

    def test_game_store_get_not_found(self):
        store = GameStore()
        result = store.get("nonexistent")
        assert result is None

    def test_game_store_update(self):
        store = GameStore()
        state = _make_state()
        game_id = store.create(state)
        updated = _make_state(phase=Phase.BIDDING)
        store.update(game_id, updated)
        retrieved = store.get(game_id)
        assert retrieved is not None
        assert retrieved.phase == Phase.BIDDING

    def test_game_store_delete(self):
        store = GameStore()
        state = _make_state()
        game_id = store.create(state)
        store.delete(game_id)
        assert store.get(game_id) is None

    def test_game_store_list_games(self):
        store = GameStore()
        store.create(_make_state())
        store.create(_make_state())
        games = store.list_games()
        assert len(games) == 2

    def test_game_store_update_nonexistent(self):
        store = GameStore()
        with pytest.raises(KeyError):
            store.update("nonexistent", _make_state())

    def test_game_store_get_returns_copy(self):
        """Retrieved state is a deep copy; mutating it does not affect the store."""
        store = GameStore()
        state = _make_state()
        game_id = store.create(state)
        retrieved = store.get(game_id)
        retrieved.phase = Phase.PLAYING
        assert store.get(game_id).phase == Phase.DEALING

    def test_game_store_create_stores_copy(self):
        """Stored state is a deep copy; mutating the original does not affect the store."""
        store = GameStore()
        state = _make_state()
        game_id = store.create(state)
        state.phase = Phase.PLAYING
        assert store.get(game_id).phase == Phase.DEALING

    def test_game_store_update_stores_copy(self):
        """Updated state is a deep copy; mutating the original does not affect the store."""
        store = GameStore()
        state = _make_state()
        game_id = store.create(state)
        updated = _make_state(phase=Phase.BIDDING)
        store.update(game_id, updated)
        updated.phase = Phase.PLAYING
        assert store.get(game_id).phase == Phase.BIDDING
