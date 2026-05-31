"""Tests for engine.game_state module."""
import pytest
from pydantic import ValidationError
from server.engine.game_state import (
    TrickSlot, CompletedTrick, PlayerState, TeamState,
    GameState, GameSettings,
)
from server.engine.types import BidAction, Phase, PlayType, StirAction
from server.engine.card import Suit, Rank, Card


class TestTrickSlot:
    def test_trick_slot_creation(self):
        slot = TrickSlot(player_index=0, cards=None)
        assert slot.player_index == 0
        assert slot.cards is None

    def test_trick_slot_with_cards(self):
        c = Card(id="D1-hearts-A", suit=Suit.HEARTS, rank=Rank.ACE,
                 is_joker=False, is_big_joker=False, points=0, deck=1)
        slot = TrickSlot(player_index=0, cards=[c])
        assert len(slot.cards) == 1


class TestCompletedTrick:
    def test_completed_trick_creation(self):
        trick = CompletedTrick(
            lead_player_index=0,
            lead_type=PlayType.SINGLE,
            slots=[],
            winner_index=0,
            points=5,
        )
        assert trick.lead_player_index == 0
        assert trick.points == 5


class TestPlayerState:
    def test_player_state_creation(self):
        player = PlayerState(
            index=0, name="test", hand=[], team_index=0,
            is_human=False, is_declarer=False,
        )
        assert player.index == 0
        assert player.hand == []


class TestTeamState:
    def test_team_state_creation(self):
        team = TeamState(index=0, tricks=[], current_level=Rank.TWO)
        assert team.index == 0
        assert team.current_level == Rank.TWO


class TestGameSettings:
    def test_game_settings_defaults(self):
        settings = GameSettings()
        assert settings.api_key == ""
        assert settings.model == "gpt-4o"
        assert settings.base_url == ""
        assert settings.target_level == Rank.ACE
        assert settings.bottom_card_count == 8


class TestGameState:
    def test_game_state_creation(self):
        state = GameState(
            phase=Phase.DEALING,
            current_level=Rank.TWO,
            players=[
                PlayerState(index=0, name="N", hand=[], team_index=0, is_human=False, is_declarer=False),
                PlayerState(index=1, name="W", hand=[], team_index=1, is_human=False, is_declarer=False),
                PlayerState(index=2, name="E", hand=[], team_index=1, is_human=False, is_declarer=False),
                PlayerState(index=3, name="S", hand=[], team_index=0, is_human=True, is_declarer=False),
            ],
            teams=[
                TeamState(index=0, tricks=[], current_level=Rank.TWO),
                TeamState(index=1, tricks=[], current_level=Rank.TWO),
            ],
            current_player_index=0,
            trump_suit=None,
            trump_rank=Rank.TWO,
            declarer_team_index=0,
            current_trick=[
                TrickSlot(player_index=0, cards=None),
                TrickSlot(player_index=1, cards=None),
                TrickSlot(player_index=2, cards=None),
                TrickSlot(player_index=3, cards=None),
            ],
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
        assert state.phase == Phase.DEALING
        assert len(state.players) == 4

    def test_game_state_camelcase_alias(self):
        state = GameState(
            phase=Phase.DEALING,
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
        data = state.model_dump(by_alias=True)
        assert "currentPlayerIndex" in data
        assert "trumpSuit" in data
        assert "declarerTeamIndex" in data

    def test_game_state_serialization_round_trip(self):
        """model_dump then model_validate produces equivalent state."""
        state = GameState(
            phase=Phase.DEALING,
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
        data = state.model_dump()
        restored = GameState.model_validate(data)
        assert restored == state

    def test_game_state_with_typed_histories(self):
        """bidding_history and stir_history accept typed action models."""
        state = GameState(
            phase=Phase.BIDDING,
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
            bidding_history=[BidAction(player_index=0, level=Rank.TWO, pass_=False)],
            stir_history=[StirAction(player_index=1, new_trump_suit=Suit.HEARTS, level=Rank.TWO)],
            defender_points=0,
            settings=GameSettings(),
        )
        assert len(state.bidding_history) == 1
        assert state.bidding_history[0].player_index == 0
        assert len(state.stir_history) == 1
        assert state.stir_history[0].new_trump_suit == Suit.HEARTS


class TestValidation:
    """Pydantic validation rejects invalid inputs."""

    def test_trick_slot_missing_player_index(self):
        with pytest.raises(ValidationError):
            TrickSlot()

    def test_game_state_missing_required_fields(self):
        with pytest.raises(ValidationError):
            GameState(phase=Phase.DEALING)

    def test_player_state_wrong_type(self):
        with pytest.raises(ValidationError):
            PlayerState(index="bad", name=123, hand=[], team_index=0, is_human=False, is_declarer=False)

    def test_game_settings_rejects_wrong_type(self):
        with pytest.raises(ValidationError):
            GameSettings(bottom_card_count="not_int")
