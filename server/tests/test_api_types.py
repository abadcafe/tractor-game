"""Tests for server.api_types module."""
from server.engine.card import Rank
from server.engine.types import Phase
from server.engine.game_state import (
    GameState, GameSettings, PlayerState, TeamState, TrickSlot,
)
from server.api_types import (
    GameStateResponse, CreateGameRequest, BidRequest,
    SetTrumpRequest, StirRequest, DiscardRequest, PlayRequest,
)


def _make_state() -> GameState:
    return GameState(
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


class TestGameStateResponse:
    def test_game_state_response_creation(self):
        resp = GameStateResponse(
            game_id="test",
            state=_make_state(),
            awaiting_action=None,
            legal_actions=None,
            valid_bid_levels=None,
        )
        assert resp.game_id == "test"
        assert resp.awaiting_action is None

    def test_game_state_response_camelcase(self):
        resp = GameStateResponse(
            game_id="test",
            state=_make_state(),
            awaiting_action="bid",
            legal_actions=[],
            valid_bid_levels=["3", "4"],
        )
        data = resp.model_dump(by_alias=True)
        assert "awaitingAction" in data
        assert "legalActions" in data
        assert "validBidLevels" in data

    def test_game_state_response_deserialize_camelcase(self):
        state = _make_state()
        raw = {
            "gameId": "test",
            "state": state.model_dump(by_alias=True),
            "awaitingAction": "bid",
            "legalActions": [],
            "validBidLevels": ["3", "4"],
        }
        resp = GameStateResponse.model_validate(raw)
        assert resp.game_id == "test"
        assert resp.awaiting_action == "bid"


class TestCreateGameRequest:
    def test_create_game_request(self):
        req = CreateGameRequest()
        assert req is not None

    def test_create_game_request_deserialize_camelcase(self):
        req = CreateGameRequest.model_validate({})
        assert req is not None


class TestBidRequest:
    def test_bid_request_creation(self):
        req = BidRequest(player_index=0, level="3", pass_=False)
        assert req.player_index == 0
        assert req.level == "3"

    def test_bid_request_pass_alias(self):
        req = BidRequest(player_index=0, level=None, pass_=True)
        data = req.model_dump(by_alias=True)
        assert "pass" in data

    def test_bid_request_deserialize_camelcase(self):
        req = BidRequest.model_validate({"playerIndex": 0, "level": "3", "pass": False})
        assert req.player_index == 0
        assert req.level == "3"
        assert req.pass_ is False


class TestSetTrumpRequest:
    def test_set_trump_request_creation(self):
        req = SetTrumpRequest(player_index=0, trump_suit="hearts")
        assert req.trump_suit == "hearts"

    def test_set_trump_request_camelcase(self):
        data = SetTrumpRequest(player_index=0, trump_suit="hearts").model_dump(by_alias=True)
        assert "trumpSuit" in data
        assert "playerIndex" in data

    def test_set_trump_request_deserialize_camelcase(self):
        req = SetTrumpRequest.model_validate({"playerIndex": 0, "trumpSuit": "hearts"})
        assert req.player_index == 0
        assert req.trump_suit == "hearts"


class TestStirRequest:
    def test_stir_request_creation(self):
        req = StirRequest(player_index=1, new_trump_suit=None, level=None, pass_=True)
        assert req.pass_ is True

    def test_stir_request_camelcase(self):
        data = StirRequest(player_index=1, new_trump_suit="spades", level="5", pass_=False).model_dump(by_alias=True)
        assert "newTrumpSuit" in data
        assert "playerIndex" in data
        assert "pass" in data

    def test_stir_request_deserialize_camelcase(self):
        req = StirRequest.model_validate({"playerIndex": 1, "newTrumpSuit": None, "level": None, "pass": True})
        assert req.player_index == 1
        assert req.pass_ is True


class TestDiscardRequest:
    def test_discard_request_creation(self):
        req = DiscardRequest(player_index=0, card_ids=["D1-hearts-A", "D1-hearts-K"])
        assert len(req.card_ids) == 2

    def test_discard_request_camelcase(self):
        data = DiscardRequest(player_index=0, card_ids=["D1-hearts-A"]).model_dump(by_alias=True)
        assert "cardIds" in data
        assert "playerIndex" in data

    def test_discard_request_deserialize_camelcase(self):
        req = DiscardRequest.model_validate({"playerIndex": 0, "cardIds": ["D1-hearts-A", "D1-hearts-K"]})
        assert req.player_index == 0
        assert len(req.card_ids) == 2


class TestPlayRequest:
    def test_play_request_creation(self):
        req = PlayRequest(player_index=0, card_ids=["D1-spades-A"])
        assert req.player_index == 0
        assert len(req.card_ids) == 1

    def test_play_request_camelcase(self):
        data = PlayRequest(player_index=0, card_ids=["D1-spades-A"]).model_dump(by_alias=True)
        assert "cardIds" in data
        assert "playerIndex" in data

    def test_play_request_deserialize_camelcase(self):
        req = PlayRequest.model_validate({"playerIndex": 0, "cardIds": ["D1-spades-A"]})
        assert req.player_index == 0
        assert len(req.card_ids) == 1
