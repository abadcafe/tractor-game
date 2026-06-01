"""Tests for game API endpoints."""
import pytest
from fastapi.testclient import TestClient
from server.server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def game_id(client):
    """Create a game and return its ID."""
    resp = client.post("/api/game")
    assert resp.status_code == 200
    return resp.json()["gameId"]


class TestApiCreateGame:
    def test_api_create_game(self, client):
        resp = client.post("/api/game")
        assert resp.status_code == 200
        data = resp.json()
        assert "gameId" in data
        assert "state" in data
        assert data["state"]["phase"] == "dealing"


class TestApiGetGame:
    def test_api_get_game(self, client, game_id):
        resp = client.get(f"/api/game/{game_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gameId"] == game_id

    def test_api_game_not_found(self, client):
        resp = client.get("/api/game/nonexistent")
        assert resp.status_code == 404


class TestApiDeal:
    def test_api_deal(self, client, game_id):
        resp = client.post(f"/api/game/{game_id}/deal")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"]["phase"] == "bidding"
        assert data["awaitingAction"] == "bid"


class TestApiBid:
    def test_api_bid(self, client, game_id):
        client.post(f"/api/game/{game_id}/deal")
        resp = client.post(f"/api/game/{game_id}/bid", json={
            "player_index": 0,
            "level": "3",
            "pass": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"]["phase"] in ("bidding", "stirring")

    def test_api_bid_pass(self, client, game_id):
        client.post(f"/api/game/{game_id}/deal")
        resp = client.post(f"/api/game/{game_id}/bid", json={
            "player_index": 0,
            "level": None,
            "pass": True,
        })
        assert resp.status_code == 200


class TestApiSetTrump:
    def test_api_set_trump(self, client, game_id):
        client.post(f"/api/game/{game_id}/deal")
        client.post(f"/api/game/{game_id}/bid", json={"player_index": 0, "level": "3", "pass": False})
        client.post(f"/api/game/{game_id}/bid", json={"player_index": 2, "level": None, "pass": True})
        client.post(f"/api/game/{game_id}/bid", json={"player_index": 3, "level": None, "pass": True})
        client.post(f"/api/game/{game_id}/bid", json={"player_index": 1, "level": None, "pass": True})
        resp = client.post(f"/api/game/{game_id}/set-trump", json={
            "player_index": 0,
            "trump_suit": "hearts",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"]["phase"] == "stirring"


class TestApiStir:
    def test_api_stir_pass(self, client, game_id):
        _setup_to_stirring(client, game_id)
        resp = client.post(f"/api/game/{game_id}/stir", json={
            "player_index": 2,
            "pass": True,
        })
        assert resp.status_code == 200


class TestApiDiscard:
    def test_api_discard(self, client, game_id):
        _setup_to_exchange(client, game_id)
        data = client.get(f"/api/game/{game_id}").json()
        declarer_idx = None
        for p in data["state"]["players"]:
            if p.get("isDeclarer"):
                declarer_idx = p["index"]
                break
        assert declarer_idx is not None
        discard_ids = [c["id"] for c in data["state"]["players"][declarer_idx]["hand"][:8]]
        resp = client.post(f"/api/game/{game_id}/discard", json={
            "player_index": declarer_idx,
            "card_ids": discard_ids,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"]["phase"] == "playing"


class TestApiPlay:
    def test_api_play(self, client, game_id):
        _setup_to_playing(client, game_id)
        data = client.get(f"/api/game/{game_id}").json()
        player_idx = data["state"]["currentPlayerIndex"]
        legal = data.get("legalActions", [])
        if legal:
            card_ids = legal[0]["cards"]
        else:
            card_ids = [data["state"]["players"][player_idx]["hand"][0]["id"]]
        resp = client.post(f"/api/game/{game_id}/play", json={
            "player_index": player_idx,
            "card_ids": card_ids,
        })
        assert resp.status_code == 200


class TestApiClearTrick:
    def test_api_clear_trick(self, client, game_id):
        _setup_to_playing(client, game_id)
        for _ in range(4):
            data = client.get(f"/api/game/{game_id}").json()
            player_idx = data["state"]["currentPlayerIndex"]
            hand = data["state"]["players"][player_idx]["hand"]
            if not hand:
                break
            play_resp = client.post(f"/api/game/{game_id}/play", json={
                "player_index": player_idx,
                "card_ids": [hand[0]["id"]],
            })
            assert play_resp.status_code == 200
        resp = client.post(f"/api/game/{game_id}/clear-trick")
        assert resp.status_code == 200


class TestApiNextRound:
    def test_api_next_round_rejects_playing_phase(self, client, game_id):
        _setup_to_playing(client, game_id)
        resp = client.post(f"/api/game/{game_id}/next-round")
        assert resp.status_code == 400


class TestApiWinningTeam:
    def test_api_winning_team_field_in_response(self, client, game_id):
        """CR-008: verify winningTeam field is present in API responses."""
        resp = client.post(f"/api/game/{game_id}/deal")
        data = resp.json()
        # When not game over, winningTeam should be null
        assert "winningTeam" in data
        assert data["winningTeam"] is None


# ---- Helpers ----

def _setup_to_stirring(client, game_id):
    client.post(f"/api/game/{game_id}/deal")
    client.post(f"/api/game/{game_id}/bid", json={"player_index": 0, "level": "3", "pass": False})
    client.post(f"/api/game/{game_id}/bid", json={"player_index": 2, "level": None, "pass": True})
    client.post(f"/api/game/{game_id}/bid", json={"player_index": 3, "level": None, "pass": True})
    client.post(f"/api/game/{game_id}/bid", json={"player_index": 1, "level": None, "pass": True})
    client.post(f"/api/game/{game_id}/set-trump", json={"player_index": 0, "trump_suit": "hearts"})


def _setup_to_exchange(client, game_id):
    _setup_to_stirring(client, game_id)
    for i in [2, 3, 1, 0]:
        client.post(f"/api/game/{game_id}/stir", json={"player_index": i, "pass": True})


def _setup_to_playing(client, game_id):
    _setup_to_exchange(client, game_id)
    data = client.get(f"/api/game/{game_id}").json()
    declarer_idx = None
    for p in data["state"]["players"]:
        if p.get("isDeclarer"):
            declarer_idx = p["index"]
            break
    discard_ids = [c["id"] for c in data["state"]["players"][declarer_idx]["hand"][:8]]
    client.post(f"/api/game/{game_id}/discard", json={
        "player_index": declarer_idx,
        "card_ids": discard_ids,
    })
