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
        # After deal, AI auto-play runs so phase may advance beyond bidding
        assert data["state"]["phase"] in ("bidding", "stirring", "exchange", "playing")
        assert data["awaitingAction"] is not None


class TestApiBid:
    def test_api_bid(self, client, game_id):
        deal_data = client.post(f"/api/game/{game_id}/deal").json()
        # After deal + AI auto-play, bid as the current player
        current_player = deal_data["state"]["currentPlayerIndex"]
        if deal_data["state"]["phase"] == "bidding":
            valid_levels = deal_data.get("validBidLevels", [])
            bid_level = valid_levels[-1] if valid_levels else None
            resp = client.post(f"/api/game/{game_id}/bid", json={
                "player_index": current_player,
                "level": bid_level,
                "pass": bid_level is None,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["state"]["phase"] in ("bidding", "stirring")
        else:
            # AI already resolved bidding — test passes trivially
            assert deal_data["state"]["phase"] in ("stirring", "exchange", "playing")

    def test_api_bid_pass(self, client, game_id):
        deal_data = client.post(f"/api/game/{game_id}/deal").json()
        current_player = deal_data["state"]["currentPlayerIndex"]
        if deal_data["state"]["phase"] == "bidding":
            resp = client.post(f"/api/game/{game_id}/bid", json={
                "player_index": current_player,
                "level": None,
                "pass": True,
            })
            assert resp.status_code == 200
        else:
            # AI already resolved bidding
            assert deal_data["state"]["phase"] in ("stirring", "exchange", "playing")


class TestApiSetTrump:
    def test_api_set_trump(self, client, game_id):
        data = _drive_to_set_trump(client, game_id)
        if data["awaitingAction"] == "set_trump":
            # Find the bid winner
            bids = data["state"]["biddingHistory"]
            winner_idx = None
            for b in reversed(bids):
                if not b["pass"]:
                    winner_idx = b["playerIndex"]
                    break
            assert winner_idx is not None
            resp = client.post(f"/api/game/{game_id}/set-trump", json={
                "player_index": winner_idx,
                "trump_suit": "hearts",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["state"]["phase"] == "stirring"
        else:
            # AI already set trump
            assert data["state"]["phase"] in ("stirring", "exchange", "playing")


class TestApiStir:
    def test_api_stir_pass(self, client, game_id):
        data = _drive_to_stirring(client, game_id)
        if data["state"]["phase"] == "stirring":
            current_player = data["state"]["currentPlayerIndex"]
            resp = client.post(f"/api/game/{game_id}/stir", json={
                "player_index": current_player,
                "pass": True,
            })
            assert resp.status_code == 200
        else:
            # AI already resolved stirring
            assert data["state"]["phase"] in ("exchange", "playing")


class TestApiDiscard:
    def test_api_discard(self, client, game_id):
        data = _drive_to_exchange(client, game_id)
        if data["state"]["phase"] == "exchange":
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
        else:
            # AI already handled exchange
            assert data["state"]["phase"] == "playing"


class TestApiPlay:
    def test_api_play(self, client, game_id):
        data = _drive_to_playing(client, game_id)
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
        data = _drive_to_playing(client, game_id)
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
        _drive_to_playing(client, game_id)
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

def _drive_to_set_trump(client, game_id):
    """Drive game to the set_trump awaiting state. Returns current API data."""
    deal_data = client.post(f"/api/game/{game_id}/deal").json()
    if deal_data["state"]["phase"] != "bidding":
        return deal_data
    # Continue bidding until resolved
    for _ in range(50):
        data = client.get(f"/api/game/{game_id}").json()
        if data["state"]["phase"] != "bidding":
            return data
        if data["awaitingAction"] == "set_trump":
            return data
        cp = data["state"]["currentPlayerIndex"]
        client.post(f"/api/game/{game_id}/bid", json={
            "player_index": cp, "level": None, "pass": True,
        })
    return client.get(f"/api/game/{game_id}").json()


def _drive_to_stirring(client, game_id):
    """Drive game to stirring phase. Returns current API data."""
    data = _drive_to_set_trump(client, game_id)
    if data["state"]["phase"] == "stirring":
        return data
    # Set trump if still in bidding
    if data["state"]["phase"] == "bidding" and data["awaitingAction"] == "set_trump":
        bids = data["state"]["biddingHistory"]
        winner_idx = None
        for b in reversed(bids):
            if not b["pass"]:
                winner_idx = b["playerIndex"]
                break
        if winner_idx is not None:
            client.post(f"/api/game/{game_id}/set-trump", json={
                "player_index": winner_idx, "trump_suit": "hearts",
            })
    return client.get(f"/api/game/{game_id}").json()


def _drive_to_exchange(client, game_id):
    """Drive game to exchange phase. Returns current API data."""
    data = _drive_to_stirring(client, game_id)
    if data["state"]["phase"] != "stirring":
        return data
    for _ in range(20):
        data = client.get(f"/api/game/{game_id}").json()
        if data["state"]["phase"] != "stirring":
            return data
        cp = data["state"]["currentPlayerIndex"]
        client.post(f"/api/game/{game_id}/stir", json={
            "player_index": cp, "pass": True,
        })
    return client.get(f"/api/game/{game_id}").json()


def _drive_to_playing(client, game_id):
    """Drive game to playing phase. Returns current API data."""
    data = _drive_to_exchange(client, game_id)
    if data["state"]["phase"] != "exchange":
        return data
    declarer_idx = None
    for p in data["state"]["players"]:
        if p.get("isDeclarer"):
            declarer_idx = p["index"]
            break
    if declarer_idx is not None:
        hand = data["state"]["players"][declarer_idx]["hand"]
        discard_ids = [c["id"] for c in hand[:8]]
        client.post(f"/api/game/{game_id}/discard", json={
            "player_index": declarer_idx, "card_ids": discard_ids,
        })
    return client.get(f"/api/game/{game_id}").json()
