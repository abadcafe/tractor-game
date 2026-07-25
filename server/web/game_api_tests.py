"""Black-box tests for the game HTTP API."""

import pytest

from server.web.application_test_client import (
    AsyncRestClient,
    SyncServerClient,
)
from server.web.application_test_client import (
    game_id_from as _game_id_from,
)
from server.web.application_test_client import (
    is_dict as _is_dict,
)
from server.web.application_test_client import (
    is_list_of_dict as _is_list_of_dict,
)

pytest_plugins: tuple[str, ...] = (
    "server.web.application_test_fixtures",
)


@pytest.mark.asyncio
async def test_create_game_returns_201(client: AsyncRestClient) -> None:
    response = await client.post("/api/game")
    assert response.status_code == 201
    data = response.json()
    assert _is_dict(data)
    assert "game_id" in data


@pytest.mark.asyncio
async def test_create_game_starts_game(client: AsyncRestClient) -> None:
    response = await client.post("/api/game")
    data = response.json()
    assert _is_dict(data)
    game_id_raw = data["game_id"]
    assert isinstance(game_id_raw, str)
    assert len(game_id_raw) > 0


@pytest.mark.asyncio
async def test_create_game_starts_with_empty_players_by_default(
    client: AsyncRestClient,
) -> None:
    response = await client.post("/api/game")
    game_id = _game_id_from(response)
    listed = await client.get("/api/game")

    document = listed.json()
    assert _is_dict(document)
    games = document["games"]
    assert _is_list_of_dict(games)
    matching = [game for game in games if game["game_id"] == game_id]
    assert len(matching) == 1
    game = matching[0]
    players = game["seats"]
    assert _is_list_of_dict(players)
    assert [player["kind"] for player in players] == [
        "empty",
        "empty",
        "empty",
        "empty",
    ]
    assert all(player["occupied"] is False for player in players)
    assert "phase" not in game


# ---- REST: List Games ----


@pytest.mark.asyncio
async def test_list_games_empty(client: AsyncRestClient) -> None:
    response = await client.get("/api/game")
    assert response.status_code == 200
    data = response.json()
    assert _is_dict(data)
    assert data["games"] == []


@pytest.mark.asyncio
async def test_list_games_with_games(client: AsyncRestClient) -> None:
    # Create a game first
    create_resp = await client.post("/api/game")
    assert create_resp.status_code == 201
    # List games
    response = await client.get("/api/game")
    assert response.status_code == 200
    data = response.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    assert len(games_raw) == 1
    first_game = games_raw[0]
    assert "game_id" in first_game
    assert first_game["user_count"] == 0
    assert first_game["capacity"] == 4
    assert first_game["user_seats"] == []
    players_raw = first_game["seats"]
    assert _is_list_of_dict(players_raw)
    assert len(players_raw) == 4
    for player_raw in players_raw:
        assert player_raw["occupied"] is False
    assert "phase" not in first_game


def test_list_games_counts_attached_user_player(
    sync_client: SyncServerClient,
) -> None:
    create_resp = sync_client.post("/api/game")
    game_id = _game_id_from(create_resp)
    attach_resp = sync_client.post(
        f"/api/game/{game_id}/seat/b?user_id=user-1"
    )
    assert attach_resp.status_code == 200

    response = sync_client.get("/api/game?user_id=user-1")
    assert response.status_code == 200
    data = response.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    matching = [
        game for game in games_raw if game["game_id"] == game_id
    ]
    assert len(matching) == 1
    assert matching[0]["user_count"] == 1
    assert matching[0]["capacity"] == 4
    assert matching[0]["user_seats"] == ["b"]
    players_raw = matching[0]["seats"]
    assert _is_list_of_dict(players_raw)
    player_one = players_raw[1]
    assert player_one["seat"] == "b"
    assert player_one["occupied"] is True
    assert player_one["connected"] is False
    assert player_one["mine"] is True
    assert player_one["kind"] == "user"


# ---- REST: Player Selection ----


@pytest.mark.asyncio
async def test_attach_player_rest_attaches_lobby_player(
    client: AsyncRestClient,
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)

    attach_resp = await client.post(
        f"/api/game/{game_id}/seat/c?user_id=user-2"
    )
    response = await client.get("/api/game?user_id=user-2")

    assert attach_resp.status_code == 200
    attach_data = attach_resp.json()
    assert _is_dict(attach_data)
    assert attach_data["ok"] is True
    data = response.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    matching = [
        game for game in games_raw if game["game_id"] == game_id
    ]
    assert len(matching) == 1
    assert matching[0]["user_count"] == 1
    assert matching[0]["user_seats"] == ["c"]
    players_raw = matching[0]["seats"]
    assert _is_list_of_dict(players_raw)
    player_two = players_raw[2]
    assert player_two["occupied"] is True
    assert player_two["connected"] is False
    assert player_two["mine"] is True


@pytest.mark.asyncio
async def test_attach_player_rest_switches_user_player(
    client: AsyncRestClient,
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)

    first_resp = await client.post(
        f"/api/game/{game_id}/seat/a?user_id=user-0"
    )
    second_resp = await client.post(
        f"/api/game/{game_id}/seat/d?user_id=user-0"
    )
    response = await client.get("/api/game?user_id=user-0")

    assert first_resp.status_code == 200
    assert second_resp.status_code == 200
    data = response.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    matching = [
        game for game in games_raw if game["game_id"] == game_id
    ]
    assert len(matching) == 1
    assert matching[0]["user_seats"] == ["d"]
    players_raw = matching[0]["seats"]
    assert _is_list_of_dict(players_raw)
    assert players_raw[0]["occupied"] is False
    assert players_raw[3]["occupied"] is True
    assert players_raw[3]["mine"] is True


@pytest.mark.asyncio
async def test_detach_player_rest_detaches_lobby_player(
    client: AsyncRestClient,
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)
    attach_resp = await client.post(
        f"/api/game/{game_id}/seat/b?user_id=user-1"
    )

    detach_resp = await client.delete(
        f"/api/game/{game_id}/seat/b?user_id=user-1"
    )
    response = await client.get("/api/game?user_id=user-1")

    assert attach_resp.status_code == 200
    assert detach_resp.status_code == 200
    data = response.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    matching = [
        game for game in games_raw if game["game_id"] == game_id
    ]
    assert len(matching) == 1
    assert matching[0]["user_count"] == 0
    assert matching[0]["user_seats"] == []
    players_raw = matching[0]["seats"]
    assert _is_list_of_dict(players_raw)
    assert players_raw[1]["occupied"] is False
    assert players_raw[1]["mine"] is False


@pytest.mark.asyncio
async def test_attach_player_rest_rejects_occupied_player(
    client: AsyncRestClient,
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)
    first_resp = await client.post(
        f"/api/game/{game_id}/seat/b?user_id=user-1"
    )

    second_resp = await client.post(
        f"/api/game/{game_id}/seat/b?user_id=user-other"
    )

    assert first_resp.status_code == 200
    assert second_resp.status_code == 409
    data = second_resp.json()
    assert _is_dict(data)
    assert data["ok"] is False
    assert data["error"] == "seat occupied"


@pytest.mark.asyncio
async def test_fill_bot_players_rest_fills_remaining_with_auto(
    client: AsyncRestClient,
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)
    attach_resp = await client.post(
        f"/api/game/{game_id}/seat/c?user_id=user-2"
    )

    fill_resp = await client.post(
        f"/api/game/{game_id}/bots?kind=auto&user_id=user-2"
    )
    list_resp = await client.get("/api/game?user_id=user-2")

    assert attach_resp.status_code == 200
    assert fill_resp.status_code == 200
    fill_data = fill_resp.json()
    assert _is_dict(fill_data)
    assert fill_data["ok"] is True
    data = list_resp.json()
    assert _is_dict(data)
    games_raw = data["games"]
    assert _is_list_of_dict(games_raw)
    matching = [
        game for game in games_raw if game["game_id"] == game_id
    ]
    assert len(matching) == 1
    assert matching[0]["user_count"] == 1
    assert matching[0]["user_seats"] == ["c"]
    players_raw = matching[0]["seats"]
    assert _is_list_of_dict(players_raw)
    assert [player["kind"] for player in players_raw] == [
        "auto",
        "auto",
        "user",
        "auto",
    ]
    assert all(player["occupied"] is True for player in players_raw)
    assert players_raw[2]["mine"] is True


@pytest.mark.asyncio
async def test_fill_bot_players_rest_requires_attached_user(
    client: AsyncRestClient,
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)

    fill_resp = await client.post(
        f"/api/game/{game_id}/bots?kind=auto&user_id=user-x"
    )

    assert fill_resp.status_code == 409
    data = fill_resp.json()
    assert _is_dict(data)
    assert data["ok"] is False
    assert data["error"] == "user does not occupy a seat"


@pytest.mark.asyncio
async def test_fill_bot_players_rest_rejects_invalid_kind(
    client: AsyncRestClient,
) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)

    fill_resp = await client.post(
        f"/api/game/{game_id}/bots?kind=random&user_id=user-x"
    )

    assert fill_resp.status_code == 400
    data = fill_resp.json()
    assert _is_dict(data)
    assert data["ok"] is False
    assert data["error"] == "invalid bot kind"


# ---- REST: Delete Game ----


@pytest.mark.asyncio
async def test_delete_game_returns_200(client: AsyncRestClient) -> None:
    create_resp = await client.post("/api/game")
    game_id = _game_id_from(create_resp)
    response = await client.delete(f"/api/game/{game_id}")
    assert response.status_code == 200
    del_data = response.json()
    assert _is_dict(del_data)
    assert del_data["ok"] is True


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_200(
    client: AsyncRestClient,
) -> None:
    """DELETE is idempotent -- returns 200 even for unknown game_id."""
    response = await client.delete("/api/game/nonexistent123")
    assert response.status_code == 200
    data = response.json()
    assert _is_dict(data)
    assert data["ok"] is True
