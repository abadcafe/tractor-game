"""Black-box tests for the web application static routes."""

import pytest

from server.web.application_test_client import (
    AsyncRestClient,
    SyncServerClient,
)

pytest_plugins: tuple[str, ...] = (
    "server.web.application_test_fixtures",
)


def test_game_spa_owns_root_and_deep_game_routes(
    sync_client: SyncServerClient,
) -> None:
    root = sync_client.get("/")
    deep = sync_client.get("/game/example/player/1?user_id=user-1")

    assert root.status_code == 200
    assert deep.status_code == 200
    assert "/game/style.css" in deep.text
    assert "/game/main.js?v=" in deep.text


def test_training_spa_is_independent_from_game_routes(
    sync_client: SyncServerClient,
) -> None:
    response = sync_client.get("/training/metrics")

    assert response.status_code == 200
    assert "Training Console" in response.text
    assert "/training/main.js?v=" in response.text


def test_static_assets_are_scoped_by_application(
    sync_client: SyncServerClient,
) -> None:
    game = sync_client.get("/game/config.js")
    training = sync_client.get("/training/main.js")

    assert game.status_code == 200
    assert training.status_code == 200
    assert game.headers.get("cache-control") == "no-store"
    assert training.headers.get("cache-control") == "no-store"


@pytest.mark.asyncio
async def test_path_traversal_encoded_returns_403(
    client: AsyncRestClient,
) -> None:
    """
    Encoded path traversal attempts bypass ASGI normalization and are
    blocked by
    the server's path traversal protection with 403."""
    response = await client.get("/training/..%2F..%2F..%2Fetc/passwd")
    assert response.status_code == 403
