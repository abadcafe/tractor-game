"""Black-box tests for the AI transcript debug HTTP API."""

from server.web.application_test_client import (
    SyncServerClient,
)
from server.web.application_test_client import (
    game_id_from as _game_id_from,
)

pytest_plugins: tuple[str, ...] = (
    "server.web.application_test_fixtures",
)


def test_ai_debug_page_returns_html(
    sync_client: SyncServerClient,
) -> None:
    create_response = sync_client.post("/api/game")
    game_id = _game_id_from(create_response)

    response = sync_client.get(f"/debug/ai/{game_id}?player=0")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Transcript" in response.text
    assert "/ai-debug/style.css" in response.text
    assert "/ai-debug/main.js" in response.text
    assert game_id not in response.text


def test_ai_debug_page_missing_game_returns_404(
    sync_client: SyncServerClient,
) -> None:
    response = sync_client.get("/debug/ai/not-a-game?player=0")

    assert response.status_code == 404
