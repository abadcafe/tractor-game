"""Black-box tests for the LLM transcript debug HTTP API."""

from server.web.application_test_client import (
    SyncServerClient,
)
from server.web.application_test_client import (
    game_id_from as _game_id_from,
)

pytest_plugins: tuple[str, ...] = (
    "server.web.application_test_fixtures",
)


def test_llm_debug_page_returns_html(
    sync_client: SyncServerClient,
) -> None:
    create_response = sync_client.post("/api/game")
    game_id = _game_id_from(create_response)

    response = sync_client.get(f"/debug/llm/{game_id}?seat=a")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "LLM Transcript" in response.text
    assert "/llm-debug/style.css" in response.text
    assert "/llm-debug/main.js" in response.text
    assert game_id not in response.text


def test_llm_debug_page_missing_game_returns_404(
    sync_client: SyncServerClient,
) -> None:
    response = sync_client.get("/debug/llm/not-a-game?seat=a")

    assert response.status_code == 404
