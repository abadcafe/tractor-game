"""Explicit static routes for the game and training SPAs."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response

_NO_STORE_HEADERS: dict[str, str] = {"Cache-Control": "no-store"}


def register_static_routes(app: FastAPI, static_dir: str) -> None:
    root = Path(static_dir).resolve(strict=False)

    async def game_root() -> Response:
        return _spa_response(root, "game", "")

    async def game_asset(path: str) -> Response:
        return _spa_response(root, "game", path)

    async def training_root() -> Response:
        return _spa_response(root, "training", "")

    async def training_asset(path: str) -> Response:
        return _spa_response(root, "training", path)

    async def ai_debug_asset(path: str) -> Response:
        return _asset_response(root, "ai-debug", path)

    async def shared_browser_asset(path: str) -> Response:
        return _asset_response(root, "browser", path)

    app.add_api_route("/", game_root, methods=["GET"])
    app.add_api_route("/game", game_root, methods=["GET"])
    app.add_api_route("/game/{path:path}", game_asset, methods=["GET"])
    app.add_api_route("/training", training_root, methods=["GET"])
    app.add_api_route(
        "/training/{path:path}", training_asset, methods=["GET"]
    )
    app.add_api_route(
        "/ai-debug/{path:path}", ai_debug_asset, methods=["GET"]
    )
    app.add_api_route(
        "/browser/{path:path}", shared_browser_asset, methods=["GET"]
    )


def _spa_response(root: Path, application: str, path: str) -> Response:
    asset = _safe_asset(root, application, path)
    if asset is None:
        return Response(status_code=403, content="Forbidden")
    if asset.is_file():
        return _frontend_file_response(asset)
    index = root / application / "index.html"
    if index.is_file():
        return _frontend_file_response(index)
    return Response(
        status_code=404,
        content="Frontend not built. Run: deno task build",
    )


def _asset_response(
    root: Path, application: str, path: str
) -> Response:
    asset = _safe_asset(root, application, path)
    if asset is None:
        return Response(status_code=403, content="Forbidden")
    if asset.is_file():
        return _frontend_file_response(asset)
    return Response(status_code=404, content="Not found")


def _safe_asset(root: Path, application: str, path: str) -> Path | None:
    application_root = (root / application).resolve(strict=False)
    candidate = (application_root / path).resolve(strict=False)
    if not candidate.is_relative_to(application_root):
        return None
    return candidate


def _frontend_file_response(path: Path) -> FileResponse:
    return FileResponse(path, headers=_NO_STORE_HEADERS)
