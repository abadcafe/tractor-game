"""Static frontend file routes."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response

_NO_STORE_HEADERS: dict[str, str] = {"Cache-Control": "no-store"}


def register_static_routes(app: FastAPI, static_dir: str) -> None:
    async def index() -> Response:
        html_path = os.path.join(static_dir, "index.html")
        if os.path.isfile(html_path):
            return _frontend_file_response(html_path)
        return Response(
            status_code=404,
            content="Frontend not built. Run: deno task build",
        )

    async def serve_static(path: str) -> Response:
        if path.startswith("api/") or path.startswith("ws/"):
            return Response(status_code=404, content="Not found")
        file_path = os.path.normpath(os.path.join(static_dir, path))
        if (
            not file_path.startswith(static_dir + os.sep)
            and file_path != static_dir
        ):
            return Response(status_code=403, content="Forbidden")
        if os.path.isfile(file_path):
            return _frontend_file_response(file_path)
        html_path = os.path.join(static_dir, "index.html")
        if os.path.isfile(html_path):
            return _frontend_file_response(html_path)
        return Response(status_code=404, content="Not found")

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/{path:path}", serve_static, methods=["GET"])


def _frontend_file_response(path: str) -> FileResponse:
    return FileResponse(path, headers=_NO_STORE_HEADERS)
