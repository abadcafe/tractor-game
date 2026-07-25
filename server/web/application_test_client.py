"""Typed clients and JSON narrowing for black-box web tests."""

from __future__ import annotations

from typing import Protocol, TypeGuard

import httpx
from starlette.testclient import WebSocketTestSession


class AsyncRestClient(Protocol):
    async def get(self, url: str) -> httpx.Response: ...

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,
    ) -> httpx.Response: ...

    async def delete(self, url: str) -> httpx.Response: ...


class SyncServerClient(Protocol):
    def get(self, url: str) -> httpx.Response: ...

    def post(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,
    ) -> httpx.Response: ...

    def delete(self, url: str) -> httpx.Response: ...

    def websocket_connect(self, url: str) -> WebSocketTestSession: ...


def is_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Narrow decoded JSON to an object with string keys."""
    return isinstance(value, dict)


def is_list_of_dict(
    value: object,
) -> TypeGuard[list[dict[str, object]]]:
    """Narrow decoded JSON to a list of objects with string keys."""
    return isinstance(value, list)


def game_id_from(response: httpx.Response) -> str:
    """Extract the public identifier returned by game creation."""
    document = response.json()
    assert is_dict(document)
    game_id = document["game_id"]
    assert isinstance(game_id, str)
    return game_id
