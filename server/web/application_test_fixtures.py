"""Fresh :class:`WebApplication` ownership for black-box web tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

import httpx
import pytest
from starlette.testclient import TestClient

from server.web.app import WebApplication
from server.web.application_test_client import (
    AsyncRestClient,
    SyncServerClient,
)


@pytest.fixture
def web_application() -> WebApplication:
    """Create state and routes owned only by the current test."""
    return WebApplication()


@pytest.fixture
async def client(
    web_application: WebApplication,
) -> AsyncGenerator[AsyncRestClient, None]:
    """Expose the application's public HTTP interface asynchronously."""
    transport = httpx.ASGITransport(app=web_application.asgi)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as async_client:
        yield async_client


@pytest.fixture
def sync_client(
    web_application: WebApplication,
) -> Generator[SyncServerClient, None, None]:
    """Expose the application's public HTTP and WebSocket interfaces."""
    with TestClient(
        web_application.asgi,
        backend_options={"use_uvloop": True},
    ) as test_client:
        yield test_client
