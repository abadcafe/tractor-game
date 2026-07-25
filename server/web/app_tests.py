"""Black-box tests for the composed web application."""

import pytest

from server.web.application_test_client import (
    AsyncRestClient,
)
from server.web.application_test_client import (
    is_dict as _is_dict,
)

pytest_plugins: tuple[str, ...] = (
    "server.web.application_test_fixtures",
)


@pytest.mark.asyncio
async def test_health_endpoint(
    client: AsyncRestClient,
) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert _is_dict(data)
    assert data["status"] == "ok"
