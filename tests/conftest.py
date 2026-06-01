"""Pytest configuration for E2E tests."""
import pytest
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _is_server_running(url: str) -> bool:
    """Check if the game server is already running."""
    try:
        resp = urlopen(Request(f"{url}/api/health"), timeout=2)
        return resp.status == 200
    except (URLError, OSError):
        return False


@pytest.fixture(scope="session")
def live_server():
    """Start the FastAPI server for E2E tests, or reuse an existing one.

    Reuse logic is necessary because a previous test session may have left
    a server running on the same port. Without reuse, subprocess.Popen
    would fail with EADDRINUSE. Each E2E test creates its own game via the
    API, so stale server state does not affect test isolation.
    """
    base_url = "http://127.0.0.1:8787"

    if _is_server_running(base_url):
        yield base_url
        return

    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "server.server:app", "--host", "127.0.0.1", "--port", "8787"],
        cwd=PROJECT_ROOT,
    )
    # Wait for server to be ready (check /docs endpoint per spec)
    for _ in range(30):
        try:
            resp = urlopen(Request(f"{base_url}/docs"), timeout=1)
            if resp.status == 200:
                break
        except (URLError, OSError):
            pass
        time.sleep(0.5)
    yield base_url
    proc.terminate()
    proc.wait()
