"""Pytest configuration for E2E tests."""
import pytest
import subprocess
import time
import requests


def _is_server_running(url: str) -> bool:
    """Check if the game server is already running."""
    try:
        resp = requests.get(f"{url}/api/health", timeout=2)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


@pytest.fixture(scope="session")
def live_server():
    """Start the FastAPI server for E2E tests, or reuse an existing one."""
    base_url = "http://127.0.0.1:8787"

    if _is_server_running(base_url):
        # Server already running — reuse it without starting/stopping
        yield base_url
        return

    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "server.server:app", "--host", "127.0.0.1", "--port", "8787"],
        cwd="/home/lfw/works/tractor-game",
    )
    # Wait for server to be ready
    for _ in range(30):
        try:
            resp = requests.get(f"{base_url}/api/health", timeout=1)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    yield base_url
    proc.terminate()
    proc.wait()
