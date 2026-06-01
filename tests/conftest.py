"""Pytest configuration for E2E tests."""
import pytest
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


@pytest.fixture(scope="session")
def live_server():
    """Start the FastAPI server for E2E tests."""
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "server.server:app", "--host", "127.0.0.1", "--port", "8787"],
        cwd=PROJECT_ROOT,
    )
    # Wait for server to be ready (check /docs endpoint per spec)
    for _ in range(30):
        try:
            resp = urlopen(Request(f"http://127.0.0.1:8787/docs"), timeout=1)
            if resp.status == 200:
                break
        except (URLError, OSError):
            pass
        time.sleep(0.5)
    yield "http://127.0.0.1:8787"
    proc.terminate()
    proc.wait()
