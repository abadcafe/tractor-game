"""Pytest configuration for E2E tests."""

import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


@pytest.fixture(scope="session")
def live_server() -> Generator[str, None, None]:
    """Start the FastAPI server for E2E tests."""
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "server.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8787",
        ],
        cwd=PROJECT_ROOT,
        stderr=subprocess.PIPE,
    )
    # Wait for server to be ready (check /docs endpoint per spec)
    server_ready = False
    for _ in range(30):
        # Check if process exited prematurely
        if proc.poll() is not None:
            stderr_output = (
                proc.stderr.read().decode() if proc.stderr else ""
            )
            raise RuntimeError(
                f"Server process exited prematurely with code"
                f"{proc.returncode}."
                f"Stderr: {stderr_output}"
            )
        try:
            resp = urlopen(
                Request("http://127.0.0.1:8787/docs"), timeout=1
            )
            if resp.status == 200:
                server_ready = True
                break
        except URLError, OSError:
            pass
        time.sleep(0.5)
    if not server_ready:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        stderr_output = (
            proc.stderr.read().decode() if proc.stderr else ""
        )
        raise RuntimeError(
            f"Server failed to start within 15 seconds. Stderr:"
            f"{stderr_output}"
        )
    yield "http://127.0.0.1:8787"
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
