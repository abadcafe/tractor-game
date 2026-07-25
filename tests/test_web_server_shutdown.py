"""Process-level shutdown tests for the production web server."""

from __future__ import annotations

import asyncio
import signal
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path
from urllib.parse import urlencode

import httpx
from pydantic import TypeAdapter

_HOST = "127.0.0.1"
_INET_ADDRESS = TypeAdapter(tuple[str, int])
_EXIT_TIMEOUT_SECONDS = 3.0
_STREAM_EOF_TIMEOUT_SECONDS = 2.0
_FORBIDDEN_LOG_OUTPUT = (
    "Waiting for connections to close",
    "Waiting for background tasks to complete",
    "timeout graceful shutdown exceeded",
    "CTRL+C to force quit",
    "CancelledError",
    "Exception in ASGI application",
    "Traceback",
    "ERROR",
)


async def test_server_sigint_closes_all_event_streams(
    tmp_path: Path,
) -> None:
    await _assert_server_shutdown(
        tmp_path,
        shutdown_signal=signal.SIGINT,
        expected_returncode=0,
    )


async def test_server_sigterm_closes_all_event_streams(
    tmp_path: Path,
) -> None:
    await _assert_server_shutdown(
        tmp_path,
        shutdown_signal=signal.SIGTERM,
        expected_returncode=-signal.SIGTERM,
    )


async def _assert_server_shutdown(
    tmp_path: Path,
    *,
    shutdown_signal: signal.Signals,
    expected_returncode: int,
) -> None:
    port = _unused_port()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = tmp_path / "server.log"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        (
            sys.executable,
            "-m",
            "server.web",
            "--host",
            _HOST,
            "--port",
            str(port),
        ),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://{_HOST}:{port}"
    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(1.0),
        ) as client:
            await _wait_until_ready(client, process)
            async with AsyncExitStack() as stack:
                streams = await _open_event_streams(
                    stack,
                    client,
                    run_dir=run_dir,
                )

                process.send_signal(shutdown_signal)

                await asyncio.wait_for(
                    asyncio.gather(
                        *(_drain_to_eof(stream) for stream in streams)
                    ),
                    timeout=_STREAM_EOF_TIMEOUT_SECONDS,
                )
                returncode = await _wait_for_exit(process)
                assert returncode == expected_returncode
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
        log_file.close()

    output = log_path.read_text(encoding="utf-8")
    assert not any(
        forbidden in output for forbidden in _FORBIDDEN_LOG_OUTPUT
    ), output


async def _wait_until_ready(
    client: httpx.AsyncClient,
    process: subprocess.Popen[str],
) -> None:
    deadline = asyncio.get_running_loop().time() + 10.0
    while asyncio.get_running_loop().time() < deadline:
        assert process.poll() is None, (
            "web server exited during startup"
        )
        try:
            response = await client.get("/health")
        except httpx.HTTPError:
            await asyncio.sleep(0.05)
            continue
        if response.status_code == 200:
            return
        await asyncio.sleep(0.05)
    assert False, "web server did not become ready"


async def _open_event_streams(
    stack: AsyncExitStack,
    client: httpx.AsyncClient,
    *,
    run_dir: Path,
) -> tuple[AsyncIterator[str], ...]:
    query = urlencode({"run_dir": str(run_dir)})
    paths = (
        f"/api/training/events/process?{query}",
        f"/api/training/events/metrics?{query}",
        f"/api/training/events/logs?{query}",
        f"/api/training/events/checkpoints?{query}",
    )
    streams: list[AsyncIterator[str]] = []
    for path in paths:
        response = await stack.enter_async_context(
            client.stream("GET", path, timeout=None)
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )
        stream = response.aiter_lines()
        assert await anext(stream) == "retry: 1000"
        streams.append(stream)
    return tuple(streams)


async def _drain_to_eof(stream: AsyncIterator[str]) -> None:
    async for _line in stream:
        pass


async def _wait_for_exit(
    process: subprocess.Popen[str],
) -> int:
    deadline = asyncio.get_running_loop().time() + _EXIT_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        returncode = process.poll()
        if returncode is not None:
            return returncode
        await asyncio.sleep(0.02)
    assert False, "web server did not exit after one signal"


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((_HOST, 0))
        address = _INET_ADDRESS.validate_python(listener.getsockname())
    return address[1]
