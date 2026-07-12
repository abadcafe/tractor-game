"""Async length-prefixed socket frames with explicit backpressure."""

from __future__ import annotations

import asyncio
import select
import socket
import struct
from collections.abc import Buffer
from dataclasses import dataclass, field

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected

_FRAME_HEADER_BYTES = 8
_DEFAULT_MAX_FRAME_BYTES = 2_147_483_647


@dataclass(frozen=True, slots=True)
class AsyncSocketPair:
    """Two connected async frame endpoints."""

    first: AsyncFrameEndpoint
    second: AsyncFrameEndpoint


@dataclass(slots=True)
class AsyncFrameEndpoint:
    """One full-duplex async frame endpoint backed by a socket."""

    socket: socket.socket
    max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES
    _send_lock: asyncio.Lock | None = field(
        init=False, default=None, repr=False
    )
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        assert self.max_frame_bytes > 0
        self.socket.setblocking(False)

    async def send_frame(
        self, payload: Buffer
    ) -> _result.Ok[None] | _result.Rejected:
        """Send one frame with socket-level backpressure."""
        if self._closed:
            return Rejected(reason="async IPC endpoint is closed")
        payload_view = memoryview(payload)
        payload_size = payload_view.nbytes
        if payload_size > self.max_frame_bytes:
            return Rejected(reason="async IPC frame exceeds limit")
        lock = self._lock()
        async with lock:
            try:
                loop = asyncio.get_running_loop()
                header = struct.pack(">Q", payload_size)
                await loop.sock_sendall(self.socket, header)
                if payload_size > 0:
                    await loop.sock_sendall(
                        self.socket, payload_view.cast("B")
                    )
            except OSError as exc:
                return Rejected(
                    reason=f"async IPC frame send failed: {exc}"
                )
        return Ok(value=None)

    async def recv_frame(
        self, *, timeout_seconds: float | None = None
    ) -> _result.Ok[bytes] | _result.Rejected:
        """Receive one complete frame."""
        if timeout_seconds is not None:
            assert timeout_seconds >= 0.0
        try:
            if timeout_seconds is None:
                return await self._recv_frame_unbounded()
            return await asyncio.wait_for(
                self._recv_frame_unbounded(),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            self.close()
            return Rejected(reason="async IPC frame receive timed out")

    async def recv_frame_into(
        self, buffer: memoryview
    ) -> _result.Ok[int] | _result.Rejected:
        """Receive one frame payload into a caller-owned buffer."""
        header_result = await self._read_exact_bytes(
            _FRAME_HEADER_BYTES
        )
        if isinstance(header_result, Rejected):
            return header_result
        payload_size = struct.unpack(">Q", header_result.value)[0]
        if payload_size > self.max_frame_bytes:
            return Rejected(reason="async IPC frame exceeds limit")
        if payload_size > buffer.nbytes:
            return Rejected(reason="async IPC frame exceeds slot")
        read_result = await self._read_exact_into(
            buffer=buffer,
            byte_count=payload_size,
        )
        if isinstance(read_result, Rejected):
            return read_result
        return Ok(value=payload_size)

    def is_readable(self) -> bool:
        """Return whether a frame can be started without blocking."""
        if self._closed:
            return False
        readable, _writable, _errors = select.select(
            (self.socket,), (), (), 0.0
        )
        return bool(readable)

    async def wait_readable(
        self, *, timeout_seconds: float | None = None
    ) -> _result.Ok[bool] | _result.Rejected:
        """Wait until the endpoint becomes readable."""
        if timeout_seconds is not None:
            assert timeout_seconds >= 0.0
        if self.is_readable():
            return Ok(value=True)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def mark_ready() -> None:
            if not future.done():
                future.set_result(None)

        try:
            loop.add_reader(self.socket.fileno(), mark_ready)
            if timeout_seconds is None:
                await future
            else:
                await asyncio.wait_for(future, timeout=timeout_seconds)
        except TimeoutError:
            return Ok(value=False)
        except OSError as exc:
            return Rejected(
                reason=f"async IPC readability wait failed: {exc}"
            )
        finally:
            loop.remove_reader(self.socket.fileno())
        return Ok(value=True)

    def close(self) -> None:
        """Close this endpoint."""
        if self._closed:
            return
        self._closed = True
        self.socket.close()

    def _lock(self) -> asyncio.Lock:
        lock = self._send_lock
        if lock is None:
            lock = asyncio.Lock()
            self._send_lock = lock
        return lock

    async def _recv_frame_unbounded(
        self,
    ) -> _result.Ok[bytes] | _result.Rejected:
        header_result = await self._read_exact_bytes(
            _FRAME_HEADER_BYTES
        )
        if isinstance(header_result, Rejected):
            return header_result
        payload_size = struct.unpack(">Q", header_result.value)[0]
        if payload_size > self.max_frame_bytes:
            return Rejected(reason="async IPC frame exceeds limit")
        payload_result = await self._read_exact_bytes(payload_size)
        if isinstance(payload_result, Rejected):
            return payload_result
        return Ok(value=payload_result.value)

    async def _read_exact_bytes(
        self, byte_count: int
    ) -> _result.Ok[bytes] | _result.Rejected:
        assert byte_count >= 0
        chunks: list[bytes] = []
        remaining = byte_count
        loop = asyncio.get_running_loop()
        while remaining > 0:
            try:
                chunk = await loop.sock_recv(self.socket, remaining)
            except OSError as exc:
                return Rejected(
                    reason=f"async IPC frame receive failed: {exc}"
                )
            if not chunk:
                return Rejected(reason="async IPC endpoint closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return Ok(value=b"".join(chunks))

    async def _read_exact_into(
        self, *, buffer: memoryview, byte_count: int
    ) -> _result.Ok[None] | _result.Rejected:
        assert byte_count >= 0
        view = buffer.cast("B")
        remaining = byte_count
        offset = 0
        loop = asyncio.get_running_loop()
        while remaining > 0:
            try:
                read_count = await loop.sock_recv_into(
                    self.socket,
                    view[offset : offset + remaining],
                )
            except OSError as exc:
                return Rejected(
                    reason=f"async IPC frame receive failed: {exc}"
                )
            if read_count == 0:
                return Rejected(reason="async IPC endpoint closed")
            offset += read_count
            remaining -= read_count
        return Ok(value=None)


def create_async_socket_pair(
    *, max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES
) -> AsyncSocketPair:
    """Create one connected async socket pair."""
    assert max_frame_bytes > 0
    first_socket, second_socket = socket.socketpair()
    return AsyncSocketPair(
        first=AsyncFrameEndpoint(
            socket=first_socket,
            max_frame_bytes=max_frame_bytes,
        ),
        second=AsyncFrameEndpoint(
            socket=second_socket,
            max_frame_bytes=max_frame_bytes,
        ),
    )


async def wait_readable_frames(
    *,
    endpoints: tuple[AsyncFrameEndpoint, ...],
    timeout_seconds: float | None,
) -> _result.Ok[tuple[AsyncFrameEndpoint, ...]] | _result.Rejected:
    """Wait until at least one endpoint is readable."""
    assert endpoints
    if timeout_seconds is not None:
        assert timeout_seconds >= 0.0
    ready = tuple(
        endpoint for endpoint in endpoints if endpoint.is_readable()
    )
    if ready:
        return Ok(value=ready)
    if timeout_seconds == 0.0:
        return Ok(value=())
    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()

    def mark_ready() -> None:
        if not future.done():
            future.set_result(None)

    try:
        for endpoint in endpoints:
            loop.add_reader(endpoint.socket.fileno(), mark_ready)
        if timeout_seconds is None:
            await future
        else:
            await asyncio.wait_for(future, timeout=timeout_seconds)
    except TimeoutError:
        return Ok(value=())
    except OSError as exc:
        return Rejected(reason=f"async IPC input wait failed: {exc}")
    finally:
        for endpoint in endpoints:
            loop.remove_reader(endpoint.socket.fileno())
    return Ok(
        value=tuple(
            endpoint for endpoint in endpoints if endpoint.is_readable()
        )
    )
