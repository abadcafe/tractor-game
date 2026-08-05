"""Black-box tests for the training-event stream lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from server.web.training_events.lifecycle import EventStreamLifecycle

_COMPLETION_TIMEOUT_SECONDS = 0.25


async def _completed_next(
    stream: AsyncGenerator[bytes, None],
) -> asyncio.Task[bytes]:
    task = asyncio.create_task(anext(stream))
    _done, pending = await asyncio.wait(
        (task,), timeout=_COMPLETION_TIMEOUT_SECONDS
    )
    assert not pending
    return task


async def test_stream_forwards_source_frames() -> None:
    async def source() -> AsyncGenerator[bytes, None]:
        yield b"first"
        yield b"second"

    lifecycle = EventStreamLifecycle()
    stream = lifecycle.stream(source())

    assert await anext(stream) == b"first"
    assert await anext(stream) == b"second"
    completed = await _completed_next(stream)
    assert isinstance(completed.exception(), StopAsyncIteration)


async def test_stream_ends_when_source_ends() -> None:
    async def source() -> AsyncGenerator[bytes, None]:
        items: tuple[bytes, ...] = ()
        for item in items:
            yield item

    lifecycle = EventStreamLifecycle()
    completed = await _completed_next(lifecycle.stream(source()))

    assert isinstance(completed.exception(), StopAsyncIteration)


async def test_close_ends_waiting_stream() -> None:
    source_waiting = asyncio.Event()
    never = asyncio.Event()

    async def source() -> AsyncGenerator[bytes, None]:
        source_waiting.set()
        _ = await never.wait()
        yield b"unreachable"

    lifecycle = EventStreamLifecycle()
    stream = lifecycle.stream(source())
    pending_frame = asyncio.create_task(anext(stream))
    _ = await source_waiting.wait()

    lifecycle.close()

    _done, pending = await asyncio.wait(
        (pending_frame,), timeout=_COMPLETION_TIMEOUT_SECONDS
    )
    assert not pending
    assert isinstance(pending_frame.exception(), StopAsyncIteration)


async def test_close_ends_all_active_streams() -> None:
    source_waiting = [asyncio.Event() for _index in range(4)]
    never = asyncio.Event()

    async def source(index: int) -> AsyncGenerator[bytes, None]:
        source_waiting[index].set()
        _ = await never.wait()
        yield b"unreachable"

    lifecycle = EventStreamLifecycle()
    streams = [lifecycle.stream(source(index)) for index in range(4)]
    pending_frames = [
        asyncio.create_task(anext(stream)) for stream in streams
    ]
    _ = await asyncio.gather(
        *(waiting.wait() for waiting in source_waiting)
    )

    lifecycle.close()

    _done, pending = await asyncio.wait(
        pending_frames, timeout=_COMPLETION_TIMEOUT_SECONDS
    )
    assert not pending
    assert all(
        isinstance(task.exception(), StopAsyncIteration)
        for task in pending_frames
    )


async def test_close_discards_frame_completed_in_same_turn() -> None:
    source_waiting = asyncio.Event()
    release_source = asyncio.Event()

    async def source() -> AsyncGenerator[bytes, None]:
        source_waiting.set()
        _ = await release_source.wait()
        yield b"must-not-be-sent"

    lifecycle = EventStreamLifecycle()
    stream = lifecycle.stream(source())
    pending_frame = asyncio.create_task(anext(stream))
    _ = await source_waiting.wait()

    release_source.set()
    lifecycle.close()

    _done, pending = await asyncio.wait(
        (pending_frame,), timeout=_COMPLETION_TIMEOUT_SECONDS
    )
    assert not pending
    assert isinstance(pending_frame.exception(), StopAsyncIteration)


async def test_stream_after_close_does_not_start_source() -> None:
    source_started = False

    async def source() -> AsyncGenerator[bytes, None]:
        nonlocal source_started
        source_started = True
        yield b"unreachable"

    lifecycle = EventStreamLifecycle()
    lifecycle.close()

    completed = await _completed_next(lifecycle.stream(source()))

    assert isinstance(completed.exception(), StopAsyncIteration)
    assert not source_started


async def test_close_is_idempotent() -> None:
    lifecycle = EventStreamLifecycle()

    lifecycle.close()
    lifecycle.close()

    async def source() -> AsyncGenerator[bytes, None]:
        yield b"unreachable"

    completed = await _completed_next(lifecycle.stream(source()))
    assert isinstance(completed.exception(), StopAsyncIteration)


async def test_stream_close_finalizes_source_once() -> None:
    source_finalizations = 0

    async def source() -> AsyncGenerator[bytes, None]:
        nonlocal source_finalizations
        try:
            yield b"first"
            _ = await asyncio.Event().wait()
        finally:
            source_finalizations += 1

    lifecycle = EventStreamLifecycle()
    stream = lifecycle.stream(source())
    assert await anext(stream) == b"first"

    await stream.aclose()
    await stream.aclose()

    assert source_finalizations == 1


async def test_lifecycle_close_finalizes_source_once() -> None:
    source_waiting = asyncio.Event()
    source_finalizations = 0

    async def source() -> AsyncGenerator[bytes, None]:
        nonlocal source_finalizations
        try:
            yield b"first"
            source_waiting.set()
            _ = await asyncio.Event().wait()
        finally:
            source_finalizations += 1

    lifecycle = EventStreamLifecycle()
    stream = lifecycle.stream(source())
    assert await anext(stream) == b"first"
    pending_frame = asyncio.create_task(anext(stream))
    _ = await source_waiting.wait()

    lifecycle.close()

    _done, pending = await asyncio.wait(
        (pending_frame,), timeout=_COMPLETION_TIMEOUT_SECONDS
    )
    assert not pending
    assert isinstance(pending_frame.exception(), StopAsyncIteration)
    assert source_finalizations == 1


async def test_stream_propagates_source_programming_error() -> None:
    async def source() -> AsyncGenerator[bytes, None]:
        yield b"first"
        raise RuntimeError("source failed")

    lifecycle = EventStreamLifecycle()
    stream = lifecycle.stream(source())
    assert await anext(stream) == b"first"

    failed = await _completed_next(stream)

    error = failed.exception()
    assert isinstance(error, RuntimeError)
    assert str(error) == "source failed"


async def test_close_propagates_simultaneous_source_error() -> None:
    source_waiting = asyncio.Event()
    release_source = asyncio.Event()

    async def source() -> AsyncGenerator[bytes, None]:
        source_waiting.set()
        _ = await release_source.wait()
        items: tuple[bytes, ...] = ()
        for item in items:
            yield item
        raise RuntimeError("source failed during close")

    lifecycle = EventStreamLifecycle()
    stream = lifecycle.stream(source())
    failed = asyncio.create_task(anext(stream))
    _ = await source_waiting.wait()

    release_source.set()
    lifecycle.close()

    _done, pending = await asyncio.wait(
        (failed,), timeout=_COMPLETION_TIMEOUT_SECONDS
    )
    assert not pending
    error = failed.exception()
    assert isinstance(error, RuntimeError)
    assert str(error) == "source failed during close"
