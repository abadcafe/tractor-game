"""Application lifecycle boundary for training event streams."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import final


@final
class EventStreamLifecycle:
    """End wrapped event streams when application shutdown begins."""

    def __init__(self) -> None:
        self._closed = asyncio.Event()

    def close(self) -> None:
        """Permanently close this lifecycle."""
        self._closed.set()

    def stream(
        self, source: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        """Wrap a byte source so closure produces normal EOF."""
        return self._stream(source)

    async def _stream(
        self, source: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        if self._closed.is_set():
            await source.aclose()
            return

        closed = asyncio.create_task(self._closed.wait())
        next_frame = asyncio.create_task(anext(source))
        try:
            while True:
                completed, _pending = await asyncio.wait(
                    (next_frame, closed),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if closed in completed or self._closed.is_set():
                    if next_frame.done():
                        try:
                            _ = next_frame.result()
                        except StopAsyncIteration:
                            return
                    else:
                        _ = next_frame.cancel()
                        with suppress(asyncio.CancelledError):
                            await next_frame
                    return

                try:
                    frame = next_frame.result()
                except StopAsyncIteration:
                    return
                yield frame
                if self._closed.is_set():
                    return
                next_frame = asyncio.create_task(anext(source))
        finally:
            if not next_frame.done():
                _ = next_frame.cancel()
                with suppress(asyncio.CancelledError):
                    await next_frame
            _ = closed.cancel()
            with suppress(asyncio.CancelledError):
                await closed
            await source.aclose()
