"""SSE framing shared by training dashboard event endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict
from starlette.responses import StreamingResponse

type EventName = Literal[
    "process",
    "metrics",
    "log",
    "invalidation",
    "replacement",
    "rejected",
]

RETRY = b"retry: 1000\n\n"
KEEP_ALIVE = b": keep-alive\n\n"

_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


class RejectedEvent(BaseModel):
    """Terminal stream rejection sent before closing the response."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    error: str


@dataclass(frozen=True, slots=True)
class ServerEvent:
    """One named UTF-8 event in the SSE wire format."""

    name: EventName
    data: str
    event_id: str | None = None

    def encode(self) -> bytes:
        assert "\n" not in self.name and "\r" not in self.name
        lines = [f"event: {self.name}"]
        if self.event_id is not None:
            assert "\n" not in self.event_id
            assert "\r" not in self.event_id
            lines.append(f"id: {self.event_id}")
        data_lines = self.data.splitlines() or [""]
        lines.extend(f"data: {line}" for line in data_lines)
        return ("\n".join(lines) + "\n\n").encode("utf-8")


def rejected_event(reason: str) -> ServerEvent:
    """Build the common terminal rejection event."""
    return ServerEvent(
        name="rejected",
        data=RejectedEvent(error=reason).model_dump_json(),
    )


def event_response(content: AsyncIterable[bytes]) -> StreamingResponse:
    """Return an unbuffered SSE response for an async byte iterator."""
    return StreamingResponse(
        content,
        media_type="text/event-stream",
        headers=_HEADERS,
    )
