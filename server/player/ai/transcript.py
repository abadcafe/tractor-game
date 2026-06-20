"""In-memory debug transcript for one AI player."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypedDict


class TranscriptRecordDict(TypedDict):
    id: int
    event_id: int
    created_at: str
    player_index: int
    seq: int
    attempt: int
    api_request: str | None
    api_response: str | None
    api_error: str | None
    tool_result: str | None


def _empty_records() -> list[TranscriptRecord]:
    return []


def _empty_subscribers() -> list[asyncio.Queue[TranscriptRecordDict]]:
    return []


@dataclass(slots=True)
class TranscriptRecord:
    id: int
    event_id: int
    created_at: str
    player_index: int
    seq: int
    attempt: int
    api_request: str | None
    api_response: str | None
    api_error: str | None
    tool_result: str | None

    def to_dict(self) -> TranscriptRecordDict:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "created_at": self.created_at,
            "player_index": self.player_index,
            "seq": self.seq,
            "attempt": self.attempt,
            "api_request": self.api_request,
            "api_response": self.api_response,
            "api_error": self.api_error,
            "tool_result": self.tool_result,
        }


@dataclass(slots=True)
class AITranscript:
    records: list[TranscriptRecord] = field(default_factory=_empty_records)
    subscribers: list[asyncio.Queue[TranscriptRecordDict]] = field(default_factory=_empty_subscribers)
    next_record_id: int = 1
    next_event_id: int = 1

    def add_record(
        self,
        *,
        player_index: int,
        seq: int,
        attempt: int,
        api_request: str | None,
        api_response: str | None,
        api_error: str | None,
        tool_result: str | None,
    ) -> TranscriptRecord:
        record = TranscriptRecord(
            id=self.next_record_id,
            event_id=0,
            created_at=datetime.now(UTC).isoformat(),
            player_index=player_index,
            seq=seq,
            attempt=attempt,
            api_request=api_request,
            api_response=api_response,
            api_error=api_error,
            tool_result=tool_result,
        )
        self.next_record_id += 1
        self.records.append(record)
        self._publish(record)
        return record

    def update_tool_result(self, record: TranscriptRecord, tool_result: str) -> None:
        record.tool_result = tool_result
        self._publish(record)

    def to_dict(self) -> list[TranscriptRecordDict]:
        return [record.to_dict() for record in self.records]

    def stream_dicts(self) -> list[TranscriptRecordDict]:
        return self.to_dict()

    def subscribe(self) -> asyncio.Queue[TranscriptRecordDict]:
        queue: asyncio.Queue[TranscriptRecordDict] = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[TranscriptRecordDict]) -> None:
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    def _publish(self, record: TranscriptRecord) -> None:
        record.event_id = self.next_event_id
        self.next_event_id += 1
        message = record.to_dict()
        for queue in self.subscribers:
            queue.put_nowait(message)
