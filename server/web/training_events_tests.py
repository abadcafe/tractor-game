"""Black-box tests for training server-sent event routes."""

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from server.training_events import (
    EventContext,
    ProcessIdentity,
    StructuredEventSink,
)
from server.web.app import WebApplication
from server.web.application_test_client import (
    AsyncRestClient,
)
from server.web.application_test_client import (
    is_dict as _is_dict,
)
from server.web.application_test_client import (
    is_list_of_dict as _is_list_of_dict,
)

pytest_plugins: tuple[str, ...] = (
    "server.web.application_test_fixtures",
)


async def test_training_logs_have_rest_history_and_cursor_tail(
    web_application: WebApplication,
    client: AsyncRestClient,
    tmp_path: Path,
) -> None:
    initialized = await client.post(
        "/api/training/init",
        json={
            "run_dir": str(tmp_path),
            "d_model": 8,
            "layers": 1,
            "heads": 1,
        },
    )
    assert initialized.status_code == 204
    query = urlencode(
        {
            "run_dir": str(tmp_path),
        }
    )

    page = await client.get(f"/api/training/logs?{query}")
    assert page.status_code == 200
    document = page.json()
    assert _is_dict(document)
    events = document["events"]
    assert _is_list_of_dict(events) and events
    rest_entry = events[0]
    rest_event = rest_entry["event"]
    assert _is_dict(rest_event)
    assert rest_event["schema_version"] == 3
    store_id = document["store_id"]
    assert isinstance(store_id, str)
    response = await _read_sse(
        web_application.asgi,
        "/api/training/events/logs?"
        f"{query}&store_id={store_id}&after_sequence=0",
    )
    assert response.status == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    event = response.events[0]
    assert event.name == "log"
    message = event.json()
    assert _is_dict(message)
    assert message == rest_entry
    sequence = message["sequence"]
    assert isinstance(sequence, int)
    assert event.event_id == f"{store_id}:{sequence}"


async def test_training_process_events_send_current_snapshot(
    web_application: WebApplication,
    tmp_path: Path,
) -> None:
    query = urlencode({"run_dir": str(tmp_path)})

    response = await _read_sse(
        web_application.asgi, f"/api/training/events/process?{query}"
    )
    assert response.events[0].name == "process"
    snapshot = response.events[0].json()
    assert snapshot == {"process": None}


async def test_training_metrics_events_send_complete_snapshot(
    web_application: WebApplication,
    tmp_path: Path,
) -> None:
    query = urlencode(
        {
            "run_dir": str(tmp_path),
            "update_limit": 200,
            "series_points": 500,
        }
    )

    response = await _read_sse(
        web_application.asgi, f"/api/training/events/metrics?{query}"
    )
    assert response.events[0].name == "metrics"
    snapshot = response.events[0].json()
    assert _is_dict(snapshot)
    assert snapshot["schema_version"] == 4
    assert snapshot["store_id"] is None
    assert snapshot["through_sequence"] == 0
    assert _is_dict(snapshot["datasets"])


async def test_training_metrics_events_push_replacement_snapshot(
    web_application: WebApplication,
    tmp_path: Path,
) -> None:
    query = urlencode(
        {
            "run_dir": str(tmp_path),
            "update_limit": 200,
            "series_points": 500,
        }
    )

    async def create_store_after_initial(_event: _SseEvent) -> None:
        sink = StructuredEventSink(
            run_dir=tmp_path,
            process=ProcessIdentity(kind="worker", index=0),
        )
        sink.emit(
            "inference.batch",
            context=EventContext(
                policy_version=0,
                rollout_id="rollout-a",
                worker_index=0,
            ),
            fields={"batch_size": 1},
        )
        sink.close()

    response = await _read_sse(
        web_application.asgi,
        f"/api/training/events/metrics?{query}",
        event_count=2,
        after_first=create_store_after_initial,
    )
    initial, updated = [event.json() for event in response.events]

    assert _is_dict(initial)
    assert initial["store_id"] is None
    assert _is_dict(updated)
    assert isinstance(updated["store_id"], str)
    updated_sequence = updated["through_sequence"]
    assert isinstance(updated_sequence, int)
    assert updated_sequence >= 1
    datasets = updated["datasets"]
    assert _is_dict(datasets)
    inference = datasets["inference"]
    assert _is_list_of_dict(inference) and len(inference) == 1


async def test_training_metrics_events_apply_projection_parameters(
    web_application: WebApplication,
    client: AsyncRestClient,
    tmp_path: Path,
) -> None:
    initialized = await client.post(
        "/api/training/init",
        json={
            "run_dir": str(tmp_path),
            "d_model": 8,
            "layers": 1,
            "heads": 1,
        },
    )
    assert initialized.status_code == 204
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    for update in (1, 2):
        sink.emit(
            "update",
            context=EventContext(
                policy_version=update - 1,
                rollout_id=f"rollout-{update}",
            ),
            fields={"total_updates": update},
        )
    sink.close()
    query = urlencode(
        {
            "run_dir": str(tmp_path),
            "update_limit": 1,
            "series_points": 1,
        }
    )

    response = await _read_sse(
        web_application.asgi, f"/api/training/events/metrics?{query}"
    )
    snapshot = response.events[0].json()
    assert _is_dict(snapshot)
    datasets = snapshot["datasets"]
    assert _is_dict(datasets)
    throughput = datasets["throughput"]
    assert _is_list_of_dict(throughput)
    assert len(throughput) == 1
    values = throughput[0]["values"]
    assert _is_dict(values)
    assert values["total_updates"] == 2


async def test_training_events_report_store_replacement(
    web_application: WebApplication,
    client: AsyncRestClient,
    tmp_path: Path,
) -> None:
    initialized = await client.post(
        "/api/training/init",
        json={
            "run_dir": str(tmp_path),
            "d_model": 8,
            "layers": 1,
            "heads": 1,
        },
    )
    assert initialized.status_code == 204
    query = urlencode(
        {
            "run_dir": str(tmp_path),
            "store_id": "0" * 32,
        }
    )

    for path in ("logs", "checkpoints"):
        response = await _read_sse(
            web_application.asgi, f"/api/training/events/{path}?{query}"
        )
        assert response.events[0].name == "replacement"
        message = response.events[0].json()
        assert _is_dict(message)
        assert message["store_id"] != "0" * 32


async def test_training_events_send_terminal_rejection(
    web_application: WebApplication,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / ("x" * 120)
    run_dir.mkdir()
    with sqlite3.connect(run_dir / "training.sqlite3") as connection:
        connection.execute("CREATE TABLE invalid_store (value TEXT)")
    query = urlencode({"run_dir": str(run_dir)})
    expected_error = (
        "unsupported training database schema: "
        f"{run_dir / 'training.sqlite3'}"
    )

    for path in ("logs", "metrics", "checkpoints"):
        response = await _read_sse(
            web_application.asgi, f"/api/training/events/{path}?{query}"
        )
        assert len(response.events) == 1
        assert response.events[0].name == "rejected"
        assert response.events[0].json() == {"error": expected_error}


async def test_training_log_events_resume_from_last_event_id(
    web_application: WebApplication,
    client: AsyncRestClient,
    tmp_path: Path,
) -> None:
    initialized = await client.post(
        "/api/training/init",
        json={
            "run_dir": str(tmp_path),
            "d_model": 8,
            "layers": 1,
            "heads": 1,
        },
    )
    assert initialized.status_code == 204
    page = await client.get(
        f"/api/training/logs?{urlencode({'run_dir': str(tmp_path)})}"
    )
    document = page.json()
    assert _is_dict(document)
    store_id = document["store_id"]
    events = document["events"]
    assert isinstance(store_id, str)
    assert _is_list_of_dict(events) and events
    first = events[0]
    first_sequence = first["sequence"]
    assert isinstance(first_sequence, int)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit("training", fields={"total_updates": 0})
    sink.close()
    query = urlencode(
        {
            "run_dir": str(tmp_path),
            "store_id": store_id,
            "after_sequence": "0",
        }
    )

    response = await _read_sse(
        web_application.asgi,
        f"/api/training/events/logs?{query}",
        headers=(
            (b"last-event-id", f"{store_id}:{first_sequence}".encode()),
        ),
    )

    message = response.events[0].json()
    assert _is_dict(message)
    sequence = message["sequence"]
    assert isinstance(sequence, int) and sequence > first_sequence


@dataclass(frozen=True, slots=True)
class _SseEvent:
    name: str
    data: str
    event_id: str | None

    def json(self) -> object:
        return json.loads(self.data)


@dataclass(frozen=True, slots=True)
class _SseResponse:
    status: int
    headers: dict[str, str]
    events: tuple[_SseEvent, ...]


async def _read_sse(
    app: ASGIApp,
    url: str,
    *,
    event_count: int = 1,
    headers: tuple[tuple[bytes, bytes], ...] = (),
    after_first: Callable[[_SseEvent], Awaitable[None]] | None = None,
) -> _SseResponse:
    """Read named SSE events through the app's public ASGI interface."""
    assert event_count > 0
    parsed = urlsplit(url)
    disconnect = asyncio.Event()
    request_sent = False
    status: int | None = None
    response_headers: dict[str, str] = {}
    events: list[_SseEvent] = []
    buffer = bytearray()
    called_after_first = False

    async def receive() -> Message:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        nonlocal status, buffer, called_after_first
        if message["type"] == "http.response.start":
            status = message["status"]
            response_headers.update(
                {
                    key.decode("latin-1"): value.decode("latin-1")
                    for key, value in message["headers"]
                }
            )
            return
        assert message["type"] == "http.response.body"
        buffer.extend(message.get("body", b""))
        blocks = bytes(buffer).split(b"\n\n")
        buffer = bytearray(blocks.pop())
        for block in blocks:
            event = _parse_sse_event(block)
            if event is None:
                continue
            events.append(event)
            if len(events) == 1 and after_first is not None:
                assert not called_after_first
                called_after_first = True
                await after_first(event)
            if len(events) >= event_count:
                disconnect.set()

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": parsed.path,
        "raw_path": parsed.path.encode("ascii"),
        "query_string": parsed.query.encode("ascii"),
        "root_path": "",
        "headers": ((b"accept", b"text/event-stream"), *headers),
        "client": ("test", 123),
        "server": ("test", 80),
        "state": {},
    }
    app_receive: Receive = receive
    app_send: Send = send
    await asyncio.wait_for(
        app(scope, app_receive, app_send), timeout=8.0
    )
    assert status is not None
    assert len(events) >= event_count
    return _SseResponse(
        status=status,
        headers=response_headers,
        events=tuple(events[:event_count]),
    )


def _parse_sse_event(block: bytes) -> _SseEvent | None:
    name: str | None = None
    event_id: str | None = None
    data: list[str] = []
    for raw_line in block.decode("utf-8").splitlines():
        if raw_line.startswith(":") or raw_line.startswith("retry:"):
            continue
        field, separator, raw_value = raw_line.partition(":")
        value = (
            raw_value[1:] if raw_value.startswith(" ") else raw_value
        )
        if separator == "":
            continue
        if field == "event":
            name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data.append(value)
    if name is None:
        return None
    return _SseEvent(name=name, data="\n".join(data), event_id=event_id)
