"""Black-box tests for the training SSE wire contract."""

from server.web.training_events.wire import (
    KEEP_ALIVE,
    RETRY,
    ServerEvent,
    rejected_event,
)


def test_server_event_encodes_named_json_event_with_cursor() -> None:
    event = ServerEvent(
        name="log",
        data='{"sequence":7}',
        event_id="0123456789abcdef0123456789abcdef:7",
    )

    assert event.encode() == (
        b"event: log\n"
        b"id: 0123456789abcdef0123456789abcdef:7\n"
        b'data: {"sequence":7}\n\n'
    )


def test_server_event_encodes_every_data_line() -> None:
    event = ServerEvent(name="metrics", data="first\nsecond")

    assert event.encode() == (
        b"event: metrics\ndata: first\ndata: second\n\n"
    )


def test_rejected_event_is_terminal_domain_data() -> None:
    assert rejected_event("broken store").encode() == (
        b'event: rejected\ndata: {"error":"broken store"}\n\n'
    )


def test_transport_control_frames_are_not_business_events() -> None:
    assert RETRY == b"retry: 1000\n\n"
    assert KEEP_ALIVE == b": keep-alive\n\n"
