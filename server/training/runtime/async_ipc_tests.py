"""Tests for async framed IPC primitives."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from server.result import Ok, Rejected
from server.training.runtime.async_ipc import (
    ProcessControlProtocol,
    create_async_process_control_link,
    create_async_socket_pair,
    wait_async_control_responses,
)


@dataclass(frozen=True, slots=True)
class _Command:
    value: str


@dataclass(frozen=True, slots=True)
class _Response:
    value: int


def _decode_command(value: object) -> _Command | None:
    if isinstance(value, _Command):
        return value
    return None


def _decode_response(value: object) -> _Response | None:
    if isinstance(value, _Response):
        return value
    return None


_CONTROL_PROTOCOL: ProcessControlProtocol[_Command, _Response] = (
    ProcessControlProtocol(
        name="test",
        decode_command=_decode_command,
        decode_response=_decode_response,
    )
)


async def test_async_frame_endpoint_round_trips_bytes() -> None:
    pair = create_async_socket_pair()
    try:
        sent = await pair.first.send_frame(b"abc")
        received = await pair.second.recv_frame(timeout_seconds=1.0)

        assert isinstance(sent, Ok)
        assert isinstance(received, Ok)
        assert received.value == b"abc"
    finally:
        pair.first.close()
        pair.second.close()


async def test_async_frame_endpoint_receives_into_buffer() -> None:
    pair = create_async_socket_pair()
    buffer = bytearray(8)
    try:
        sent = await pair.first.send_frame(b"abcdef")
        received = await pair.second.recv_frame_into(memoryview(buffer))

        assert isinstance(sent, Ok)
        assert isinstance(received, Ok)
        assert received.value == 6
        assert bytes(buffer[:6]) == b"abcdef"
    finally:
        pair.first.close()
        pair.second.close()


async def test_async_frame_endpoint_rejects_oversized_frame() -> None:
    pair = create_async_socket_pair(max_frame_bytes=4)
    try:
        sent = await pair.first.send_frame(b"abcde")

        assert isinstance(sent, Rejected)
        assert sent.reason == "async IPC frame exceeds limit"
    finally:
        pair.first.close()
        pair.second.close()


async def test_recv_frame_rejects_oversized_inbound_frame() -> None:
    pair = create_async_socket_pair(max_frame_bytes=4)
    try:
        raw = struct.pack(">Q", 5) + b"abcde"
        pair.first.socket.sendall(raw)
        received = await pair.second.recv_frame(timeout_seconds=1.0)

        assert isinstance(received, Rejected)
        assert received.reason == "async IPC frame exceeds limit"
    finally:
        pair.first.close()
        pair.second.close()


async def test_async_control_link_round_trips_command_response() -> (
    None
):
    link = create_async_process_control_link(
        protocol=_CONTROL_PROTOCOL,
    )
    try:
        send_command = await link.coordinator.send_command(
            _Command(value="load")
        )
        command = await link.child.recv_command()
        send_response = await link.child.send_response(
            _Response(value=7)
        )
        response = await link.coordinator.recv_response()

        assert isinstance(send_command, Ok)
        assert isinstance(command, Ok)
        assert command.value == _Command(value="load")
        assert isinstance(send_response, Ok)
        assert isinstance(response, Ok)
        assert response.value == _Response(value=7)
    finally:
        link.coordinator.close()
        link.child.close()


async def test_wait_control_responses_returns_ready_endpoint() -> None:
    first = create_async_process_control_link(
        protocol=_CONTROL_PROTOCOL,
    )
    second = create_async_process_control_link(
        protocol=_CONTROL_PROTOCOL,
    )
    try:
        sent = await second.child.send_response(_Response(value=11))
        ready = await wait_async_control_responses(
            endpoints=(first.coordinator, second.coordinator),
            timeout_seconds=1.0,
        )
        assert isinstance(sent, Ok)
        assert isinstance(ready, Ok)
        assert ready.value == (second.coordinator,)
        response = await ready.value[0].recv_response()
        assert isinstance(response, Ok)
        assert response.value == _Response(value=11)
    finally:
        first.coordinator.close()
        first.child.close()
        second.coordinator.close()
        second.child.close()


async def test_wait_async_control_responses_rejects_timeout() -> None:
    link = create_async_process_control_link(
        protocol=_CONTROL_PROTOCOL,
    )
    try:
        ready = await wait_async_control_responses(
            endpoints=(link.coordinator,),
            timeout_seconds=0.01,
        )
        assert isinstance(ready, Rejected)
        assert ready.reason == "process control response timed out"
    finally:
        link.coordinator.close()
        link.child.close()


async def test_async_control_link_rejects_wrong_payload_type() -> None:
    link = create_async_process_control_link(
        protocol=_CONTROL_PROTOCOL,
    )
    try:
        sent = await link.coordinator.frame_endpoint.send_frame(
            b"\x80\x05N."
        )
        command = await link.child.recv_command()

        assert isinstance(sent, Ok)
        assert isinstance(command, Rejected)
        assert "unexpected control message type" in command.reason
    finally:
        link.coordinator.close()
        link.child.close()
