"""Tests for typed process control links."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.context import SpawnContext

from server.result import Ok, Rejected
from server.training.runtime.process_control import (
    ProcessControlProtocol,
    create_process_control_link,
    wait_control_responses,
)


@dataclass(frozen=True, slots=True)
class _Command:
    value: str


@dataclass(frozen=True, slots=True)
class _Response:
    value: int


_CONTROL_PROTOCOL: ProcessControlProtocol[_Command, _Response] = (
    ProcessControlProtocol(name="test")
)


def test_process_control_link_round_trips_command_response() -> None:
    link = create_process_control_link(
        context=_spawn_context(),
        protocol=_CONTROL_PROTOCOL,
    )
    try:
        send_command = link.coordinator.send_command(
            _Command(value="load")
        )
        command = link.child.recv_command()
        send_response = link.child.send_response(_Response(value=7))
        response = link.coordinator.recv_response()

        assert isinstance(send_command, Ok)
        assert isinstance(command, Ok)
        assert command.value == _Command(value="load")
        assert isinstance(send_response, Ok)
        assert isinstance(response, Ok)
        assert response.value == _Response(value=7)
    finally:
        link.coordinator.close()
        link.child.close()


def test_wait_control_responses_returns_ready_endpoint() -> None:
    context = _spawn_context()
    first = create_process_control_link(
        context=context,
        protocol=_CONTROL_PROTOCOL,
    )
    second = create_process_control_link(
        context=context,
        protocol=_CONTROL_PROTOCOL,
    )
    try:
        sent = second.child.send_response(_Response(value=11))
        ready = wait_control_responses(
            endpoints=(first.coordinator, second.coordinator),
            timeout_seconds=1.0,
        )
        assert isinstance(sent, Ok)
        assert isinstance(ready, Ok)
        assert ready.value == (second.coordinator,)
        response = ready.value[0].recv_response()
        assert isinstance(response, Ok)
        assert response.value == _Response(value=11)
    finally:
        first.coordinator.close()
        first.child.close()
        second.coordinator.close()
        second.child.close()


def test_wait_control_responses_rejects_timeout() -> None:
    link = create_process_control_link(
        context=_spawn_context(),
        protocol=_CONTROL_PROTOCOL,
    )
    try:
        ready = wait_control_responses(
            endpoints=(link.coordinator,),
            timeout_seconds=0.01,
        )
        assert isinstance(ready, Rejected)
        assert ready.reason == "process control response timed out"
    finally:
        link.coordinator.close()
        link.child.close()


def test_child_wait_command_or_connections_reports_command() -> None:
    link = create_process_control_link(
        context=_spawn_context(),
        protocol=_CONTROL_PROTOCOL,
    )
    try:
        sent = link.coordinator.send_command(_Command(value="stop"))
        ready = link.child.wait_command_or_connections(
            connections=(),
            timeout_seconds=1.0,
        )
        assert isinstance(sent, Ok)
        assert isinstance(ready, Ok)
        assert ready.value.command_ready
        assert ready.value.connections == ()
    finally:
        link.coordinator.close()
        link.child.close()


def test_child_wait_command_or_connections_reports_extra_input() -> (
    None
):
    context = _spawn_context()
    link = create_process_control_link(
        context=context,
        protocol=_CONTROL_PROTOCOL,
    )
    receiver, sender = context.Pipe(duplex=False)
    try:
        sender.send(_Response(value=23))
        ready = link.child.wait_command_or_connections(
            connections=(receiver,),
            timeout_seconds=1.0,
        )
        assert isinstance(ready, Ok)
        assert not ready.value.command_ready
        assert ready.value.connections == (receiver,)
        assert receiver.recv() == _Response(value=23)
    finally:
        receiver.close()
        sender.close()
        link.coordinator.close()
        link.child.close()


def test_process_control_link_rejects_closed_send() -> None:
    link = create_process_control_link(
        context=_spawn_context(),
        protocol=_CONTROL_PROTOCOL,
    )
    try:
        link.child.close()
        sent = link.coordinator.send_command(_Command(value="load"))
        assert isinstance(sent, Rejected)
        assert sent.reason.startswith(
            "process control command send failed:"
        )
    finally:
        link.coordinator.close()


def _spawn_context() -> SpawnContext:
    return mp.get_context("spawn")
