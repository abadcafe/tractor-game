"""Pipe-backed command/response control links for child processes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.context import SpawnContext
from typing import cast

from server import result as _result
from server.result import Ok, Rejected


@dataclass(frozen=True, slots=True)
class ProcessControlProtocol[CommandT, ResponseT]:
    """Typed command/response protocol for one child-process role."""

    name: str

    def __post_init__(self) -> None:
        assert self.name


@dataclass(frozen=True, slots=True)
class ControlReady:
    """Readable command and extra input state for one child process."""

    command_ready: bool
    connections: tuple[Connection, ...]


@dataclass(slots=True)
class CoordinatorControlEndpoint[CommandT, ResponseT]:
    """Coordinator side of a child-process control link."""

    _command_sender: Connection
    _response_receiver: Connection

    @property
    def response_connection(self) -> Connection:
        """Return the readable response connection for wait sets."""
        return self._response_receiver

    def send_command(
        self, command: CommandT
    ) -> _result.Ok[None] | _result.Rejected:
        """Send one command to the child process."""
        try:
            self._command_sender.send(command)
        except (EOFError, OSError) as exc:
            return Rejected(
                reason=f"process control command send failed: {exc}"
            )
        return Ok(value=None)

    def recv_response(
        self,
    ) -> _result.Ok[ResponseT] | _result.Rejected:
        """Receive one response from the child process."""
        try:
            response = self._response_receiver.recv()
        except (EOFError, OSError) as exc:
            return Rejected(
                reason=f"process control response receive failed: {exc}"
            )
        return Ok(value=cast(ResponseT, response))

    def close(self) -> None:
        """Close coordinator-owned control endpoints."""
        self._command_sender.close()
        self._response_receiver.close()


@dataclass(slots=True)
class ChildControlEndpoint[CommandT, ResponseT]:
    """Child side of a coordinator command/response control link."""

    _command_receiver: Connection
    _response_sender: Connection

    def poll_command(self, timeout_seconds: float) -> bool:
        """Return whether a command is readable."""
        assert timeout_seconds >= 0.0
        return self._command_receiver.poll(timeout_seconds)

    def recv_command(
        self,
    ) -> _result.Ok[CommandT] | _result.Rejected:
        """Receive one coordinator command."""
        try:
            command = self._command_receiver.recv()
        except (EOFError, OSError) as exc:
            return Rejected(
                reason=f"process control command receive failed: {exc}"
            )
        return Ok(value=cast(CommandT, command))

    def send_response(
        self, response: ResponseT
    ) -> _result.Ok[None] | _result.Rejected:
        """Send one response to the coordinator."""
        try:
            self._response_sender.send(response)
        except (EOFError, OSError) as exc:
            return Rejected(
                reason=f"process control response send failed: {exc}"
            )
        return Ok(value=None)

    def wait_command_or_connections(
        self,
        *,
        connections: tuple[Connection, ...],
        timeout_seconds: float | None,
    ) -> _result.Ok[ControlReady] | _result.Rejected:
        """Wait for a command or one of the caller-owned connections."""
        if timeout_seconds is not None:
            assert timeout_seconds >= 0.0
        try:
            ready = wait(
                connections + (self._command_receiver,),
                timeout=timeout_seconds,
            )
        except OSError as exc:
            return Rejected(
                reason=f"process control input wait failed: {exc}"
            )
        ready_connections = _ready_connections(ready)
        command_ready = _connection_in_ready(
            self._command_receiver, ready_connections
        )
        return Ok(
            value=ControlReady(
                command_ready=command_ready,
                connections=tuple(
                    connection
                    for connection in connections
                    if _connection_in_ready(
                        connection, ready_connections
                    )
                ),
            )
        )

    def close(self) -> None:
        """Close child-owned control endpoints."""
        self._command_receiver.close()
        self._response_sender.close()


@dataclass(frozen=True, slots=True)
class ProcessControlLink[CommandT, ResponseT]:
    """Coordinator and child endpoints for one control link."""

    coordinator: CoordinatorControlEndpoint[CommandT, ResponseT]
    child: ChildControlEndpoint[CommandT, ResponseT]


def create_process_control_link[CommandT, ResponseT](
    *,
    context: SpawnContext,
    protocol: ProcessControlProtocol[CommandT, ResponseT],
) -> ProcessControlLink[CommandT, ResponseT]:
    """Create one typed command/response link."""
    assert protocol.name
    command_receiver, command_sender = context.Pipe(duplex=False)
    response_receiver, response_sender = context.Pipe(duplex=False)
    return ProcessControlLink(
        coordinator=CoordinatorControlEndpoint(
            _command_sender=command_sender,
            _response_receiver=response_receiver,
        ),
        child=ChildControlEndpoint(
            _command_receiver=command_receiver,
            _response_sender=response_sender,
        ),
    )


def wait_control_responses[CommandT, ResponseT](
    *,
    endpoints: tuple[
        CoordinatorControlEndpoint[CommandT, ResponseT], ...
    ],
    timeout_seconds: float,
) -> (
    _result.Ok[
        tuple[CoordinatorControlEndpoint[CommandT, ResponseT], ...]
    ]
    | _result.Rejected
):
    """Wait until at least one coordinator endpoint has a response."""
    assert endpoints
    assert timeout_seconds >= 0.0
    try:
        ready = wait(
            tuple(
                endpoint.response_connection for endpoint in endpoints
            ),
            timeout=timeout_seconds,
        )
    except OSError as exc:
        return Rejected(
            reason=f"process control response wait failed: {exc}"
        )
    ready_connections = _ready_connections(ready)
    if not ready_connections:
        return Rejected(reason="process control response timed out")
    return Ok(
        value=tuple(
            endpoint
            for endpoint in endpoints
            if _connection_in_ready(
                endpoint.response_connection, ready_connections
            )
        )
    )


def _connection_in_ready(
    connection: Connection, ready: tuple[Connection, ...]
) -> bool:
    return any(connection is item for item in ready)


def _ready_connections(
    ready: Iterable[object],
) -> tuple[Connection, ...]:
    connections: list[Connection] = []
    for item in ready:
        if not isinstance(item, Connection):
            continue
        connections.append(cast(Connection, item))
    return tuple(connections)
