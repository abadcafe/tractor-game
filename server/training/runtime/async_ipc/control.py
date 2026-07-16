"""Typed async command/response control links."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.runtime.async_ipc.frame import (
    AsyncFrameEndpoint,
    AsyncSocketPair,
    create_async_socket_pair,
    wait_readable_frames,
)

type MessageDecoder[MessageT] = Callable[[object], MessageT | None]


@dataclass(frozen=True, slots=True)
class ProcessControlProtocol[CommandT, ResponseT]:
    """Typed command/response protocol for one child-process role."""

    name: str
    decode_command: MessageDecoder[CommandT]
    decode_response: MessageDecoder[ResponseT]

    def __post_init__(self) -> None:
        assert self.name


@dataclass(slots=True)
class AsyncCoordinatorControlEndpoint[CommandT, ResponseT]:
    """Coordinator side of one async child-process control link."""

    _endpoint: AsyncFrameEndpoint
    _protocol: ProcessControlProtocol[CommandT, ResponseT]

    @property
    def frame_endpoint(self) -> AsyncFrameEndpoint:
        """Return the readable frame endpoint for wait sets."""
        return self._endpoint

    async def send_command(
        self, command: CommandT
    ) -> _result.Ok[None] | _result.Rejected:
        """Send one command to the child process."""
        payload = _encode_message(command)
        send_result = await self._endpoint.send_frame(payload)
        if isinstance(send_result, Rejected):
            return Rejected(
                reason=(
                    f"{self._protocol.name} control command send "
                    f"failed: {send_result.reason}"
                )
            )
        return Ok(value=None)

    async def recv_response(
        self,
    ) -> _result.Ok[ResponseT] | _result.Rejected:
        """Receive one response from the child process."""
        frame_result = await self._endpoint.recv_frame()
        if isinstance(frame_result, Rejected):
            return Rejected(
                reason=(
                    f"{self._protocol.name} control response receive "
                    f"failed: {frame_result.reason}"
                )
            )
        decoded = _decode_message(
            data=frame_result.value,
            decoder=self._protocol.decode_response,
        )
        if isinstance(decoded, Rejected):
            return Rejected(
                reason=(
                    f"{self._protocol.name} control response decode "
                    f"failed: {decoded.reason}"
                )
            )
        return Ok(value=decoded.value)

    def close(self) -> None:
        """Close coordinator-owned control endpoint."""
        self._endpoint.close()


@dataclass(slots=True)
class AsyncChildControlEndpoint[CommandT, ResponseT]:
    """Child side of one async coordinator command/response link."""

    _endpoint: AsyncFrameEndpoint
    _protocol: ProcessControlProtocol[CommandT, ResponseT]

    @property
    def frame_endpoint(self) -> AsyncFrameEndpoint:
        """Return the readable frame endpoint for wait sets."""
        return self._endpoint

    async def command_ready(self, timeout_seconds: float) -> bool:
        """Return whether a command is readable."""
        assert timeout_seconds >= 0.0
        ready_result = await self._endpoint.wait_readable(
            timeout_seconds=timeout_seconds
        )
        if isinstance(ready_result, Rejected):
            return False
        return ready_result.value

    async def recv_command(
        self,
    ) -> _result.Ok[CommandT] | _result.Rejected:
        """Receive one coordinator command."""
        frame_result = await self._endpoint.recv_frame()
        if isinstance(frame_result, Rejected):
            return Rejected(
                reason=(
                    f"{self._protocol.name} control command receive "
                    f"failed: {frame_result.reason}"
                )
            )
        decoded = _decode_message(
            data=frame_result.value,
            decoder=self._protocol.decode_command,
        )
        if isinstance(decoded, Rejected):
            return Rejected(
                reason=(
                    f"{self._protocol.name} control command decode "
                    f"failed: {decoded.reason}"
                )
            )
        return Ok(value=decoded.value)

    async def send_response(
        self, response: ResponseT
    ) -> _result.Ok[None] | _result.Rejected:
        """Send one response to the coordinator."""
        payload = _encode_message(response)
        send_result = await self._endpoint.send_frame(payload)
        if isinstance(send_result, Rejected):
            return Rejected(
                reason=(
                    f"{self._protocol.name} control response send "
                    f"failed: {send_result.reason}"
                )
            )
        return Ok(value=None)

    def close(self) -> None:
        """Close child-owned control endpoint."""
        self._endpoint.close()


@dataclass(frozen=True, slots=True)
class AsyncProcessControlLink[CommandT, ResponseT]:
    """Coordinator and child endpoints for one async control link."""

    coordinator: AsyncCoordinatorControlEndpoint[CommandT, ResponseT]
    child: AsyncChildControlEndpoint[CommandT, ResponseT]


def create_async_process_control_link[CommandT, ResponseT](
    *,
    protocol: ProcessControlProtocol[CommandT, ResponseT],
) -> AsyncProcessControlLink[CommandT, ResponseT]:
    """Create one async typed command/response link."""
    pair = create_async_socket_pair()
    return _control_link_from_pair(protocol=protocol, pair=pair)


async def wait_async_control_responses[CommandT, ResponseT](
    *,
    endpoints: tuple[
        AsyncCoordinatorControlEndpoint[CommandT, ResponseT], ...
    ],
    timeout_seconds: float,
) -> (
    _result.Ok[
        tuple[AsyncCoordinatorControlEndpoint[CommandT, ResponseT], ...]
    ]
    | _result.Rejected
):
    """Wait until at least one coordinator endpoint has a response."""
    assert endpoints
    assert timeout_seconds >= 0.0
    ready_result = await poll_async_control_responses(
        endpoints=endpoints,
        timeout_seconds=timeout_seconds,
    )
    if isinstance(ready_result, Rejected):
        return ready_result
    if not ready_result.value:
        return Rejected(reason="process control response timed out")
    return ready_result


async def poll_async_control_responses[CommandT, ResponseT](
    *,
    endpoints: tuple[
        AsyncCoordinatorControlEndpoint[CommandT, ResponseT], ...
    ],
    timeout_seconds: float,
) -> (
    _result.Ok[
        tuple[AsyncCoordinatorControlEndpoint[CommandT, ResponseT], ...]
    ]
    | _result.Rejected
):
    """Return ready endpoints, including an empty timeout result."""
    assert endpoints
    assert timeout_seconds >= 0.0
    ready_result = await wait_readable_frames(
        endpoints=tuple(
            endpoint.frame_endpoint for endpoint in endpoints
        ),
        timeout_seconds=timeout_seconds,
    )
    if isinstance(ready_result, Rejected):
        return ready_result
    return Ok(
        value=tuple(
            endpoint
            for endpoint in endpoints
            if _endpoint_in_ready(
                endpoint.frame_endpoint,
                ready_result.value,
            )
        )
    )


def _control_link_from_pair[CommandT, ResponseT](
    *,
    protocol: ProcessControlProtocol[CommandT, ResponseT],
    pair: AsyncSocketPair,
) -> AsyncProcessControlLink[CommandT, ResponseT]:
    return AsyncProcessControlLink(
        coordinator=AsyncCoordinatorControlEndpoint(
            _endpoint=pair.first,
            _protocol=protocol,
        ),
        child=AsyncChildControlEndpoint(
            _endpoint=pair.second,
            _protocol=protocol,
        ),
    )


def _endpoint_in_ready(
    endpoint: AsyncFrameEndpoint,
    ready: tuple[AsyncFrameEndpoint, ...],
) -> bool:
    return any(endpoint is item for item in ready)


def _encode_message(message: object) -> bytes:
    return pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)


def _decode_message[MessageT](
    *, data: bytes, decoder: MessageDecoder[MessageT]
) -> _result.Ok[MessageT] | _result.Rejected:
    try:
        payload = cast(object, pickle.loads(data))
    except (pickle.PickleError, EOFError, ValueError) as exc:
        return Rejected(reason=f"invalid pickle payload: {exc}")
    decoded = decoder(payload)
    if decoded is None:
        return Rejected(reason="unexpected control message type")
    return Ok(value=decoded)
