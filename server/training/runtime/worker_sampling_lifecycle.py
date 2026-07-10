"""Worker sampling lifecycle for coordinator-managed rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, assert_never

from server import result as _result
from server.result import Ok, Rejected
from server.training.runtime.async_ipc import (
    AsyncCoordinatorControlEndpoint,
    ControlCommandBroadcastFailure,
    broadcast_control_commands,
    wait_async_control_responses,
)
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.messages import (
    WorkerCommand,
    WorkerCommandRejected,
    WorkerResponse,
    WorkerSamplingAlreadyStopped,
    WorkerSamplingStarted,
    WorkerSamplingStopped,
    WorkerSnapshotCompleted,
    WorkerStartSamplingCommand,
    WorkerStateLoaded,
    WorkerStopSamplingCommand,
    WorkerUpdateCompleted,
)


class WorkerControlHandle(Protocol):
    """Coordinator-owned control endpoint for one worker."""

    @property
    def index(self) -> int:
        """Return the worker index used for deterministic ordering."""
        ...

    @property
    def control(
        self,
    ) -> AsyncCoordinatorControlEndpoint[WorkerCommand, WorkerResponse]:
        """Return the coordinator-side worker control endpoint."""
        ...


@dataclass(frozen=True, slots=True)
class WorkerSamplingSession:
    """Active sampling session across all commanded workers."""

    policy_version: int
    commanded_handles: tuple[WorkerControlHandle, ...]
    started_handles: tuple[WorkerControlHandle, ...]

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.commanded_handles
        assert self.started_handles
        assert len(self.commanded_handles) == len(self.started_handles)


async def start_worker_sampling_session(
    *,
    handles: tuple[WorkerControlHandle, ...],
    execution_config: ExecutionConfig,
    policy_version: int,
) -> _result.Ok[WorkerSamplingSession] | _result.Rejected:
    """Start sampling on every worker or clean up commanded workers."""
    assert handles
    send_result = await broadcast_control_commands(
        targets=handles,
        sender=_worker_control_sender,
        command=lambda _handle: WorkerStartSamplingCommand(
            policy_version=policy_version,
            game_env_count=execution_config.game_envs_per_worker,
        ),
    )
    if isinstance(send_result, ControlCommandBroadcastFailure):
        return await _abort_worker_sampling_start(
            handles=send_result.sent_targets,
            policy_version=policy_version,
            timeout_seconds=(
                execution_config.timeouts.sampling_stop_seconds
            ),
            failure=send_result.rejection,
        )
    return await _receive_worker_sampling_session_started(
        commanded_handles=send_result.value,
        policy_version=policy_version,
        start_timeout_seconds=(
            execution_config.timeouts.sampling_start_seconds
        ),
        stop_timeout_seconds=(
            execution_config.timeouts.sampling_stop_seconds
        ),
    )


async def stop_worker_sampling_session(
    *,
    session: WorkerSamplingSession,
    timeout_seconds: float,
) -> _result.Ok[tuple[WorkerSamplingStopped, ...]] | _result.Rejected:
    """Stop a fully started worker sampling session."""
    send_result = await _send_worker_stop_sampling_commands(
        handles=session.started_handles,
        policy_version=session.policy_version,
    )
    if isinstance(send_result, Rejected):
        return send_result
    return await _receive_worker_sampling_stopped(
        handles=session.started_handles,
        policy_version=session.policy_version,
        timeout_seconds=timeout_seconds,
    )


def _cleanup_rejection(
    *, failure: Rejected, cleanup: Rejected
) -> Rejected:
    reason = (
        f"{failure.reason}; sampling cleanup failed: {cleanup.reason}"
    )
    return Rejected(reason=reason)


async def _send_worker_stop_sampling_commands(
    *,
    handles: tuple[WorkerControlHandle, ...],
    policy_version: int,
) -> _result.Ok[None] | _result.Rejected:
    assert handles
    send_result = await broadcast_control_commands(
        targets=handles,
        sender=_worker_control_sender,
        command=lambda _handle: WorkerStopSamplingCommand(
            policy_version=policy_version
        ),
    )
    if isinstance(send_result, ControlCommandBroadcastFailure):
        return send_result.rejection
    return Ok(value=None)


def _worker_control_sender(
    handle: WorkerControlHandle,
) -> AsyncCoordinatorControlEndpoint[WorkerCommand, WorkerResponse]:
    return handle.control


async def _receive_worker_sampling_session_started(
    *,
    commanded_handles: tuple[WorkerControlHandle, ...],
    policy_version: int,
    start_timeout_seconds: float,
    stop_timeout_seconds: float,
) -> _result.Ok[WorkerSamplingSession] | _result.Rejected:
    started_handles: list[WorkerControlHandle] = []
    pending = list(commanded_handles)
    while pending:
        ready_result = await _wait_worker_responses(
            handles=tuple(pending),
            timeout_seconds=start_timeout_seconds,
        )
        if isinstance(ready_result, Rejected):
            return await _abort_worker_sampling_start(
                handles=commanded_handles,
                policy_version=policy_version,
                timeout_seconds=stop_timeout_seconds,
                failure=ready_result,
            )
        for handle in ready_result.value:
            response_result = await handle.control.recv_response()
            if isinstance(response_result, Rejected):
                return await _abort_worker_sampling_start(
                    handles=commanded_handles,
                    policy_version=policy_version,
                    timeout_seconds=stop_timeout_seconds,
                    failure=response_result,
                )
            response = response_result.value
            pending.remove(handle)
            if isinstance(response, WorkerCommandRejected):
                return await _abort_worker_sampling_start(
                    handles=commanded_handles,
                    policy_version=policy_version,
                    timeout_seconds=stop_timeout_seconds,
                    failure=_worker_command_rejection(response),
                )
            if isinstance(response, WorkerSamplingStarted):
                if response.policy_version != policy_version:
                    return await _abort_worker_sampling_start(
                        handles=commanded_handles,
                        policy_version=policy_version,
                        timeout_seconds=stop_timeout_seconds,
                        failure=Rejected(
                            reason=(
                                "worker returned stale sampling start "
                                "policy version"
                            )
                        ),
                    )
                started_handles.append(handle)
                continue
            return await _abort_worker_sampling_start(
                handles=commanded_handles,
                policy_version=policy_version,
                timeout_seconds=stop_timeout_seconds,
                failure=Rejected(
                    reason=_unexpected_worker_response_reason(
                        response=response,
                        stage="sampling start",
                    )
                ),
            )
    return Ok(
        value=WorkerSamplingSession(
            policy_version=policy_version,
            commanded_handles=commanded_handles,
            started_handles=tuple(
                sorted(started_handles, key=lambda item: item.index)
            ),
        )
    )


async def _abort_worker_sampling_start(
    *,
    handles: tuple[WorkerControlHandle, ...],
    policy_version: int,
    timeout_seconds: float,
    failure: Rejected,
) -> Rejected:
    if not handles:
        return failure
    send_result = await _send_worker_stop_sampling_commands(
        handles=handles,
        policy_version=policy_version,
    )
    if isinstance(send_result, Rejected):
        return _cleanup_rejection(failure=failure, cleanup=send_result)
    stopped_result = await _receive_worker_sampling_abort_stopped(
        handles=handles,
        policy_version=policy_version,
        timeout_seconds=timeout_seconds,
    )
    if isinstance(stopped_result, Rejected):
        return _cleanup_rejection(
            failure=failure, cleanup=stopped_result
        )
    return failure


async def _receive_worker_sampling_abort_stopped(
    *,
    handles: tuple[WorkerControlHandle, ...],
    policy_version: int,
    timeout_seconds: float,
) -> _result.Ok[tuple[WorkerSamplingStopped, ...]] | _result.Rejected:
    responses: list[WorkerSamplingStopped] = []
    pending = list(handles)
    while pending:
        ready_result = await _wait_worker_responses(
            handles=tuple(pending),
            timeout_seconds=timeout_seconds,
        )
        if isinstance(ready_result, Rejected):
            return ready_result
        for handle in ready_result.value:
            response_result = await handle.control.recv_response()
            if isinstance(response_result, Rejected):
                return response_result
            response = response_result.value
            if isinstance(response, WorkerCommandRejected):
                if (
                    response.command == "start_sampling"
                    and response.policy_version == policy_version
                ):
                    continue
                return _worker_command_rejection(response)
            if isinstance(response, WorkerSamplingStarted):
                if response.policy_version != policy_version:
                    return Rejected(
                        reason=(
                            "worker returned stale sampling start "
                            "policy version during abort"
                        )
                    )
                continue
            if isinstance(response, WorkerSamplingStopped):
                if response.policy_version != policy_version:
                    return Rejected(
                        reason=(
                            "worker returned stale sampling stop "
                            "policy version during abort"
                        )
                    )
                pending.remove(handle)
                responses.append(response)
                continue
            if isinstance(response, WorkerSamplingAlreadyStopped):
                if response.policy_version != policy_version:
                    return Rejected(
                        reason=(
                            "worker returned stale sampling "
                            "already-stopped policy version "
                            "during abort"
                        )
                    )
                pending.remove(handle)
                continue
            return Rejected(
                reason=_unexpected_worker_response_reason(
                    response=response,
                    stage="sampling abort",
                )
            )
    return Ok(
        value=tuple(
            sorted(responses, key=lambda item: item.worker_index)
        )
    )


async def _receive_worker_sampling_stopped(
    *,
    handles: tuple[WorkerControlHandle, ...],
    policy_version: int,
    timeout_seconds: float,
) -> _result.Ok[tuple[WorkerSamplingStopped, ...]] | _result.Rejected:
    responses: list[WorkerSamplingStopped] = []
    pending = list(handles)
    while pending:
        ready_result = await _wait_worker_responses(
            handles=tuple(pending),
            timeout_seconds=timeout_seconds,
        )
        if isinstance(ready_result, Rejected):
            return ready_result
        for handle in ready_result.value:
            response_result = await handle.control.recv_response()
            if isinstance(response_result, Rejected):
                return response_result
            response = response_result.value
            pending.remove(handle)
            match response:
                case WorkerCommandRejected():
                    return _worker_command_rejection(response)
                case WorkerSamplingStarted():
                    return Rejected(
                        reason=(
                            "worker returned start during sampling stop"
                        )
                    )
                case WorkerSamplingAlreadyStopped():
                    return Rejected(
                        reason=(
                            "worker returned already-stopped during "
                            "sampling stop"
                        )
                    )
                case WorkerUpdateCompleted():
                    return Rejected(
                        reason="worker returned update during sampling"
                    )
                case WorkerStateLoaded():
                    return Rejected(
                        reason=(
                            "worker returned state sync during sampling"
                        )
                    )
                case WorkerSnapshotCompleted():
                    return Rejected(
                        reason=(
                            "worker returned snapshot during sampling"
                        )
                    )
                case WorkerSamplingStopped():
                    if response.policy_version != policy_version:
                        return Rejected(
                            reason=(
                                "worker returned stale sampling policy "
                                "version"
                            )
                        )
                    responses.append(response)
                case _:
                    assert_never(response)
    return Ok(
        value=tuple(
            sorted(responses, key=lambda item: item.worker_index)
        )
    )


async def _wait_worker_responses(
    *,
    handles: tuple[WorkerControlHandle, ...],
    timeout_seconds: float,
) -> _result.Ok[tuple[WorkerControlHandle, ...]] | _result.Rejected:
    ready_result = await wait_async_control_responses(
        endpoints=tuple(handle.control for handle in handles),
        timeout_seconds=timeout_seconds,
    )
    if isinstance(ready_result, Rejected):
        return ready_result
    return Ok(
        value=tuple(
            _worker_handle_for_control(
                handles=handles,
                control=control,
            )
            for control in ready_result.value
        )
    )


def _worker_handle_for_control(
    *,
    handles: tuple[WorkerControlHandle, ...],
    control: AsyncCoordinatorControlEndpoint[
        WorkerCommand, WorkerResponse
    ],
) -> WorkerControlHandle:
    for handle in handles:
        if handle.control is control:
            return handle
    raise AssertionError("ready worker control endpoint is unknown")


def _worker_command_rejection(
    response: WorkerCommandRejected,
) -> Rejected:
    return Rejected(
        reason=f"worker-{response.worker_index}: {response.reason}"
    )


def _unexpected_worker_response_reason(
    *, response: WorkerResponse, stage: str
) -> str:
    if isinstance(response, WorkerSamplingStarted):
        action = "start"
    elif isinstance(response, WorkerSamplingStopped):
        action = "stop"
    elif isinstance(response, WorkerSamplingAlreadyStopped):
        action = "already-stopped"
    elif isinstance(response, WorkerUpdateCompleted):
        action = "update"
    elif isinstance(response, WorkerStateLoaded):
        action = "state sync"
    elif isinstance(response, WorkerSnapshotCompleted):
        action = "snapshot"
    else:
        action = response.command
    return f"worker returned {action} during {stage}"
