"""Worker sampling lifecycle for coordinator-managed rollout."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, assert_never

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.runtime.async_ipc import (
    AsyncCoordinatorControlEndpoint,
    ControlCommandBroadcastFailure,
    broadcast_control_commands,
    poll_async_control_responses,
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
from server.training.stop import TrainingStopRequest

_STOP_CHECK_INTERVAL_SECONDS = 0.05


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
    rollout_id: str
    commanded_handles: tuple[WorkerControlHandle, ...]
    started_handles: tuple[WorkerControlHandle, ...]

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.rollout_id
        assert self.commanded_handles
        assert self.started_handles
        assert len(self.commanded_handles) == len(self.started_handles)


@dataclass(frozen=True, slots=True)
class WorkerSamplingCleanupFailed:
    """Sampling start failed and commanded workers were not cleaned."""

    rejection: Rejected

    def __post_init__(self) -> None:
        assert self.rejection.reason


@dataclass(frozen=True, slots=True)
class WorkerSamplingStartStopped:
    """A stop prevented a sampling session from becoming active."""


@dataclass(frozen=True, slots=True)
class _WorkerSamplingStartStopRequested:
    """Internal marker returned while waiting for start responses."""


type WorkerSamplingStartOutcome = (
    WorkerSamplingSession | WorkerSamplingStartStopped
)


async def start_worker_sampling_session(
    *,
    handles: tuple[WorkerControlHandle, ...],
    execution_config: ExecutionConfig,
    policy_version: int,
    rollout_id: str,
    stop_request: TrainingStopRequest,
) -> (
    _result.Ok[WorkerSamplingStartOutcome]
    | _result.Rejected
    | WorkerSamplingCleanupFailed
):
    """Start sampling on every worker or clean up commanded workers."""
    assert handles
    if stop_request.is_requested():
        return Ok(value=WorkerSamplingStartStopped())
    send_result = await broadcast_control_commands(
        targets=handles,
        sender=_worker_control_sender,
        command=lambda _handle: WorkerStartSamplingCommand(
            policy_version=policy_version,
            rollout_id=rollout_id,
            game_env_count=execution_config.game_envs_per_worker,
        ),
    )
    if isinstance(send_result, ControlCommandBroadcastFailure):
        return await _abort_worker_sampling_start(
            handles=send_result.sent_targets,
            policy_version=policy_version,
            rollout_id=rollout_id,
            timeout_seconds=(
                execution_config.timeouts.sampling_stop_seconds
            ),
            failure=send_result.rejection,
        )
    return await _receive_worker_sampling_session_started(
        commanded_handles=send_result.value,
        policy_version=policy_version,
        rollout_id=rollout_id,
        start_timeout_seconds=(
            execution_config.timeouts.sampling_start_seconds
        ),
        stop_timeout_seconds=(
            execution_config.timeouts.sampling_stop_seconds
        ),
        stop_request=stop_request,
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
        rollout_id=session.rollout_id,
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
    rollout_id: str,
) -> _result.Ok[None] | _result.Rejected:
    assert handles
    send_result = await broadcast_control_commands(
        targets=handles,
        sender=_worker_control_sender,
        command=lambda _handle: WorkerStopSamplingCommand(
            policy_version=policy_version,
            rollout_id=rollout_id,
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
    rollout_id: str,
    start_timeout_seconds: float,
    stop_timeout_seconds: float,
    stop_request: TrainingStopRequest,
) -> (
    _result.Ok[WorkerSamplingStartOutcome]
    | _result.Rejected
    | WorkerSamplingCleanupFailed
):
    started_handles: list[WorkerControlHandle] = []
    pending = list(commanded_handles)
    deadline = time.monotonic() + start_timeout_seconds
    while pending:
        ready_result = await _wait_worker_start_responses(
            handles=tuple(pending),
            deadline=deadline,
            stop_request=stop_request,
        )
        if isinstance(ready_result, _WorkerSamplingStartStopRequested):
            return await _stop_worker_sampling_after_start_request(
                handles=commanded_handles,
                policy_version=policy_version,
                rollout_id=rollout_id,
                timeout_seconds=stop_timeout_seconds,
            )
        if isinstance(ready_result, Rejected):
            return await _abort_worker_sampling_start(
                handles=commanded_handles,
                policy_version=policy_version,
                rollout_id=rollout_id,
                timeout_seconds=stop_timeout_seconds,
                failure=ready_result,
            )
        for handle in ready_result.value:
            response_result = await handle.control.recv_response()
            if isinstance(response_result, Rejected):
                return await _abort_worker_sampling_start(
                    handles=commanded_handles,
                    policy_version=policy_version,
                    rollout_id=rollout_id,
                    timeout_seconds=stop_timeout_seconds,
                    failure=response_result,
                )
            response = response_result.value
            pending.remove(handle)
            if isinstance(response, WorkerCommandRejected):
                return await _abort_worker_sampling_start(
                    handles=commanded_handles,
                    policy_version=policy_version,
                    rollout_id=rollout_id,
                    timeout_seconds=stop_timeout_seconds,
                    failure=_worker_command_rejection(response),
                )
            if isinstance(response, WorkerSamplingStarted):
                if response.policy_version != policy_version:
                    return await _abort_worker_sampling_start(
                        handles=commanded_handles,
                        policy_version=policy_version,
                        rollout_id=rollout_id,
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
                rollout_id=rollout_id,
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
            rollout_id=rollout_id,
            commanded_handles=commanded_handles,
            started_handles=tuple(
                sorted(started_handles, key=lambda item: item.index)
            ),
        )
    )


async def _stop_worker_sampling_after_start_request(
    *,
    handles: tuple[WorkerControlHandle, ...],
    policy_version: int,
    rollout_id: str,
    timeout_seconds: float,
) -> (
    _result.Ok[WorkerSamplingStartOutcome] | WorkerSamplingCleanupFailed
):
    send_result = await _send_worker_stop_sampling_commands(
        handles=handles,
        policy_version=policy_version,
        rollout_id=rollout_id,
    )
    if isinstance(send_result, Rejected):
        return WorkerSamplingCleanupFailed(
            rejection=Rejected(
                reason=(
                    "stop-requested sampling cleanup failed: "
                    f"{send_result.reason}"
                )
            )
        )
    stopped_result = await _receive_worker_sampling_abort_stopped(
        handles=handles,
        policy_version=policy_version,
        timeout_seconds=timeout_seconds,
    )
    if isinstance(stopped_result, Rejected):
        return WorkerSamplingCleanupFailed(
            rejection=Rejected(
                reason=(
                    "stop-requested sampling cleanup failed: "
                    f"{stopped_result.reason}"
                )
            )
        )
    return Ok(value=WorkerSamplingStartStopped())


async def _abort_worker_sampling_start(
    *,
    handles: tuple[WorkerControlHandle, ...],
    policy_version: int,
    rollout_id: str,
    timeout_seconds: float,
    failure: Rejected,
) -> Rejected | WorkerSamplingCleanupFailed:
    if not handles:
        return failure
    send_result = await _send_worker_stop_sampling_commands(
        handles=handles,
        policy_version=policy_version,
        rollout_id=rollout_id,
    )
    if isinstance(send_result, Rejected):
        return WorkerSamplingCleanupFailed(
            rejection=_cleanup_rejection(
                failure=failure,
                cleanup=send_result,
            )
        )
    stopped_result = await _receive_worker_sampling_abort_stopped(
        handles=handles,
        policy_version=policy_version,
        timeout_seconds=timeout_seconds,
    )
    if isinstance(stopped_result, Rejected):
        return WorkerSamplingCleanupFailed(
            rejection=_cleanup_rejection(
                failure=failure,
                cleanup=stopped_result,
            )
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


async def _wait_worker_start_responses(
    *,
    handles: tuple[WorkerControlHandle, ...],
    deadline: float,
    stop_request: TrainingStopRequest,
) -> (
    _result.Ok[tuple[WorkerControlHandle, ...]]
    | _result.Rejected
    | _WorkerSamplingStartStopRequested
):
    while True:
        if stop_request.is_requested():
            return _WorkerSamplingStartStopRequested()
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return Rejected(reason="process control response timed out")
        ready_result = await poll_async_control_responses(
            endpoints=tuple(handle.control for handle in handles),
            timeout_seconds=min(
                remaining, _STOP_CHECK_INTERVAL_SECONDS
            ),
        )
        if isinstance(ready_result, Rejected):
            return ready_result
        if ready_result.value:
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
