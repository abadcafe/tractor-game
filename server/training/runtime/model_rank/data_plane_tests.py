"""Tests for model-rank inference data-plane scheduling."""

from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext

import torch

from server import result as _result
from server.player.test_helpers import card, make_snapshot
from server.result import Ok, Rejected
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.observation import Observation, build_observation
from server.training.policy_inference_batch import (
    PolicyRequestBatchBuilder,
    PolicyRequestInput,
    PolicyRequestRoute,
)
from server.training.policy_inference_batch.types import (
    PolicyRequestWireFrame,
)
from server.training.runtime.model_rank.data_plane import (
    ModelRankDataPlane,
)
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyRequestReceiver,
    ConnectionPolicyRequestSender,
)
from server.training.runtime.model_rank.messages import (
    ModelRankCommand,
    ModelRankLoadStateCommand,
    ModelRankResponse,
)
from server.training.runtime.model_rank.staging import (
    ModelRankInferenceBatch,
    PolicyRequestIngress,
)
from server.training.runtime.process_control import (
    ProcessControlLink,
    ProcessControlProtocol,
    create_process_control_link,
)
from server.training.runtime.state import RuntimeTrainingState
from server.training.sampling import PolicyDecisionKey

type _ModelRankControlLink = ProcessControlLink[
    ModelRankCommand, ModelRankResponse
]

_MODEL_RANK_CONTROL_PROTOCOL: ProcessControlProtocol[
    ModelRankCommand, ModelRankResponse
] = ProcessControlProtocol(name="model-rank-test")
_TEST_MAX_OBSERVATION_TOKENS = 45


@dataclass(frozen=True, slots=True)
class _RequestLink:
    receiver: Connection
    sender: Connection


@dataclass(slots=True)
class _RequestSendTask:
    thread: threading.Thread
    result: list[Ok[None] | Rejected]


@dataclass(frozen=True, slots=True)
class _PendingRequest:
    worker_index: int
    policy_version: int


def _worker_log() -> list[tuple[int, ...]]:
    return []


def _reason_log() -> list[str]:
    return []


@dataclass(slots=True)
class _DataPlaneRecorder:
    processed_workers: list[tuple[int, ...]] = field(
        default_factory=_worker_log
    )
    rejected_workers: list[tuple[int, ...]] = field(
        default_factory=_worker_log
    )
    rejected_reasons: list[str] = field(default_factory=_reason_log)

    def process(
        self, batch: ModelRankInferenceBatch
    ) -> _result.Ok[None] | _result.Rejected:
        self.processed_workers.append(
            tuple(route.worker_index for route in batch.routes)
        )
        return Ok(value=None)

    def reject(
        self, *, routes: tuple[PolicyRequestRoute, ...], reason: str
    ) -> _result.Ok[None] | _result.Rejected:
        self.rejected_workers.append(
            tuple(route.worker_index for route in routes)
        )
        self.rejected_reasons.append(reason)
        return Ok(value=None)


def test_run_until_command_drains_requests_before_ready_command() -> (
    None
):
    context = _spawn_context()
    control_link = create_process_control_link(
        context=context,
        protocol=_MODEL_RANK_CONTROL_PROTOCOL,
    )
    link0 = _request_link(context)
    link1 = _request_link(context)
    recorder = _DataPlaneRecorder()
    try:
        command_sent = control_link.coordinator.send_command(
            ModelRankLoadStateCommand(
                state=_runtime_state(), policy_version=8
            )
        )
        assert isinstance(command_sent, Ok)
        send_tasks = (
            _start_request_send(
                link=link0, worker_index=0, policy_version=7
            ),
            _start_request_send(
                link=link1, worker_index=1, policy_version=7
            ),
        )
        data_plane = _data_plane(
            control_link=control_link,
            request_receivers=(
                ConnectionPolicyRequestReceiver(link0.receiver),
                ConnectionPolicyRequestReceiver(link1.receiver),
            ),
            batch_size=4,
            max_observation_tokens=_TEST_MAX_OBSERVATION_TOKENS,
        )

        command_result = data_plane.run_until_command(
            policy_version=7,
            process_batch=recorder.process,
            reject_batch=recorder.reject,
        )

        assert isinstance(command_result, Ok)
        _finish_request_sends(send_tasks)
        assert isinstance(
            command_result.value, ModelRankLoadStateCommand
        )
        assert command_result.value.policy_version == 8
        assert len(recorder.processed_workers) == 1
        assert set(recorder.processed_workers[0]) == {0, 1}
        assert recorder.rejected_workers == []
    finally:
        control_link.coordinator.close()
        control_link.child.close()
        _close_connections(
            link0.receiver,
            link0.sender,
            link1.receiver,
            link1.sender,
        )


def test_run_until_command_returns_command_when_data_channel_idle() -> (
    None
):
    context = _spawn_context()
    control_link = create_process_control_link(
        context=context,
        protocol=_MODEL_RANK_CONTROL_PROTOCOL,
    )
    recorder = _DataPlaneRecorder()
    try:
        command_sent = control_link.coordinator.send_command(
            ModelRankLoadStateCommand(
                state=_runtime_state(), policy_version=2
            )
        )
        assert isinstance(command_sent, Ok)
        data_plane = _data_plane(
            control_link=control_link,
            request_receivers=(),
            batch_size=4,
        )

        command_result = data_plane.run_until_command(
            policy_version=1,
            process_batch=recorder.process,
            reject_batch=recorder.reject,
        )

        assert isinstance(command_result, Ok)
        assert isinstance(
            command_result.value, ModelRankLoadStateCommand
        )
        assert command_result.value.policy_version == 2
        assert recorder.processed_workers == []
        assert recorder.rejected_workers == []
    finally:
        control_link.coordinator.close()
        control_link.child.close()


def test_run_until_command_rejects_mismatched_policy_version() -> None:
    context = _spawn_context()
    control_link = create_process_control_link(
        context=context,
        protocol=_MODEL_RANK_CONTROL_PROTOCOL,
    )
    link = _request_link(context)
    recorder = _DataPlaneRecorder()
    max_observation_tokens = 45
    try:
        _send_request(
            link=link,
            worker_index=2,
            policy_version=4,
            max_observation_tokens=max_observation_tokens,
        )
        command_sent = control_link.coordinator.send_command(
            ModelRankLoadStateCommand(
                state=_runtime_state(), policy_version=9
            )
        )
        assert isinstance(command_sent, Ok)
        data_plane = _data_plane(
            control_link=control_link,
            request_receivers=(
                ConnectionPolicyRequestReceiver(link.receiver),
            ),
            batch_size=4,
            max_observation_tokens=max_observation_tokens,
        )

        command_result = data_plane.run_until_command(
            policy_version=5,
            process_batch=recorder.process,
            reject_batch=recorder.reject,
        )

        assert isinstance(command_result, Ok)
        assert isinstance(
            command_result.value, ModelRankLoadStateCommand
        )
        assert command_result.value.policy_version == 9
        assert recorder.processed_workers == []
        assert recorder.rejected_workers == [(2,)]
        assert recorder.rejected_reasons == [
            "policy request version does not match command"
        ]
    finally:
        control_link.coordinator.close()
        control_link.child.close()
        _close_connections(link.receiver, link.sender)


def test_run_until_command_rejects_only_mismatched_policy_routes() -> (
    None
):
    context = _spawn_context()
    control_link = create_process_control_link(
        context=context,
        protocol=_MODEL_RANK_CONTROL_PROTOCOL,
    )
    link0 = _request_link(context)
    link1 = _request_link(context)
    link2 = _request_link(context)
    recorder = _DataPlaneRecorder()
    max_observation_tokens = 45
    try:
        command_sent = control_link.coordinator.send_command(
            ModelRankLoadStateCommand(
                state=_runtime_state(), policy_version=9
            )
        )
        assert isinstance(command_sent, Ok)
        _send_request(
            link=link0,
            worker_index=0,
            policy_version=4,
            max_observation_tokens=max_observation_tokens,
        )
        _send_request(
            link=link1,
            worker_index=1,
            policy_version=5,
            max_observation_tokens=max_observation_tokens,
        )
        _send_request(
            link=link2,
            worker_index=2,
            policy_version=4,
            max_observation_tokens=max_observation_tokens,
        )
        data_plane = _data_plane(
            control_link=control_link,
            request_receivers=(
                ConnectionPolicyRequestReceiver(link0.receiver),
                ConnectionPolicyRequestReceiver(link1.receiver),
                ConnectionPolicyRequestReceiver(link2.receiver),
            ),
            batch_size=4,
            max_observation_tokens=max_observation_tokens,
        )

        command_result = data_plane.run_until_command(
            policy_version=5,
            process_batch=recorder.process,
            reject_batch=recorder.reject,
        )

        assert isinstance(command_result, Ok)
        assert isinstance(
            command_result.value, ModelRankLoadStateCommand
        )
        assert command_result.value.policy_version == 9
        assert recorder.processed_workers == [(1,)]
        assert recorder.rejected_workers == [(0, 2)]
        assert recorder.rejected_reasons == [
            "policy request version does not match command"
        ]
    finally:
        control_link.coordinator.close()
        control_link.child.close()
        _close_connections(
            link0.receiver,
            link0.sender,
            link1.receiver,
            link1.sender,
            link2.receiver,
            link2.sender,
        )


def test_run_until_command_batches_until_batch_size() -> None:
    context = _spawn_context()
    control_link = create_process_control_link(
        context=context,
        protocol=_MODEL_RANK_CONTROL_PROTOCOL,
    )
    link0 = _request_link(context)
    link1 = _request_link(context)
    link2 = _request_link(context)
    recorder = _DataPlaneRecorder()
    max_observation_tokens = 45
    try:
        command_sent = control_link.coordinator.send_command(
            ModelRankLoadStateCommand(
                state=_runtime_state(), policy_version=6
            )
        )
        assert isinstance(command_sent, Ok)
        _send_request(
            link=link0,
            worker_index=0,
            policy_version=3,
            max_observation_tokens=max_observation_tokens,
        )
        _send_request(
            link=link1,
            worker_index=1,
            policy_version=3,
            max_observation_tokens=max_observation_tokens,
        )
        _send_request(
            link=link2,
            worker_index=2,
            policy_version=3,
            max_observation_tokens=max_observation_tokens,
        )
        data_plane = _data_plane(
            control_link=control_link,
            request_receivers=(
                ConnectionPolicyRequestReceiver(link0.receiver),
                ConnectionPolicyRequestReceiver(link1.receiver),
                ConnectionPolicyRequestReceiver(link2.receiver),
            ),
            batch_size=2,
            max_observation_tokens=max_observation_tokens,
        )

        command_result = data_plane.run_until_command(
            policy_version=3,
            process_batch=recorder.process,
            reject_batch=recorder.reject,
        )

        assert isinstance(command_result, Ok)
        assert isinstance(
            command_result.value, ModelRankLoadStateCommand
        )
        assert command_result.value.policy_version == 6
        assert recorder.processed_workers == [(0, 1), (2,)]
        assert recorder.rejected_workers == []
    finally:
        control_link.coordinator.close()
        control_link.child.close()
        _close_connections(
            link0.receiver,
            link0.sender,
            link1.receiver,
            link1.sender,
            link2.receiver,
            link2.sender,
        )


def _spawn_context() -> SpawnContext:
    return mp.get_context("spawn")


def _request_link(context: SpawnContext) -> _RequestLink:
    receiver, sender = context.Pipe(duplex=False)
    return _RequestLink(receiver=receiver, sender=sender)


def _data_plane(
    *,
    control_link: _ModelRankControlLink,
    request_receivers: tuple[ConnectionPolicyRequestReceiver, ...],
    batch_size: int,
    max_observation_tokens: int = 512,
) -> ModelRankDataPlane:
    return ModelRankDataPlane(
        control=control_link.child,
        request_receivers=request_receivers,
        ingress=PolicyRequestIngress(
            batch_size=batch_size,
            max_observation_tokens=max_observation_tokens,
            device=torch.device("cpu"),
        ),
    )


def _start_request_send(
    *, link: _RequestLink, worker_index: int, policy_version: int
) -> _RequestSendTask:
    return _start_request_send_batch(
        link=link,
        requests=(
            _PendingRequest(
                worker_index=worker_index,
                policy_version=policy_version,
            ),
        ),
    )


def _start_request_send_batch(
    *, link: _RequestLink, requests: tuple[_PendingRequest, ...]
) -> _RequestSendTask:
    assert requests
    inputs = tuple(
        _request_input(
            worker_index=request.worker_index,
            policy_version=request.policy_version,
        )
        for request in requests
    )
    result: list[Ok[None] | Rejected] = []
    sender = ConnectionPolicyRequestSender(link.sender)

    def send_requests() -> None:
        frame_result = _request_frame(
            requests=inputs,
            batch_capacity=len(inputs),
            max_observation_tokens=_TEST_MAX_OBSERVATION_TOKENS,
        )
        assert isinstance(frame_result, Ok)
        send_result = sender.send(frame_result.value)
        result.append(send_result)

    thread = threading.Thread(target=send_requests)
    thread.start()
    return _RequestSendTask(thread=thread, result=result)


def _send_request(
    *,
    link: _RequestLink,
    worker_index: int,
    policy_version: int,
    max_observation_tokens: int,
) -> None:
    sender = ConnectionPolicyRequestSender(link.sender)
    frame_result = _request_frame(
        requests=(
            _request_input(
                worker_index=worker_index,
                policy_version=policy_version,
            ),
        ),
        batch_capacity=1,
        max_observation_tokens=max_observation_tokens,
    )
    assert isinstance(frame_result, Ok)
    send_result = sender.send(frame_result.value)
    assert isinstance(send_result, Ok)


def _request_frame(
    *,
    requests: tuple[PolicyRequestInput, ...],
    batch_capacity: int,
    max_observation_tokens: int,
) -> Ok[PolicyRequestWireFrame] | Rejected:
    preparer = PolicyRequestBatchBuilder(
        batch_capacity=batch_capacity,
        max_observation_tokens=max_observation_tokens,
    )
    batch_result = preparer.compile_batch(requests)
    if isinstance(batch_result, Rejected):
        return batch_result
    return Ok(value=preparer.encode_wire_frame(batch_result.value))


def _request_input(
    *,
    worker_index: int,
    policy_version: int,
) -> PolicyRequestInput:
    observation = _observation()
    return PolicyRequestInput(
        route=PolicyRequestRoute(
            worker_index=worker_index,
            request_id=worker_index,
        ),
        observation=observation,
        legal_actions=_legal_actions(observation),
        decision_key=_decision_key(policy_version=policy_version),
    )


def _finish_request_sends(tasks: tuple[_RequestSendTask, ...]) -> None:
    for task in tasks:
        task.thread.join(timeout=5.0)
        assert not task.thread.is_alive()
        assert task.result
        assert all(isinstance(result, Ok) for result in task.result)


def _observation() -> Observation:
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2", 1)],
        trump_rank="2",
    )
    return build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )


def _legal_actions(observation: Observation) -> LegalActionIndex:
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2", 1)],
        trump_rank="2",
    )
    return build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )


def _decision_key(*, policy_version: int) -> PolicyDecisionKey:
    return PolicyDecisionKey(
        base_seed=0,
        policy_version=policy_version,
        episode_id=0,
        player_index=0,
        decision_index=0,
    )


def _runtime_state() -> RuntimeTrainingState:
    return RuntimeTrainingState(
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={
            "kind": "adamw",
            "step_count": 0,
            "exp_avgs": [],
            "exp_avg_sqs": [],
        },
    )


def _close_connections(*connections: Connection) -> None:
    for connection in connections:
        connection.close()
