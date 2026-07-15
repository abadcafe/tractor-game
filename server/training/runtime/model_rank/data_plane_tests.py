"""Tests for model-rank inference data-plane scheduling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import torch

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.game.players.test_helpers import card, make_snapshot
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.observation import Observation, build_observation
from server.training.policy_inference_batch import (
    PolicyRequestCompiler,
    PolicyRequestInput,
    PolicyRequestRoute,
)
from server.training.policy_inference_batch.types import (
    PolicyRequestWireFrame,
)
from server.training.runtime.async_ipc import (
    AsyncProcessControlLink,
    ProcessControlProtocol,
    create_async_process_control_link,
    create_async_socket_pair,
)
from server.training.runtime.model_rank.data_plane import (
    ModelRankDataPlane,
)
from server.training.runtime.model_rank.inference_transport import (
    AsyncPolicyPeer,
)
from server.training.runtime.model_rank.messages import (
    ModelRankCommand,
    ModelRankLoadStateCommand,
    ModelRankResponse,
    decode_model_rank_command,
    decode_model_rank_response,
)
from server.training.runtime.model_rank.staging import (
    ModelRankInferenceBatch,
    PolicyRequestIngress,
)
from server.training.runtime.state import RuntimeTrainingState
from server.training.sampling import PolicyDecisionKey

type _ModelRankControlLink = AsyncProcessControlLink[
    ModelRankCommand, ModelRankResponse
]

_MODEL_RANK_CONTROL_PROTOCOL: ProcessControlProtocol[
    ModelRankCommand, ModelRankResponse
] = ProcessControlProtocol(
    name="model-rank-test",
    decode_command=decode_model_rank_command,
    decode_response=decode_model_rank_response,
)
_TEST_MAX_OBSERVATION_TOKENS = 45


@dataclass(frozen=True, slots=True)
class _RequestLink:
    worker_peer: AsyncPolicyPeer
    model_rank_peer: AsyncPolicyPeer

    def close(self) -> None:
        self.worker_peer.close()
        self.model_rank_peer.close()


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

    async def process(
        self, batch: ModelRankInferenceBatch
    ) -> _result.Ok[None] | _result.Rejected:
        self.processed_workers.append(
            tuple(route.worker_index for route in batch.routes)
        )
        return Ok(value=None)

    async def reject(
        self, *, routes: tuple[PolicyRequestRoute, ...], reason: str
    ) -> _result.Ok[None] | _result.Rejected:
        self.rejected_workers.append(
            tuple(route.worker_index for route in routes)
        )
        self.rejected_reasons.append(reason)
        return Ok(value=None)


async def test_run_until_command_drains_requests_before_command() -> (
    None
):
    control_link = _control_link()
    link0 = _request_link(worker_index=0)
    link1 = _request_link(worker_index=1)
    recorder = _DataPlaneRecorder()
    try:
        command_sent = await control_link.coordinator.send_command(
            ModelRankLoadStateCommand(
                state=_runtime_state(), policy_version=8
            )
        )
        assert isinstance(command_sent, Ok)
        data_plane = _data_plane(
            control_link=control_link,
            request_peers=(
                link0.model_rank_peer,
                link1.model_rank_peer,
            ),
            batch_size=4,
            max_observation_tokens=_TEST_MAX_OBSERVATION_TOKENS,
        )

        async with asyncio.TaskGroup() as senders:
            senders.create_task(
                _send_request(
                    link=link0,
                    worker_index=0,
                    policy_version=7,
                )
            )
            senders.create_task(
                _send_request(
                    link=link1,
                    worker_index=1,
                    policy_version=7,
                )
            )
            await _wait_for_ready_requests((link0, link1))
            command_result = await data_plane.run_until_command(
                policy_version=7,
                process_batch=recorder.process,
                reject_batch=recorder.reject,
            )

        assert isinstance(command_result, Ok)
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
        link0.close()
        link1.close()


async def test_run_until_command_returns_idle_command() -> None:
    control_link = _control_link()
    recorder = _DataPlaneRecorder()
    try:
        command_sent = await control_link.coordinator.send_command(
            ModelRankLoadStateCommand(
                state=_runtime_state(), policy_version=2
            )
        )
        assert isinstance(command_sent, Ok)
        data_plane = _data_plane(
            control_link=control_link,
            request_peers=(),
            batch_size=4,
        )

        command_result = await data_plane.run_until_command(
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


async def test_run_until_command_rejects_stale_policy_version() -> None:
    control_link = _control_link()
    link = _request_link(worker_index=2)
    recorder = _DataPlaneRecorder()
    max_observation_tokens = 45
    try:
        command_sent = await control_link.coordinator.send_command(
            ModelRankLoadStateCommand(
                state=_runtime_state(), policy_version=9
            )
        )
        assert isinstance(command_sent, Ok)
        data_plane = _data_plane(
            control_link=control_link,
            request_peers=(link.model_rank_peer,),
            batch_size=4,
            max_observation_tokens=max_observation_tokens,
        )

        async with asyncio.TaskGroup() as senders:
            senders.create_task(
                _send_request(
                    link=link,
                    worker_index=2,
                    policy_version=4,
                    max_observation_tokens=max_observation_tokens,
                )
            )
            await _wait_for_ready_requests((link,))
            command_result = await data_plane.run_until_command(
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
        link.close()


async def test_run_until_command_rejects_only_stale_routes() -> None:
    control_link = _control_link()
    link0 = _request_link(worker_index=0)
    link1 = _request_link(worker_index=1)
    link2 = _request_link(worker_index=2)
    recorder = _DataPlaneRecorder()
    max_observation_tokens = 45
    try:
        command_sent = await control_link.coordinator.send_command(
            ModelRankLoadStateCommand(
                state=_runtime_state(), policy_version=9
            )
        )
        assert isinstance(command_sent, Ok)
        data_plane = _data_plane(
            control_link=control_link,
            request_peers=(
                link0.model_rank_peer,
                link1.model_rank_peer,
                link2.model_rank_peer,
            ),
            batch_size=4,
            max_observation_tokens=max_observation_tokens,
        )

        async with asyncio.TaskGroup() as senders:
            senders.create_task(
                _send_request(
                    link=link0,
                    worker_index=0,
                    policy_version=4,
                    max_observation_tokens=max_observation_tokens,
                )
            )
            senders.create_task(
                _send_request(
                    link=link1,
                    worker_index=1,
                    policy_version=5,
                    max_observation_tokens=max_observation_tokens,
                )
            )
            senders.create_task(
                _send_request(
                    link=link2,
                    worker_index=2,
                    policy_version=4,
                    max_observation_tokens=max_observation_tokens,
                )
            )
            await _wait_for_ready_requests((link0, link1, link2))
            command_result = await data_plane.run_until_command(
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
        link0.close()
        link1.close()
        link2.close()


async def test_run_until_command_batches_until_batch_size() -> None:
    control_link = _control_link()
    link0 = _request_link(worker_index=0)
    link1 = _request_link(worker_index=1)
    link2 = _request_link(worker_index=2)
    recorder = _DataPlaneRecorder()
    max_observation_tokens = 45
    try:
        command_sent = await control_link.coordinator.send_command(
            ModelRankLoadStateCommand(
                state=_runtime_state(), policy_version=6
            )
        )
        assert isinstance(command_sent, Ok)
        data_plane = _data_plane(
            control_link=control_link,
            request_peers=(
                link0.model_rank_peer,
                link1.model_rank_peer,
                link2.model_rank_peer,
            ),
            batch_size=2,
            max_observation_tokens=max_observation_tokens,
        )

        async with asyncio.TaskGroup() as senders:
            senders.create_task(
                _send_request(
                    link=link0,
                    worker_index=0,
                    policy_version=3,
                    max_observation_tokens=max_observation_tokens,
                )
            )
            senders.create_task(
                _send_request(
                    link=link1,
                    worker_index=1,
                    policy_version=3,
                    max_observation_tokens=max_observation_tokens,
                )
            )
            senders.create_task(
                _send_request(
                    link=link2,
                    worker_index=2,
                    policy_version=3,
                    max_observation_tokens=max_observation_tokens,
                )
            )
            await _wait_for_ready_requests((link0, link1, link2))
            command_result = await data_plane.run_until_command(
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
        link0.close()
        link1.close()
        link2.close()


async def test_ingress_reuses_cuda_single_frame_slot() -> None:
    if not torch.cuda.is_available():
        return
    link = _request_link(worker_index=0)
    ingress = PolicyRequestIngress(
        batch_size=4,
        max_observation_tokens=_TEST_MAX_OBSERVATION_TOKENS,
        device=torch.device("cuda:0"),
    )
    try:
        first_batch = await _receive_single_frame_batch(
            ingress=ingress,
            link=link,
            worker_index=0,
            policy_version=11,
        )
        first_component_ptr = int(
            first_batch.device_batch.observation_batch.component_ids.data_ptr()
        )

        second_batch = await _receive_single_frame_batch(
            ingress=ingress,
            link=link,
            worker_index=0,
            policy_version=12,
        )
        second_component_ptr = int(
            second_batch.device_batch.observation_batch.component_ids.data_ptr()
        )

        assert second_component_ptr == first_component_ptr
        assert second_batch.device_batch.policy_versions == (12,)
        generation_count = int(
            second_batch.device_batch.generation_step_counts.cpu()[
                0
            ].item()
        )
        assert generation_count > 0
    finally:
        link.close()


async def test_ingress_materializes_mps_thresholds_as_float32() -> None:
    if not torch.backends.mps.is_available():
        return
    link = _request_link(worker_index=0)
    ingress = PolicyRequestIngress(
        batch_size=4,
        max_observation_tokens=_TEST_MAX_OBSERVATION_TOKENS,
        device=torch.device("mps"),
    )
    try:
        batch = await _receive_single_frame_batch(
            ingress=ingress,
            link=link,
            worker_index=0,
            policy_version=11,
        )

        thresholds = batch.device_batch.sampling_thresholds
        assert thresholds.device.type == "mps"
        assert thresholds.dtype == torch.float32
        assert bool(torch.isfinite(thresholds).all().cpu().item())
        assert bool((thresholds >= 0.0).all().cpu().item())
        assert bool((thresholds < 1.0).all().cpu().item())
    finally:
        link.close()


async def test_ingress_aggregates_mps_thresholds_as_float32() -> None:
    if not torch.backends.mps.is_available():
        return
    link0 = _request_link(worker_index=0)
    link1 = _request_link(worker_index=1)
    ingress = PolicyRequestIngress(
        batch_size=4,
        max_observation_tokens=_TEST_MAX_OBSERVATION_TOKENS,
        device=torch.device("mps:0"),
    )
    try:
        ingress.begin_batch()
        async with asyncio.TaskGroup() as senders:
            senders.create_task(
                _send_request(
                    link=link0,
                    worker_index=0,
                    policy_version=13,
                )
            )
            senders.create_task(
                _send_request(
                    link=link1,
                    worker_index=1,
                    policy_version=13,
                )
            )
            await _wait_for_ready_requests((link0, link1))
            first_result = await ingress.receive_from(
                link0.model_rank_peer
            )
            second_result = await ingress.receive_from(
                link1.model_rank_peer
            )
        assert isinstance(first_result, Ok)
        assert isinstance(second_result, Ok)
        batch_result = ingress.finish_batch()
        assert isinstance(batch_result, Ok)

        batch = batch_result.value.device_batch
        assert batch.policy_versions == (13, 13)
        assert batch.sampling_thresholds.device == torch.device("mps:0")
        assert batch.sampling_thresholds.dtype == torch.float32
        assert bool(
            (batch.sampling_thresholds < 1.0).all().cpu().item()
        )
    finally:
        link0.close()
        link1.close()


def _control_link() -> _ModelRankControlLink:
    return create_async_process_control_link(
        protocol=_MODEL_RANK_CONTROL_PROTOCOL
    )


def _request_link(*, worker_index: int) -> _RequestLink:
    pair = create_async_socket_pair()
    return _RequestLink(
        worker_peer=AsyncPolicyPeer(
            worker_index=worker_index,
            endpoint=pair.first,
        ),
        model_rank_peer=AsyncPolicyPeer(
            worker_index=worker_index,
            endpoint=pair.second,
        ),
    )


def _data_plane(
    *,
    control_link: _ModelRankControlLink,
    request_peers: tuple[AsyncPolicyPeer, ...],
    batch_size: int,
    max_observation_tokens: int = 512,
) -> ModelRankDataPlane:
    return ModelRankDataPlane(
        control=control_link.child,
        request_peers=request_peers,
        ingress=PolicyRequestIngress(
            batch_size=batch_size,
            max_observation_tokens=max_observation_tokens,
            device=torch.device("cpu"),
        ),
    )


async def _send_request(
    *,
    link: _RequestLink,
    worker_index: int,
    policy_version: int,
    max_observation_tokens: int = _TEST_MAX_OBSERVATION_TOKENS,
) -> None:
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
    send_result = await link.worker_peer.send_request(
        frame_result.value
    )
    assert isinstance(send_result, Ok)


async def _wait_for_ready_requests(
    links: tuple[_RequestLink, ...],
) -> None:
    for link in links:
        ready_result = (
            await link.model_rank_peer.endpoint.wait_readable(
                timeout_seconds=5.0
            )
        )
        assert isinstance(ready_result, Ok)
        assert ready_result.value


async def _receive_single_frame_batch(
    *,
    ingress: PolicyRequestIngress,
    link: _RequestLink,
    worker_index: int,
    policy_version: int,
) -> ModelRankInferenceBatch:
    ingress.begin_batch()
    async with asyncio.TaskGroup() as senders:
        senders.create_task(
            _send_request(
                link=link,
                worker_index=worker_index,
                policy_version=policy_version,
            )
        )
        await _wait_for_ready_requests((link,))
        receive_result = await ingress.receive_from(
            link.model_rank_peer
        )
    assert isinstance(receive_result, Ok)
    batch_result = ingress.finish_batch()
    assert isinstance(batch_result, Ok)
    return batch_result.value


def _request_frame(
    *,
    requests: tuple[PolicyRequestInput, ...],
    batch_capacity: int,
    max_observation_tokens: int,
) -> Ok[PolicyRequestWireFrame] | Rejected:
    compiler = PolicyRequestCompiler(
        batch_capacity=batch_capacity,
        max_observation_tokens=max_observation_tokens,
    )
    batch_result = compiler.compile_batch(requests)
    if isinstance(batch_result, Rejected):
        return batch_result
    return Ok(value=batch_result.value.frame)


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
        rollout_id=f"rollout-{policy_version}",
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
