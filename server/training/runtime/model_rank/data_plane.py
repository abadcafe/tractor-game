"""Command-scoped model-rank inference data plane."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Protocol, cast

from server import result as _result
from server.result import Rejected
from server.training.policy_inference_wire import PolicyRequestRoute
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyRequestReceiver,
)
from server.training.runtime.model_rank.messages import (
    ModelRankCommand,
    ModelRankResponse,
)
from server.training.runtime.model_rank.staging import (
    PolicyRequestStager,
    StagedPolicyRequestBatch,
)
from server.training.runtime.process_control import (
    ChildControlEndpoint,
    ControlReady,
)


class ModelRankBatchHandler(Protocol):
    """Process one inference batch inside a data-plane transfer."""

    def __call__(
        self, batch: StagedPolicyRequestBatch
    ) -> _result.Ok[None] | _result.Rejected: ...


class ModelRankRejectHandler(Protocol):
    """Reject one request batch inside a data-plane transfer."""

    def __call__(
        self, *, routes: tuple[PolicyRequestRoute, ...], reason: str
    ) -> _result.Ok[None] | _result.Rejected: ...


@dataclass(slots=True)
class ModelRankDataPlane:
    """Drain inference requests until the next coordinator command."""

    control: ChildControlEndpoint[ModelRankCommand, ModelRankResponse]
    request_receivers: tuple[ConnectionPolicyRequestReceiver, ...]
    stager: PolicyRequestStager

    def run_until_command(
        self,
        *,
        policy_version: int,
        process_batch: ModelRankBatchHandler,
        reject_batch: ModelRankRejectHandler,
    ) -> _result.Ok[ModelRankCommand] | _result.Rejected:
        """Process this policy version until a command is readable."""
        assert policy_version >= 0
        while True:
            ready_requests = self._ready_request_receivers(
                timeout_seconds=0.0
            )
            if ready_requests:
                request_result = self._handle_ready_requests(
                    initial_ready=ready_requests,
                    policy_version=policy_version,
                    process_batch=process_batch,
                    reject_batch=reject_batch,
                )
                if isinstance(request_result, Rejected):
                    return request_result
                continue
            if self.control.poll_command(0.0):
                return self.control.recv_command()
            ready = self._wait_for_input()
            if isinstance(ready, Rejected):
                return ready
            ready_value = ready.value
            ready_requests = self._request_receivers_from_ready(
                ready_value.connections
            )
            if ready_requests:
                request_result = self._handle_ready_requests(
                    initial_ready=ready_requests,
                    policy_version=policy_version,
                    process_batch=process_batch,
                    reject_batch=reject_batch,
                )
                if isinstance(request_result, Rejected):
                    return request_result
                continue
            assert ready_value.command_ready
            return self.control.recv_command()

    def _handle_ready_requests(
        self,
        *,
        initial_ready: tuple[ConnectionPolicyRequestReceiver, ...],
        policy_version: int,
        process_batch: ModelRankBatchHandler,
        reject_batch: ModelRankRejectHandler,
    ) -> _result.Ok[None] | _result.Rejected:
        batch_result = self._receive_ready_batch(
            initial_ready=initial_ready
        )
        if isinstance(batch_result, Rejected):
            return batch_result
        batch = batch_result.value
        if any(
            version != policy_version
            for version in batch.device_batch.policy_versions
        ):
            return reject_batch(
                routes=batch.routes,
                reason="policy request version does not match command",
            )
        return process_batch(batch)

    def _receive_ready_batch(
        self,
        *,
        initial_ready: tuple[ConnectionPolicyRequestReceiver, ...],
    ) -> _result.Ok[StagedPolicyRequestBatch] | _result.Rejected:
        assert initial_ready
        self.stager.begin_batch()
        ready_receivers = initial_ready
        while ready_receivers and self.stager.can_receive():
            for receiver in ready_receivers:
                if not self.stager.can_receive():
                    break
                receive_result = self.stager.receive_from(receiver)
                if isinstance(receive_result, Rejected):
                    self.stager.discard_batch()
                    return receive_result
            if self.stager.can_receive():
                ready_receivers = self._ready_request_receivers(
                    timeout_seconds=0.0
                )
        return self.stager.finish_batch()

    def _ready_request_receivers(
        self, *, timeout_seconds: float
    ) -> tuple[ConnectionPolicyRequestReceiver, ...]:
        if not self.request_receivers:
            return ()
        ready = wait(
            tuple(
                receiver.connection
                for receiver in self.request_receivers
            ),
            timeout=timeout_seconds,
        )
        return self._request_receivers_from_ready(
            _ready_connections(ready)
        )

    def _wait_for_input(
        self,
    ) -> _result.Ok[ControlReady] | _result.Rejected:
        request_connections = tuple(
            receiver.connection for receiver in self.request_receivers
        )
        return self.control.wait_command_or_connections(
            connections=request_connections,
            timeout_seconds=None,
        )

    def _request_receivers_from_ready(
        self, ready: tuple[Connection, ...]
    ) -> tuple[ConnectionPolicyRequestReceiver, ...]:
        return tuple(
            receiver
            for receiver in self.request_receivers
            if _connection_in_ready(receiver.connection, ready)
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
