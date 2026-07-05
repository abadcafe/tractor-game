"""Inline model-rank implementation used inside CPU workers."""

from __future__ import annotations

from typing import Protocol

from server import result as _result
from server.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.policy_request_frame import (
    CompletedPolicyResponseFrame,
    PolicyRequestBatchFrame,
    build_policy_request_frame,
    decode_policy_response,
)
from server.training.policy_sampling import ModelRankPolicyDecision
from server.training.ppo import PPOUpdateStats
from server.training.runtime.model_rank.update import ModelUpdateResult
from server.training.runtime.state import RuntimeTrainingState
from server.training.runtime.update_wave import SynchronizedUpdateShard
from server.training.sampling import PolicyDecisionKey


class ModelReplicaProtocol(Protocol):
    """Model replica operations used by inline model ranks."""

    def load_state(self, *, snapshot: RuntimeTrainingState) -> None: ...

    def decide_batch(
        self, requests: PolicyRequestBatchFrame
    ) -> tuple[
        _result.Ok[ModelRankPolicyDecision] | _result.Rejected, ...
    ]: ...

    def update_shard(
        self, *, shard: SynchronizedUpdateShard
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected: ...

    def snapshot(self) -> RuntimeTrainingState: ...


class DirectPolicyClient:
    """Policy client that calls a same-process model replica."""

    def __init__(self, *, replica: ModelReplicaProtocol) -> None:
        self._replica = replica

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        frame_result = build_policy_request_frame(
            observation=observation,
            legal_actions=legal_actions,
            decision_key=decision_key,
        )
        if isinstance(frame_result, Rejected):
            return frame_result
        result = self._replica.decide_batch(
            PolicyRequestBatchFrame(frames=(frame_result.value,))
        )[0]
        if isinstance(result, Rejected):
            return result
        return decode_policy_response(
            legal_actions=legal_actions,
            response=CompletedPolicyResponseFrame(
                trace_token_ids=result.value.trace_token_ids,
                decision_handle=result.value.decision_handle,
                choice_count=result.value.choice_count,
            ),
        )


class InlineModelRank:
    """Model rank hosted inside a rollout worker process."""

    def __init__(self, *, replica: ModelReplicaProtocol) -> None:
        self._replica = replica

    def load_state(
        self,
        *,
        state: RuntimeTrainingState,
        policy_version: int,
    ) -> Ok[None] | Rejected:
        assert policy_version >= 0
        self._replica.load_state(snapshot=state)
        return Ok(value=None)

    def update(
        self,
        *,
        shard: SynchronizedUpdateShard,
        policy_version: int,
    ) -> _result.Ok[ModelUpdateResult] | _result.Rejected:
        assert policy_version >= 0
        assert shard.policy_version == policy_version
        update_result = self._replica.update_shard(shard=shard)
        if isinstance(update_result, Rejected):
            return update_result
        return Ok(
            value=ModelUpdateResult(
                update_stats=update_result.value,
                state=self._replica.snapshot(),
            )
        )

    def snapshot(
        self,
    ) -> _result.Ok[RuntimeTrainingState] | _result.Rejected:
        return Ok(value=self._replica.snapshot())
