"""Worker-local model-rank implementation."""

from __future__ import annotations

from typing import Protocol

from server import result as _result
from server.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.policy_inference_wire import (
    CompletedPolicyResponse,
    PolicyRequestRoute,
    PolicyRequestWireBatch,
    build_policy_request_wire,
    decode_policy_response,
)
from server.training.policy_sampling import (
    ModelRankPolicyDecision,
    RankReturnBatch,
)
from server.training.ppo import PPOUpdateStats
from server.training.runtime.state import RuntimeTrainingState
from server.training.sampling import PolicyDecisionKey


class ModelReplicaProtocol(Protocol):
    """Model replica operations used by same-process model ranks."""

    def load_state(self, *, snapshot: RuntimeTrainingState) -> None: ...

    def decide_wires(
        self, requests: PolicyRequestWireBatch
    ) -> tuple[
        _result.Ok[ModelRankPolicyDecision] | _result.Rejected, ...
    ]: ...

    def update_returns(
        self, *, returns: RankReturnBatch
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected: ...

    def snapshot(self) -> RuntimeTrainingState: ...


class DirectPolicyClient:
    """Policy client that calls a same-process model replica."""

    def __init__(
        self,
        *,
        replica: ModelReplicaProtocol,
        max_observation_tokens: int,
    ) -> None:
        assert max_observation_tokens > 0
        self._replica = replica
        self._max_observation_tokens = max_observation_tokens

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        request_result = build_policy_request_wire(
            max_observation_tokens=self._max_observation_tokens,
            worker_index=0,
            request_id=decision_key.decision_index,
            observation=observation,
            legal_actions=legal_actions,
            decision_key=decision_key,
        )
        if isinstance(request_result, Rejected):
            return request_result
        result = self._replica.decide_wires(
            PolicyRequestWireBatch(requests=(request_result.value,))
        )[0]
        if isinstance(result, Rejected):
            return result
        return decode_policy_response(
            legal_actions=legal_actions,
            response=CompletedPolicyResponse(
                route=PolicyRequestRoute(
                    worker_index=0,
                    request_id=decision_key.decision_index,
                ),
                trace_token_ids=result.value.trace_token_ids,
                decision_handle_model_rank=(
                    result.value.decision_handle.model_rank_index
                ),
                decision_handle_policy_version=(
                    result.value.decision_handle.policy_version
                ),
                decision_handle_slot_index=(
                    result.value.decision_handle.slot_index
                ),
                decision_handle_slot_generation=(
                    result.value.decision_handle.slot_generation
                ),
                choice_count=result.value.choice_count,
            ),
        )


class LocalModelRank:
    """Model rank hosted inside the current worker process."""

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
        returns: RankReturnBatch,
        policy_version: int,
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected:
        assert policy_version >= 0
        assert returns.policy_version == policy_version
        update_result = self._replica.update_returns(returns=returns)
        if isinstance(update_result, Rejected):
            return update_result
        return Ok(value=update_result.value)

    def snapshot(
        self,
    ) -> _result.Ok[RuntimeTrainingState] | _result.Rejected:
        return Ok(value=self._replica.snapshot())
