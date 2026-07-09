"""Worker-local batched model-rank implementation."""

from __future__ import annotations

from typing import Protocol

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_sampling import (
    RankReturnTargets,
)
from server.training.ppo import PPOUpdateStats
from server.training.runtime.state import RuntimeTrainingState


class ModelReplicaProtocol(Protocol):
    """Model replica operations used by same-process model ranks."""

    def load_state(self, *, snapshot: RuntimeTrainingState) -> None: ...

    def update_returns(
        self, *, returns: RankReturnTargets
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected: ...

    def snapshot(self) -> RuntimeTrainingState: ...


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
        returns: RankReturnTargets,
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
