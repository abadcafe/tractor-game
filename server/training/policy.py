"""Policy interfaces used by TrainingPlayer and trainers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from server.result import Ok
from server.training.action_tokens import (
    BEGIN_TOKEN_ID,
    MAX_ACTION_TOKENS,
    STOP_TOKEN_ID,
    ActionQuery,
    GeneratedAction,
    decode_action_tokens,
    valid_next_token_ids,
)
from server.training.observation import Observation


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Policy output plus trace values for one generated action."""

    action: GeneratedAction
    log_probability: float
    value_estimate: float
    entropy: float
    token_count: int


class TrainingPolicy(Protocol):
    """Policy abstraction consumed by TrainingPlayer."""

    def decide(
        self,
        observation: Observation,
        query: ActionQuery,
    ) -> PolicyDecision: ...


class RandomTrainingPolicy:
    """Verified random token policy for smoke runs."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def decide(
        self,
        observation: Observation,
        query: ActionQuery,
    ) -> PolicyDecision:
        prefix = (BEGIN_TOKEN_ID,)
        log_probability = 0.0
        entropy = 0.0
        for _ in range(MAX_ACTION_TOKENS - 1):
            allowed = valid_next_token_ids(query, prefix)
            assert allowed
            token_id = self._rng.choice(allowed)
            probability = 1.0 / len(allowed)
            log_probability += math.log(probability)
            entropy += math.log(len(allowed))
            prefix = (*prefix, token_id)
            if token_id == STOP_TOKEN_ID:
                break
        decoded = decode_action_tokens(query, prefix)
        assert isinstance(decoded, Ok)
        return PolicyDecision(
            action=decoded.value,
            log_probability=log_probability,
            value_estimate=0.0,
            entropy=entropy,
            token_count=len(decoded.value.token_ids),
        )
