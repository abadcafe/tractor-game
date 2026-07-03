"""Policy interfaces used by TrainingPlayer and trainers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from server.result import Ok
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.semantic_actions import (
    MAX_ARGUMENT_TOKENS,
    GeneratedAction,
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Policy output plus trace values for one generated action."""

    action: GeneratedAction
    log_probability: float
    value_estimate: float
    entropy: float
    choice_count: int


class TrainingPolicy(Protocol):
    """Policy abstraction consumed by TrainingPlayer."""

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
    ) -> PolicyDecision: ...


class RandomTrainingPolicy:
    """Verified random semantic policy for smoke runs."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
    ) -> PolicyDecision:
        prefix = SemanticArgumentPrefix(arguments=())
        arguments: list[SemanticArgument] = []
        log_probability = 0.0
        entropy = 0.0
        choice_count = 0
        for _ in range(MAX_ARGUMENT_TOKENS):
            allowed = legal_actions.allowed_next(prefix)
            if not allowed:
                break
            choice_count += len(allowed)
            argument = self._rng.choice(allowed)
            probability = 1.0 / len(allowed)
            log_probability += math.log(probability)
            entropy += math.log(len(allowed))
            arguments.append(argument)
            if argument.kind in ("pass", "stop"):
                break
            assert argument.kind == "select_face_count"
            prefix = SemanticArgumentPrefix(
                arguments=(*prefix.arguments, argument)
            )
        else:
            assert False
        trace = SemanticArgumentTrace(arguments=tuple(arguments))
        decoded = legal_actions.decode(trace)
        assert isinstance(decoded, Ok)
        return PolicyDecision(
            action=decoded.value,
            log_probability=log_probability,
            value_estimate=0.0,
            entropy=entropy,
            choice_count=choice_count,
        )
