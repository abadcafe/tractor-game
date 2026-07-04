"""Policy interfaces used by TrainingPlayer and trainers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

import torch

from server.result import Ok, Rejected
from server.training.choice_trace import (
    SemanticChoiceStep,
    SemanticChoiceTrace,
    semantic_choice_step_from_offset,
)
from server.training.config import ModelConfig
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.semantic_actions.arguments import (
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.semantic_actions.values import GeneratedAction
from server.training.tensorize import (
    ObservationTensorBatch,
    tensorize_observation,
)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Policy output plus trace values for one generated action."""

    action: GeneratedAction
    observation_batch: ObservationTensorBatch
    choice_trace: SemanticChoiceTrace
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
    ) -> Ok[PolicyDecision] | Rejected: ...


class RandomTrainingPolicy:
    """Verified random semantic policy for smoke runs."""

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        self._model_config = model_config
        self._device = device
        self._rng = random.Random(seed)

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
    ) -> Ok[PolicyDecision] | Rejected:
        observation_batch = tensorize_observation(
            observation=observation,
            max_observation_tokens=self._model_config.max_tokens,
            device=self._device,
        )
        prefix = SemanticArgumentPrefix(arguments=())
        arguments: list[SemanticArgument] = []
        choice_steps: list[SemanticChoiceStep] = []
        log_probability = 0.0
        entropy = 0.0
        choice_count = 0
        for _ in range(SEMANTIC_CODEC.max_argument_tokens):
            allowed = legal_actions.allowed_next(prefix)
            if not allowed:
                break
            choice_count += len(allowed)
            selected_argument_offset = self._rng.randrange(len(allowed))
            argument = allowed[selected_argument_offset]
            choice_steps.append(
                semantic_choice_step_from_offset(
                    allowed=allowed,
                    selected_argument_offset=selected_argument_offset,
                )
            )
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
        return Ok(
            value=PolicyDecision(
                action=decoded.value,
                observation_batch=observation_batch,
                choice_trace=SemanticChoiceTrace(
                    steps=tuple(choice_steps)
                ),
                log_probability=log_probability,
                value_estimate=0.0,
                entropy=entropy,
                choice_count=choice_count,
            )
        )
