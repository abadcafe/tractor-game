"""Policy interfaces used by TrainingPlayer and trainers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from server.result import Ok
from server.training.observation import Observation
from server.training.selection_actions import (
    MAX_HAND_CARD_SLOTS,
    ActionQuery,
    GeneratedAction,
    SelectionChoice,
    SelectionState,
    SelectionTrace,
    decode_selection_action,
    valid_selection_choices,
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
        query: ActionQuery,
    ) -> PolicyDecision: ...


class RandomTrainingPolicy:
    """Verified random selection policy for smoke runs."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def decide(
        self,
        observation: Observation,
        query: ActionQuery,
    ) -> PolicyDecision:
        state = SelectionState(selected_slots=())
        choices: list[SelectionChoice] = []
        log_probability = 0.0
        entropy = 0.0
        for _ in range(MAX_HAND_CARD_SLOTS + 2):
            allowed = valid_selection_choices(query, state)
            if not allowed:
                break
            choice = self._rng.choice(allowed)
            probability = 1.0 / len(allowed)
            log_probability += math.log(probability)
            entropy += math.log(len(allowed))
            choices.append(choice)
            if choice.kind in ("pass", "stop"):
                break
            assert choice.kind == "select_card"
            assert choice.slot is not None
            state = SelectionState(
                selected_slots=(*state.selected_slots, choice.slot)
            )
        else:
            assert False
        trace = SelectionTrace(choices=tuple(choices))
        decoded = decode_selection_action(query, trace)
        assert isinstance(decoded, Ok)
        return PolicyDecision(
            action=decoded.value,
            log_probability=log_probability,
            value_estimate=0.0,
            entropy=entropy,
            choice_count=len(decoded.value.selection_trace.choices),
        )
