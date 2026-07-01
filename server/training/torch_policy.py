"""Torch-backed TrainingPolicy implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from server.result import Ok
from server.training.config import ModelConfig
from server.training.model import TractorPolicyModel
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.selection_actions import (
    MAX_HAND_CARD_SLOTS,
    ActionQuery,
    SelectionChoice,
    SelectionState,
    SelectionTrace,
    decode_selection_action,
    valid_selection_choices,
)
from server.training.selection_torch import (
    forward_selection_head,
    logits_for_choices,
)
from server.training.tensorize import (
    tensorize_observation,
    tensorize_selection_state,
)


class TorchTrainingPolicy:
    """Sample selection traces from a torch policy/value model."""

    def __init__(
        self,
        *,
        model: TractorPolicyModel,
        config: ModelConfig,
        device: torch.device,
        temperature: float = 1.0,
    ) -> None:
        assert temperature > 0.0
        self.model = model
        self.config = config
        self.device = device
        self.temperature = temperature

    def decide(
        self,
        observation: Observation,
        query: ActionQuery,
    ) -> PolicyDecision:
        self.model.eval()
        with torch.no_grad():
            observation_batch = tensorize_observation(
                observation=observation,
                max_observation_tokens=self.config.max_tokens,
                device=self.device,
            )
            state = SelectionState(selected_slots=())
            choices: list[SelectionChoice] = []
            log_probability = 0.0
            entropy = 0.0
            value_estimate = 0.0
            for _ in range(MAX_HAND_CARD_SLOTS + 2):
                selection_batch = tensorize_selection_state(
                    query=query,
                    state=state,
                    device=self.device,
                )
                output = forward_selection_head(
                    model=self.model,
                    query=query,
                    observation=observation_batch,
                    selection=selection_batch,
                )
                value_estimate = float(
                    output.values[0].detach().cpu().item()
                )
                allowed = valid_selection_choices(query, state)
                if not allowed:
                    break
                logits = logits_for_choices(output, allowed)
                masked_logits = logits / self.temperature
                probabilities = torch.softmax(masked_logits, dim=0)
                log_probabilities = torch.log_softmax(
                    masked_logits, dim=0
                )
                sampled = torch.multinomial(
                    probabilities, num_samples=1
                )
                choice_index = int(sampled[0].detach().cpu().item())
                choice = allowed[choice_index]
                log_probability += _float_tensor(
                    log_probabilities[choice_index]
                )
                entropy += _float_tensor(
                    -(probabilities * log_probabilities).sum()
                )
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
            value_estimate=value_estimate,
            entropy=entropy,
            choice_count=len(decoded.value.selection_trace.choices),
        )


def _float_tensor(value: Tensor) -> float:
    return float(value.detach().cpu().item())
