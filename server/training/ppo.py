"""PPO updates for rewarded selection decisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from server.training.config import ModelConfig, TrainConfig
from server.training.model import TractorPolicyModel
from server.training.selection_actions import (
    SelectionChoice,
    SelectionState,
    valid_selection_choices,
)
from server.training.selection_torch import (
    forward_selection_head,
    logits_for_choices,
)
from server.training.tensorize import (
    ObservationTensorBatch,
    tensorize_observation,
    tensorize_selection_state,
)
from server.training.trajectory import RewardedDecisionStep


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    """Scalar loss stats for metrics."""

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float


class PPOTrainer:
    """Small clipped-PPO trainer over selection traces."""

    def __init__(
        self,
        *,
        model: TractorPolicyModel,
        model_config: ModelConfig,
        train_config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.train_config = train_config
        self.device = device

    def update(
        self,
        steps: tuple[RewardedDecisionStep, ...],
    ) -> PPOUpdateStats:
        """Run one optimizer step over rewarded decisions."""
        assert steps
        self.model.train()
        policy_losses: list[Tensor] = []
        value_losses: list[Tensor] = []
        entropies: list[Tensor] = []

        for rewarded in steps:
            step = rewarded.step
            log_prob, value, entropy = self._trace_log_prob(rewarded)
            old_log_prob = torch.tensor(
                step.log_probability,
                dtype=torch.float32,
                device=self.device,
            )
            reward = torch.tensor(
                rewarded.reward,
                dtype=torch.float32,
                device=self.device,
            )
            advantage = reward - value.detach()
            ratio = torch.exp(log_prob - old_log_prob)
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
            policy_losses.append(
                -torch.minimum(
                    ratio * advantage,
                    clipped_ratio * advantage,
                )
            )
            value_losses.append(nn.functional.mse_loss(value, reward))
            entropies.append(entropy)

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        self.model.zero_grad(set_to_none=True)
        torch.autograd.backward(total_loss)
        self._sgd_step()
        return PPOUpdateStats(
            policy_loss=_float_tensor(policy_loss),
            value_loss=_float_tensor(value_loss),
            entropy=_float_tensor(entropy),
            total_loss=_float_tensor(total_loss),
        )

    def optimizer_state(self) -> dict[str, object]:
        """Return serializable optimizer state for checkpoint schema."""
        return {"kind": "manual_sgd"}

    def load_optimizer_state(self, state: dict[str, object]) -> None:
        """Accept optimizer state from checkpoints."""
        kind = state.get("kind")
        assert kind == "manual_sgd"

    def _trace_log_prob(
        self,
        step: RewardedDecisionStep,
    ) -> tuple[Tensor, Tensor, Tensor]:
        observation_batch = tensorize_observation(
            observation=step.step.observation,
            max_observation_tokens=self.model_config.max_tokens,
            device=self.device,
        )
        state = SelectionState(selected_slots=())
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        value = torch.tensor(
            0.0, dtype=torch.float32, device=self.device
        )
        trace = step.step.action.selection_trace
        if not trace.choices:
            value = self._value_for_state(
                step=step,
                observation_batch=observation_batch,
                state=state,
            )
        for choice in trace.choices:
            log_prob, value, entropy = self._choice_log_prob(
                step=step,
                observation_batch=observation_batch,
                state=state,
                choice=choice,
            )
            log_probs.append(log_prob)
            entropies.append(entropy)
            if choice.kind == "select_card":
                assert choice.slot is not None
                state = SelectionState(
                    selected_slots=(*state.selected_slots, choice.slot)
                )
        return (
            _sum_tensors(log_probs, device=self.device),
            value,
            _sum_tensors(entropies, device=self.device),
        )

    def _choice_log_prob(
        self,
        *,
        step: RewardedDecisionStep,
        observation_batch: ObservationTensorBatch,
        state: SelectionState,
        choice: SelectionChoice,
    ) -> tuple[Tensor, Tensor, Tensor]:
        selection_batch = tensorize_selection_state(
            query=step.step.action_query,
            state=state,
            device=self.device,
        )
        output = forward_selection_head(
            model=self.model,
            query=step.step.action_query,
            observation=observation_batch,
            selection=selection_batch,
        )
        allowed = valid_selection_choices(step.step.action_query, state)
        assert choice in allowed
        choice_index = allowed.index(choice)
        logits = logits_for_choices(output, allowed)
        log_probabilities = torch.log_softmax(logits, dim=0)
        probabilities = torch.softmax(logits, dim=0)
        return (
            log_probabilities[choice_index],
            output.values[0],
            -(probabilities * log_probabilities).sum(),
        )

    def _value_for_state(
        self,
        *,
        step: RewardedDecisionStep,
        observation_batch: ObservationTensorBatch,
        state: SelectionState,
    ) -> Tensor:
        selection_batch = tensorize_selection_state(
            query=step.step.action_query,
            state=state,
            device=self.device,
        )
        output = forward_selection_head(
            model=self.model,
            query=step.step.action_query,
            observation=observation_batch,
            selection=selection_batch,
        )
        return output.values[0]

    def _sgd_step(self) -> None:
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.grad is None:
                    continue
                parameter.add_(
                    parameter.grad,
                    alpha=-self.train_config.learning_rate,
                )


def _sum_tensors(
    values: list[Tensor], *, device: torch.device
) -> Tensor:
    if not values:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    return torch.stack(values).sum()


def _float_tensor(value: Tensor) -> float:
    return float(value.detach().cpu().item())
