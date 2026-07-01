"""PPO updates for rewarded action-token decisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from server.training.action_tokens import (
    ACTION_TOKEN_VOCAB_SIZE,
    BEGIN_TOKEN_ID,
    valid_next_token_ids,
)
from server.training.config import ModelConfig, TrainConfig
from server.training.model import TractorPolicyModel
from server.training.tensorize import (
    tensorize_action_prefix,
    tensorize_observation,
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
    """Small clipped-PPO trainer over action-token sequences."""

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
            sequence = step.action.token_ids
            assert sequence[0] == BEGIN_TOKEN_ID
            log_prob, value, entropy = self._sequence_log_prob(rewarded)
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

    def _sequence_log_prob(
        self,
        step: RewardedDecisionStep,
    ) -> tuple[Tensor, Tensor, Tensor]:
        observation_ids = tensorize_observation(
            observation=step.step.observation,
            max_observation_tokens=self.model_config.max_tokens,
            device=self.device,
        )
        sequence = step.step.action.token_ids
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        value = torch.tensor(
            0.0, dtype=torch.float32, device=self.device
        )
        prefix = (sequence[0],)
        for token_id in sequence[1:]:
            action_prefix_ids = tensorize_action_prefix(
                prefix=prefix,
                device=self.device,
            )
            logits, values = self.model.forward_action(
                observation_ids,
                action_prefix_ids,
            )
            value = values[0]
            allowed = valid_next_token_ids(
                step.step.action_query,
                prefix,
            )
            assert token_id in allowed
            masked_logits = _masked_logits(
                logits[0],
                allowed,
                device=self.device,
            )
            log_probabilities = torch.log_softmax(masked_logits, dim=0)
            probabilities = torch.softmax(masked_logits, dim=0)
            log_probs.append(log_probabilities[token_id])
            entropies.append(-(probabilities * log_probabilities).sum())
            prefix = (*prefix, token_id)
        return (
            torch.stack(log_probs).sum(),
            value,
            torch.stack(entropies).sum(),
        )

    def _sgd_step(self) -> None:
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.grad is None:
                    continue
                parameter.add_(
                    parameter.grad,
                    alpha=-self.train_config.learning_rate,
                )


def _masked_logits(
    logits: Tensor,
    allowed_token_ids: tuple[int, ...],
    *,
    device: torch.device,
) -> Tensor:
    mask = torch.full(
        (ACTION_TOKEN_VOCAB_SIZE,),
        float("-inf"),
        dtype=logits.dtype,
        device=device,
    )
    index = torch.tensor(
        list(allowed_token_ids),
        dtype=torch.long,
        device=device,
    )
    mask[index] = logits[index]
    return mask


def _float_tensor(value: Tensor) -> float:
    return float(value.detach().cpu().item())
