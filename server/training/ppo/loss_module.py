"""PPO loss module with a standard ``nn.Module.forward`` boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from server import result as _result
from server.training.config import TrainConfig
from server.training.model import TractorPolicyModel
from server.training.ppo.evaluation import evaluate_trace_batch
from server.training.ppo.math import (
    PPOObjectiveConfig,
    clipped_ppo_objective,
)
from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.profile import PPOProfileAccumulator


@dataclass(frozen=True, slots=True)
class MinibatchLoss:
    """Loss tensors and diagnostics for one optimizer step."""

    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    total_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


@dataclass(frozen=True, slots=True)
class PPOLossForwardOutput:
    """PPO loss forward result returned through DDP or bare modules."""

    loss: MinibatchLoss | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        assert (self.loss is None) != (self.rejection_reason is None)


class PPOLossModule(nn.Module):
    """Train-time PPO objective module wrapped by DDP when needed."""

    def __init__(
        self,
        *,
        model: TractorPolicyModel,
        train_config: TrainConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._model = model
        self._train_config = train_config
        self._device = device

    def policy_model(self) -> TractorPolicyModel:
        """Return the owned policy model for inference/checkpointing."""
        return self._model

    def forward(
        self,
        minibatch: TensorizedPPOMinibatch,
        profile: PPOProfileAccumulator,
    ) -> PPOLossForwardOutput:
        """Return one minibatch PPO objective."""
        if minibatch.is_empty():
            zero = self._zero_loss_touching_all_parameters()
            return PPOLossForwardOutput(
                loss=MinibatchLoss(
                    policy_loss=zero,
                    value_loss=zero,
                    entropy=zero,
                    total_loss=zero,
                    approx_kl=zero,
                    clip_fraction=zero,
                ),
                rejection_reason=None,
            )
        evaluated_result = evaluate_trace_batch(
            model=self._model,
            minibatch=minibatch,
            device=self._device,
            profile=profile,
        )
        if isinstance(evaluated_result, _result.Rejected):
            return PPOLossForwardOutput(
                loss=None,
                rejection_reason=evaluated_result.reason,
            )
        evaluated = evaluated_result.value
        objective = clipped_ppo_objective(
            old_log_probabilities=minibatch.old_log_probabilities,
            new_log_probabilities=evaluated.log_probabilities,
            advantages=minibatch.advantages,
            old_values=minibatch.old_values,
            new_values=evaluated.values,
            return_values=minibatch.return_values,
            entropies=evaluated.entropies,
            config=PPOObjectiveConfig(
                ppo_clip=self._train_config.ppo_clip,
                value_clip=self._train_config.value_clip,
                value_coef=self._train_config.value_coef,
                entropy_coef=self._train_config.entropy_coef,
            ),
        )
        return PPOLossForwardOutput(
            loss=MinibatchLoss(
                policy_loss=objective.policy_loss,
                value_loss=objective.value_loss,
                entropy=objective.entropy,
                total_loss=objective.total_loss,
                approx_kl=objective.approx_kl,
                clip_fraction=objective.clip_fraction,
            ),
            rejection_reason=None,
        )

    def _zero_loss_touching_all_parameters(self) -> Tensor:
        zero: Tensor | None = None
        for parameter in self._model.parameters():
            term = parameter.sum() * 0.0
            zero = term if zero is None else zero + term
        assert zero is not None
        return zero
