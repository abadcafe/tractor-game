"""PPO loss module with a standard ``nn.Module.forward`` boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final, override

import torch
from torch import Tensor, nn

from server.foundation import result as _result
from server.policy_model.network import PolicyModel
from server.training.config import TrainConfig
from server.training.ppo.advantages import explained_variance
from server.training.ppo.evaluation import evaluate_trace_batch
from server.training.ppo.math import (
    PPOObjectiveConfig,
    clipped_ppo_objective,
)
from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.profile import PPOProfileAccumulator
from server.training.ppo.validation import (
    PPO_TRACE_EVALUATION_FAILED,
    validation_ok,
)
from server.training.ppo.value_model import ObservationValueModel


@dataclass(frozen=True, slots=True)
class MinibatchLoss:
    """Loss tensors and diagnostics for one optimizer step."""

    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    objective_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor
    value_clip_fraction: Tensor
    explained_variance: Tensor


@dataclass(frozen=True, slots=True)
class PPOLossForwardOutput:
    """PPO loss forward result returned through DDP or bare modules."""

    loss: MinibatchLoss
    validation_code: Tensor

    def __post_init__(self) -> None:
        assert self.validation_code.shape == ()


type PPOLossForwardTensors = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


@final
class PPOLossModule(nn.Module):
    """Train-time PPO objective module wrapped by DDP when needed."""

    def __init__(
        self,
        *,
        model: PolicyModel,
        value_model: ObservationValueModel,
        train_config: TrainConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._model = model
        self._value_model = value_model
        self._train_config = train_config
        self._device = device

    def policy_model(self) -> PolicyModel:
        """Return the owned policy model for inference/checkpointing."""
        return self._model

    @override
    def forward(
        self,
        minibatch: TensorizedPPOMinibatch,
        profile: PPOProfileAccumulator,
    ) -> PPOLossForwardTensors:
        """Return one minibatch PPO objective as DDP-visible tensors."""
        if minibatch.is_empty():
            zero = self._zero_loss_touching_all_parameters()
            return _loss_tensors(
                MinibatchLoss(
                    policy_loss=zero,
                    value_loss=zero,
                    entropy=zero,
                    objective_loss=zero,
                    approx_kl=zero,
                    clip_fraction=zero,
                    value_clip_fraction=zero,
                    explained_variance=zero,
                ),
                validation_code=validation_ok(self._device),
            )
        evaluated_result = evaluate_trace_batch(
            model=self._model,
            minibatch=minibatch,
            device=self._device,
            profile=profile,
        )
        if isinstance(evaluated_result, _result.Rejected):
            zero = self._zero_loss_touching_all_parameters()
            return _loss_tensors(
                MinibatchLoss(
                    policy_loss=zero,
                    value_loss=zero,
                    entropy=zero,
                    objective_loss=zero,
                    approx_kl=zero,
                    clip_fraction=zero,
                    value_clip_fraction=zero,
                    explained_variance=zero,
                ),
                validation_code=torch.full(
                    (),
                    PPO_TRACE_EVALUATION_FAILED,
                    dtype=torch.long,
                    device=self._device,
                ),
            )
        evaluated = evaluated_result.value
        observation_batch = minibatch.observation_batch
        assert observation_batch is not None
        new_values = self._value_model.forward(observation_batch)
        objective = clipped_ppo_objective(
            old_log_probabilities=minibatch.old_log_probabilities,
            new_log_probabilities=evaluated.log_probabilities,
            advantages=minibatch.advantages,
            entropies=evaluated.entropies,
            old_values=minibatch.old_values,
            new_values=new_values,
            value_targets=minibatch.value_targets,
            config=PPOObjectiveConfig(
                ppo_clip=self._train_config.ppo_clip,
                value_clip=self._train_config.value_clip,
                value_coef=self._train_config.value_coef,
                entropy_coef=self._train_config.entropy_coef,
            ),
        )
        return _loss_tensors(
            MinibatchLoss(
                policy_loss=objective.policy_loss,
                value_loss=objective.value_loss,
                entropy=objective.entropy,
                objective_loss=objective.objective_loss,
                approx_kl=objective.approx_kl,
                clip_fraction=objective.clip_fraction,
                value_clip_fraction=objective.value_clip_fraction,
                explained_variance=explained_variance(
                    predictions=new_values.detach(),
                    targets=minibatch.value_targets,
                ),
            ),
            validation_code=validation_ok(self._device),
        )

    def _zero_loss_touching_all_parameters(self) -> Tensor:
        zero: Tensor | None = None
        for parameter in self.parameters():
            term = parameter.sum() * 0.0
            zero = term if zero is None else zero + term
        assert zero is not None
        return zero


def _loss_tensors(
    loss: MinibatchLoss, *, validation_code: Tensor
) -> PPOLossForwardTensors:
    return (
        loss.policy_loss,
        loss.value_loss,
        loss.entropy,
        loss.objective_loss,
        loss.approx_kl,
        loss.clip_fraction,
        loss.value_clip_fraction,
        loss.explained_variance,
        validation_code,
    )


def loss_forward_output_from_tensors(
    tensors: PPOLossForwardTensors,
) -> PPOLossForwardOutput:
    """Convert DDP-visible loss tensors back to trainer output."""
    loss = MinibatchLoss(
        policy_loss=tensors[0],
        value_loss=tensors[1],
        entropy=tensors[2],
        objective_loss=tensors[3],
        approx_kl=tensors[4],
        clip_fraction=tensors[5],
        value_clip_fraction=tensors[6],
        explained_variance=tensors[7],
    )
    return PPOLossForwardOutput(loss=loss, validation_code=tensors[8])
