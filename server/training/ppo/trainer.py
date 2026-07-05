"""PPO trainer update loop."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from server import result as _result
from server.training.config import ModelConfig, TrainConfig
from server.training.model import TractorPolicyModel
from server.training.ppo.adamw import AdamWState
from server.training.ppo.evaluation import evaluate_trace_batch
from server.training.ppo.math import (
    PPOObjectiveConfig,
    clipped_ppo_objective,
)
from server.training.ppo.profile import (
    PPOProfileAccumulator,
    PPOUpdateProfile,
)
from server.training.ppo.rollout import (
    RolloutSample,
    minibatches,
    normalize_advantages,
    rollout_samples,
    shuffled_samples,
)
from server.training.ppo.stats import (
    PPOUpdateStats,
    ppo_update_stats_are_finite,
)
from server.training.trajectory import RolloutBatch


@dataclass(frozen=True, slots=True)
class MinibatchLoss:
    """Loss tensors and diagnostics for one optimizer step."""

    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    total_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


class PPOTrainer:
    """Clipped PPO trainer over semantic argument traces."""

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
        self.optimizer = AdamWState(
            parameters=tuple(self.model.parameters()),
            learning_rate=train_config.learning_rate,
            beta1=train_config.adam_beta1,
            beta2=train_config.adam_beta2,
            weight_decay=train_config.weight_decay,
        )

    def update(
        self,
        batch: RolloutBatch,
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected:
        """Run PPO epochs over one rewarded rollout."""
        assert not batch.is_empty()
        self.model.train()
        profile = PPOProfileAccumulator.start(
            device=self.device,
            mode=self.train_config.ppo_profile,
        )
        samples = normalize_advantages(
            rollout_samples(
                batch,
                gae_lambda=self.train_config.gae_lambda,
            )
        )
        losses: list[MinibatchLoss] = []
        parameters = tuple(self.model.parameters())
        for _ in range(self.train_config.ppo_epochs):
            for minibatch in minibatches(
                shuffled_samples(samples),
                minibatch_size=self.train_config.minibatch_size,
            ):
                loss_start = profile.mark()
                loss_result = self._minibatch_loss(
                    minibatch, profile=profile
                )
                profile.record_elapsed(
                    "minibatch_loss_seconds", loss_start
                )
                if isinstance(loss_result, _result.Rejected):
                    self.model.zero_grad(set_to_none=True)
                    return loss_result
                loss = loss_result.value
                self.model.zero_grad(set_to_none=True)
                loss_check = _validate_minibatch_loss(loss)
                if isinstance(loss_check, _result.Rejected):
                    self.model.zero_grad(set_to_none=True)
                    return loss_check
                backward_start = profile.mark()
                torch.autograd.backward(loss.total_loss)
                profile.record_elapsed(
                    "backward_seconds", backward_start
                )
                gradient_check = _validate_gradients(parameters)
                if isinstance(gradient_check, _result.Rejected):
                    self.model.zero_grad(set_to_none=True)
                    return gradient_check
                _clip_grad_norm(
                    parameters,
                    max_norm=self.train_config.max_grad_norm,
                )
                clipped_gradient_check = _validate_gradients(parameters)
                if isinstance(clipped_gradient_check, _result.Rejected):
                    self.model.zero_grad(set_to_none=True)
                    return clipped_gradient_check
                optimizer_start = profile.mark()
                self.optimizer.step()
                profile.record_elapsed(
                    "optimizer_step_seconds", optimizer_start
                )
                losses.append(loss)
        stats = _mean_stats(losses, profile=profile.finish())
        if not ppo_update_stats_are_finite(stats):
            return _result.Rejected(
                reason="PPO update stats must be finite"
            )
        return _result.Ok(value=stats)

    def optimizer_state(self) -> dict[str, object]:
        """Return serializable AdamW optimizer state."""
        state: dict[str, object] = self.optimizer.state_dict()
        return state

    def load_optimizer_state(self, state: dict[str, object]) -> None:
        """Load AdamW optimizer state from a checkpoint."""
        self.optimizer.load_state_dict(state)

    def _minibatch_loss(
        self,
        samples: tuple[RolloutSample, ...],
        *,
        profile: PPOProfileAccumulator,
    ) -> _result.Ok[MinibatchLoss] | _result.Rejected:
        evaluated_result = evaluate_trace_batch(
            model=self.model,
            samples=samples,
            device=self.device,
            profile=profile,
        )
        if isinstance(evaluated_result, _result.Rejected):
            return evaluated_result
        evaluated = evaluated_result.value
        old_log_probabilities = _float_tensor_vector(
            tuple(sample.old_log_probability for sample in samples),
            device=self.device,
        )
        old_values = _float_tensor_vector(
            tuple(sample.old_value_estimate for sample in samples),
            device=self.device,
        )
        advantages = _float_tensor_vector(
            tuple(sample.advantage for sample in samples),
            device=self.device,
        )
        return_values = _float_tensor_vector(
            tuple(sample.return_value for sample in samples),
            device=self.device,
        )
        objective = clipped_ppo_objective(
            old_log_probabilities=old_log_probabilities,
            new_log_probabilities=evaluated.log_probabilities,
            advantages=advantages,
            old_values=old_values,
            new_values=evaluated.values,
            return_values=return_values,
            entropies=evaluated.entropies,
            config=PPOObjectiveConfig(
                ppo_clip=self.train_config.ppo_clip,
                value_clip=self.train_config.value_clip,
                value_coef=self.train_config.value_coef,
                entropy_coef=self.train_config.entropy_coef,
            ),
        )
        return _result.Ok(
            value=MinibatchLoss(
                policy_loss=objective.policy_loss,
                value_loss=objective.value_loss,
                entropy=objective.entropy,
                total_loss=objective.total_loss,
                approx_kl=objective.approx_kl,
                clip_fraction=objective.clip_fraction,
            )
        )


def _mean_stats(
    losses: list[MinibatchLoss],
    *,
    profile: PPOUpdateProfile,
) -> PPOUpdateStats:
    assert losses
    return PPOUpdateStats(
        policy_loss=_mean_float(
            tuple(loss.policy_loss for loss in losses)
        ),
        value_loss=_mean_float(
            tuple(loss.value_loss for loss in losses)
        ),
        entropy=_mean_float(tuple(loss.entropy for loss in losses)),
        total_loss=_mean_float(
            tuple(loss.total_loss for loss in losses)
        ),
        approx_kl=_mean_float(tuple(loss.approx_kl for loss in losses)),
        clip_fraction=_mean_float(
            tuple(loss.clip_fraction for loss in losses)
        ),
        profile=profile,
    )


def _mean_float(values: tuple[Tensor, ...]) -> float:
    assert values
    return _float_tensor(torch.stack(list(values)).mean())


def _validate_minibatch_loss(
    loss: MinibatchLoss,
) -> _result.Ok[None] | _result.Rejected:
    fields = (
        ("policy_loss", loss.policy_loss),
        ("value_loss", loss.value_loss),
        ("entropy", loss.entropy),
        ("total_loss", loss.total_loss),
        ("approx_kl", loss.approx_kl),
        ("clip_fraction", loss.clip_fraction),
    )
    for field_name, value in fields:
        if not _all_finite(value):
            return _result.Rejected(
                reason=f"PPO {field_name} must be finite"
            )
    return _result.Ok(value=None)


def _validate_gradients(
    parameters: tuple[Tensor, ...],
) -> _result.Ok[None] | _result.Rejected:
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not _all_finite(gradient):
            return _result.Rejected(
                reason="PPO gradients must be finite"
            )
    return _result.Ok(value=None)


def _float_tensor_vector(
    values: tuple[float, ...], *, device: torch.device
) -> Tensor:
    assert values
    return torch.tensor(values, dtype=torch.float32, device=device)


def _float_tensor(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def _all_finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu().item())


def _clip_grad_norm(
    parameters: tuple[Tensor, ...], *, max_norm: float
) -> None:
    if max_norm <= 0.0:
        return
    gradients = tuple(
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    )
    if not gradients:
        return
    total_squared_norm = 0.0
    for gradient in gradients:
        detached = gradient.detach()
        total_squared_norm += _float_tensor((detached * detached).sum())
    clip_coef = max_norm / (math.sqrt(total_squared_norm) + 0.000001)
    if clip_coef >= 1.0:
        return
    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(clip_coef)
