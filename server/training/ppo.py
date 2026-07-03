"""PPO updates for rewarded semantic decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeGuard, cast

import torch
from torch import Tensor, nn

from server.sm.constants import get_team_index
from server.training.config import ModelConfig, TrainConfig
from server.training.model import TractorPolicyModel
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentPrefix,
)
from server.training.semantic_torch import (
    forward_argument_head,
    logits_for_arguments,
)
from server.training.tensorize import (
    ArgumentPrefixTensorBatch,
    ObservationTensorBatch,
    tensorize_argument_prefix,
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
    approx_kl: float
    clip_fraction: float


@dataclass(frozen=True, slots=True)
class RolloutSample:
    """One decision annotated with PPO targets."""

    step: RewardedDecisionStep
    advantage: float
    return_value: float
    old_log_probability: float
    old_value_estimate: float


@dataclass(frozen=True, slots=True)
class MinibatchLoss:
    """Loss tensors and diagnostics for one optimizer step."""

    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    total_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


class AdamWState:
    """Strictly typed AdamW optimizer state."""

    def __init__(
        self,
        *,
        parameters: tuple[Tensor, ...],
        learning_rate: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        eps: float = 0.00000001,
    ) -> None:
        self._parameters = parameters
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._weight_decay = weight_decay
        self._eps = eps
        self._step_count = 0
        self._exp_avgs: list[Tensor | None] = [None for _ in parameters]
        self._exp_avg_sqs: list[Tensor | None] = [
            None for _ in parameters
        ]

    def step(self) -> None:
        """Apply one AdamW update using current parameter gradients."""
        self._step_count += 1
        with torch.no_grad():
            for index, parameter in enumerate(self._parameters):
                gradient = parameter.grad
                if gradient is None:
                    continue
                exp_avg = self._exp_avgs[index]
                exp_avg_sq = self._exp_avg_sqs[index]
                if exp_avg is None:
                    exp_avg = torch.zeros_like(parameter)
                    self._exp_avgs[index] = exp_avg
                if exp_avg_sq is None:
                    exp_avg_sq = torch.zeros_like(parameter)
                    self._exp_avg_sqs[index] = exp_avg_sq
                if self._weight_decay != 0.0:
                    parameter.mul_(
                        1.0 - self._learning_rate * self._weight_decay
                    )
                exp_avg.mul_(self._beta1).add_(
                    gradient, alpha=1.0 - self._beta1
                )
                exp_avg_sq.mul_(self._beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - self._beta2,
                )
                bias_correction1 = 1.0 - self._beta1**self._step_count
                bias_correction2 = 1.0 - self._beta2**self._step_count
                step_size = (
                    self._learning_rate
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )
                denominator = exp_avg_sq.sqrt().add_(self._eps)
                parameter.addcdiv_(
                    exp_avg, denominator, value=-step_size
                )

    def state_dict(self) -> dict[str, object]:
        """Return a torch-saveable optimizer state payload."""
        return {
            "kind": "typed_adamw",
            "step_count": self._step_count,
            "exp_avgs": tuple(self._exp_avgs),
            "exp_avg_sqs": tuple(self._exp_avg_sqs),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Load optimizer state from a checkpoint payload."""
        kind = state["kind"]
        assert kind == "typed_adamw"
        step_count = state["step_count"]
        assert isinstance(step_count, int)
        exp_avgs = state["exp_avgs"]
        exp_avg_sqs = state["exp_avg_sqs"]
        assert _is_optional_tensor_tuple(exp_avgs)
        assert _is_optional_tensor_tuple(exp_avg_sqs)
        assert len(exp_avgs) == len(self._parameters)
        assert len(exp_avg_sqs) == len(self._parameters)
        self._step_count = step_count
        self._exp_avgs = list(exp_avgs)
        self._exp_avg_sqs = list(exp_avg_sqs)


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
        steps: tuple[RewardedDecisionStep, ...],
    ) -> PPOUpdateStats:
        """Run PPO epochs over one rewarded rollout."""
        assert steps
        self.model.train()
        samples = _normalize_advantages(self._rollout_samples(steps))
        losses: list[MinibatchLoss] = []
        for _ in range(self.train_config.ppo_epochs):
            for batch in _minibatches(
                _shuffled_samples(samples),
                minibatch_size=self.train_config.minibatch_size,
            ):
                loss = self._minibatch_loss(batch)
                self.model.zero_grad(set_to_none=True)
                torch.autograd.backward(loss.total_loss)
                _clip_grad_norm(
                    tuple(self.model.parameters()),
                    max_norm=self.train_config.max_grad_norm,
                )
                self.optimizer.step()
                losses.append(loss)
        return _mean_stats(losses)

    def optimizer_state(self) -> dict[str, object]:
        """Return serializable AdamW optimizer state."""
        state: dict[str, object] = self.optimizer.state_dict()
        return state

    def load_optimizer_state(self, state: dict[str, object]) -> None:
        """Load AdamW optimizer state from a checkpoint."""
        self.optimizer.load_state_dict(state)

    def _rollout_samples(
        self,
        steps: tuple[RewardedDecisionStep, ...],
    ) -> tuple[RolloutSample, ...]:
        team0 = tuple(
            step
            for step in steps
            if get_team_index(step.step.player_index) == 0
        )
        team1 = tuple(
            step
            for step in steps
            if get_team_index(step.step.player_index) == 1
        )
        return (*self._team_samples(team0), *self._team_samples(team1))

    def _team_samples(
        self,
        steps: tuple[RewardedDecisionStep, ...],
    ) -> tuple[RolloutSample, ...]:
        if not steps:
            return ()
        rewards = [0.0 for _ in steps]
        rewards[-1] = steps[-1].reward
        values = [step.step.value_estimate for step in steps]
        advantages = [0.0 for _ in steps]
        gae = 0.0
        for index in range(len(steps) - 1, -1, -1):
            next_value = (
                0.0 if index == len(steps) - 1 else values[index + 1]
            )
            delta = (
                rewards[index]
                + self.train_config.gamma * next_value
                - values[index]
            )
            gae = (
                delta
                + self.train_config.gamma
                * self.train_config.gae_lambda
                * gae
            )
            advantages[index] = gae
        return tuple(
            RolloutSample(
                step=step,
                advantage=advantages[index],
                return_value=advantages[index] + values[index],
                old_log_probability=step.step.log_probability,
                old_value_estimate=step.step.value_estimate,
            )
            for index, step in enumerate(steps)
        )

    def _minibatch_loss(
        self,
        samples: tuple[RolloutSample, ...],
    ) -> MinibatchLoss:
        policy_losses: list[Tensor] = []
        value_losses: list[Tensor] = []
        entropies: list[Tensor] = []
        approx_kls: list[Tensor] = []
        clip_fractions: list[Tensor] = []
        for sample in samples:
            log_prob, value, entropy = self._trace_log_prob(sample.step)
            old_log_prob = _scalar_tensor(
                sample.old_log_probability, device=self.device
            )
            old_value = _scalar_tensor(
                sample.old_value_estimate, device=self.device
            )
            advantage = _scalar_tensor(
                sample.advantage, device=self.device
            )
            return_value = _scalar_tensor(
                sample.return_value, device=self.device
            )
            ratio = torch.exp(log_prob - old_log_prob)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.train_config.ppo_clip,
                1.0 + self.train_config.ppo_clip,
            )
            policy_losses.append(
                -torch.minimum(
                    ratio * advantage,
                    clipped_ratio * advantage,
                )
            )
            value_clipped = old_value + torch.clamp(
                value - old_value,
                -self.train_config.value_clip,
                self.train_config.value_clip,
            )
            value_loss = torch.maximum(
                nn.functional.mse_loss(value, return_value),
                nn.functional.mse_loss(value_clipped, return_value),
            )
            value_losses.append(value_loss)
            entropies.append(entropy)
            approx_kls.append(old_log_prob - log_prob)
            clip_fractions.append(
                ratio.sub(1.0)
                .abs()
                .gt(self.train_config.ppo_clip)
                .to(dtype=torch.float32)
            )
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        total_loss = (
            policy_loss
            + self.train_config.value_coef * value_loss
            - self.train_config.entropy_coef * entropy
        )
        return MinibatchLoss(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            total_loss=total_loss,
            approx_kl=torch.stack(approx_kls).mean(),
            clip_fraction=torch.stack(clip_fractions).mean(),
        )

    def _trace_log_prob(
        self,
        step: RewardedDecisionStep,
    ) -> tuple[Tensor, Tensor, Tensor]:
        observation_batch = tensorize_observation(
            observation=step.step.observation,
            max_observation_tokens=self.model_config.max_tokens,
            device=self.device,
        )
        prefix = SemanticArgumentPrefix(arguments=())
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        value = self._value_for_prefix(
            observation_batch=observation_batch,
            prefix=prefix,
        )
        trace = step.step.action.semantic_trace
        for argument in trace.arguments:
            log_prob, value, entropy = self._argument_log_prob(
                step=step,
                observation_batch=observation_batch,
                prefix=prefix,
                argument=argument,
            )
            log_probs.append(log_prob)
            entropies.append(entropy)
            if argument.kind == "select_face_count":
                prefix = SemanticArgumentPrefix(
                    arguments=(*prefix.arguments, argument)
                )
        return (
            _sum_tensors(log_probs, device=self.device),
            value,
            _sum_tensors(entropies, device=self.device),
        )

    def _argument_log_prob(
        self,
        *,
        step: RewardedDecisionStep,
        observation_batch: ObservationTensorBatch,
        prefix: SemanticArgumentPrefix,
        argument: SemanticArgument,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prefix_batch = tensorize_argument_prefix(
            prefix=prefix,
            device=self.device,
        )
        output = forward_argument_head(
            model=self.model,
            observation=observation_batch,
            prefix=prefix_batch,
        )
        allowed = step.step.legal_actions.allowed_next(prefix)
        assert argument in allowed
        argument_index = allowed.index(argument)
        logits = logits_for_arguments(output, allowed)
        log_probabilities = torch.log_softmax(logits, dim=0)
        probabilities = torch.softmax(logits, dim=0)
        return (
            log_probabilities[argument_index],
            output.values[0],
            -(probabilities * log_probabilities).sum(),
        )

    def _value_for_prefix(
        self,
        *,
        observation_batch: ObservationTensorBatch,
        prefix: SemanticArgumentPrefix,
    ) -> Tensor:
        prefix_batch: ArgumentPrefixTensorBatch = (
            tensorize_argument_prefix(
                prefix=prefix,
                device=self.device,
            )
        )
        output = forward_argument_head(
            model=self.model,
            observation=observation_batch,
            prefix=prefix_batch,
        )
        return output.values[0]


def _normalize_advantages(
    samples: tuple[RolloutSample, ...],
) -> tuple[RolloutSample, ...]:
    assert samples
    mean = sum(sample.advantage for sample in samples) / len(samples)
    variance = sum(
        (sample.advantage - mean) * (sample.advantage - mean)
        for sample in samples
    ) / len(samples)
    stddev = math.sqrt(variance)
    if stddev <= 0.000001:
        return tuple(
            RolloutSample(
                step=sample.step,
                advantage=sample.advantage - mean,
                return_value=sample.return_value,
                old_log_probability=sample.old_log_probability,
                old_value_estimate=sample.old_value_estimate,
            )
            for sample in samples
        )
    return tuple(
        RolloutSample(
            step=sample.step,
            advantage=(sample.advantage - mean) / (stddev + 0.000001),
            return_value=sample.return_value,
            old_log_probability=sample.old_log_probability,
            old_value_estimate=sample.old_value_estimate,
        )
        for sample in samples
    )


def _minibatches(
    samples: tuple[RolloutSample, ...],
    *,
    minibatch_size: int,
) -> tuple[tuple[RolloutSample, ...], ...]:
    assert minibatch_size > 0
    result: list[tuple[RolloutSample, ...]] = []
    for start in range(0, len(samples), minibatch_size):
        result.append(samples[start : start + minibatch_size])
    return tuple(result)


def _shuffled_samples(
    samples: tuple[RolloutSample, ...],
) -> tuple[RolloutSample, ...]:
    order = torch.randperm(len(samples))
    result: list[RolloutSample] = []
    for position in range(order.numel()):
        index = int(order[position].item())
        result.append(samples[index])
    return tuple(result)


def _mean_stats(losses: list[MinibatchLoss]) -> PPOUpdateStats:
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
    )


def _mean_float(values: tuple[Tensor, ...]) -> float:
    assert values
    return _float_tensor(torch.stack(list(values)).mean())


def _sum_tensors(
    values: list[Tensor], *, device: torch.device
) -> Tensor:
    if not values:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    return torch.stack(values).sum()


def _scalar_tensor(value: float, *, device: torch.device) -> Tensor:
    return torch.tensor(value, dtype=torch.float32, device=device)


def _float_tensor(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def _clip_grad_norm(
    parameters: tuple[Tensor, ...], *, max_norm: float
) -> None:
    if max_norm <= 0.0:
        return
    grads = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grads:
        return
    total_squared_norm = 0.0
    for gradient in grads:
        detached = gradient.detach()
        total_squared_norm += _float_tensor((detached * detached).sum())
    clip_coef = max_norm / (math.sqrt(total_squared_norm) + 0.000001)
    if clip_coef >= 1.0:
        return
    with torch.no_grad():
        for gradient in grads:
            gradient.mul_(clip_coef)


def _is_optional_tensor_tuple(
    value: object,
) -> TypeGuard[tuple[Tensor | None, ...]]:
    if not isinstance(value, tuple):
        return False
    items = cast(tuple[object, ...], value)
    for item in items:
        if not _is_optional_tensor(item):
            return False
    return True


def _is_optional_tensor(value: object) -> TypeGuard[Tensor | None]:
    return value is None or isinstance(value, Tensor)
