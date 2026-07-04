"""PPO updates for rewarded semantic decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from server import result as _result
from server.sm.constants import get_team_index
from server.training.adamw import AdamWState
from server.training.argument_distribution import argument_distribution
from server.training.config import ModelConfig, TrainConfig
from server.training.model import TractorPolicyModel
from server.training.ppo_math import (
    PPOObjectiveConfig,
    ValueStep,
    clipped_ppo_objective,
    generalized_advantage_targets,
)
from server.training.semantic_actions.arguments import (
    SemanticArgumentPrefix,
)
from server.training.semantic_torch import forward_argument_head
from server.training.tensorize import (
    ObservationTensorBatch,
    tensorize_argument_prefixes,
    tensorize_observations,
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


def ppo_update_stats_are_finite(stats: PPOUpdateStats) -> bool:
    """Return whether all scalar PPO diagnostics are finite."""
    return (
        math.isfinite(stats.policy_loss)
        and math.isfinite(stats.value_loss)
        and math.isfinite(stats.entropy)
        and math.isfinite(stats.total_loss)
        and math.isfinite(stats.approx_kl)
        and math.isfinite(stats.clip_fraction)
    )


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


@dataclass(frozen=True, slots=True)
class TraceBatchEval:
    """Current-model scores for a minibatch of recorded traces."""

    log_probabilities: Tensor
    values: Tensor
    entropies: Tensor


@dataclass(frozen=True, slots=True)
class _TraceAccumulator:
    """Per-sample tensors collected from batched prefix forwards."""

    log_probabilities: tuple[Tensor, ...]
    entropies: tuple[Tensor, ...]


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
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected:
        """Run PPO epochs over one rewarded rollout."""
        assert steps
        self.model.train()
        samples = _normalize_advantages(self._rollout_samples(steps))
        losses: list[MinibatchLoss] = []
        parameters = tuple(self.model.parameters())
        for _ in range(self.train_config.ppo_epochs):
            for batch in _minibatches(
                _shuffled_samples(samples),
                minibatch_size=self.train_config.minibatch_size,
            ):
                loss_result = self._minibatch_loss(batch)
                if isinstance(loss_result, _result.Rejected):
                    self.model.zero_grad(set_to_none=True)
                    return loss_result
                loss = loss_result.value
                self.model.zero_grad(set_to_none=True)
                loss_check = _validate_minibatch_loss(loss)
                if isinstance(loss_check, _result.Rejected):
                    self.model.zero_grad(set_to_none=True)
                    return loss_check
                torch.autograd.backward(loss.total_loss)
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
                self.optimizer.step()
                losses.append(loss)
        stats = _mean_stats(losses)
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
        value_steps = tuple(
            ValueStep(
                reward=step.reward if index == len(steps) - 1 else 0.0,
                value_estimate=step.step.value_estimate,
            )
            for index, step in enumerate(steps)
        )
        targets = generalized_advantage_targets(
            steps=value_steps,
            gamma=self.train_config.gamma,
            gae_lambda=self.train_config.gae_lambda,
        )
        return tuple(
            RolloutSample(
                step=step,
                advantage=targets[index].advantage,
                return_value=targets[index].return_value,
                old_log_probability=step.step.log_probability,
                old_value_estimate=step.step.value_estimate,
            )
            for index, step in enumerate(steps)
        )

    def _minibatch_loss(
        self,
        samples: tuple[RolloutSample, ...],
    ) -> _result.Ok[MinibatchLoss] | _result.Rejected:
        evaluated_result = self._trace_batch_eval(samples)
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

    def _trace_batch_eval(
        self,
        samples: tuple[RolloutSample, ...],
    ) -> _result.Ok[TraceBatchEval] | _result.Rejected:
        assert samples
        observations = tuple(
            sample.step.step.observation for sample in samples
        )
        observation_batch = tensorize_observations(
            observations=observations,
            max_observation_tokens=self.model_config.max_tokens,
            device=self.device,
        )
        empty_prefixes = tuple(
            SemanticArgumentPrefix(arguments=()) for _ in samples
        )
        value_output = forward_argument_head(
            model=self.model,
            observation=observation_batch,
            prefix=tensorize_argument_prefixes(
                prefixes=empty_prefixes,
                device=self.device,
            ),
        )
        accumulators = tuple(
            _TraceAccumulator(log_probabilities=(), entropies=())
            for _ in samples
        )
        prefixes = list(empty_prefixes)
        max_trace_length = max(
            len(sample.step.step.action.semantic_trace.arguments)
            for sample in samples
        )
        for argument_index in range(max_trace_length):
            active_indices = tuple(
                index
                for index, sample in enumerate(samples)
                if argument_index
                < len(sample.step.step.action.semantic_trace.arguments)
            )
            step_eval_result = self._argument_batch_eval(
                samples=samples,
                active_indices=active_indices,
                observation_batch=observation_batch,
                prefixes=tuple(
                    prefixes[index] for index in active_indices
                ),
                argument_index=argument_index,
            )
            if isinstance(step_eval_result, _result.Rejected):
                return step_eval_result
            step_eval = step_eval_result.value
            for row_index, sample_index in enumerate(active_indices):
                accumulator = accumulators[sample_index]
                accumulators = _replace_accumulator(
                    accumulators,
                    index=sample_index,
                    accumulator=_TraceAccumulator(
                        log_probabilities=(
                            *accumulator.log_probabilities,
                            step_eval.log_probabilities[row_index],
                        ),
                        entropies=(
                            *accumulator.entropies,
                            step_eval.entropies[row_index],
                        ),
                    ),
                )
                argument = samples[
                    sample_index
                ].step.step.action.semantic_trace.arguments[
                    argument_index
                ]
                if argument.kind == "select_face_count":
                    current_prefix = prefixes[sample_index]
                    prefixes[sample_index] = SemanticArgumentPrefix(
                        arguments=(*current_prefix.arguments, argument)
                    )
        return _result.Ok(
            value=TraceBatchEval(
                log_probabilities=torch.stack(
                    [
                        _sum_tensors(
                            accumulator.log_probabilities,
                            device=self.device,
                        )
                        for accumulator in accumulators
                    ]
                ),
                values=value_output.values,
                entropies=torch.stack(
                    [
                        _sum_tensors(
                            accumulator.entropies,
                            device=self.device,
                        )
                        for accumulator in accumulators
                    ]
                ),
            )
        )

    def _argument_batch_eval(
        self,
        *,
        samples: tuple[RolloutSample, ...],
        active_indices: tuple[int, ...],
        observation_batch: ObservationTensorBatch,
        prefixes: tuple[SemanticArgumentPrefix, ...],
        argument_index: int,
    ) -> _result.Ok[TraceBatchEval] | _result.Rejected:
        assert active_indices
        active_observation_batch = _select_observations(
            observation_batch,
            active_indices=active_indices,
        )
        prefix_batch = tensorize_argument_prefixes(
            prefixes=prefixes,
            device=self.device,
        )
        output = forward_argument_head(
            model=self.model,
            observation=active_observation_batch,
            prefix=prefix_batch,
        )
        log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        for row_index, sample_index in enumerate(active_indices):
            sample = samples[sample_index]
            prefix = prefixes[row_index]
            argument = sample.step.step.action.semantic_trace.arguments[
                argument_index
            ]
            allowed = sample.step.step.legal_actions.allowed_next(
                prefix
            )
            assert argument in allowed
            selected_argument_index = allowed.index(argument)
            distribution_result = argument_distribution(
                argument_logits=output.argument_logits[row_index],
                choices=allowed,
            )
            if isinstance(distribution_result, _result.Rejected):
                return distribution_result
            distribution = distribution_result.value
            log_probabilities.append(
                distribution.log_probabilities[selected_argument_index]
            )
            entropies.append(distribution.entropy)
        return _result.Ok(
            value=TraceBatchEval(
                log_probabilities=torch.stack(log_probabilities),
                values=output.values,
                entropies=torch.stack(entropies),
            )
        )


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
    for field, value in fields:
        if not _all_finite(value):
            return _result.Rejected(
                reason=f"PPO {field} must be finite"
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


def _sum_tensors(
    values: tuple[Tensor, ...], *, device: torch.device
) -> Tensor:
    if not values:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    return torch.stack(list(values)).sum()


def _float_tensor_vector(
    values: tuple[float, ...], *, device: torch.device
) -> Tensor:
    assert values
    return torch.tensor(values, dtype=torch.float32, device=device)


def _replace_accumulator(
    accumulators: tuple[_TraceAccumulator, ...],
    *,
    index: int,
    accumulator: _TraceAccumulator,
) -> tuple[_TraceAccumulator, ...]:
    result = list(accumulators)
    result[index] = accumulator
    return tuple(result)


def _select_observations(
    batch: ObservationTensorBatch,
    *,
    active_indices: tuple[int, ...],
) -> ObservationTensorBatch:
    index = torch.tensor(
        active_indices,
        dtype=torch.long,
        device=batch.token_type_ids.device,
    )
    return ObservationTensorBatch(
        token_type_ids=batch.token_type_ids.index_select(0, index),
        segment_ids=batch.segment_ids.index_select(0, index),
        field_ids=batch.field_ids.index_select(0, index),
        value_ids=batch.value_ids.index_select(0, index),
        suit_ids=batch.suit_ids.index_select(0, index),
        rank_ids=batch.rank_ids.index_select(0, index),
        points_ids=batch.points_ids.index_select(0, index),
        color_ids=batch.color_ids.index_select(0, index),
        role_ids=batch.role_ids.index_select(0, index),
        trick_age_ids=batch.trick_age_ids.index_select(0, index),
        trick_state_ids=batch.trick_state_ids.index_select(0, index),
        play_order_ids=batch.play_order_ids.index_select(0, index),
        count_ids=batch.count_ids.index_select(0, index),
        play_width_ids=batch.play_width_ids.index_select(0, index),
        event_age_ids=batch.event_age_ids.index_select(0, index),
        numeric_values=batch.numeric_values.index_select(0, index),
        numeric_masks=batch.numeric_masks.index_select(0, index),
    )


def _float_tensor(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def _all_finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu().item())


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
