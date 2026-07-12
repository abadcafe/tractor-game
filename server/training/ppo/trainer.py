"""PPO trainer update loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from server.foundation import result as _result
from server.training.config import TrainConfig
from server.training.model import TractorPolicyModel
from server.training.ppo.collectives import (
    all_reduce_max,
    all_reduce_sum,
)
from server.training.ppo.device_targets import shuffled_index_tensor
from server.training.ppo.distributed import (
    PPOLossForwarder,
    PPOUpdatePartition,
    build_ppo_loss_forwarder,
    single_update_partition,
)
from server.training.ppo.gradients import (
    clip_grad_norm_on_device,
)
from server.training.ppo.loss_module import (
    MinibatchLoss,
    PPOLossModule,
)
from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.optimizer import PPOOptimizer
from server.training.ppo.prepared_batch import (
    PPOEpochSchedule,
    PreparedPPOBatch,
    empty_ppo_minibatch,
    prepare_ppo_batch,
    prepare_ppo_epoch_schedule,
    prepared_ppo_epoch_minibatch,
)
from server.training.ppo.profile import (
    PPOProfileAccumulator,
    PPOUpdateProfile,
)
from server.training.ppo.stats import (
    PPOUpdateStats,
    ppo_update_stats_are_finite,
)
from server.training.ppo.sync import (
    positive_count_value,
    synchronized_count_sum,
    synchronized_count_vector_sum,
)
from server.training.ppo.update_input import PPOUpdateInput
from server.training.ppo.validation import (
    PPO_APPROX_KL_NONFINITE,
    PPO_CLIP_FRACTION_NONFINITE,
    PPO_ENTROPY_NONFINITE,
    PPO_POLICY_LOSS_NONFINITE,
    PPO_TOTAL_LOSS_NONFINITE,
    PPO_VALUE_LOSS_NONFINITE,
    TensorValidationCheck,
    combine_validation_codes,
    gradient_validation_code,
    non_finite_validation_code,
    validation_rejection_reason,
)
from server.training.runtime.config import PPOProfileMode
from server.training.sampling import ShuffleKey, shuffled_indices


@dataclass(frozen=True, slots=True)
class _MinibatchSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        assert self.start >= 0
        assert self.end >= self.start

    def count(self) -> int:
        """Return sample count in this contiguous epoch span."""
        return self.end - self.start


class PPOTrainer:
    """Clipped PPO trainer over semantic argument traces."""

    def __init__(
        self,
        *,
        model: TractorPolicyModel,
        train_config: TrainConfig,
        device: torch.device,
        profile_mode: PPOProfileMode,
        update_partition: PPOUpdatePartition | None = None,
    ) -> None:
        self.model = model
        self.train_config = train_config
        self.device = device
        self.profile_mode: PPOProfileMode = profile_mode
        self.update_partition = (
            single_update_partition()
            if update_partition is None
            else update_partition
        )
        self.loss_module = PPOLossModule(
            model=model,
            train_config=train_config,
            device=device,
        )
        forwarder_result = build_ppo_loss_forwarder(
            module=self.loss_module,
            partition=self.update_partition,
            device=device,
        )
        if isinstance(forwarder_result, _result.Rejected):
            self._loss_forwarder: PPOLossForwarder | None = None
            self._loss_forwarder_rejection = forwarder_result
        else:
            self._loss_forwarder = forwarder_result.value
            self._loss_forwarder_rejection = None
        self.optimizer = PPOOptimizer(
            parameters=tuple(self.model.parameters()),
            learning_rate=train_config.learning_rate,
            beta1=train_config.adam_beta1,
            beta2=train_config.adam_beta2,
            weight_decay=train_config.weight_decay,
        )

    def update(
        self,
        update_input: PPOUpdateInput,
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected:
        """Run one synchronized PPO update."""
        if (
            update_input.local_batch is None
            and self.update_partition.world_size == 1
        ):
            return _result.Rejected(
                reason="single-rank PPO update requires local batch"
            )
        if self._loss_forwarder_rejection is not None:
            return self._loss_forwarder_rejection
        loss_forwarder = self._loss_forwarder
        assert loss_forwarder is not None
        loss_forwarder.train()
        profile = PPOProfileAccumulator.start(
            device=self.device,
            mode=self.profile_mode,
        )
        global_sample_count_tensor_result = synchronized_count_sum(
            value=update_input.local_transition_count(),
            partition=self.update_partition,
            device=self.device,
        )
        if isinstance(
            global_sample_count_tensor_result, _result.Rejected
        ):
            return global_sample_count_tensor_result
        global_sample_count_result = positive_count_value(
            count=global_sample_count_tensor_result.value
        )
        if isinstance(global_sample_count_result, _result.Rejected):
            return global_sample_count_result
        batch = update_input.local_batch
        prepared_batch: PreparedPPOBatch | None = None
        raw_advantages = torch.empty(
            (0,), dtype=torch.float32, device=self.device
        )
        if batch is not None:
            raw_advantages = batch.raw_advantages
            partition_check = _validate_update_partition(
                sample_count=batch.sample_count(),
                minibatch_size=self.train_config.minibatch_size,
            )
            if isinstance(partition_check, _result.Rejected):
                return partition_check
        normalized_advantages_result = _sync_normalized_advantages(
            advantages=raw_advantages,
            partition=self.update_partition,
            device=self.device,
        )
        if isinstance(normalized_advantages_result, _result.Rejected):
            return normalized_advantages_result
        if batch is not None:
            observation_batch_start = profile.mark()
            prepared_batch = prepare_ppo_batch(
                source=batch,
                advantages=normalized_advantages_result.value,
            )
            profile.record_elapsed(
                "observation_batch_seconds",
                observation_batch_start,
            )
        partition_check = _validate_update_partition(
            sample_count=global_sample_count_result.value,
            minibatch_size=self.train_config.minibatch_size,
        )
        if isinstance(partition_check, _result.Rejected):
            return partition_check
        stat_sums = torch.zeros(
            (6,), dtype=torch.float32, device=self.device
        )
        stat_count = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        parameters = tuple(self.model.parameters())
        for epoch in range(self.train_config.ppo_epochs):
            local_epoch_schedule = _local_epoch_schedule(
                prepared_batch=prepared_batch,
                train_config=self.train_config,
                policy_version=update_input.policy_version,
                epoch=epoch,
                device=self.device,
            )
            local_minibatches = _local_epoch_minibatch_spans(
                epoch_schedule=local_epoch_schedule,
                minibatch_size=self.train_config.minibatch_size,
            )
            minibatch_step_count = _global_minibatch_step_bound(
                global_sample_count=global_sample_count_result.value,
                minibatch_size=self.train_config.minibatch_size,
            )
            local_counts = _local_minibatch_counts(
                local_minibatches,
                step_count=minibatch_step_count,
                device=self.device,
            )
            global_counts_result = synchronized_count_vector_sum(
                values=local_counts,
                partition=self.update_partition,
                device=self.device,
            )
            if isinstance(global_counts_result, _result.Rejected):
                return global_counts_result
            global_counts = global_counts_result.value
            global_count_values = _count_tensor_values(global_counts)
            for step_index, global_count_value in enumerate(
                global_count_values
            ):
                if global_count_value == 0:
                    break
                local_span = _local_minibatch_or_empty(
                    local_minibatches,
                    step_index=step_index,
                )
                local_count = local_span.count()
                global_count = global_counts[step_index]
                tensorized_minibatch = _tensorized_minibatch_for_step(
                    prepared_batch=prepared_batch,
                    epoch_schedule=local_epoch_schedule,
                    span=local_span,
                    global_count=global_count,
                    device=self.device,
                )
                loss_start = profile.mark()
                forward_output = loss_forwarder(
                    tensorized_minibatch,
                    profile,
                )
                profile.record_elapsed(
                    "minibatch_loss_seconds", loss_start
                )
                loss = forward_output.loss
                self.model.zero_grad(set_to_none=True)
                loss_validation_code = combine_validation_codes(
                    forward_output.validation_code,
                    _loss_validation_code(loss),
                )
                loss_validation_code_result = _sync_validation_code(
                    code=loss_validation_code,
                    partition=self.update_partition,
                )
                if isinstance(
                    loss_validation_code_result, _result.Rejected
                ):
                    self.model.zero_grad(set_to_none=True)
                    return loss_validation_code_result
                loss_rejection = validation_rejection_reason(
                    loss_validation_code_result.value
                )
                if loss_rejection is not None:
                    self.model.zero_grad(set_to_none=True)
                    return _result.Rejected(reason=loss_rejection)
                backward_start = profile.mark()
                torch.autograd.backward(
                    loss.total_loss
                    * _ddp_loss_scale(
                        local_count=local_count,
                        global_count=global_count,
                        world_size=self.update_partition.world_size,
                    )
                )
                profile.record_elapsed(
                    "backward_seconds", backward_start
                )
                gradient_code = gradient_validation_code(parameters)
                clip_grad_norm_on_device(
                    parameters,
                    max_norm=self.train_config.max_grad_norm,
                )
                gradient_code = combine_validation_codes(
                    gradient_code,
                    gradient_validation_code(parameters),
                )
                gradient_code_result = _sync_validation_code(
                    code=gradient_code,
                    partition=self.update_partition,
                )
                if isinstance(gradient_code_result, _result.Rejected):
                    self.model.zero_grad(set_to_none=True)
                    return gradient_code_result
                clipped_gradient_rejection = (
                    validation_rejection_reason(
                        gradient_code_result.value
                    )
                )
                if clipped_gradient_rejection is not None:
                    self.model.zero_grad(set_to_none=True)
                    return _result.Rejected(
                        reason=clipped_gradient_rejection
                    )
                optimizer_start = profile.mark()
                self.optimizer.step()
                profile.record_elapsed(
                    "optimizer_step_seconds", optimizer_start
                )
                if local_count > 0:
                    stat_sums = stat_sums + _loss_stat_tensor(
                        loss
                    ) * float(local_count)
                    stat_count = stat_count + torch.tensor(
                        float(local_count),
                        dtype=torch.float32,
                        device=self.device,
                    )
        stats_result = _finalize_update_stats(
            stat_sums=stat_sums,
            stat_count=stat_count,
            profile=profile.finish(),
            partition=self.update_partition,
        )
        if isinstance(stats_result, _result.Rejected):
            return stats_result
        stats = stats_result.value
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


def _local_epoch_schedule(
    *,
    prepared_batch: PreparedPPOBatch | None,
    train_config: TrainConfig,
    policy_version: int,
    epoch: int,
    device: torch.device,
) -> PPOEpochSchedule | None:
    if prepared_batch is None:
        return None
    sample_count = prepared_batch.sample_count
    shuffled_order = shuffled_indices(
        key=ShuffleKey(
            base_seed=train_config.seed,
            policy_version=policy_version,
            epoch=epoch,
        ),
        length=sample_count,
    )
    shuffled_order_tensor = shuffled_index_tensor(
        indices=shuffled_order,
        device=device,
    )
    return prepare_ppo_epoch_schedule(
        batch=prepared_batch,
        indices=shuffled_order_tensor,
    )


def _local_epoch_minibatch_spans(
    *,
    epoch_schedule: PPOEpochSchedule | None,
    minibatch_size: int,
) -> tuple[_MinibatchSpan, ...]:
    if epoch_schedule is None:
        return ()
    return _index_minibatch_spans(
        sample_count=epoch_schedule.sample_count,
        minibatch_size=minibatch_size,
    )


def _tensorized_minibatch_for_step(
    *,
    prepared_batch: PreparedPPOBatch | None,
    epoch_schedule: PPOEpochSchedule | None,
    span: _MinibatchSpan,
    global_count: Tensor,
    device: torch.device,
) -> TensorizedPPOMinibatch:
    if prepared_batch is None or epoch_schedule is None:
        return empty_ppo_minibatch(
            device=device,
            global_count=global_count,
        )
    assert prepared_batch.sample_count == epoch_schedule.sample_count
    return prepared_ppo_epoch_minibatch(
        batch=prepared_batch,
        schedule=epoch_schedule,
        start=span.start,
        end=span.end,
        global_count=global_count,
    )


def _validate_update_partition(
    *,
    sample_count: int,
    minibatch_size: int,
) -> _result.Ok[None] | _result.Rejected:
    assert sample_count > 0
    if minibatch_size <= 0:
        return _result.Rejected(reason="PPO minibatch_size must be > 0")
    return _result.Ok(value=None)


def _global_minibatch_step_bound(
    *, global_sample_count: int, minibatch_size: int
) -> int:
    assert global_sample_count > 0
    assert minibatch_size > 0
    return (global_sample_count + minibatch_size - 1) // minibatch_size


def _count_tensor_values(counts: Tensor) -> tuple[int, ...]:
    assert counts.ndim == 1
    cpu_counts = counts.detach().cpu()
    return tuple(
        int(cpu_counts[index].item())
        for index in range(int(cpu_counts.shape[0]))
    )


def _sync_validation_code(
    *, code: Tensor, partition: PPOUpdatePartition
) -> _result.Ok[Tensor] | _result.Rejected:
    assert code.shape == ()
    if partition.world_size == 1:
        return _result.Ok(value=code)
    if not dist.is_initialized():
        return _result.Rejected(
            reason=(
                "distributed PPO validation sync requires process group"
            )
        )
    return _result.Ok(value=all_reduce_max(code))


def _index_minibatch_spans(
    *,
    sample_count: int,
    minibatch_size: int,
) -> tuple[_MinibatchSpan, ...]:
    assert minibatch_size > 0
    assert sample_count > 0
    result: list[_MinibatchSpan] = []
    for start in range(0, sample_count, minibatch_size):
        result.append(
            _MinibatchSpan(
                start=start,
                end=min(start + minibatch_size, sample_count),
            )
        )
    return tuple(result)


def _local_minibatch_or_empty(
    minibatches: tuple[_MinibatchSpan, ...],
    *,
    step_index: int,
) -> _MinibatchSpan:
    assert step_index >= 0
    if step_index < len(minibatches):
        return minibatches[step_index]
    return _MinibatchSpan(start=0, end=0)


def _local_minibatch_counts(
    minibatches: tuple[_MinibatchSpan, ...],
    *,
    step_count: int,
    device: torch.device,
) -> Tensor:
    assert step_count >= 0
    counts = torch.zeros((step_count,), dtype=torch.long, device=device)
    if not minibatches:
        return counts
    values = tuple(minibatch.count() for minibatch in minibatches)
    counts[: len(values)] = torch.tensor(
        values, dtype=torch.long, device=device
    )
    return counts


def _ddp_loss_scale(
    *, local_count: int, global_count: Tensor, world_size: int
) -> Tensor:
    assert local_count >= 0
    assert global_count.shape == ()
    assert world_size > 0
    numerator = torch.tensor(
        float(world_size * local_count),
        dtype=torch.float32,
        device=global_count.device,
    )
    return numerator / global_count.to(dtype=torch.float32)


def _sync_normalized_advantages(
    *,
    advantages: Tensor,
    partition: PPOUpdatePartition,
    device: torch.device,
) -> _result.Ok[Tensor] | _result.Rejected:
    assert advantages.ndim == 1
    if partition.world_size == 1:
        return _result.Ok(value=_normalize_advantages(advantages))
    if not dist.is_initialized():
        return _result.Rejected(
            reason=(
                "distributed PPO advantage sync requires process group"
            )
        )
    local_count = torch.tensor(
        float(int(advantages.shape[0])),
        dtype=torch.float32,
        device=device,
    )
    advantage_sum = advantages.sum()
    advantage_square_sum = (advantages * advantages).sum()
    totals = torch.stack(
        (
            advantage_sum,
            advantage_square_sum,
            local_count,
        )
    )
    totals = all_reduce_sum(totals)
    count = totals[2]
    mean = totals[0] / count
    variance = torch.clamp(totals[1] / count - mean * mean, min=0.0)
    stddev = torch.sqrt(variance)
    centered = advantages - mean
    normalized = torch.where(
        stddev <= 0.000001,
        centered,
        centered / (stddev + 0.000001),
    )
    return _result.Ok(value=normalized)


def _normalize_advantages(advantages: Tensor) -> Tensor:
    assert advantages.ndim == 1
    assert int(advantages.shape[0]) > 0
    mean = advantages.mean()
    variance = ((advantages - mean) * (advantages - mean)).mean()
    stddev = torch.sqrt(variance)
    centered = advantages - mean
    normalized = centered / (stddev + 0.000001)
    return torch.where(stddev <= 0.000001, centered, normalized)


def _loss_stat_tensor(loss: MinibatchLoss) -> Tensor:
    return torch.stack(
        (
            loss.policy_loss.detach(),
            loss.value_loss.detach(),
            loss.entropy.detach(),
            loss.total_loss.detach(),
            loss.approx_kl.detach(),
            loss.clip_fraction.detach(),
        )
    )


def _finalize_update_stats(
    *,
    stat_sums: Tensor,
    stat_count: Tensor,
    profile: PPOUpdateProfile,
    partition: PPOUpdatePartition,
) -> _result.Ok[PPOUpdateStats] | _result.Rejected:
    assert stat_sums.shape == (6,)
    assert stat_count.shape == ()
    totals = torch.cat((stat_sums, stat_count.reshape(1)))
    if partition.world_size > 1:
        if not dist.is_initialized():
            return _result.Rejected(
                reason=(
                    "distributed PPO stats sync requires process group"
                )
            )
        totals = all_reduce_sum(totals)
    count = totals[6]
    count_value = _float_tensor(count)
    if count_value <= 0.0:
        return _result.Rejected(
            reason="PPO update stats require rollout decisions"
        )
    means = totals[:6] / count
    return _result.Ok(
        value=PPOUpdateStats(
            policy_loss=_float_tensor(means[0]),
            value_loss=_float_tensor(means[1]),
            entropy=_float_tensor(means[2]),
            total_loss=_float_tensor(means[3]),
            approx_kl=_float_tensor(means[4]),
            clip_fraction=_float_tensor(means[5]),
            profile=profile,
        )
    )


def _loss_validation_code(loss: MinibatchLoss) -> Tensor:
    return non_finite_validation_code(
        (
            TensorValidationCheck(
                tensor=loss.policy_loss,
                code=PPO_POLICY_LOSS_NONFINITE,
            ),
            TensorValidationCheck(
                tensor=loss.value_loss,
                code=PPO_VALUE_LOSS_NONFINITE,
            ),
            TensorValidationCheck(
                tensor=loss.entropy,
                code=PPO_ENTROPY_NONFINITE,
            ),
            TensorValidationCheck(
                tensor=loss.total_loss,
                code=PPO_TOTAL_LOSS_NONFINITE,
            ),
            TensorValidationCheck(
                tensor=loss.approx_kl,
                code=PPO_APPROX_KL_NONFINITE,
            ),
            TensorValidationCheck(
                tensor=loss.clip_fraction,
                code=PPO_CLIP_FRACTION_NONFINITE,
            ),
        )
    )


def _float_tensor(value: Tensor) -> float:
    return float(value.detach().cpu().item())
