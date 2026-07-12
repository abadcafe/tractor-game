"""Resolve resume options into executable training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from server.foundation import result as _result
from server.training.config import (
    CheckpointPolicy,
    ModelConfig,
    TrainConfig,
)
from server.training.interface import TrainingResumeOptions
from server.training.runtime import (
    CpuSet,
    ExecutionConfig,
    ExecutionTimeouts,
    ModelRankPlacement,
    PPOProfileMode,
    parse_model_rank_placement,
)
from server.training.runtime.config import parse_cpu_set
from server.training.torch_checkpoints.load import (
    read_torch_checkpoint_metadata,
)

CHECKPOINTS_DIR_NAME = "checkpoints"


@dataclass(frozen=True, slots=True)
class TrainConfigOverrides:
    learning_rate: float | None = None
    ppo_clip: float | None = None
    value_clip: float | None = None
    entropy_coef: float | None = None
    value_coef: float | None = None
    max_grad_norm: float | None = None
    ppo_epochs: int | None = None
    minibatch_size: int | None = None
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    weight_decay: float | None = None


@dataclass(frozen=True, slots=True)
class ExecutionConfigOverrides:
    worker_cpus: CpuSet | None = None
    model_ranks: ModelRankPlacement | None = None
    ppo_profile: PPOProfileMode | None = None
    round_timeout_seconds: float | None = None
    sampling_start_timeout_seconds: float | None = None
    rollout_sample_timeout_seconds: float | None = None
    sampling_stop_timeout_seconds: float | None = None
    state_sync_timeout_seconds: float | None = None
    update_timeout_seconds: float | None = None
    model_inference_batch_size: int | None = None
    game_envs_per_worker: int | None = None
    samples_per_update: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedTrainingResume:
    run_dir: Path
    checkpoint_path: Path
    model_config: ModelConfig
    train_config: TrainConfig
    checkpoint_policy: CheckpointPolicy
    execution_config: ExecutionConfig
    max_samples: int


def resolve_resume_options(
    request: TrainingResumeOptions,
) -> _result.Ok[ResolvedTrainingResume] | _result.Rejected:
    """Load checkpoint-owned state and resolve process-owned policy."""
    run_dir = request.run_dir
    checkpoint_path = (
        run_dir / CHECKPOINTS_DIR_NAME / request.checkpoint
    )
    execution_result = _resolve_execution_config(request)
    if isinstance(execution_result, _result.Rejected):
        return execution_result
    metadata_result = read_torch_checkpoint_metadata(checkpoint_path)
    if isinstance(metadata_result, _result.Rejected):
        return metadata_result
    base = metadata_result.value.train_config
    return _result.Ok(
        value=ResolvedTrainingResume(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            model_config=metadata_result.value.model_config,
            train_config=_override_train_config(
                base, _train_overrides(request)
            ),
            checkpoint_policy=CheckpointPolicy(
                every_updates=request.checkpoint_every_updates,
                retention_updates=request.checkpoint_retention_updates,
            ),
            execution_config=execution_result.value,
            max_samples=request.max_samples,
        )
    )


def resolve_execution_config(
    overrides: ExecutionConfigOverrides,
) -> _result.Ok[ExecutionConfig]:
    base = ExecutionConfig()
    return _result.Ok(
        value=ExecutionConfig(
            worker_cpus=_or_base(
                overrides.worker_cpus, base.worker_cpus
            ),
            model_ranks=_or_base(
                overrides.model_ranks, base.model_ranks
            ),
            ppo_profile=_or_base(
                overrides.ppo_profile, base.ppo_profile
            ),
            timeouts=ExecutionTimeouts(
                round_seconds=_or_base(
                    overrides.round_timeout_seconds,
                    base.timeouts.round_seconds,
                ),
                sampling_start_seconds=_or_base(
                    overrides.sampling_start_timeout_seconds,
                    base.timeouts.sampling_start_seconds,
                ),
                rollout_sample_seconds=_or_base(
                    overrides.rollout_sample_timeout_seconds,
                    base.timeouts.rollout_sample_seconds,
                ),
                sampling_stop_seconds=_or_base(
                    overrides.sampling_stop_timeout_seconds,
                    base.timeouts.sampling_stop_seconds,
                ),
                state_sync_seconds=_or_base(
                    overrides.state_sync_timeout_seconds,
                    base.timeouts.state_sync_seconds,
                ),
                update_seconds=_or_base(
                    overrides.update_timeout_seconds,
                    base.timeouts.update_seconds,
                ),
            ),
            model_inference_batch_size=_or_base(
                overrides.model_inference_batch_size,
                base.model_inference_batch_size,
            ),
            game_envs_per_worker=_or_base(
                overrides.game_envs_per_worker,
                base.game_envs_per_worker,
            ),
            samples_per_update=_or_base(
                overrides.samples_per_update, base.samples_per_update
            ),
        )
    )


def _resolve_execution_config(
    request: TrainingResumeOptions,
) -> _result.Ok[ExecutionConfig] | _result.Rejected:
    cpu_result = _parse_optional_cpu_set(request.worker_cpus)
    if isinstance(cpu_result, _result.Rejected):
        return cpu_result
    ranks_result = _parse_optional_model_ranks(request.model_ranks)
    if isinstance(ranks_result, _result.Rejected):
        return ranks_result
    return resolve_execution_config(
        ExecutionConfigOverrides(
            worker_cpus=cpu_result.value,
            model_ranks=ranks_result.value,
            ppo_profile=request.ppo_profile,
            round_timeout_seconds=request.round_timeout_seconds,
            sampling_start_timeout_seconds=(
                request.sampling_start_timeout_seconds
            ),
            rollout_sample_timeout_seconds=(
                request.rollout_sample_timeout_seconds
            ),
            sampling_stop_timeout_seconds=(
                request.sampling_stop_timeout_seconds
            ),
            state_sync_timeout_seconds=request.state_sync_timeout_seconds,
            update_timeout_seconds=request.update_timeout_seconds,
            model_inference_batch_size=request.model_inference_batch_size,
            game_envs_per_worker=request.game_envs_per_worker,
            samples_per_update=request.samples_per_update,
        )
    )


def _train_overrides(
    request: TrainingResumeOptions,
) -> TrainConfigOverrides:
    return TrainConfigOverrides(
        learning_rate=request.learning_rate,
        ppo_clip=request.ppo_clip,
        value_clip=request.value_clip,
        entropy_coef=request.entropy_coef,
        value_coef=request.value_coef,
        max_grad_norm=request.max_grad_norm,
        ppo_epochs=request.ppo_epochs,
        minibatch_size=request.minibatch_size,
        adam_beta1=request.adam_beta1,
        adam_beta2=request.adam_beta2,
        weight_decay=request.weight_decay,
    )


def _override_train_config(
    base: TrainConfig, overrides: TrainConfigOverrides
) -> TrainConfig:
    return TrainConfig(
        seed=base.seed,
        learning_rate=_or_base(
            overrides.learning_rate, base.learning_rate
        ),
        ppo_clip=_or_base(overrides.ppo_clip, base.ppo_clip),
        value_clip=_or_base(overrides.value_clip, base.value_clip),
        entropy_coef=_or_base(
            overrides.entropy_coef, base.entropy_coef
        ),
        value_coef=_or_base(overrides.value_coef, base.value_coef),
        max_grad_norm=_or_base(
            overrides.max_grad_norm, base.max_grad_norm
        ),
        ppo_epochs=_or_base(overrides.ppo_epochs, base.ppo_epochs),
        minibatch_size=_or_base(
            overrides.minibatch_size, base.minibatch_size
        ),
        adam_beta1=_or_base(overrides.adam_beta1, base.adam_beta1),
        adam_beta2=_or_base(overrides.adam_beta2, base.adam_beta2),
        weight_decay=_or_base(
            overrides.weight_decay, base.weight_decay
        ),
    )


def _parse_optional_cpu_set(
    value: str | None,
) -> _result.Ok[CpuSet | None] | _result.Rejected:
    if value is None:
        return _result.Ok(value=None)
    parsed = parse_cpu_set(value)
    if isinstance(parsed, _result.Rejected):
        return parsed
    return _result.Ok(value=parsed.value)


def _parse_optional_model_ranks(
    value: str | None,
) -> _result.Ok[ModelRankPlacement | None] | _result.Rejected:
    if value is None:
        return _result.Ok(value=None)
    parsed = parse_model_rank_placement(value)
    if isinstance(parsed, _result.Rejected):
        return parsed
    return _result.Ok(value=parsed.value)


def _or_base[T](override: T | None, base: T) -> T:
    return base if override is None else override
