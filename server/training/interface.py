"""Small public lifecycle interface for the deep training package."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from server.foundation import result as _result
from server.training.stop import TrainingStopRequest

type ManagedCheckpointName = Annotated[
    str, Field(pattern=r"^(latest|update-[1-9][0-9]*)\.json$")
]


class TrainingInitOptions(BaseModel):
    """Portable state used to create a new training run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_dir: Path
    replace_existing: Literal["yes"] | None = None
    d_model: int = Field(default=128, gt=0)
    layers: int = Field(default=3, gt=0)
    heads: int = Field(default=4, gt=0)
    max_tokens: int = Field(default=768, ge=512)
    seed: int = Field(default=0, ge=0)
    learning_rate: float = Field(
        default=0.0003, gt=0.0, allow_inf_nan=False
    )
    ppo_clip: float = Field(
        default=0.2, gt=0.0, le=1.0, allow_inf_nan=False
    )
    value_clip: float = Field(default=0.2, gt=0.0, allow_inf_nan=False)
    entropy_coef: float = Field(
        default=0.01, ge=0.0, allow_inf_nan=False
    )
    value_coef: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    max_grad_norm: float = Field(
        default=0.5, ge=0.0, allow_inf_nan=False
    )
    ppo_epochs: int = Field(default=4, gt=0)
    minibatch_size: int = Field(default=64, gt=0)
    adam_beta1: float = Field(
        default=0.9, ge=0.0, lt=1.0, allow_inf_nan=False
    )
    adam_beta2: float = Field(
        default=0.999, ge=0.0, lt=1.0, allow_inf_nan=False
    )
    weight_decay: float = Field(
        default=0.0, ge=0.0, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def validate_model_shape(self) -> TrainingInitOptions:
        if self.d_model % self.heads != 0:
            raise ValueError("--d-model must be divisible by --heads")
        return self


class TrainingResumeOptions(BaseModel):
    """Checkpoint and process policy for one resumed training run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_dir: Path
    checkpoint: ManagedCheckpointName
    worker_cpus: str | None = None
    model_ranks: str | None = None
    ppo_profile: Literal["off", "basic", "detailed"] | None = None
    max_samples: int = Field(default=0, ge=0)
    learning_rate: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    checkpoint_every_updates: int = Field(default=50, gt=0)
    checkpoint_retention_updates: int = Field(default=5, ge=0)
    round_timeout_seconds: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    sampling_start_timeout_seconds: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    rollout_sample_timeout_seconds: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    sampling_stop_timeout_seconds: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    state_sync_timeout_seconds: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    update_timeout_seconds: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    model_inference_batch_size: int | None = Field(default=None, gt=0)
    game_envs_per_worker: int | None = Field(default=None, gt=0)
    samples_per_update: int | None = Field(default=None, gt=0)
    ppo_clip: float | None = Field(
        default=None, gt=0.0, le=1.0, allow_inf_nan=False
    )
    value_clip: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    entropy_coef: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    value_coef: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    max_grad_norm: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    ppo_epochs: int | None = Field(default=None, gt=0)
    minibatch_size: int | None = Field(default=None, gt=0)
    adam_beta1: float | None = Field(
        default=None, ge=0.0, lt=1.0, allow_inf_nan=False
    )
    adam_beta2: float | None = Field(
        default=None, ge=0.0, lt=1.0, allow_inf_nan=False
    )
    weight_decay: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )


class InitializedRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_dir: Path
    checkpoint_path: Path


class TrainingRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    checkpoint_path: Path
    total_rounds: int = Field(ge=0)
    total_samples: int = Field(ge=0)
    total_updates: int = Field(ge=0)


class TrainingService:
    """Public interface hiding all training implementation modules."""

    def initialize(
        self, options: TrainingInitOptions
    ) -> _result.Ok[InitializedRun] | _result.Rejected:
        return initialize_run(options)

    def resume(
        self,
        options: TrainingResumeOptions,
        stop_request: TrainingStopRequest,
        on_ready: Callable[[], None] | None = None,
    ) -> _result.Ok[TrainingRunResult] | _result.Rejected:
        return resume_run(options, stop_request, on_ready=on_ready)


def initialize_run(
    options: TrainingInitOptions,
) -> _result.Ok[InitializedRun] | _result.Rejected:
    """Create a portable zero-update checkpoint and event store."""
    from server.training.config import ModelConfig, TrainConfig
    from server.training.run_setup import initialize_training_run

    result = initialize_training_run(
        run_dir=options.run_dir,
        model_config=ModelConfig(
            d_model=options.d_model,
            layers=options.layers,
            heads=options.heads,
            max_tokens=options.max_tokens,
        ),
        train_config=TrainConfig(
            seed=options.seed,
            learning_rate=options.learning_rate,
            ppo_clip=options.ppo_clip,
            value_clip=options.value_clip,
            entropy_coef=options.entropy_coef,
            value_coef=options.value_coef,
            max_grad_norm=options.max_grad_norm,
            ppo_epochs=options.ppo_epochs,
            minibatch_size=options.minibatch_size,
            adam_beta1=options.adam_beta1,
            adam_beta2=options.adam_beta2,
            weight_decay=options.weight_decay,
        ),
        replace_existing=options.replace_existing == "yes",
    )
    if isinstance(result, _result.Rejected):
        return result
    return _result.Ok(
        value=InitializedRun(
            run_dir=result.value.run_dir,
            checkpoint_path=result.value.checkpoint_path,
        )
    )


def resume_run(
    options: TrainingResumeOptions,
    stop_request: TrainingStopRequest,
    *,
    on_ready: Callable[[], None] | None = None,
) -> _result.Ok[TrainingRunResult] | _result.Rejected:
    """Validate, load, and execute resumed training."""
    from server.training.resume_config import resolve_resume_options
    from server.training.resume_setup import (
        canonicalize_resume_timeline,
    )
    from server.training.runtime.affinity import preflight_cpu_affinity
    from server.training.runtime.checkpoint_state import (
        load_runtime_checkpoint_state,
    )
    from server.training.runtime.coordinator import (
        run_training_coordinator,
    )
    from server.training.training_state import (
        validate_model_rank_runtime,
    )

    resolved_result = resolve_resume_options(options)
    if isinstance(resolved_result, _result.Rejected):
        return resolved_result
    resolved = resolved_result.value
    model_rank_result = validate_model_rank_runtime(
        resolved.execution_config
    )
    if isinstance(model_rank_result, _result.Rejected):
        return model_rank_result
    for worker_index in range(
        resolved.execution_config.worker_process_count()
    ):
        affinity_result = preflight_cpu_affinity(
            label=f"worker-{worker_index}",
            cpus=resolved.execution_config.worker_cpu_set(worker_index),
        )
        if isinstance(affinity_result, _result.Rejected):
            return affinity_result
    load_result = load_runtime_checkpoint_state(
        path=resolved.checkpoint_path,
        model_config=resolved.model_config,
        train_config=resolved.train_config,
        execution_config=resolved.execution_config,
    )
    if isinstance(load_result, _result.Rejected):
        return load_result
    timeline_result = canonicalize_resume_timeline(
        run_dir=resolved.run_dir,
        selected_checkpoint=resolved.checkpoint_path,
    )
    if isinstance(timeline_result, _result.Rejected):
        return timeline_result
    result = run_training_coordinator(
        run_dir=resolved.run_dir,
        runtime_id=str(uuid4()),
        model_config=resolved.model_config,
        train_config=resolved.train_config,
        checkpoint_policy=resolved.checkpoint_policy,
        execution_config=resolved.execution_config,
        max_samples=resolved.max_samples,
        resume=resolved.run_dir / "checkpoints" / "latest.json",
        stop_request=stop_request,
        on_ready=on_ready,
    )
    if isinstance(result, _result.Rejected):
        return result
    value = result.value
    return _result.Ok(
        value=TrainingRunResult(
            checkpoint_path=value.checkpoint_path,
            total_rounds=value.total_rounds,
            total_samples=value.total_samples,
            total_updates=value.total_updates,
        )
    )
