"""Validated requests and results for the training lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from server.policy_model.network import MIN_ATTENTION_HEAD_DIMENSION

type ManagedCheckpointName = Annotated[
    str, Field(pattern=r"^(latest|update-[1-9][0-9]*)\.json$")
]


class TrainingInitOptions(BaseModel):
    """Portable state used to create a new training run."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    run_dir: Path
    replace_existing: Literal["yes"] | None = None
    d_model: int = Field(default=128, gt=0)
    layers: int = Field(default=3, gt=0)
    heads: int = Field(default=4, gt=0)
    seed: int = Field(default=0, ge=0)
    learning_rate: float = Field(
        default=0.0003, gt=0.0, allow_inf_nan=False
    )
    ppo_clip: float = Field(
        default=0.2, gt=0.0, le=1.0, allow_inf_nan=False
    )
    value_clip: float = Field(
        default=0.2, gt=0.0, le=1.0, allow_inf_nan=False
    )
    gae_lambda: float = Field(
        default=0.95, ge=0.0, le=1.0, allow_inf_nan=False
    )
    value_coef: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    entropy_coef: float = Field(
        default=0.01, ge=0.0, allow_inf_nan=False
    )
    policy_max_grad_norm: float = Field(
        default=0.5, ge=0.0, allow_inf_nan=False
    )
    value_max_grad_norm: float = Field(
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
        if self.d_model // self.heads < MIN_ATTENTION_HEAD_DIMENSION:
            raise ValueError(
                "--d-model divided by --heads must be at least "
                + f"{MIN_ATTENTION_HEAD_DIMENSION}"
            )
        return self


class TrainingResumeOptions(BaseModel):
    """Checkpoint and process policy for one resumed training run."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    run_dir: Path
    checkpoint: ManagedCheckpointName
    worker_cpus: str | None = None
    model_ranks: str | None = None
    ppo_profile: Literal["off", "basic", "detailed"] | None = None
    max_rounds: int = Field(default=0, ge=0)
    learning_rate: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    checkpoint_every_updates: int = Field(gt=0)
    checkpoint_retention_updates: int = Field(default=5, ge=0)
    round_timeout_seconds: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    sampling_start_timeout_seconds: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    rollout_collection_timeout_seconds: float | None = Field(
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
    rounds_per_update: int | None = Field(default=None, gt=0)
    ppo_clip: float | None = Field(
        default=None, gt=0.0, le=1.0, allow_inf_nan=False
    )
    value_clip: float | None = Field(
        default=None, gt=0.0, le=1.0, allow_inf_nan=False
    )
    gae_lambda: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    value_coef: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    entropy_coef: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    policy_max_grad_norm: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    value_max_grad_norm: float | None = Field(
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
    """Initialized run and its first checkpoint."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    run_dir: Path
    checkpoint_path: Path


class TrainingRunResult(BaseModel):
    """Cumulative counters and checkpoint returned by training."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    checkpoint_path: Path
    total_rounds: int = Field(ge=0)
    total_trainable_decisions: int = Field(ge=0)
    total_updates: int = Field(ge=0)
