"""Typed argv contracts for the external training CLI process."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

MIN_CLI_MAX_TOKENS = 512
DEFAULT_RUN_DIR = Path("training_runs")


def _parse_request_path(value: object) -> object:
    if isinstance(value, str):
        return Path(value)
    return value


type ManagedCheckpointName = Annotated[
    str, Field(pattern=r"^(latest|update-[1-9][0-9]*)\.json$")
]
type RequestPath = Annotated[Path, BeforeValidator(_parse_request_path)]


class TrainingInitRequest(BaseModel):
    """Portable state required to initialize a new training run."""

    model_config = ConfigDict(extra="forbid", strict=True)

    run_dir: RequestPath = DEFAULT_RUN_DIR
    replace_existing: Literal["yes"] | None = None
    d_model: Annotated[int, Field(gt=0)] = 128
    layers: Annotated[int, Field(gt=0)] = 3
    heads: Annotated[int, Field(gt=0)] = 4
    max_tokens: Annotated[int, Field(ge=MIN_CLI_MAX_TOKENS)] = 768
    seed: Annotated[int, Field(ge=0)] = 0
    learning_rate: Annotated[
        float, Field(gt=0.0, allow_inf_nan=False)
    ] = 0.0003
    ppo_clip: Annotated[
        float, Field(gt=0.0, le=1.0, allow_inf_nan=False)
    ] = 0.2
    value_clip: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = (
        0.2
    )
    entropy_coef: Annotated[
        float, Field(ge=0.0, allow_inf_nan=False)
    ] = 0.01
    value_coef: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = (
        0.5
    )
    max_grad_norm: Annotated[
        float, Field(ge=0.0, allow_inf_nan=False)
    ] = 0.5
    ppo_epochs: Annotated[int, Field(gt=0)] = 4
    minibatch_size: Annotated[int, Field(gt=0)] = 64
    adam_beta1: Annotated[
        float, Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    ] = 0.9
    adam_beta2: Annotated[
        float, Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    ] = 0.999
    weight_decay: Annotated[
        float, Field(ge=0.0, allow_inf_nan=False)
    ] = 0.0

    @model_validator(mode="after")
    def validate_model_shape(self) -> TrainingInitRequest:
        if self.d_model % self.heads != 0:
            raise ValueError("--d-model must be divisible by --heads")
        return self

    def to_cli_argv(self) -> tuple[str, ...]:
        argv = ["--run-dir", str(self.run_dir), "init"]
        _append_optional(
            argv, "--replace-existing", self.replace_existing
        )
        for flag, value in _portable_values(self):
            _append(argv, flag, value)
        return tuple(argv)


class TrainingResumeRequest(BaseModel):
    """Checkpoint and process policy for one training session."""

    model_config = ConfigDict(extra="forbid", strict=True)

    run_dir: RequestPath = DEFAULT_RUN_DIR
    checkpoint: ManagedCheckpointName
    worker_cpus: str | None = None
    model_ranks: str | None = None
    ppo_profile: Literal["off", "basic", "detailed"] | None = None
    max_samples: Annotated[int, Field(ge=0)] = 0
    learning_rate: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None
    ) = None
    checkpoint_every_updates: Annotated[int, Field(gt=0)] = 50
    checkpoint_retention_updates: Annotated[int, Field(ge=0)] = 5
    round_timeout_seconds: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None
    ) = None
    sampling_start_timeout_seconds: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None
    ) = None
    rollout_sample_timeout_seconds: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None
    ) = None
    sampling_stop_timeout_seconds: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None
    ) = None
    state_sync_timeout_seconds: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None
    ) = None
    update_timeout_seconds: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None
    ) = None
    model_inference_batch_size: Annotated[int, Field(gt=0)] | None = (
        None
    )
    game_envs_per_worker: Annotated[int, Field(gt=0)] | None = None
    samples_per_update: Annotated[int, Field(gt=0)] | None = None
    ppo_clip: (
        Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)]
        | None
    ) = None
    value_clip: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None
    ) = None
    entropy_coef: (
        Annotated[float, Field(ge=0.0, allow_inf_nan=False)] | None
    ) = None
    value_coef: (
        Annotated[float, Field(ge=0.0, allow_inf_nan=False)] | None
    ) = None
    max_grad_norm: (
        Annotated[float, Field(ge=0.0, allow_inf_nan=False)] | None
    ) = None
    ppo_epochs: Annotated[int, Field(gt=0)] | None = None
    minibatch_size: Annotated[int, Field(gt=0)] | None = None
    adam_beta1: (
        Annotated[float, Field(ge=0.0, lt=1.0, allow_inf_nan=False)]
        | None
    ) = None
    adam_beta2: (
        Annotated[float, Field(ge=0.0, lt=1.0, allow_inf_nan=False)]
        | None
    ) = None
    weight_decay: (
        Annotated[float, Field(ge=0.0, allow_inf_nan=False)] | None
    ) = None

    def to_cli_argv(self) -> tuple[str, ...]:
        argv = [
            "--run-dir",
            str(self.run_dir),
            "resume",
            self.checkpoint,
            "--max-samples",
            str(self.max_samples),
            "--checkpoint-every-updates",
            str(self.checkpoint_every_updates),
            "--checkpoint-retention-updates",
            str(self.checkpoint_retention_updates),
        ]
        values: tuple[tuple[str, str | int | float | None], ...] = (
            ("--worker-cpus", self.worker_cpus),
            ("--model-ranks", self.model_ranks),
            ("--ppo-profile", self.ppo_profile),
            ("--learning-rate", self.learning_rate),
            ("--round-timeout-seconds", self.round_timeout_seconds),
            (
                "--sampling-start-timeout-seconds",
                self.sampling_start_timeout_seconds,
            ),
            (
                "--rollout-sample-timeout-seconds",
                self.rollout_sample_timeout_seconds,
            ),
            (
                "--sampling-stop-timeout-seconds",
                self.sampling_stop_timeout_seconds,
            ),
            (
                "--state-sync-timeout-seconds",
                self.state_sync_timeout_seconds,
            ),
            ("--update-timeout-seconds", self.update_timeout_seconds),
            (
                "--model-inference-batch-size",
                self.model_inference_batch_size,
            ),
            ("--game-envs-per-worker", self.game_envs_per_worker),
            ("--samples-per-update", self.samples_per_update),
            ("--ppo-clip", self.ppo_clip),
            ("--value-clip", self.value_clip),
            ("--entropy-coef", self.entropy_coef),
            ("--value-coef", self.value_coef),
            ("--max-grad-norm", self.max_grad_norm),
            ("--ppo-epochs", self.ppo_epochs),
            ("--minibatch-size", self.minibatch_size),
            ("--adam-beta1", self.adam_beta1),
            ("--adam-beta2", self.adam_beta2),
            ("--weight-decay", self.weight_decay),
        )
        for flag, value in values:
            _append_optional(argv, flag, value)
        return tuple(argv)


def init_command_argv(request: TrainingInitRequest) -> tuple[str, ...]:
    """Return the exact argv for a short initialization process."""
    return (
        sys.executable,
        "-m",
        "server.training_cli",
        *request.to_cli_argv(),
    )


def resume_command_argv(
    request: TrainingResumeRequest,
) -> tuple[str, ...]:
    """Return the exact argv for a detached training process."""
    return (
        sys.executable,
        "-m",
        "server.training_cli",
        *request.to_cli_argv(),
    )


def _portable_values(
    request: TrainingInitRequest,
) -> tuple[tuple[str, int | float], ...]:
    return (
        ("--d-model", request.d_model),
        ("--layers", request.layers),
        ("--heads", request.heads),
        ("--max-tokens", request.max_tokens),
        ("--seed", request.seed),
        ("--learning-rate", request.learning_rate),
        ("--ppo-clip", request.ppo_clip),
        ("--value-clip", request.value_clip),
        ("--entropy-coef", request.entropy_coef),
        ("--value-coef", request.value_coef),
        ("--max-grad-norm", request.max_grad_norm),
        ("--ppo-epochs", request.ppo_epochs),
        ("--minibatch-size", request.minibatch_size),
        ("--adam-beta1", request.adam_beta1),
        ("--adam-beta2", request.adam_beta2),
        ("--weight-decay", request.weight_decay),
    )


def _append(
    argv: list[str], flag: str, value: str | int | float
) -> None:
    argv.extend((flag, str(value)))


def _append_optional(
    argv: list[str], flag: str, value: str | int | float | None
) -> None:
    if value is not None:
        _append(argv, flag, value)
