"""Training CLI entry point.

The command prepares a run directory and dashboard.  Long training is
only started when the user explicitly invokes this module.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from server import result as _result
from server.training.config import (
    ModelConfig,
    PPOProfileMode,
    TrainConfig,
    TrainingDevice,
)
from server.training.loop import run_training_loop
from server.training.run_setup import (
    initialize_training_run,
    prepare_training_run,
)
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata,
)

DEFAULT_RUN_DIR = Path("training_runs/manual")
CHECKPOINTS_DIR_NAME = "checkpoints"
MIN_CLI_MAX_TOKENS = 512


@dataclass(frozen=True, slots=True)
class TrainConfigOverrides:
    """Explicit CLI overrides for train config fields."""

    device: TrainingDevice | None = None
    ppo_profile: PPOProfileMode | None = None
    seed: int | None = None
    learning_rate: float | None = None
    checkpoint_every_updates: int | None = None
    checkpoint_retention_updates: int | None = None
    max_round_seconds: float | None = None
    gae_lambda: float | None = None
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


def resolve_model_config(
    *,
    cli_model_config: ModelConfig | None,
    resume_path: Path | None,
) -> _result.Ok[ModelConfig] | _result.Rejected:
    """Use checkpoint model shape when resuming a run."""
    if resume_path is None:
        assert cli_model_config is not None
        return _result.Ok(value=cli_model_config)
    metadata_result = read_torch_checkpoint_metadata(resume_path)
    if isinstance(metadata_result, _result.Rejected):
        return metadata_result
    return _result.Ok(value=metadata_result.value.model_config)


def resolve_train_config(
    *,
    cli_overrides: TrainConfigOverrides,
    resume_path: Path | None,
) -> _result.Ok[TrainConfig] | _result.Rejected:
    """Resolve train config from checkpoint plus CLI overrides."""
    if resume_path is None:
        base = TrainConfig()
    else:
        metadata_result = read_torch_checkpoint_metadata(resume_path)
        if isinstance(metadata_result, _result.Rejected):
            return metadata_result
        base = metadata_result.value.train_config
    return _result.Ok(
        value=TrainConfig(
            device=base.device
            if cli_overrides.device is None
            else cli_overrides.device,
            ppo_profile=base.ppo_profile
            if cli_overrides.ppo_profile is None
            else cli_overrides.ppo_profile,
            seed=base.seed
            if cli_overrides.seed is None
            else cli_overrides.seed,
            learning_rate=base.learning_rate
            if cli_overrides.learning_rate is None
            else cli_overrides.learning_rate,
            checkpoint_every_updates=base.checkpoint_every_updates
            if cli_overrides.checkpoint_every_updates is None
            else cli_overrides.checkpoint_every_updates,
            checkpoint_retention_updates=base.checkpoint_retention_updates
            if cli_overrides.checkpoint_retention_updates is None
            else cli_overrides.checkpoint_retention_updates,
            max_round_seconds=base.max_round_seconds
            if cli_overrides.max_round_seconds is None
            else cli_overrides.max_round_seconds,
            gae_lambda=base.gae_lambda
            if cli_overrides.gae_lambda is None
            else cli_overrides.gae_lambda,
            ppo_clip=base.ppo_clip
            if cli_overrides.ppo_clip is None
            else cli_overrides.ppo_clip,
            value_clip=base.value_clip
            if cli_overrides.value_clip is None
            else cli_overrides.value_clip,
            entropy_coef=base.entropy_coef
            if cli_overrides.entropy_coef is None
            else cli_overrides.entropy_coef,
            value_coef=base.value_coef
            if cli_overrides.value_coef is None
            else cli_overrides.value_coef,
            max_grad_norm=base.max_grad_norm
            if cli_overrides.max_grad_norm is None
            else cli_overrides.max_grad_norm,
            ppo_epochs=base.ppo_epochs
            if cli_overrides.ppo_epochs is None
            else cli_overrides.ppo_epochs,
            minibatch_size=base.minibatch_size
            if cli_overrides.minibatch_size is None
            else cli_overrides.minibatch_size,
            adam_beta1=base.adam_beta1
            if cli_overrides.adam_beta1 is None
            else cli_overrides.adam_beta1,
            adam_beta2=base.adam_beta2
            if cli_overrides.adam_beta2 is None
            else cli_overrides.adam_beta2,
            weight_decay=base.weight_decay
            if cli_overrides.weight_decay is None
            else cli_overrides.weight_decay,
        )
    )


def _validated_run_dir(
    *,
    parser: argparse.ArgumentParser,
    cli_run_dir: Path | None,
    resume_path: Path | None,
) -> Path:
    if resume_path is None:
        if cli_run_dir is None:
            return DEFAULT_RUN_DIR
        return cli_run_dir
    resume_run_dir = _infer_resume_run_dir(resume_path)
    if resume_run_dir is None:
        parser.error(
            "--resume must point to "
            "<run-dir>/checkpoints/<checkpoint>.json"
        )
        assert False
    if cli_run_dir is None:
        return resume_run_dir
    if _canonical_path(cli_run_dir) != _canonical_path(resume_run_dir):
        parser.error(
            "--run-dir must match the run directory that owns --resume"
        )
        assert False
    return cli_run_dir


def _validate_resume_seed_override(
    *,
    parser: argparse.ArgumentParser,
    resume_path: Path | None,
    seed: int | None,
) -> _result.Ok[None] | _result.Rejected:
    if resume_path is None or seed is None:
        return _result.Ok(value=None)
    metadata_result = read_torch_checkpoint_metadata(resume_path)
    if isinstance(metadata_result, _result.Rejected):
        return metadata_result
    checkpoint_seed = metadata_result.value.train_config.seed
    if seed != checkpoint_seed:
        return _result.Rejected(
            "--seed must match the checkpoint seed when using --resume"
        )
    return _result.Ok(value=None)


def _infer_resume_run_dir(resume_path: Path) -> Path | None:
    if resume_path.suffix != ".json":
        return None
    if resume_path.parent.name != CHECKPOINTS_DIR_NAME:
        return None
    return resume_path.parent.parent


def _canonical_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _non_negative_int_arg(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _positive_int_arg(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _max_tokens_arg(text: str) -> int:
    value = _positive_int_arg(text)
    if value < MIN_CLI_MAX_TOKENS:
        raise argparse.ArgumentTypeError(
            f"must be >= {MIN_CLI_MAX_TOKENS}"
        )
    return value


def _finite_float_arg(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def _positive_float_arg(text: str) -> float:
    value = _finite_float_arg(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _non_negative_float_arg(text: str) -> float:
    value = _finite_float_arg(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _unit_interval_float_arg(text: str) -> float:
    value = _finite_float_arg(text)
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def _positive_unit_float_arg(text: str) -> float:
    value = _finite_float_arg(text)
    if value <= 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("must be > 0 and <= 1")
    return value


def _adam_beta_arg(text: str) -> float:
    value = _finite_float_arg(text)
    if value < 0.0 or value >= 1.0:
        raise argparse.ArgumentTypeError("must be >= 0 and < 1")
    return value


def _ppo_profile_arg(value: object) -> PPOProfileMode | None:
    if value is None:
        return None
    if value == "off":
        return "off"
    if value == "basic":
        return "basic"
    if value == "detailed":
        return "detailed"
    assert False


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--force-new-run", action="store_true")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=None
    )
    parser.add_argument(
        "--ppo-profile",
        choices=("off", "basic", "detailed"),
        default=None,
    )
    parser.add_argument(
        "--max-rounds", type=_non_negative_int_arg, default=0
    )
    parser.add_argument(
        "--d-model", type=_positive_int_arg, default=128
    )
    parser.add_argument("--layers", type=_positive_int_arg, default=3)
    parser.add_argument("--heads", type=_positive_int_arg, default=4)
    parser.add_argument(
        "--max-tokens", type=_max_tokens_arg, default=768
    )
    parser.add_argument(
        "--seed", type=_non_negative_int_arg, default=None
    )
    parser.add_argument(
        "--learning-rate", type=_positive_float_arg, default=None
    )
    parser.add_argument(
        "--checkpoint-every-updates",
        type=_positive_int_arg,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-retention-updates",
        type=_non_negative_int_arg,
        default=None,
    )
    parser.add_argument(
        "--max-round-seconds", type=_positive_float_arg, default=None
    )
    parser.add_argument(
        "--gae-lambda", type=_unit_interval_float_arg, default=None
    )
    parser.add_argument(
        "--ppo-clip", type=_positive_unit_float_arg, default=None
    )
    parser.add_argument(
        "--value-clip", type=_positive_float_arg, default=None
    )
    parser.add_argument(
        "--entropy-coef", type=_non_negative_float_arg, default=None
    )
    parser.add_argument(
        "--value-coef", type=_non_negative_float_arg, default=None
    )
    parser.add_argument(
        "--max-grad-norm", type=_non_negative_float_arg, default=None
    )
    parser.add_argument(
        "--ppo-epochs", type=_positive_int_arg, default=None
    )
    parser.add_argument(
        "--minibatch-size", type=_positive_int_arg, default=None
    )
    parser.add_argument(
        "--adam-beta1", type=_adam_beta_arg, default=None
    )
    parser.add_argument(
        "--adam-beta2", type=_adam_beta_arg, default=None
    )
    parser.add_argument(
        "--weight-decay", type=_non_negative_float_arg, default=None
    )
    args = parser.parse_args(argv)
    run_dir_arg: object = args.run_dir
    assert run_dir_arg is None or isinstance(run_dir_arg, str)
    resume_arg: object = args.resume
    assert resume_arg is None or isinstance(resume_arg, str)
    cli_run_dir = None if run_dir_arg is None else Path(run_dir_arg)
    resume_path = None if resume_arg is None else Path(resume_arg)
    run_dir = _validated_run_dir(
        parser=parser,
        cli_run_dir=cli_run_dir,
        resume_path=resume_path,
    )
    if args.init_only and resume_path is not None:
        parser.error("--init-only cannot be combined with --resume")
    if args.force_new_run and resume_path is not None:
        parser.error("--force-new-run cannot be combined with --resume")
    if resume_path is None and args.d_model % args.heads != 0:
        parser.error("--d-model must be divisible by --heads")
    seed_validation = _validate_resume_seed_override(
        parser=parser,
        resume_path=resume_path,
        seed=args.seed,
    )
    if isinstance(seed_validation, _result.Rejected):
        parser.error(seed_validation.reason)
    cli_model_config = (
        ModelConfig(
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            max_tokens=args.max_tokens,
        )
        if resume_path is None
        else None
    )
    model_config_result = resolve_model_config(
        cli_model_config=cli_model_config,
        resume_path=resume_path,
    )
    if isinstance(model_config_result, _result.Rejected):
        parser.error(model_config_result.reason)
    model_config = model_config_result.value
    train_config_result = resolve_train_config(
        cli_overrides=TrainConfigOverrides(
            device=args.device,
            ppo_profile=_ppo_profile_arg(args.ppo_profile),
            seed=args.seed,
            learning_rate=args.learning_rate,
            checkpoint_every_updates=args.checkpoint_every_updates,
            checkpoint_retention_updates=(
                args.checkpoint_retention_updates
            ),
            max_round_seconds=args.max_round_seconds,
            gae_lambda=args.gae_lambda,
            ppo_clip=args.ppo_clip,
            value_clip=args.value_clip,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            weight_decay=args.weight_decay,
        ),
        resume_path=resume_path,
    )
    if isinstance(train_config_result, _result.Rejected):
        parser.error(train_config_result.reason)
    train_config = train_config_result.value
    if resume_path is None:
        initialized_result = initialize_training_run(
            run_dir=run_dir,
            run_id=run_dir.name,
            model_config=model_config,
            train_config=train_config,
            force_new_run=args.force_new_run,
        )
        if isinstance(initialized_result, _result.Rejected):
            parser.error(initialized_result.reason)
        initialized = initialized_result.value
        if args.init_only:
            print(f"dashboard: {initialized.dashboard_path}")
            print(f"checkpoint: {initialized.checkpoint_path}")
            return
        dashboard_path = initialized.dashboard_path
        training_resume = initialized.checkpoint_path
    else:
        prepared_result = prepare_training_run(run_dir=run_dir)
        if isinstance(prepared_result, _result.Rejected):
            parser.error(prepared_result.reason)
        prepared = prepared_result.value
        dashboard_path = prepared.dashboard_path
        training_resume = resume_path
    result = run_training_loop(
        run_dir=run_dir,
        run_id=run_dir.name,
        model_config=model_config,
        train_config=train_config,
        max_rounds=args.max_rounds,
        resume=training_resume,
    )
    if isinstance(result, _result.Rejected):
        parser.error(result.reason)
    loop_result = result.value
    print(f"dashboard: {dashboard_path}")
    print(f"checkpoint: {loop_result.checkpoint_path}")
    print(f"rounds: {loop_result.total_rounds}")
    print(f"updates: {loop_result.total_updates}")


if __name__ == "__main__":
    main()
