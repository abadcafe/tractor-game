"""Training CLI entry point.

The command prepares a run directory and dashboard.  Long training is
only started when the user explicitly invokes this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from server.training.config import (
    ModelConfig,
    TrainConfig,
    TrainingDevice,
)
from server.training.loop import run_training_loop
from server.training.run_setup import prepare_training_run
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata,
)


@dataclass(frozen=True, slots=True)
class TrainConfigOverrides:
    """Explicit CLI overrides for train config fields."""

    device: TrainingDevice | None = None
    learning_rate: float | None = None
    checkpoint_every_updates: int | None = None
    max_round_seconds: float | None = None
    gamma: float | None = None
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
    cli_model_config: ModelConfig,
    resume_path: Path | None,
) -> ModelConfig:
    """Use checkpoint model shape when resuming a run."""
    if resume_path is None:
        return cli_model_config
    return read_torch_checkpoint_metadata(resume_path).model_config


def resolve_train_config(
    *,
    cli_overrides: TrainConfigOverrides,
    resume_path: Path | None,
) -> TrainConfig:
    """Resolve train config from checkpoint plus CLI overrides."""
    base = (
        TrainConfig()
        if resume_path is None
        else read_torch_checkpoint_metadata(resume_path).train_config
    )
    return TrainConfig(
        device=base.device
        if cli_overrides.device is None
        else cli_overrides.device,
        learning_rate=base.learning_rate
        if cli_overrides.learning_rate is None
        else cli_overrides.learning_rate,
        checkpoint_every_updates=base.checkpoint_every_updates
        if cli_overrides.checkpoint_every_updates is None
        else cli_overrides.checkpoint_every_updates,
        max_round_seconds=base.max_round_seconds
        if cli_overrides.max_round_seconds is None
        else cli_overrides.max_round_seconds,
        gamma=base.gamma
        if cli_overrides.gamma is None
        else cli_overrides.gamma,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="training_runs/manual")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=None
    )
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--checkpoint-every-updates", type=int, default=None
    )
    parser.add_argument("--max-round-seconds", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--ppo-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--adam-beta1", type=float, default=None)
    parser.add_argument("--adam-beta2", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    resume_path = None if args.resume is None else Path(args.resume)
    cli_model_config = ModelConfig(
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        max_tokens=args.max_tokens,
    )
    model_config = resolve_model_config(
        cli_model_config=cli_model_config,
        resume_path=resume_path,
    )
    train_config = resolve_train_config(
        cli_overrides=TrainConfigOverrides(
            device=args.device,
            learning_rate=args.learning_rate,
            checkpoint_every_updates=args.checkpoint_every_updates,
            max_round_seconds=args.max_round_seconds,
            gamma=args.gamma,
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
    prepared = prepare_training_run(
        run_dir=run_dir,
        run_id=run_dir.name,
        model_config=model_config,
        train_config=train_config,
    )
    if args.init_only:
        print(f"dashboard: {prepared.dashboard_path}")
        print(f"checkpoint: {prepared.checkpoint_path}")
        return
    result = run_training_loop(
        run_dir=run_dir,
        run_id=run_dir.name,
        model_config=model_config,
        train_config=train_config,
        max_rounds=args.max_rounds,
        resume=resume_path,
    )
    print(f"dashboard: {prepared.dashboard_path}")
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"rounds: {result.total_rounds}")
    print(f"updates: {result.total_updates}")


if __name__ == "__main__":
    main()
