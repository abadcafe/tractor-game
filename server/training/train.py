"""Training CLI entry point.

The command prepares a run directory and dashboard.  Long training is
only started when the user explicitly invokes this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from server.training.config import ModelConfig, TrainConfig
from server.training.loop import run_training_loop
from server.training.run_setup import prepare_training_run
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata,
)


def resolve_model_config(
    *,
    cli_model_config: ModelConfig,
    resume_path: Path | None,
) -> ModelConfig:
    """Use checkpoint model shape when resuming a run."""
    if resume_path is None:
        return cli_model_config
    return read_torch_checkpoint_metadata(resume_path).model_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="training_runs/manual")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu"
    )
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--max-round-seconds", type=float, default=30.0)
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
    train_config = TrainConfig(
        device=args.device,
        learning_rate=args.learning_rate,
        max_round_seconds=args.max_round_seconds,
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
