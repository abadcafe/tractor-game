"""Argument parser and command dispatch for standalone training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from server.foundation import result as _result
from server.training import (
    TrainingInitOptions,
    TrainingResumeOptions,
    TrainingService,
    TrainingStopRequest,
    training_stop_signals,
)
from server.training_cli.process_inspection import ProcessInspector
from server.training_cli.summary import (
    build_training_summary,
    format_training_summary,
)

type SummaryFormat = Literal["text", "json"]


def main(
    argv: Sequence[str] | None = None,
    *,
    stop_request: TrainingStopRequest | None = None,
) -> None:
    parser = _argument_parser()
    namespace = parser.parse_args(argv)
    values = vars(namespace)
    command = values.pop("command")
    run_dir = values.pop("run_dir")
    assert isinstance(run_dir, Path)
    try:
        if command == "init":
            _execute_init(
                parser,
                TrainingInitOptions.model_validate(
                    {"run_dir": run_dir, **values}
                ),
            )
            return
        if command == "resume":
            options = TrainingResumeOptions.model_validate(
                {"run_dir": run_dir, **values}
            )
            if stop_request is not None:
                _execute_resume(parser, options, stop_request)
                return
            request = TrainingStopRequest()
            with training_stop_signals(request):
                _execute_resume(parser, options, request)
            return
    except ValidationError as error:
        parser.error(_validation_reason(error))
    assert command == "summary"
    output_format = values.pop("format")
    assert not values
    assert output_format in ("text", "json")
    _execute_summary(
        parser, run_dir=run_dir, output_format=output_format
    )


def _execute_init(
    parser: argparse.ArgumentParser, options: TrainingInitOptions
) -> None:
    process_result = ProcessInspector().inspect(
        options.run_dir.resolve()
    )
    if isinstance(process_result, _result.Rejected):
        parser.error(process_result.reason)
    if process_result.value is not None:
        parser.error(
            "training process is already running: "
            f"PID {process_result.value.pid}"
        )
    result = TrainingService().initialize(options)
    if isinstance(result, _result.Rejected):
        parser.error(result.reason)
    print(f"checkpoint: {result.value.checkpoint_path}")


def _execute_resume(
    parser: argparse.ArgumentParser,
    options: TrainingResumeOptions,
    stop_request: TrainingStopRequest,
) -> None:
    result = TrainingService().resume(options, stop_request)
    if isinstance(result, _result.Rejected):
        parser.error(result.reason)
    value = result.value
    print(f"outcome: {value.outcome}")
    print(f"checkpoint: {value.checkpoint_path}")
    print(f"rounds: {value.total_rounds}")
    print(f"samples: {value.total_samples}")
    print(f"updates: {value.total_updates}")


def _execute_summary(
    parser: argparse.ArgumentParser,
    *,
    run_dir: Path,
    output_format: SummaryFormat,
) -> None:
    result = build_training_summary(run_dir)
    if isinstance(result, _result.Rejected):
        parser.error(result.reason)
    if output_format == "json":
        print(result.value.model_dump_json())
    else:
        print(format_training_summary(result.value))


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m server.training_cli"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("training_runs")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_init_arguments(subparsers.add_parser("init"))
    _add_resume_arguments(subparsers.add_parser("resume"))
    _add_summary_arguments(subparsers.add_parser("summary"))
    return parser


def _add_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--replace-existing", choices=("yes",), default=None
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)


def _add_resume_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("checkpoint")
    parser.add_argument("--worker-cpus", default=None)
    parser.add_argument("--model-ranks", default=None)
    parser.add_argument(
        "--ppo-profile",
        choices=("off", "basic", "detailed"),
        default=None,
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--checkpoint-every-updates", type=int, default=50
    )
    parser.add_argument(
        "--checkpoint-retention-updates", type=int, default=5
    )
    for name in (
        "round",
        "sampling-start",
        "rollout-sample",
        "sampling-stop",
        "state-sync",
        "update",
    ):
        parser.add_argument(f"--{name}-timeout-seconds", type=float)
    parser.add_argument("--model-inference-batch-size", type=int)
    parser.add_argument("--game-envs-per-worker", type=int)
    parser.add_argument("--samples-per-update", type=int)
    parser.add_argument("--ppo-clip", type=float)
    parser.add_argument("--value-clip", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument("--value-coef", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--adam-beta1", type=float)
    parser.add_argument("--adam-beta2", type=float)
    parser.add_argument("--weight-decay", type=float)


def _add_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format", choices=("text", "json"), default="text"
    )


def _validation_reason(error: ValidationError) -> str:
    first = error.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first["loc"])
    message = first["msg"]
    return f"{location}: {message}" if location else message
