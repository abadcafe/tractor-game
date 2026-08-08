"""Argument parser and command dispatch for standalone training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, TypeAdapter, ValidationError

from server.foundation import result as _result
from server.training_cli.summary import (
    build_training_summary,
    format_training_summary,
)

if TYPE_CHECKING:
    from server.training import (
        TrainingInitOptions,
        TrainingResumeOptions,
        TrainingStopRequest,
    )

type SummaryFormat = Literal["text", "json"]
_ARGUMENT_VALUES = TypeAdapter(
    dict[str, object],
    config=ConfigDict(strict=True),
)


def main(
    argv: Sequence[str] | None = None,
    *,
    stop_request: TrainingStopRequest | None = None,
) -> None:
    parser = _argument_parser()
    namespace = parser.parse_args(argv)
    values = _ARGUMENT_VALUES.validate_python(vars(namespace))
    command = values.pop("command")
    run_dir = values.pop("run_dir")
    assert isinstance(command, str)
    assert isinstance(run_dir, Path)
    try:
        if command == "init":
            from server.training import TrainingInitOptions

            _execute_init(
                parser,
                TrainingInitOptions.model_validate(
                    {"run_dir": run_dir, **values}
                ),
            )
            return
        if command == "resume":
            from server.training import (
                TrainingResumeOptions,
                TrainingStopRequest,
                training_stop_signals,
            )

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
        reason = _validation_reason(error)
        parser.error(reason)
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
    from server.training import TrainingService

    result = TrainingService().initialize(options)
    if isinstance(result, _result.Rejected):
        parser.error(result.reason)
    print(f"checkpoint: {result.value.checkpoint_path}")


def _execute_resume(
    parser: argparse.ArgumentParser,
    options: TrainingResumeOptions,
    stop_request: TrainingStopRequest,
) -> None:
    from server.training import TrainingService

    result = TrainingService().resume(options, stop_request)
    if isinstance(result, _result.Rejected):
        parser.error(result.reason)
    value = result.value
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
    _ = parser.add_argument(
        "--run-dir", type=Path, default=Path("training_runs")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_init_arguments(subparsers.add_parser("init"))
    _add_resume_arguments(subparsers.add_parser("resume"))
    _add_summary_arguments(subparsers.add_parser("summary"))
    return parser


def _add_init_arguments(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--replace-existing", choices=("yes",), default=None
    )
    _ = parser.add_argument("--d-model", type=int, default=128)
    _ = parser.add_argument("--layers", type=int, default=3)
    _ = parser.add_argument("--heads", type=int, default=4)
    _ = parser.add_argument("--seed", type=int, default=0)
    _ = parser.add_argument(
        "--learning-rate", type=float, default=0.0003
    )
    _ = parser.add_argument("--ppo-clip", type=float, default=0.2)
    _ = parser.add_argument("--entropy-coef", type=float, default=0.01)
    _ = parser.add_argument(
        "--policy-max-grad-norm", type=float, default=0.5
    )
    _ = parser.add_argument("--ppo-epochs", type=int, default=4)
    _ = parser.add_argument("--minibatch-size", type=int, default=64)
    _ = parser.add_argument("--adam-beta1", type=float, default=0.9)
    _ = parser.add_argument("--adam-beta2", type=float, default=0.999)
    _ = parser.add_argument("--weight-decay", type=float, default=0.0)


def _add_resume_arguments(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("checkpoint")
    _ = parser.add_argument("--worker-cpus", default=None)
    _ = parser.add_argument("--model-ranks", default=None)
    _ = parser.add_argument(
        "--ppo-profile",
        choices=("off", "basic", "detailed"),
        default=None,
    )
    _ = parser.add_argument("--max-samples", type=int, default=0)
    _ = parser.add_argument("--learning-rate", type=float, default=None)
    _ = parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=5,
        help="save a periodic checkpoint every N updates (default: 5)",
    )
    _ = parser.add_argument(
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
        _ = parser.add_argument(f"--{name}-timeout-seconds", type=float)
    _ = parser.add_argument("--model-inference-batch-size", type=int)
    _ = parser.add_argument("--game-envs-per-worker", type=int)
    _ = parser.add_argument("--samples-per-update", type=int)
    _ = parser.add_argument("--ppo-clip", type=float)
    _ = parser.add_argument("--entropy-coef", type=float)
    _ = parser.add_argument("--policy-max-grad-norm", type=float)
    _ = parser.add_argument("--ppo-epochs", type=int)
    _ = parser.add_argument("--minibatch-size", type=int)
    _ = parser.add_argument("--adam-beta1", type=float)
    _ = parser.add_argument("--adam-beta2", type=float)
    _ = parser.add_argument("--weight-decay", type=float)


def _add_summary_arguments(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--format", choices=("text", "json"), default="text"
    )


def _validation_reason(error: ValidationError) -> str:
    first = error.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first["loc"])
    message = first["msg"]
    return f"{location}: {message}" if location else message
