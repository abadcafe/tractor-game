"""Black-box tests for commands sent to the standalone training CLI."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from server.training_control.commands import (
    TrainingInitRequest,
    TrainingResumeRequest,
    init_command_argv,
    resume_command_argv,
)


def test_init_command_places_shared_run_dir_before_subcommand() -> None:
    command = init_command_argv(
        TrainingInitRequest(run_dir=Path("run with spaces"))
    )

    assert command[1:3] == ("-m", "server.training_cli")
    assert command[3:6] == (
        "--run-dir",
        "run with spaces",
        "init",
    )
    assert all(";" not in argument for argument in command)


def test_resume_command_places_shared_run_dir_before_subcommand() -> (
    None
):
    command = resume_command_argv(
        TrainingResumeRequest(
            run_dir=Path("training_runs"), checkpoint="update-12.json"
        )
    )

    assert command[1:3] == ("-m", "server.training_cli")
    assert command[3:6] == (
        "--run-dir",
        "training_runs",
        "resume",
    )
    assert command[6] == "update-12.json"
    assert "--checkpoint" not in command


def test_resume_request_rejects_unmanaged_checkpoint_path() -> None:
    rejected = False
    try:
        TrainingResumeRequest(checkpoint="../other/latest.json")
    except ValidationError:
        rejected = True

    assert rejected


def test_init_and_resume_expose_lifecycle_specific_parameters() -> None:
    init_fields = set(TrainingInitRequest.model_fields)
    resume_fields = set(TrainingResumeRequest.model_fields)

    assert "d_model" in init_fields
    assert "d_model" not in resume_fields
    assert "checkpoint" in resume_fields
    assert "checkpoint" not in init_fields
    assert "replace_existing" in init_fields
    assert "replace_existing" not in resume_fields
    assert "worker_cpus" in resume_fields
    assert "worker_cpus" not in init_fields
