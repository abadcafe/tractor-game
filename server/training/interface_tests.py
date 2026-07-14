"""Black-box tests for the public training lifecycle interface."""

from __future__ import annotations

from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training import (
    TrainingInitOptions,
    TrainingResumeOptions,
    TrainingService,
    TrainingStopRequest,
)


def test_initialize_run_through_public_interface(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    service = TrainingService()

    initialized = service.initialize(
        TrainingInitOptions(
            run_dir=run_dir,
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=512,
            seed=23,
        )
    )

    assert isinstance(initialized, Ok)
    assert initialized.value.checkpoint_path == (
        run_dir / "checkpoints" / "latest.json"
    )


def test_initialize_requires_explicit_replacement_confirmation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    service = TrainingService()
    options = TrainingInitOptions(
        run_dir=run_dir,
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    first = service.initialize(options)
    assert isinstance(first, Ok)

    repeated = service.initialize(options)

    assert isinstance(repeated, Rejected)
    replaced = service.initialize(
        options.model_copy(update={"replace_existing": "yes"})
    )
    assert isinstance(replaced, Ok)


def test_initialize_then_resume_through_public_interface(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    service = TrainingService()
    initialized = service.initialize(
        TrainingInitOptions(
            run_dir=run_dir,
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=512,
        )
    )
    assert isinstance(initialized, Ok)
    stop_request = TrainingStopRequest()
    stop_request.request_stop()

    resumed = service.resume(
        TrainingResumeOptions(
            run_dir=run_dir,
            checkpoint="latest.json",
        ),
        stop_request,
    )

    assert isinstance(resumed, Ok)
    assert resumed.value.total_rounds == 0
    assert resumed.value.total_samples == 0
    assert resumed.value.total_updates == 0
    assert (
        resumed.value.checkpoint_path
        == initialized.value.checkpoint_path
    )
