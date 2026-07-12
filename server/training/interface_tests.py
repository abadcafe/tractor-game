"""Black-box tests for the public training lifecycle interface."""

from __future__ import annotations

from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training import TrainingInitOptions, TrainingService


def test_initialize_and_inspect_run_through_public_interface(
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
    inspected = service.inspect(run_dir)
    assert isinstance(inspected, Ok)
    assert inspected.value.total_updates == 0
    catalog = service.checkpoint_catalog(run_dir)
    assert isinstance(catalog, Ok)
    manifests = catalog.value["manifests"]
    assert isinstance(manifests, list)
    assert len(manifests) == 1


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
