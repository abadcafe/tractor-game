"""Black-box tests for complete initialized-run validation."""

from __future__ import annotations

from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training.config import TrainConfig
from server.training.model import ModelConfig
from server.training.run_setup import initialize_training_run
from server.training.run_validation import validate_training_run
from server.training_events import (
    ProcessIdentity,
    StructuredEventSink,
)


def test_validate_training_run_accepts_initialized_run(
    tmp_path: Path,
) -> None:
    initialized = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=8, layers=1, heads=1),
        train_config=TrainConfig(),
    )
    assert isinstance(initialized, Ok)

    result = validate_training_run(tmp_path)

    assert isinstance(result, Ok)
    assert result.value.total_updates == 0
    assert result.value.model_config_values["d_model"] == 8


def test_validate_training_run_accepts_observation_tail(
    tmp_path: Path,
) -> None:
    initialized = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=8, layers=1, heads=1),
        train_config=TrainConfig(),
    )
    assert isinstance(initialized, Ok)
    sink = StructuredEventSink(
        run_dir=tmp_path,
        process=ProcessIdentity(kind="coordinator"),
    )
    sink.emit("update", fields={"total_updates": 1})
    sink.close()

    result = validate_training_run(tmp_path)

    assert isinstance(result, Ok)
    assert result.value.total_updates == 0


def test_validate_training_run_rejects_state_hash_mismatch(
    tmp_path: Path,
) -> None:
    initialized = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=8, layers=1, heads=1),
        train_config=TrainConfig(),
    )
    assert isinstance(initialized, Ok)
    state_path = next(
        (tmp_path / "checkpoints" / "objects").glob("*/state.pt")
    )
    state_path.write_bytes(b"corrupt")

    result = validate_training_run(tmp_path)

    assert isinstance(result, Rejected)
    assert "sha256" in result.reason


def test_validate_training_run_rejects_missing_database(
    tmp_path: Path,
) -> None:
    initialized = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=8, layers=1, heads=1),
        train_config=TrainConfig(),
    )
    assert isinstance(initialized, Ok)
    (tmp_path / "training.sqlite3").unlink()

    result = validate_training_run(tmp_path)

    assert isinstance(result, Rejected)
    assert "database" in result.reason
