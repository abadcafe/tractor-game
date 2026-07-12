"""Black-box tests for portable training run initialization."""

from __future__ import annotations

from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import read_metrics
from server.training.run_setup import initialize_training_run
from server.training.torch_checkpoints.load import (
    read_torch_checkpoint_metadata,
)


def test_initialize_training_run_writes_zero_update_state(
    tmp_path: Path,
) -> None:
    prepared = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(),
    )

    assert isinstance(prepared, Ok)
    checkpoint_path = tmp_path / "checkpoints" / "latest.json"
    assert prepared.value.checkpoint_path == checkpoint_path
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert isinstance(metadata, Ok)
    assert metadata.value.model_config == ModelConfig(d_model=128)
    assert metadata.value.train_config == TrainConfig()
    assert metadata.value.total_rounds == 0
    assert metadata.value.total_updates == 0
    metrics = read_metrics(tmp_path)
    assert isinstance(metrics, Ok)
    assert len(metrics.value) == 1
    assert metrics.value[0].checkpoint_path == str(checkpoint_path)


def test_initialize_training_run_rejects_existing_run(
    tmp_path: Path,
) -> None:
    first = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(seed=1),
    )
    assert isinstance(first, Ok)

    second = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=64),
        train_config=TrainConfig(seed=2),
    )

    assert isinstance(second, Rejected)
    assert "training run already exists" in second.reason
    metadata = read_torch_checkpoint_metadata(
        first.value.checkpoint_path
    )
    assert isinstance(metadata, Ok)
    assert metadata.value.model_config == ModelConfig(d_model=128)


def test_initialize_training_run_replace_existing_rebuilds_state(
    tmp_path: Path,
) -> None:
    first = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(seed=1),
    )
    assert isinstance(first, Ok)

    replaced = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=64),
        train_config=TrainConfig(seed=2),
        replace_existing=True,
    )

    assert isinstance(replaced, Ok)
    metadata = read_torch_checkpoint_metadata(
        replaced.value.checkpoint_path
    )
    assert isinstance(metadata, Ok)
    assert metadata.value.model_config == ModelConfig(d_model=64)
    assert metadata.value.train_config == TrainConfig(seed=2)
    metrics = read_metrics(tmp_path)
    assert isinstance(metrics, Ok)
    assert len(metrics.value) == 1
