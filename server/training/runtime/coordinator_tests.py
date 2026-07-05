"""Tests for the training coordinator public boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.result import Ok
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import read_metrics
from server.training.run_setup import initialize_training_run
from server.training.runtime.affinity import current_cpu_affinity
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.coordinator import run_training_coordinator
from server.training.runtime.telemetry import telemetry_path
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata,
)


def test_run_training_coordinator_spawns_worker_and_commits_progress(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    train_config = TrainConfig(
        checkpoint_every_updates=1,
        checkpoint_retention_updates=1,
        ppo_epochs=1,
        minibatch_size=512,
    )
    execution_config = ExecutionConfig()
    initialized = initialize_training_run(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    assert isinstance(initialized, Ok)

    result = run_training_coordinator(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        max_rounds=1,
        resume=initialized.value.checkpoint_path,
    )

    assert isinstance(result, Ok)
    assert result.value.total_rounds == 1
    assert result.value.total_updates == 1
    metadata = read_torch_checkpoint_metadata(
        result.value.checkpoint_path
    )
    assert isinstance(metadata, Ok)
    assert metadata.value.total_rounds == 1
    assert metadata.value.total_updates == 1
    metrics = read_metrics(tmp_path)
    assert [metric.total_games for metric in metrics] == [0, 1]
    assert telemetry_path(tmp_path).exists()


def test_run_training_coordinator_synchronizes_partial_cpu_update_wave(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    train_config = TrainConfig(
        checkpoint_every_updates=1,
        checkpoint_retention_updates=1,
        ppo_epochs=1,
        minibatch_size=512,
    )
    worker_cpus = current_cpu_affinity()[:2]
    if len(worker_cpus) < 2:
        pytest.skip("multi-rank CPU update requires two available CPUs")
    execution_config = ExecutionConfig(worker_cpus=worker_cpus)
    initialized = initialize_training_run(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    assert isinstance(initialized, Ok)

    result = run_training_coordinator(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        max_rounds=1,
        resume=initialized.value.checkpoint_path,
    )

    assert isinstance(result, Ok)
    assert result.value.total_rounds == 1
    assert result.value.total_updates == 1
