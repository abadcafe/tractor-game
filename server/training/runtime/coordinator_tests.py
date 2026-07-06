"""Tests for the training coordinator public boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from server.result import Ok
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import read_metrics
from server.training.run_setup import initialize_training_run
from server.training.runtime.affinity import current_cpu_affinity
from server.training.runtime.checkpoint_state import (
    create_initial_runtime_checkpoint_state,
    save_runtime_checkpoint_state,
)
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.coordinator import run_training_coordinator
from server.training.runtime.state import RuntimeTrainingState
from server.training.runtime.telemetry import telemetry_path
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata,
)


def test_run_training_coordinator_spawns_worker_and_commits_progress(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=2,
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


@pytest.mark.timeout(120.0)
def test_run_training_coordinator_synchronizes_partial_cpu_update_wave(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=2,
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
    initial = create_initial_runtime_checkpoint_state(
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    shifted_state = RuntimeTrainingState(
        model_state={
            key: value + torch.full_like(value, 0.125)
            for key, value in initial.state.model_state.items()
        },
        optimizer_state=initial.state.optimizer_state,
    )
    checkpoint_path = tmp_path / "checkpoints" / "latest.json"
    saved = save_runtime_checkpoint_state(
        manifest_paths=(checkpoint_path,),
        state=shifted_state,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        total_rounds=1,
        total_updates=1,
        retained_update_count=1,
    )
    assert isinstance(saved, Ok)

    result = run_training_coordinator(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        max_rounds=1,
        resume=checkpoint_path,
    )

    assert isinstance(result, Ok)
    assert result.value.total_rounds == 2
    assert result.value.total_updates == 2
