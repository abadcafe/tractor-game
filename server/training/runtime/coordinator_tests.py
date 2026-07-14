"""Tests for the training coordinator public boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import torch
from pydantic import TypeAdapter

from server.foundation.json_value import JsonObject
from server.foundation.result import Ok, Rejected
from server.training.config import (
    CheckpointPolicy,
    ModelConfig,
    TrainConfig,
)
from server.training.run_setup import initialize_training_run
from server.training.runtime.affinity import current_cpu_affinity
from server.training.runtime.checkpoint_state import (
    create_initial_runtime_checkpoint_state,
    save_runtime_checkpoint_state,
)
from server.training.runtime.config import (
    ExecutionConfig,
    ExecutionTimeouts,
)
from server.training.runtime.coordinator import run_training_coordinator
from server.training.runtime.state import RuntimeTrainingState
from server.training.runtime.training_runtime import (
    open_training_runtime,
)
from server.training.stop import TrainingStopRequest
from server.training.torch_checkpoints.load import (
    read_torch_checkpoint_metadata,
)
from server.training_events import NullEventSink
from server.training_events.store import (
    database_path,
    initialize_database,
)

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def test_run_training_coordinator_spawns_worker_and_commits_progress(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=2,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    train_config = TrainConfig(ppo_epochs=1, minibatch_size=512)
    checkpoint_policy = CheckpointPolicy(
        every_updates=50, retention_updates=1
    )
    execution_config = ExecutionConfig(
        samples_per_update=32,
    )
    initialized = initialize_training_run(
        run_dir=tmp_path,
        model_config=model_config,
        train_config=train_config,
    )
    assert isinstance(initialized, Ok)

    result = run_training_coordinator(
        run_dir=tmp_path,
        runtime_id="test-runtime",
        model_config=model_config,
        train_config=train_config,
        checkpoint_policy=checkpoint_policy,
        execution_config=execution_config,
        max_samples=1,
        resume=initialized.value.checkpoint_path,
        stop_request=TrainingStopRequest(),
    )

    assert isinstance(result, Ok)
    assert result.value.total_rounds >= 1
    assert result.value.total_samples > 0
    assert result.value.total_updates == 1
    metadata = read_torch_checkpoint_metadata(
        result.value.checkpoint_path
    )
    assert isinstance(metadata, Ok)
    assert metadata.value.total_rounds == result.value.total_rounds
    assert metadata.value.total_samples == result.value.total_samples
    assert metadata.value.total_updates == 1
    assert not (tmp_path / "checkpoints" / "update-1.json").exists()
    event_types = _event_types(tmp_path)
    assert "update" in event_types
    assert "training" in event_types
    assert "decision" in event_types


@pytest.mark.timeout(120.0)
def test_run_training_coordinator_synchronizes_cpu_arena_update(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=2,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    train_config = TrainConfig(ppo_epochs=1, minibatch_size=512)
    checkpoint_policy = CheckpointPolicy(
        every_updates=1, retention_updates=1
    )
    worker_cpus = current_cpu_affinity()[:2]
    if len(worker_cpus) < 2:
        pytest.skip("multi-rank CPU update requires two available CPUs")
    execution_config = ExecutionConfig(
        worker_cpus=worker_cpus,
        samples_per_update=32,
    )
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
        total_samples=17,
        total_updates=1,
        retained_update_count=1,
    )
    assert isinstance(saved, Ok)
    database = initialize_database(tmp_path)
    assert isinstance(database, Ok)

    result = run_training_coordinator(
        run_dir=tmp_path,
        runtime_id="test-runtime",
        model_config=model_config,
        train_config=train_config,
        checkpoint_policy=checkpoint_policy,
        execution_config=execution_config,
        max_samples=1,
        resume=checkpoint_path,
        stop_request=TrainingStopRequest(),
    )

    assert isinstance(result, Ok)
    assert result.value.total_rounds >= 2
    assert result.value.total_samples > 17
    assert result.value.total_updates == 2


def test_coordinator_honors_pre_requested_stop_and_saves_checkpoint(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=2, layers=1, heads=1, max_tokens=512
    )
    train_config = TrainConfig()
    execution_config = ExecutionConfig(samples_per_update=32)
    initialized = initialize_training_run(
        run_dir=tmp_path,
        model_config=model_config,
        train_config=train_config,
    )
    assert isinstance(initialized, Ok)
    stop_request = TrainingStopRequest()
    stop_request.request_stop()
    ready_calls = 0

    def on_ready() -> None:
        nonlocal ready_calls
        ready_calls += 1

    result = run_training_coordinator(
        run_dir=tmp_path,
        runtime_id="test-runtime",
        model_config=model_config,
        train_config=train_config,
        checkpoint_policy=CheckpointPolicy(),
        execution_config=execution_config,
        max_samples=0,
        resume=initialized.value.checkpoint_path,
        stop_request=stop_request,
        on_ready=on_ready,
    )

    assert isinstance(result, Ok)
    assert result.value.total_updates == 0
    assert result.value.checkpoint_path.exists()
    assert ready_calls == 1


def test_coordinator_does_not_signal_ready_before_checkpoint_load(
    tmp_path: Path,
) -> None:
    database = initialize_database(tmp_path)
    assert isinstance(database, Ok)
    ready_calls = 0

    def on_ready() -> None:
        nonlocal ready_calls
        ready_calls += 1

    result = run_training_coordinator(
        run_dir=tmp_path,
        runtime_id="test-runtime",
        model_config=ModelConfig(
            d_model=2, layers=1, heads=1, max_tokens=512
        ),
        train_config=TrainConfig(),
        checkpoint_policy=CheckpointPolicy(),
        execution_config=ExecutionConfig(),
        max_samples=0,
        resume=tmp_path / "checkpoints" / "missing.json",
        stop_request=TrainingStopRequest(),
        on_ready=on_ready,
    )

    assert isinstance(result, Rejected)
    assert ready_calls == 0


@pytest.mark.asyncio
async def test_runtime_stops_sampling_after_rollout_wait_failure(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=2,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    train_config = TrainConfig(
        ppo_epochs=1,
        minibatch_size=512,
    )
    execution_config = ExecutionConfig(
        samples_per_update=4096,
        timeouts=ExecutionTimeouts(
            round_seconds=120.0,
            sampling_start_seconds=10.0,
            rollout_sample_seconds=0.2,
            sampling_stop_seconds=10.0,
            state_sync_seconds=10.0,
            update_seconds=2.0,
        ),
    )
    runtime_result = open_training_runtime(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        event_sink=NullEventSink(),
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    assert isinstance(runtime_result, Ok)
    runtime = runtime_result.value
    try:
        initial = create_initial_runtime_checkpoint_state(
            model_config=model_config,
            train_config=train_config,
            execution_config=execution_config,
        )
        load_result = await runtime.load_state(
            state=initial.state,
            policy_version=0,
        )
        assert isinstance(load_result, Ok)

        update_result = await runtime.run_update(
            policy_version=0, rollout_id="rollout-0"
        )

        assert isinstance(update_result, Rejected)
        assert "rollout sample target timed out" in update_result.reason
        snapshot_result = await runtime.snapshot(policy_version=0)
        assert isinstance(snapshot_result, Ok)
    finally:
        await runtime.close()


def _event_types(run_dir: Path) -> tuple[str, ...]:
    with sqlite3.connect(database_path(run_dir)) as connection:
        rows = connection.execute(
            "SELECT event_json FROM training_logs ORDER BY sequence"
        ).fetchall()
    event_types: list[str] = []
    for row in rows:
        event_json = row[0]
        assert isinstance(event_json, str)
        event = _JSON_OBJECT_ADAPTER.validate_json(event_json)
        event_type = event["event"]
        assert isinstance(event_type, str)
        event_types.append(event_type)
    return tuple(event_types)
