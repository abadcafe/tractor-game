"""Tests for the training coordinator public boundary."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from pydantic import TypeAdapter

from server.foundation.json_value import JsonObject
from server.foundation.result import Ok, Rejected
from server.training.config import (
    CheckpointPolicy,
    TrainConfig,
)
from server.training.model import ModelConfig
from server.training.ppo import PPOUpdateStats
from server.training.ppo.profile import blank_update_profile
from server.training.run_setup import initialize_training_run
from server.training.runtime import coordinator as coordinator_module
from server.training.runtime.checkpoint_state import (
    create_initial_runtime_checkpoint_state,
    save_runtime_checkpoint_state,
)
from server.training.runtime.config import (
    ExecutionConfig,
    ExecutionTimeouts,
)
from server.training.runtime.coordinator import run_training_coordinator
from server.training.runtime.shared_rollout_arena import (
    RolloutArenaSnapshot,
)
from server.training.runtime.state import RuntimeTrainingState
from server.training.runtime.training_runtime import (
    TrainingCycleOutcome,
    TrainingRuntime,
    TrainingStopDiscardedPartialRollout,
    TrainingUpdateResult,
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


@dataclass(slots=True)
class _StoppingRuntime:
    outcome: TrainingCycleOutcome
    loaded_state: RuntimeTrainingState | None = None
    update_calls: int = 0
    snapshot_calls: int = 0
    close_calls: int = 0

    async def load_state(
        self, *, state: RuntimeTrainingState, policy_version: int
    ) -> Ok[None] | Rejected:
        assert policy_version == 0
        self.loaded_state = state
        return Ok(value=None)

    async def run_update(
        self,
        *,
        policy_version: int,
        rollout_id: str,
        stop_request: TrainingStopRequest,
    ) -> Ok[TrainingCycleOutcome] | Rejected:
        assert policy_version == 0
        assert rollout_id
        assert not stop_request.is_requested()
        self.update_calls += 1
        stop_request.request_stop()
        return Ok(value=self.outcome)

    async def snapshot(
        self, *, policy_version: int
    ) -> Ok[RuntimeTrainingState] | Rejected:
        assert policy_version in (0, 1)
        state = self.loaded_state
        assert state is not None
        self.snapshot_calls += 1
        return Ok(value=state)

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.timeout(180.0)
def test_run_training_coordinator_spawns_worker_and_commits_progress(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
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


@pytest.mark.timeout(180.0)
def test_run_training_coordinator_synchronizes_cpu_arena_update(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
    )
    train_config = TrainConfig(ppo_epochs=1, minibatch_size=512)
    checkpoint_policy = CheckpointPolicy(
        every_updates=1, retention_updates=1
    )
    execution_config = ExecutionConfig(
        worker_cpu_layout=(None, None),
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
    model_config = ModelConfig(d_model=8, layers=1, heads=1)
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

    result = run_training_coordinator(
        run_dir=tmp_path,
        runtime_id="test-runtime",
        model_config=model_config,
        train_config=train_config,
        checkpoint_policy=CheckpointPolicy(every_updates=3),
        execution_config=execution_config,
        max_samples=0,
        resume=initialized.value.checkpoint_path,
        stop_request=stop_request,
    )

    assert isinstance(result, Ok)
    assert result.value.total_updates == 0
    assert result.value.checkpoint_path.exists()


def test_stop_commits_partial_update_and_saves_one_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _StoppingRuntime(
        outcome=TrainingUpdateResult(
            snapshot=_rollout_snapshot(
                policy_version=0,
                round_count=1,
                sample_count=64,
            ),
            update_stats=_update_stats(),
        )
    )
    _install_runtime(monkeypatch=monkeypatch, runtime=runtime)
    initialized = initialize_training_run(
        run_dir=tmp_path,
        model_config=_model_config(),
        train_config=TrainConfig(),
    )
    assert isinstance(initialized, Ok)

    result = run_training_coordinator(
        run_dir=tmp_path,
        runtime_id="test-runtime",
        model_config=_model_config(),
        train_config=TrainConfig(),
        checkpoint_policy=CheckpointPolicy(every_updates=1),
        execution_config=ExecutionConfig(samples_per_update=1024),
        max_samples=0,
        resume=initialized.value.checkpoint_path,
        stop_request=TrainingStopRequest(),
    )

    assert isinstance(result, Ok)
    assert result.value.total_rounds == 1
    assert result.value.total_samples == 64
    assert result.value.total_updates == 1
    assert runtime.update_calls == 1
    assert runtime.snapshot_calls == 1
    assert runtime.close_calls == 1
    assert _checkpoint_kinds(tmp_path) == ("final",)


def test_stop_discards_small_rollout_without_committing_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _StoppingRuntime(
        outcome=TrainingStopDiscardedPartialRollout(
            snapshot=_rollout_snapshot(
                policy_version=0,
                round_count=1,
                sample_count=63,
            ),
            minimum_sample_count=64,
        )
    )
    _install_runtime(monkeypatch=monkeypatch, runtime=runtime)
    initialized = initialize_training_run(
        run_dir=tmp_path,
        model_config=_model_config(),
        train_config=TrainConfig(),
    )
    assert isinstance(initialized, Ok)

    result = run_training_coordinator(
        run_dir=tmp_path,
        runtime_id="test-runtime",
        model_config=_model_config(),
        train_config=TrainConfig(),
        checkpoint_policy=CheckpointPolicy(every_updates=1),
        execution_config=ExecutionConfig(samples_per_update=1024),
        max_samples=0,
        resume=initialized.value.checkpoint_path,
        stop_request=TrainingStopRequest(),
    )

    assert isinstance(result, Ok)
    assert result.value.total_rounds == 0
    assert result.value.total_samples == 0
    assert result.value.total_updates == 0
    assert runtime.update_calls == 1
    assert runtime.snapshot_calls == 1
    assert runtime.close_calls == 1
    assert _checkpoint_kinds(tmp_path) == ("final",)
    rollout_fields = _last_event_fields(tmp_path, event_type="rollout")
    assert rollout_fields["termination"] == "stop_requested"
    assert rollout_fields["discarded_sample_count"] == 63
    assert rollout_fields["minimum_update_sample_count"] == 64


@pytest.mark.asyncio
async def test_runtime_stops_sampling_after_rollout_wait_failure(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
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
            policy_version=0,
            rollout_id="rollout-0",
            stop_request=TrainingStopRequest(),
        )

        assert isinstance(update_result, Rejected)
        assert "rollout sample target timed out" in update_result.reason
        snapshot_result = await runtime.snapshot(policy_version=0)
        assert isinstance(snapshot_result, Ok)
    finally:
        await runtime.close()


def _install_runtime(
    *, monkeypatch: pytest.MonkeyPatch, runtime: _StoppingRuntime
) -> None:
    def open_runtime(
        **_kwargs: object,
    ) -> Ok[TrainingRuntime] | Rejected:
        boundary: TrainingRuntime = runtime
        return Ok(value=boundary)

    monkeypatch.setattr(
        coordinator_module, "open_training_runtime", open_runtime
    )


def _model_config() -> ModelConfig:
    return ModelConfig(d_model=8, layers=1, heads=1)


def _rollout_snapshot(
    *, policy_version: int, round_count: int, sample_count: int
) -> RolloutArenaSnapshot:
    return RolloutArenaSnapshot(
        policy_version=policy_version,
        capacity=1024,
        round_count=round_count,
        sample_count=sample_count,
        generated_action_count=sample_count,
        accepted_action_count=sample_count,
        action_choice_count=sample_count,
        game_over_count=0,
        dropped_sample_count=0,
        cancelled_env_count=1,
        total_step_count=sample_count,
        max_step_count=1,
        team0_reward_sum=0.0,
        team1_reward_sum=0.0,
        elapsed_seconds_max=1.0,
    )


def _update_stats() -> PPOUpdateStats:
    return PPOUpdateStats(
        policy_loss=1.0,
        value_loss=2.0,
        entropy=3.0,
        total_loss=4.0,
        approx_kl=5.0,
        clip_fraction=0.5,
        profile=blank_update_profile(update_seconds=0.1),
    )


def _events(run_dir: Path) -> tuple[JsonObject, ...]:
    with sqlite3.connect(database_path(run_dir)) as connection:
        rows = connection.execute(
            "SELECT event_json FROM training_logs ORDER BY sequence"
        ).fetchall()
    events: list[JsonObject] = []
    for row in rows:
        event_json = row[0]
        assert isinstance(event_json, str)
        events.append(_JSON_OBJECT_ADAPTER.validate_json(event_json))
    return tuple(events)


def _event_types(run_dir: Path) -> tuple[str, ...]:
    event_types: list[str] = []
    for event in _events(run_dir):
        event_type = event["event"]
        assert isinstance(event_type, str)
        event_types.append(event_type)
    return tuple(event_types)


def _checkpoint_kinds(run_dir: Path) -> tuple[str, ...]:
    kinds: list[str] = []
    for event in _events(run_dir):
        if event["event"] != "checkpoint":
            continue
        fields = event["fields"]
        assert isinstance(fields, dict)
        kind = fields["kind"]
        assert isinstance(kind, str)
        kinds.append(kind)
    return tuple(kinds)


def _last_event_fields(run_dir: Path, *, event_type: str) -> JsonObject:
    matches = tuple(
        event
        for event in _events(run_dir)
        if event["event"] == event_type
    )
    assert matches
    fields = matches[-1]["fields"]
    assert isinstance(fields, dict)
    return fields
