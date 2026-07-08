"""Training coordinator loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from server import result as _result
from server.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import (
    TrainingMetric,
    append_metric,
    validate_training_metric,
)
from server.training.ppo import PPOUpdateStats
from server.training.runtime.checkpoint_state import (
    RuntimeCheckpointState,
    create_initial_runtime_checkpoint_state,
    load_runtime_checkpoint_state,
    save_runtime_checkpoint_state,
)
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.result import TrainingLoopResult
from server.training.runtime.state import RuntimeTrainingState
from server.training.runtime.telemetry import (
    IntervalTelemetrySink,
    JsonlTelemetrySink,
    ProcessStage,
    TelemetryEvent,
    TelemetrySink,
)
from server.training.runtime.training_runtime import (
    TrainingRuntime,
    TrainingUpdateResult,
    open_training_runtime,
)

type _CoordinatorStage = Literal["rollout", "update", "checkpoint"]

_CHECKPOINTS_DIR_NAME = "checkpoints"


def run_training_coordinator(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    max_samples: int,
    resume: Path | None,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    """Run synchronized worker training and commit canonical state."""
    assert max_samples >= 0
    setup_result = _setup_coordinator_runtime(
        execution_config=execution_config
    )
    if isinstance(setup_result, Rejected):
        return setup_result
    telemetry_sink = IntervalTelemetrySink(
        sink=JsonlTelemetrySink(run_dir),
        min_interval_seconds=(
            execution_config.telemetry_interval_seconds
        ),
    )
    start_result = telemetry_sink.append(
        TelemetryEvent(
            run_id=run_id,
            process_label="coordinator",
            stage="coordinator",
            total_rounds=0,
            total_updates=0,
            progress_numerator=0,
            progress_denominator=_progress_denominator(max_samples),
            unix_seconds=time.time(),
        )
    )
    if isinstance(start_result, Rejected):
        return start_result
    state_result = _load_or_create_state(
        resume=resume,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    if isinstance(state_result, Rejected):
        return state_result
    runtime_result = open_training_runtime(
        run_dir=run_dir,
        run_id=run_id,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    if isinstance(runtime_result, Rejected):
        return runtime_result
    runtime = runtime_result.value
    try:
        training_result = _run_synchronized_training(
            run_dir=run_dir,
            run_id=run_id,
            model_config=model_config,
            train_config=train_config,
            execution_config=execution_config,
            state=state_result.value,
            max_samples=max_samples,
            telemetry_sink=telemetry_sink,
            runtime=runtime,
        )
    finally:
        runtime.close()
    if isinstance(training_result, Rejected):
        return training_result
    complete_result = _record_coordinator_complete(
        telemetry_sink=telemetry_sink,
        run_id=run_id,
        max_samples=max_samples,
        total_rounds=training_result.value.total_rounds,
        total_samples=training_result.value.total_samples,
        total_updates=training_result.value.total_updates,
    )
    if isinstance(complete_result, Rejected):
        return complete_result
    return training_result


def _setup_coordinator_runtime(
    *,
    execution_config: ExecutionConfig,
) -> _result.Ok[None] | _result.Rejected:
    assert execution_config.worker_process_count() > 0
    return Ok(value=None)


def _progress_denominator(max_samples: int) -> int:
    assert max_samples >= 0
    if max_samples == 0:
        return 1
    return max_samples


def _progress_numerator(
    *, max_samples: int, processed_samples: int
) -> int:
    assert max_samples >= 0
    assert processed_samples >= 0
    if max_samples == 0:
        return 0
    return min(processed_samples, max_samples)


def _complete_progress_numerator(*, max_samples: int) -> int:
    assert max_samples >= 0
    if max_samples == 0:
        return 1
    return max_samples


def _should_continue_training(
    *,
    max_samples: int,
    start_total_samples: int,
    total_samples: int,
) -> bool:
    assert max_samples >= 0
    assert start_total_samples >= 0
    assert total_samples >= start_total_samples
    if max_samples == 0:
        return True
    return total_samples - start_total_samples < max_samples


def _load_or_create_state(
    *,
    resume: Path | None,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
) -> _result.Ok[RuntimeCheckpointState] | _result.Rejected:
    if resume is None:
        return Ok(
            value=create_initial_runtime_checkpoint_state(
                model_config=model_config,
                train_config=train_config,
                execution_config=execution_config,
            )
        )
    return load_runtime_checkpoint_state(
        path=resume,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )


def _run_synchronized_training(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    state: RuntimeCheckpointState,
    max_samples: int,
    telemetry_sink: TelemetrySink,
    runtime: TrainingRuntime,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    start = time.monotonic()
    total_rounds = state.total_rounds
    total_samples = state.total_samples
    total_updates = state.total_updates
    runtime_state = state.state
    start_total_rounds = total_rounds
    start_total_samples = total_samples
    load_result = runtime.load_state(
        state=runtime_state,
        policy_version=total_updates,
    )
    if isinstance(load_result, Rejected):
        return load_result
    while _should_continue_training(
        max_samples=max_samples,
        start_total_samples=start_total_samples,
        total_samples=total_samples,
    ):
        processed_samples = total_samples - start_total_samples
        stage_result = _record_coordinator_stage(
            telemetry_sink=telemetry_sink,
            run_id=run_id,
            stage="rollout",
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=_progress_numerator(
                max_samples=max_samples,
                processed_samples=processed_samples,
            ),
            progress_denominator=_progress_denominator(max_samples),
        )
        if isinstance(stage_result, Rejected):
            return stage_result
        update_result = runtime.run_update(
            policy_version=total_updates,
        )
        if isinstance(update_result, Rejected):
            return update_result
        update = update_result.value
        update_stage = _record_coordinator_stage(
            telemetry_sink=telemetry_sink,
            run_id=run_id,
            stage="update",
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=_progress_numerator(
                max_samples=max_samples,
                processed_samples=processed_samples,
            ),
            progress_denominator=_progress_denominator(max_samples),
        )
        if isinstance(update_stage, Rejected):
            return update_stage
        total_updates += 1
        total_rounds += update.snapshot.round_count
        total_samples += update.snapshot.sample_count
        checkpoint_stage = _record_coordinator_stage(
            telemetry_sink=telemetry_sink,
            run_id=run_id,
            stage="checkpoint",
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=_progress_numerator(
                max_samples=max_samples,
                processed_samples=total_samples - start_total_samples,
            ),
            progress_denominator=_progress_denominator(max_samples),
        )
        if isinstance(checkpoint_stage, Rejected):
            return checkpoint_stage
        snapshot_result = runtime.snapshot(policy_version=total_updates)
        if isinstance(snapshot_result, Rejected):
            return snapshot_result
        runtime_state = snapshot_result.value
        checkpoint_result = _save_checkpoint(
            run_dir=run_dir,
            model_config=model_config,
            train_config=train_config,
            execution_config=execution_config,
            state=runtime_state,
            total_rounds=total_rounds,
            total_samples=total_samples,
            total_updates=total_updates,
            update_stats=update.update_stats,
        )
        if isinstance(checkpoint_result, Rejected):
            return checkpoint_result
        elapsed = max(time.monotonic() - start, 0.000001)
        metric_result = _append_update_metric(
            run_dir=run_dir,
            run_id=run_id,
            checkpoint_path=checkpoint_result.value,
            start_total_rounds=start_total_rounds,
            start_total_samples=start_total_samples,
            total_rounds=total_rounds,
            total_samples=total_samples,
            total_updates=total_updates,
            elapsed_seconds=elapsed,
            update=update,
        )
        if isinstance(metric_result, Rejected):
            return metric_result
    return Ok(
        value=TrainingLoopResult(
            total_rounds=total_rounds,
            total_samples=total_samples,
            total_updates=total_updates,
            checkpoint_path=run_dir
            / _CHECKPOINTS_DIR_NAME
            / "latest.json",
        )
    )


def _save_checkpoint(
    *,
    run_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    state: RuntimeTrainingState,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
    update_stats: PPOUpdateStats,
) -> _result.Ok[Path] | _result.Rejected:
    latest_checkpoint = run_dir / _CHECKPOINTS_DIR_NAME / "latest.json"
    archive_checkpoint = _archive_checkpoint_path(
        run_dir=run_dir,
        total_updates=total_updates,
        train_config=train_config,
    )
    checkpoint_path = (
        latest_checkpoint
        if archive_checkpoint is None
        else archive_checkpoint
    )
    save_result = save_runtime_checkpoint_state(
        manifest_paths=(
            (latest_checkpoint,)
            if archive_checkpoint is None
            else (archive_checkpoint, latest_checkpoint)
        ),
        state=state,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        total_rounds=total_rounds,
        total_samples=total_samples,
        total_updates=total_updates,
        retained_update_count=train_config.checkpoint_retention_updates,
    )
    if isinstance(save_result, Rejected):
        return save_result
    prune_failure = save_result.value.post_commit_prune_failure
    if prune_failure is not None:
        return prune_failure
    return Ok(value=checkpoint_path)


def _archive_checkpoint_path(
    *,
    run_dir: Path,
    total_updates: int,
    train_config: TrainConfig,
) -> Path | None:
    if train_config.checkpoint_retention_updates == 0:
        return None
    if (
        total_updates > 0
        and total_updates % train_config.checkpoint_every_updates == 0
    ):
        return (
            run_dir
            / _CHECKPOINTS_DIR_NAME
            / f"update-{total_updates}.json"
        )
    return None


def _append_update_metric(
    *,
    run_dir: Path,
    run_id: str,
    checkpoint_path: Path,
    start_total_rounds: int,
    start_total_samples: int,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
    elapsed_seconds: float,
    update: TrainingUpdateResult,
) -> _result.Ok[None] | _result.Rejected:
    snapshot = update.snapshot
    process_rounds = total_rounds - start_total_rounds
    process_samples = total_samples - start_total_samples
    assert process_rounds > 0
    assert process_samples > 0
    stats = update.update_stats
    metric = TrainingMetric(
        run_id=run_id,
        total_games=total_rounds,
        total_samples=total_samples,
        total_updates=total_updates,
        process_games_per_second=process_rounds / elapsed_seconds,
        process_samples_per_second=process_samples / elapsed_seconds,
        last_round_decisions_per_second=(
            0.0
            if snapshot.elapsed_seconds_max <= 0.0
            else snapshot.sample_count / snapshot.elapsed_seconds_max
        ),
        last_team0_reward=snapshot.average_team0_reward(),
        last_team1_reward=snapshot.average_team1_reward(),
        last_generated_action_count=snapshot.generated_action_count,
        last_accepted_action_count=snapshot.accepted_action_count,
        last_decision_count=snapshot.sample_count,
        last_average_action_choices=0.0
        if snapshot.generated_action_count == 0
        else snapshot.action_choice_count
        / snapshot.generated_action_count,
        policy_loss=stats.policy_loss,
        value_loss=stats.value_loss,
        entropy=stats.entropy,
        approx_kl=stats.approx_kl,
        clip_fraction=stats.clip_fraction,
        ppo_update_seconds=stats.profile.update_seconds,
        ppo_minibatch_loss_seconds=(
            stats.profile.minibatch_loss_seconds
        ),
        ppo_observation_batch_seconds=(
            stats.profile.observation_batch_seconds
        ),
        ppo_observation_encode_seconds=(
            stats.profile.observation_encode_seconds
        ),
        ppo_value_head_seconds=stats.profile.value_head_seconds,
        ppo_argument_select_seconds=(
            stats.profile.argument_select_seconds
        ),
        ppo_argument_prefix_tensorize_seconds=(
            stats.profile.argument_prefix_tensorize_seconds
        ),
        ppo_argument_decode_seconds=(
            stats.profile.argument_decode_seconds
        ),
        ppo_argument_distribution_seconds=(
            stats.profile.argument_distribution_seconds
        ),
        ppo_backward_seconds=stats.profile.backward_seconds,
        ppo_optimizer_step_seconds=(
            stats.profile.optimizer_step_seconds
        ),
        ppo_argument_decode_fraction=(
            stats.profile.argument_decode_fraction
        ),
        ppo_argument_prefix_batch_count=(
            stats.profile.argument_prefix_batch_count
        ),
        ppo_argument_prefix_row_count=(
            stats.profile.argument_prefix_row_count
        ),
        ppo_argument_prefix_token_count=(
            stats.profile.argument_prefix_token_count
        ),
        ppo_argument_prefix_valid_token_count=(
            stats.profile.argument_prefix_valid_token_count
        ),
        ppo_argument_prefix_padding_token_count=(
            stats.profile.argument_prefix_padding_token_count
        ),
        checkpoint_path=str(checkpoint_path),
    )
    validation = validate_training_metric(metric)
    if isinstance(validation, Rejected):
        return validation
    return append_metric(run_dir, metric)


def _record_coordinator_stage(
    *,
    telemetry_sink: TelemetrySink,
    run_id: str,
    stage: _CoordinatorStage,
    total_rounds: int,
    total_updates: int,
    progress_numerator: int,
    progress_denominator: int,
) -> _result.Ok[None] | _result.Rejected:
    process_stage: ProcessStage = stage
    return telemetry_sink.append(
        TelemetryEvent(
            run_id=run_id,
            process_label="coordinator",
            stage=process_stage,
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=progress_numerator,
            progress_denominator=progress_denominator,
            unix_seconds=time.time(),
        )
    )


def _record_coordinator_complete(
    *,
    telemetry_sink: TelemetrySink,
    run_id: str,
    max_samples: int,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
) -> _result.Ok[None] | _result.Rejected:
    return telemetry_sink.append(
        TelemetryEvent(
            run_id=run_id,
            process_label="coordinator",
            stage="complete",
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=_complete_progress_numerator(
                max_samples=max_samples
            ),
            progress_denominator=_progress_denominator(max_samples),
            unix_seconds=time.time(),
        )
    )
