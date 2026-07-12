"""Training coordinator loop."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Literal

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.config import (
    CheckpointPolicy,
    ModelConfig,
    TrainConfig,
)
from server.training.metrics import (
    TrainingMetric,
    append_metric,
    validate_training_metric,
)
from server.training.runtime.checkpoint_state import (
    RuntimeCheckpointState,
    load_runtime_checkpoint_state,
    save_runtime_checkpoint_state,
)
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.result import TrainingLoopResult
from server.training.runtime.state import RuntimeTrainingState
from server.training.runtime.training_runtime import (
    TrainingRuntime,
    TrainingUpdateResult,
    open_training_runtime,
)
from server.training.stop import TrainingStopRequest
from server.training.telemetry import (
    IntervalTelemetrySink,
    ProcessStage,
    SqliteTelemetrySink,
    TelemetryEvent,
    TelemetryMeasurement,
    TelemetrySink,
    prune_telemetry,
)

type _CoordinatorStage = Literal["rollout", "update", "checkpoint"]

_CHECKPOINTS_DIR_NAME = "checkpoints"
_TELEMETRY_PRUNE_INTERVAL_SECONDS = 600.0


def run_training_coordinator(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_policy: CheckpointPolicy,
    execution_config: ExecutionConfig,
    max_samples: int,
    resume: Path,
    stop_request: TrainingStopRequest,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    """Run synchronized worker training and commit canonical state."""
    return asyncio.run(
        _run_training_coordinator_async(
            run_dir=run_dir,
            run_id=run_id,
            model_config=model_config,
            train_config=train_config,
            checkpoint_policy=checkpoint_policy,
            execution_config=execution_config,
            max_samples=max_samples,
            resume=resume,
            stop_request=stop_request,
        )
    )


async def _run_training_coordinator_async(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_policy: CheckpointPolicy,
    execution_config: ExecutionConfig,
    max_samples: int,
    resume: Path,
    stop_request: TrainingStopRequest,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    """Run synchronized worker training inside the async runtime."""
    assert max_samples >= 0
    setup_result = _setup_coordinator_runtime(
        execution_config=execution_config
    )
    if isinstance(setup_result, Rejected):
        return setup_result
    telemetry_sink = IntervalTelemetrySink(
        sink=SqliteTelemetrySink(run_dir),
        min_interval_seconds=(
            execution_config.telemetry_interval_seconds
        ),
    )
    prune_result = prune_telemetry(run_dir)
    if isinstance(prune_result, Rejected):
        return prune_result
    start_result = telemetry_sink.append(
        TelemetryEvent(
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
    state_result = load_runtime_checkpoint_state(
        path=resume,
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
        training_result = await _run_synchronized_training(
            run_dir=run_dir,
            run_id=run_id,
            model_config=model_config,
            train_config=train_config,
            checkpoint_policy=checkpoint_policy,
            execution_config=execution_config,
            state=state_result.value,
            max_samples=max_samples,
            telemetry_sink=telemetry_sink,
            runtime=runtime,
            stop_request=stop_request,
        )
    finally:
        await runtime.close()
    if isinstance(training_result, Rejected):
        failed_result = telemetry_sink.append(
            TelemetryEvent(
                process_label="coordinator",
                stage="failed",
                total_rounds=state_result.value.total_rounds,
                total_updates=state_result.value.total_updates,
                progress_numerator=0,
                progress_denominator=_progress_denominator(max_samples),
                unix_seconds=time.time(),
            )
        )
        if isinstance(failed_result, Rejected):
            return Rejected(
                reason=(
                    f"{training_result.reason}; {failed_result.reason}"
                )
            )
        return training_result
    complete_result = _record_coordinator_complete(
        telemetry_sink=telemetry_sink,
        outcome=training_result.value.outcome,
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


async def _run_synchronized_training(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_policy: CheckpointPolicy,
    execution_config: ExecutionConfig,
    state: RuntimeCheckpointState,
    max_samples: int,
    telemetry_sink: TelemetrySink,
    runtime: TrainingRuntime,
    stop_request: TrainingStopRequest,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    start = time.monotonic()
    next_telemetry_prune = start + _TELEMETRY_PRUNE_INTERVAL_SECONDS
    total_rounds = state.total_rounds
    total_samples = state.total_samples
    total_updates = state.total_updates
    start_total_rounds = total_rounds
    start_total_samples = total_samples
    load_result = await runtime.load_state(
        state=state.state,
        policy_version=total_updates,
    )
    if isinstance(load_result, Rejected):
        return load_result
    while (
        _should_continue_training(
            max_samples=max_samples,
            start_total_samples=start_total_samples,
            total_samples=total_samples,
        )
        and not stop_request.is_requested()
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
        update_cycle_start = time.monotonic()
        update_result = await runtime.run_update(
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
            measurements=(
                TelemetryMeasurement(
                    key="coordinator_update_cycle_seconds",
                    value=max(
                        time.monotonic() - update_cycle_start,
                        0.0,
                    ),
                ),
            ),
        )
        if isinstance(update_stage, Rejected):
            return update_stage
        total_updates += 1
        total_rounds += update.snapshot.round_count
        total_samples += update.snapshot.sample_count
        checkpoint_path_result = await _maybe_save_checkpoint(
            run_dir=run_dir,
            run_id=run_id,
            model_config=model_config,
            train_config=train_config,
            checkpoint_policy=checkpoint_policy,
            execution_config=execution_config,
            telemetry_sink=telemetry_sink,
            runtime=runtime,
            max_samples=max_samples,
            start_total_samples=start_total_samples,
            total_rounds=total_rounds,
            total_samples=total_samples,
            total_updates=total_updates,
        )
        if isinstance(checkpoint_path_result, Rejected):
            return checkpoint_path_result
        elapsed = max(time.monotonic() - start, 0.000001)
        metric_result = _append_update_metric(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path_result.value,
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
        if time.monotonic() >= next_telemetry_prune:
            prune_result = prune_telemetry(run_dir)
            if isinstance(prune_result, Rejected):
                return prune_result
            next_telemetry_prune = (
                time.monotonic() + _TELEMETRY_PRUNE_INTERVAL_SECONDS
            )
    final_checkpoint_result = await _save_final_checkpoint(
        run_dir=run_dir,
        model_config=model_config,
        train_config=train_config,
        checkpoint_policy=checkpoint_policy,
        execution_config=execution_config,
        runtime=runtime,
        total_rounds=total_rounds,
        total_samples=total_samples,
        total_updates=total_updates,
    )
    if isinstance(final_checkpoint_result, Rejected):
        return final_checkpoint_result
    return Ok(
        value=TrainingLoopResult(
            total_rounds=total_rounds,
            total_samples=total_samples,
            total_updates=total_updates,
            checkpoint_path=final_checkpoint_result.value,
            outcome="stopped"
            if stop_request.is_requested()
            else "completed",
        )
    )


async def _save_final_checkpoint(
    *,
    run_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_policy: CheckpointPolicy,
    execution_config: ExecutionConfig,
    runtime: TrainingRuntime,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
) -> _result.Ok[Path] | _result.Rejected:
    snapshot_result = await runtime.snapshot(
        policy_version=total_updates
    )
    if isinstance(snapshot_result, Rejected):
        return snapshot_result
    return _save_checkpoint(
        run_dir=run_dir,
        model_config=model_config,
        train_config=train_config,
        checkpoint_policy=checkpoint_policy,
        execution_config=execution_config,
        state=snapshot_result.value,
        total_rounds=total_rounds,
        total_samples=total_samples,
        total_updates=total_updates,
    )


async def _maybe_save_checkpoint(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_policy: CheckpointPolicy,
    execution_config: ExecutionConfig,
    telemetry_sink: TelemetrySink,
    runtime: TrainingRuntime,
    max_samples: int,
    start_total_samples: int,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
) -> _result.Ok[Path | None] | _result.Rejected:
    if not _checkpoint_due(
        total_updates=total_updates,
        checkpoint_policy=checkpoint_policy,
    ):
        return Ok(value=None)
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
    snapshot_result = await runtime.snapshot(
        policy_version=total_updates
    )
    if isinstance(snapshot_result, Rejected):
        return snapshot_result
    save_result = _save_checkpoint(
        run_dir=run_dir,
        model_config=model_config,
        train_config=train_config,
        checkpoint_policy=checkpoint_policy,
        execution_config=execution_config,
        state=snapshot_result.value,
        total_rounds=total_rounds,
        total_samples=total_samples,
        total_updates=total_updates,
    )
    if isinstance(save_result, Rejected):
        return save_result
    return Ok(value=save_result.value)


def _checkpoint_due(
    *, total_updates: int, checkpoint_policy: CheckpointPolicy
) -> bool:
    assert total_updates > 0
    return total_updates % checkpoint_policy.every_updates == 0


def _save_checkpoint(
    *,
    run_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_policy: CheckpointPolicy,
    execution_config: ExecutionConfig,
    state: RuntimeTrainingState,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
) -> _result.Ok[Path] | _result.Rejected:
    latest_checkpoint = run_dir / _CHECKPOINTS_DIR_NAME / "latest.json"
    archive_checkpoint = _archive_checkpoint_path(
        run_dir=run_dir,
        total_updates=total_updates,
        checkpoint_policy=checkpoint_policy,
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
        retained_update_count=checkpoint_policy.retention_updates,
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
    checkpoint_policy: CheckpointPolicy,
) -> Path | None:
    if total_updates == 0 or checkpoint_policy.retention_updates == 0:
        return None
    assert _checkpoint_due(
        total_updates=total_updates,
        checkpoint_policy=checkpoint_policy,
    )
    return (
        run_dir / _CHECKPOINTS_DIR_NAME / f"update-{total_updates}.json"
    )


def _append_update_metric(
    *,
    run_dir: Path,
    checkpoint_path: Path | None,
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
        ppo_argument_trace_batch_count=(
            stats.profile.argument_trace_batch_count
        ),
        ppo_argument_trace_row_count=(
            stats.profile.argument_trace_row_count
        ),
        ppo_argument_trace_token_count=(
            stats.profile.argument_trace_token_count
        ),
        ppo_argument_trace_valid_token_count=(
            stats.profile.argument_trace_valid_token_count
        ),
        ppo_argument_trace_padding_token_count=(
            stats.profile.argument_trace_padding_token_count
        ),
        checkpoint_path=(
            None if checkpoint_path is None else str(checkpoint_path)
        ),
    )
    validation = validate_training_metric(metric)
    if isinstance(validation, Rejected):
        return validation
    appended = append_metric(run_dir, metric)
    if isinstance(appended, Rejected):
        return appended
    return Ok(value=None)


def _record_coordinator_stage(
    *,
    telemetry_sink: TelemetrySink,
    run_id: str,
    stage: _CoordinatorStage,
    total_rounds: int,
    total_updates: int,
    progress_numerator: int,
    progress_denominator: int,
    measurements: tuple[TelemetryMeasurement, ...] = (),
) -> _result.Ok[None] | _result.Rejected:
    process_stage: ProcessStage = stage
    return telemetry_sink.append(
        TelemetryEvent(
            process_label="coordinator",
            stage=process_stage,
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=progress_numerator,
            progress_denominator=progress_denominator,
            unix_seconds=time.time(),
            measurements=measurements,
        )
    )


def _record_coordinator_complete(
    *,
    telemetry_sink: TelemetrySink,
    outcome: Literal["completed", "stopped"],
    max_samples: int,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
) -> _result.Ok[None] | _result.Rejected:
    return telemetry_sink.append(
        TelemetryEvent(
            process_label="coordinator",
            stage=outcome,
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=_complete_progress_numerator(
                max_samples=max_samples
            ),
            progress_denominator=_progress_denominator(max_samples),
            unix_seconds=time.time(),
        )
    )
