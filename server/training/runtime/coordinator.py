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
from server.training.runtime.messages import WorkerRoundSummary
from server.training.runtime.result import TrainingLoopResult
from server.training.runtime.state import (
    RuntimeTrainingState,
    select_canonical_runtime_training_state,
)
from server.training.runtime.telemetry import (
    IntervalTelemetrySink,
    JsonlTelemetrySink,
    ProcessStage,
    TelemetryEvent,
    TelemetrySink,
)
from server.training.runtime.training_runtime import (
    TrainingRuntime,
    TrainingWaveRequest,
    TrainingWaveResult,
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
    max_rounds: int,
    resume: Path | None,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    """Run synchronized worker training and commit canonical state."""
    assert max_rounds >= 0
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
            progress_denominator=max_rounds,
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
    if max_rounds == 0:
        return _commit_checkpoint_only(
            run_dir=run_dir,
            run_id=run_id,
            model_config=model_config,
            train_config=train_config,
            execution_config=execution_config,
            state=state_result.value,
            telemetry_sink=telemetry_sink,
            max_rounds=max_rounds,
        )
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
            max_rounds=max_rounds,
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
        max_rounds=max_rounds,
        total_rounds=training_result.value.total_rounds,
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


def _commit_checkpoint_only(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    state: RuntimeCheckpointState,
    telemetry_sink: TelemetrySink,
    max_rounds: int,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    checkpoint_result = _save_checkpoint(
        run_dir=run_dir,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        state=state.state,
        total_rounds=state.total_rounds,
        total_updates=state.total_updates,
        update_stats=None,
    )
    if isinstance(checkpoint_result, Rejected):
        return checkpoint_result
    complete_result = _record_coordinator_complete(
        telemetry_sink=telemetry_sink,
        run_id=run_id,
        max_rounds=max_rounds,
        total_rounds=state.total_rounds,
        total_updates=state.total_updates,
    )
    if isinstance(complete_result, Rejected):
        return complete_result
    return Ok(
        value=TrainingLoopResult(
            total_rounds=state.total_rounds,
            total_updates=state.total_updates,
            checkpoint_path=checkpoint_result.value,
        )
    )


def _run_synchronized_training(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    state: RuntimeCheckpointState,
    max_rounds: int,
    telemetry_sink: TelemetrySink,
    runtime: TrainingRuntime,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    start = time.monotonic()
    total_rounds = state.total_rounds
    total_updates = state.total_updates
    runtime_state = state.state
    start_total_rounds = total_rounds
    while total_rounds - start_total_rounds < max_rounds:
        processed_rounds = total_rounds - start_total_rounds
        active_worker_count = min(
            execution_config.worker_process_count(),
            max_rounds - processed_rounds,
        )
        stage_result = _record_coordinator_stage(
            telemetry_sink=telemetry_sink,
            run_id=run_id,
            stage="rollout",
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=processed_rounds,
            progress_denominator=max_rounds,
        )
        if isinstance(stage_result, Rejected):
            return stage_result
        wave_result = runtime.run_wave(
            TrainingWaveRequest(
                state=runtime_state,
                policy_version=total_updates,
                first_episode_id=total_rounds,
                active_worker_count=active_worker_count,
            )
        )
        if isinstance(wave_result, Rejected):
            return wave_result
        wave = wave_result.value
        if wave.update_stats is not None:
            update_stage = _record_coordinator_stage(
                telemetry_sink=telemetry_sink,
                run_id=run_id,
                stage="update",
                total_rounds=total_rounds,
                total_updates=total_updates,
                progress_numerator=processed_rounds,
                progress_denominator=max_rounds,
            )
            if isinstance(update_stage, Rejected):
                return update_stage
            canonical_state_result = (
                select_canonical_runtime_training_state(wave.states)
            )
            if isinstance(canonical_state_result, Rejected):
                return canonical_state_result
            runtime_state = canonical_state_result.value
            total_updates += 1
        total_rounds += active_worker_count
        checkpoint_stage = _record_coordinator_stage(
            telemetry_sink=telemetry_sink,
            run_id=run_id,
            stage="checkpoint",
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=total_rounds - start_total_rounds,
            progress_denominator=max_rounds,
        )
        if isinstance(checkpoint_stage, Rejected):
            return checkpoint_stage
        checkpoint_result = _save_checkpoint(
            run_dir=run_dir,
            model_config=model_config,
            train_config=train_config,
            execution_config=execution_config,
            state=runtime_state,
            total_rounds=total_rounds,
            total_updates=total_updates,
            update_stats=wave.update_stats,
        )
        if isinstance(checkpoint_result, Rejected):
            return checkpoint_result
        elapsed = max(time.monotonic() - start, 0.000001)
        metric_result = _append_wave_metric(
            run_dir=run_dir,
            run_id=run_id,
            checkpoint_path=checkpoint_result.value,
            start_total_rounds=start_total_rounds,
            total_rounds=total_rounds,
            total_updates=total_updates,
            elapsed_seconds=elapsed,
            wave=wave,
        )
        if isinstance(metric_result, Rejected):
            return metric_result
    return Ok(
        value=TrainingLoopResult(
            total_rounds=total_rounds,
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
    total_updates: int,
    update_stats: PPOUpdateStats | None,
) -> _result.Ok[Path] | _result.Rejected:
    latest_checkpoint = run_dir / _CHECKPOINTS_DIR_NAME / "latest.json"
    archive_checkpoint = (
        None
        if update_stats is None
        else _archive_checkpoint_path(
            run_dir=run_dir,
            total_updates=total_updates,
            train_config=train_config,
        )
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


def _append_wave_metric(
    *,
    run_dir: Path,
    run_id: str,
    checkpoint_path: Path,
    start_total_rounds: int,
    total_rounds: int,
    total_updates: int,
    elapsed_seconds: float,
    wave: TrainingWaveResult,
) -> _result.Ok[None] | _result.Rejected:
    summary = _aggregate_round_summaries(wave.summaries)
    process_rounds = total_rounds - start_total_rounds
    assert process_rounds > 0
    stats = wave.update_stats
    metric = TrainingMetric(
        run_id=run_id,
        total_games=total_rounds,
        total_updates=total_updates,
        process_games_per_second=process_rounds / elapsed_seconds,
        last_round_decisions_per_second=(
            summary.decision_count / summary.elapsed_seconds
        ),
        last_team0_reward=summary.team0_reward,
        last_team1_reward=summary.team1_reward,
        last_generated_action_count=summary.generated_action_count,
        last_accepted_action_count=summary.accepted_action_count,
        last_decision_count=summary.decision_count,
        last_average_action_choices=0.0
        if summary.generated_action_count == 0
        else summary.action_choice_count
        / summary.generated_action_count,
        policy_loss=None if stats is None else stats.policy_loss,
        value_loss=None if stats is None else stats.value_loss,
        entropy=None if stats is None else stats.entropy,
        approx_kl=None if stats is None else stats.approx_kl,
        clip_fraction=None if stats is None else stats.clip_fraction,
        ppo_update_seconds=None
        if stats is None
        else stats.profile.update_seconds,
        ppo_minibatch_loss_seconds=None
        if stats is None
        else stats.profile.minibatch_loss_seconds,
        ppo_observation_batch_seconds=None
        if stats is None
        else stats.profile.observation_batch_seconds,
        ppo_observation_encode_seconds=None
        if stats is None
        else stats.profile.observation_encode_seconds,
        ppo_value_head_seconds=None
        if stats is None
        else stats.profile.value_head_seconds,
        ppo_argument_select_seconds=None
        if stats is None
        else stats.profile.argument_select_seconds,
        ppo_argument_prefix_tensorize_seconds=None
        if stats is None
        else stats.profile.argument_prefix_tensorize_seconds,
        ppo_argument_decode_seconds=None
        if stats is None
        else stats.profile.argument_decode_seconds,
        ppo_argument_distribution_seconds=None
        if stats is None
        else stats.profile.argument_distribution_seconds,
        ppo_backward_seconds=None
        if stats is None
        else stats.profile.backward_seconds,
        ppo_optimizer_step_seconds=None
        if stats is None
        else stats.profile.optimizer_step_seconds,
        ppo_argument_decode_fraction=None
        if stats is None
        else stats.profile.argument_decode_fraction,
        ppo_argument_prefix_batch_count=None
        if stats is None
        else stats.profile.argument_prefix_batch_count,
        ppo_argument_prefix_row_count=None
        if stats is None
        else stats.profile.argument_prefix_row_count,
        ppo_argument_prefix_token_count=None
        if stats is None
        else stats.profile.argument_prefix_token_count,
        ppo_argument_prefix_valid_token_count=None
        if stats is None
        else stats.profile.argument_prefix_valid_token_count,
        ppo_argument_prefix_padding_token_count=None
        if stats is None
        else stats.profile.argument_prefix_padding_token_count,
        checkpoint_path=str(checkpoint_path),
    )
    validation = validate_training_metric(metric)
    if isinstance(validation, Rejected):
        return validation
    return append_metric(run_dir, metric)


def _aggregate_round_summaries(
    summaries: tuple[WorkerRoundSummary, ...],
) -> WorkerRoundSummary:
    assert summaries
    count = len(summaries)
    return WorkerRoundSummary(
        team0_reward=sum(summary.team0_reward for summary in summaries)
        / count,
        team1_reward=sum(summary.team1_reward for summary in summaries)
        / count,
        generated_action_count=sum(
            summary.generated_action_count for summary in summaries
        ),
        accepted_action_count=sum(
            summary.accepted_action_count for summary in summaries
        ),
        action_choice_count=sum(
            summary.action_choice_count for summary in summaries
        ),
        decision_count=sum(
            summary.decision_count for summary in summaries
        ),
        elapsed_seconds=max(
            summary.elapsed_seconds for summary in summaries
        ),
        game_over=any(summary.game_over for summary in summaries),
    )


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
    max_rounds: int,
    total_rounds: int,
    total_updates: int,
) -> _result.Ok[None] | _result.Rejected:
    return telemetry_sink.append(
        TelemetryEvent(
            run_id=run_id,
            process_label="coordinator",
            stage="complete",
            total_rounds=total_rounds,
            total_updates=total_updates,
            progress_numerator=max_rounds,
            progress_denominator=max_rounds,
            unix_seconds=time.time(),
        )
    )
