"""Training coordinator loop."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable
from uuid import uuid4

from server.foundation import result as _result
from server.foundation.json_value import JsonObject
from server.foundation.result import Ok, Rejected
from server.training.config import (
    CheckpointPolicy,
    ModelConfig,
    TrainConfig,
)
from server.training.runtime.checkpoint_state import (
    RuntimeCheckpointState,
    load_runtime_checkpoint_state,
    save_runtime_checkpoint_state,
)
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.result import TrainingLoopResult
from server.training.runtime.shared_rollout_arena import (
    RolloutArenaSnapshot,
)
from server.training.runtime.state import RuntimeTrainingState
from server.training.runtime.training_runtime import (
    TrainingRuntime,
    TrainingUpdateResult,
    open_training_runtime,
)
from server.training.stop import TrainingStopRequest
from server.training_events import (
    EventContext,
    ProcessIdentity,
    StructuredEventSink,
)

_CHECKPOINTS_DIR_NAME = "checkpoints"


def run_training_coordinator(
    *,
    run_dir: Path,
    runtime_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_policy: CheckpointPolicy,
    execution_config: ExecutionConfig,
    max_samples: int,
    resume: Path,
    stop_request: TrainingStopRequest,
    on_ready: Callable[[], None] | None = None,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    """Run synchronized worker training and commit canonical state."""
    return asyncio.run(
        _run_training_coordinator_async(
            run_dir=run_dir,
            runtime_id=runtime_id,
            model_config=model_config,
            train_config=train_config,
            checkpoint_policy=checkpoint_policy,
            execution_config=execution_config,
            max_samples=max_samples,
            resume=resume,
            stop_request=stop_request,
            on_ready=on_ready,
        )
    )


async def _run_training_coordinator_async(
    *,
    run_dir: Path,
    runtime_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_policy: CheckpointPolicy,
    execution_config: ExecutionConfig,
    max_samples: int,
    resume: Path,
    stop_request: TrainingStopRequest,
    on_ready: Callable[[], None] | None,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    """Run synchronized worker training inside the async runtime."""
    assert max_samples >= 0
    setup_result = _setup_coordinator_runtime(
        execution_config=execution_config
    )
    if isinstance(setup_result, Rejected):
        return setup_result
    event_sink = StructuredEventSink(
        run_dir=run_dir,
        process=ProcessIdentity(kind="coordinator", index=0),
    )
    state_result = load_runtime_checkpoint_state(
        path=resume,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    if isinstance(state_result, Rejected):
        event_sink.emit(
            "training",
            error=state_result.reason,
        )
        event_sink.close()
        return state_result
    runtime_result = open_training_runtime(
        run_dir=run_dir,
        run_id=runtime_id,
        event_sink=event_sink,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
    )
    if isinstance(runtime_result, Rejected):
        event_sink.emit(
            "training",
            error=runtime_result.reason,
        )
        event_sink.close()
        return runtime_result
    runtime = runtime_result.value
    try:
        training_result = await _run_synchronized_training(
            run_dir=run_dir,
            runtime_id=runtime_id,
            model_config=model_config,
            train_config=train_config,
            checkpoint_policy=checkpoint_policy,
            execution_config=execution_config,
            state=state_result.value,
            max_samples=max_samples,
            event_sink=event_sink,
            runtime=runtime,
            stop_request=stop_request,
            on_ready=on_ready,
        )
    finally:
        await runtime.close()
    if isinstance(training_result, Rejected):
        event_sink.emit(
            "training",
            fields={
                "total_rounds": state_result.value.total_rounds,
                "total_updates": state_result.value.total_updates,
            },
            error=training_result.reason,
        )
        event_sink.close()
        return training_result
    event_sink.emit(
        "training",
        fields={
            "checkpoint_path": str(
                training_result.value.checkpoint_path
            ),
            "total_rounds": training_result.value.total_rounds,
            "total_samples": training_result.value.total_samples,
            "total_updates": training_result.value.total_updates,
        },
    )
    event_sink.close()
    return training_result


def _setup_coordinator_runtime(
    *,
    execution_config: ExecutionConfig,
) -> _result.Ok[None] | _result.Rejected:
    assert execution_config.worker_process_count() > 0
    return Ok(value=None)


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
    runtime_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_policy: CheckpointPolicy,
    execution_config: ExecutionConfig,
    state: RuntimeCheckpointState,
    max_samples: int,
    event_sink: StructuredEventSink,
    runtime: TrainingRuntime,
    stop_request: TrainingStopRequest,
    on_ready: Callable[[], None] | None,
) -> _result.Ok[TrainingLoopResult] | _result.Rejected:
    start = time.monotonic()
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
    if on_ready is not None:
        on_ready()
    while (
        _should_continue_training(
            max_samples=max_samples,
            start_total_samples=start_total_samples,
            total_samples=total_samples,
        )
        and not stop_request.is_requested()
    ):
        rollout_id = str(uuid4())
        context = EventContext(
            policy_version=total_updates,
            rollout_id=rollout_id,
        )
        update_cycle_start = time.monotonic()
        update_result = await runtime.run_update(
            policy_version=total_updates,
            rollout_id=rollout_id,
        )
        if isinstance(update_result, Rejected):
            duration_seconds = max(
                time.monotonic() - update_cycle_start, 0.0
            )
            event_sink.emit(
                "rollout",
                context=context,
                fields={"duration_seconds": duration_seconds},
                error=update_result.reason,
            )
            event_sink.emit(
                "update",
                context=context,
                fields={"duration_seconds": duration_seconds},
                error=update_result.reason,
            )
            return update_result
        update = update_result.value
        update_cycle_seconds = max(
            time.monotonic() - update_cycle_start, 0.0
        )
        event_sink.emit(
            "rollout",
            context=context,
            fields={
                **_rollout_fields(update.snapshot),
                "duration_seconds": update_cycle_seconds,
            },
        )
        total_updates += 1
        total_rounds += update.snapshot.round_count
        total_samples += update.snapshot.sample_count
        elapsed = max(time.monotonic() - start, 0.000001)
        _record_update_completed(
            event_sink=event_sink,
            context=context,
            start_total_rounds=start_total_rounds,
            start_total_samples=start_total_samples,
            total_rounds=total_rounds,
            total_samples=total_samples,
            total_updates=total_updates,
            elapsed_seconds=elapsed,
            update_cycle_seconds=update_cycle_seconds,
            update=update,
        )
        checkpoint_result = await _maybe_save_checkpoint(
            run_dir=run_dir,
            rollout_id=rollout_id,
            model_config=model_config,
            train_config=train_config,
            checkpoint_policy=checkpoint_policy,
            execution_config=execution_config,
            event_sink=event_sink,
            runtime=runtime,
            max_samples=max_samples,
            start_total_samples=start_total_samples,
            total_rounds=total_rounds,
            total_samples=total_samples,
            total_updates=total_updates,
        )
        if isinstance(checkpoint_result, Rejected):
            return checkpoint_result
    final_checkpoint_result = await _save_final_checkpoint(
        run_dir=run_dir,
        model_config=model_config,
        train_config=train_config,
        checkpoint_policy=checkpoint_policy,
        execution_config=execution_config,
        runtime=runtime,
        event_sink=event_sink,
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
    event_sink: StructuredEventSink,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
) -> _result.Ok[Path] | _result.Rejected:
    context = EventContext(policy_version=total_updates)
    started = time.monotonic()
    snapshot_result = await runtime.snapshot(
        policy_version=total_updates
    )
    if isinstance(snapshot_result, Rejected):
        event_sink.emit(
            "checkpoint",
            context=context,
            fields={
                "kind": "final",
                "duration_seconds": max(
                    time.monotonic() - started, 0.0
                ),
            },
            error=snapshot_result.reason,
        )
        return snapshot_result
    result = _save_checkpoint(
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
    if isinstance(result, Rejected):
        event_sink.emit(
            "checkpoint",
            context=context,
            fields={
                "kind": "final",
                "duration_seconds": max(
                    time.monotonic() - started, 0.0
                ),
            },
            error=result.reason,
        )
        return result
    event_sink.emit(
        "checkpoint",
        context=context,
        fields={
            "kind": "final",
            "checkpoint_path": str(result.value),
            "total_rounds": total_rounds,
            "total_samples": total_samples,
            "total_updates": total_updates,
            "duration_seconds": max(time.monotonic() - started, 0.0),
        },
    )
    return result


async def _maybe_save_checkpoint(
    *,
    run_dir: Path,
    rollout_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_policy: CheckpointPolicy,
    execution_config: ExecutionConfig,
    event_sink: StructuredEventSink,
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
    context = EventContext(
        policy_version=total_updates,
        rollout_id=rollout_id,
    )
    started = time.monotonic()
    snapshot_result = await runtime.snapshot(
        policy_version=total_updates
    )
    if isinstance(snapshot_result, Rejected):
        event_sink.emit(
            "checkpoint",
            context=context,
            fields={
                "kind": "scheduled",
                "duration_seconds": max(
                    time.monotonic() - started, 0.0
                ),
            },
            error=snapshot_result.reason,
        )
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
        event_sink.emit(
            "checkpoint",
            context=context,
            fields={
                "kind": "scheduled",
                "duration_seconds": max(
                    time.monotonic() - started, 0.0
                ),
            },
            error=save_result.reason,
        )
        return save_result
    event_sink.emit(
        "checkpoint",
        context=context,
        fields={
            "kind": "scheduled",
            "checkpoint_path": str(save_result.value),
            "total_rounds": total_rounds,
            "total_samples": total_samples,
            "total_updates": total_updates,
            "duration_seconds": max(time.monotonic() - started, 0.0),
        },
    )
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
    if (
        total_updates == 0
        or checkpoint_policy.retention_updates == 0
        or not _checkpoint_due(
            total_updates=total_updates,
            checkpoint_policy=checkpoint_policy,
        )
    ):
        return None
    return (
        run_dir / _CHECKPOINTS_DIR_NAME / f"update-{total_updates}.json"
    )


def _record_update_completed(
    *,
    event_sink: StructuredEventSink,
    context: EventContext,
    start_total_rounds: int,
    start_total_samples: int,
    total_rounds: int,
    total_samples: int,
    total_updates: int,
    elapsed_seconds: float,
    update_cycle_seconds: float,
    update: TrainingUpdateResult,
) -> None:
    snapshot = update.snapshot
    process_rounds = total_rounds - start_total_rounds
    process_samples = total_samples - start_total_samples
    assert process_rounds > 0
    assert process_samples > 0
    stats = update.update_stats
    profile = stats.profile
    event_sink.emit(
        "update",
        context=context,
        fields={
            "total_rounds": total_rounds,
            "total_samples": total_samples,
            "total_updates": total_updates,
            "round_count": snapshot.round_count,
            "sample_count": snapshot.sample_count,
            "duration_seconds": update_cycle_seconds,
            "process_rounds_per_second": process_rounds
            / elapsed_seconds,
            "process_samples_per_second": process_samples
            / elapsed_seconds,
            "rollout_decisions_per_second": (
                0.0
                if snapshot.elapsed_seconds_max <= 0.0
                else snapshot.sample_count
                / snapshot.elapsed_seconds_max
            ),
            "team0_reward": snapshot.average_team0_reward(),
            "team1_reward": snapshot.average_team1_reward(),
            "generated_action_count": snapshot.generated_action_count,
            "accepted_action_count": snapshot.accepted_action_count,
            "decision_count": snapshot.sample_count,
            "average_action_choices": 0.0
            if snapshot.generated_action_count == 0
            else snapshot.action_choice_count
            / snapshot.generated_action_count,
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy": stats.entropy,
            "approx_kl": stats.approx_kl,
            "clip_fraction": stats.clip_fraction,
            "update_cycle_seconds": update_cycle_seconds,
            "ppo_update_seconds": profile.update_seconds,
            "ppo_minibatch_loss_seconds": (
                profile.minibatch_loss_seconds
            ),
            "ppo_observation_batch_seconds": (
                profile.observation_batch_seconds
            ),
            "ppo_observation_encode_seconds": (
                profile.observation_encode_seconds
            ),
            "ppo_value_head_seconds": profile.value_head_seconds,
            "ppo_argument_select_seconds": (
                profile.argument_select_seconds
            ),
            "ppo_argument_decode_seconds": (
                profile.argument_decode_seconds
            ),
            "ppo_argument_distribution_seconds": (
                profile.argument_distribution_seconds
            ),
            "ppo_backward_seconds": profile.backward_seconds,
            "ppo_optimizer_step_seconds": (
                profile.optimizer_step_seconds
            ),
            "ppo_argument_decode_fraction": (
                profile.argument_decode_fraction
            ),
            "ppo_argument_trace_batch_count": (
                profile.argument_trace_batch_count
            ),
            "ppo_argument_trace_row_count": (
                profile.argument_trace_row_count
            ),
            "ppo_argument_trace_token_count": (
                profile.argument_trace_token_count
            ),
            "ppo_argument_trace_valid_token_count": (
                profile.argument_trace_valid_token_count
            ),
            "ppo_argument_trace_padding_token_count": (
                profile.argument_trace_padding_token_count
            ),
        },
    )


def _rollout_fields(snapshot: RolloutArenaSnapshot) -> JsonObject:
    return {
        "round_count": snapshot.round_count,
        "sample_count": snapshot.sample_count,
        "generated_action_count": snapshot.generated_action_count,
        "accepted_action_count": snapshot.accepted_action_count,
        "action_choice_count": snapshot.action_choice_count,
        "game_over_count": snapshot.game_over_count,
        "dropped_sample_count": snapshot.dropped_sample_count,
        "cancelled_env_count": snapshot.cancelled_env_count,
        "team0_reward": snapshot.average_team0_reward(),
        "team1_reward": snapshot.average_team1_reward(),
        "elapsed_seconds": snapshot.elapsed_seconds_max,
    }
