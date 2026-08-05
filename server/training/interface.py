"""Small public lifecycle interface for the deep training package."""

from __future__ import annotations

from uuid import uuid4

from server.foundation import result as _result
from server.training.contracts import (
    InitializedRun as InitializedRun,
)
from server.training.contracts import (
    TrainingInitOptions as TrainingInitOptions,
)
from server.training.contracts import (
    TrainingResumeOptions as TrainingResumeOptions,
)
from server.training.contracts import (
    TrainingRunResult as TrainingRunResult,
)
from server.training.stop import TrainingStopRequest


class TrainingService:
    """Public interface hiding all training implementation modules."""

    def initialize(
        self, options: TrainingInitOptions
    ) -> _result.Ok[InitializedRun] | _result.Rejected:
        return initialize_run(options)

    def resume(
        self,
        options: TrainingResumeOptions,
        stop_request: TrainingStopRequest,
    ) -> _result.Ok[TrainingRunResult] | _result.Rejected:
        return resume_run(options, stop_request)


def initialize_run(
    options: TrainingInitOptions,
) -> _result.Ok[InitializedRun] | _result.Rejected:
    """Create a portable zero-update checkpoint and event store."""
    from server.policy_model.network import ModelConfig
    from server.training.config import TrainConfig
    from server.training.lifecycle.run_setup import (
        initialize_training_run,
    )

    result = initialize_training_run(
        run_dir=options.run_dir,
        model_config=ModelConfig(
            d_model=options.d_model,
            layers=options.layers,
            heads=options.heads,
            action_value_layers=options.action_value_layers,
        ),
        train_config=TrainConfig(
            seed=options.seed,
            learning_rate=options.learning_rate,
            ppo_clip=options.ppo_clip,
            entropy_coef=options.entropy_coef,
            policy_max_grad_norm=options.policy_max_grad_norm,
            action_value_max_grad_norm=(
                options.action_value_max_grad_norm
            ),
            ppo_epochs=options.ppo_epochs,
            minibatch_size=options.minibatch_size,
            adam_beta1=options.adam_beta1,
            adam_beta2=options.adam_beta2,
            weight_decay=options.weight_decay,
        ),
        replace_existing=options.replace_existing == "yes",
    )
    if isinstance(result, _result.Rejected):
        return result
    return _result.Ok(
        value=InitializedRun(
            run_dir=result.value.run_dir,
            checkpoint_path=result.value.checkpoint_path,
        )
    )


def resume_run(
    options: TrainingResumeOptions,
    stop_request: TrainingStopRequest,
) -> _result.Ok[TrainingRunResult] | _result.Rejected:
    """Validate, load, and execute resumed training."""
    from server.training.lifecycle.resume_config import (
        resolve_resume_options,
    )
    from server.training.lifecycle.resume_setup import (
        canonicalize_resume_timeline,
    )
    from server.training.lifecycle.state import (
        validate_model_rank_runtime,
    )
    from server.training.runtime.affinity import preflight_cpu_affinity
    from server.training.runtime.checkpoint_state import (
        load_runtime_checkpoint_state,
    )
    from server.training.runtime.coordinator import (
        run_training_coordinator,
    )

    resolved_result = resolve_resume_options(options)
    if isinstance(resolved_result, _result.Rejected):
        return resolved_result
    resolved = resolved_result.value
    model_rank_result = validate_model_rank_runtime(
        resolved.execution_config
    )
    if isinstance(model_rank_result, _result.Rejected):
        return model_rank_result
    for worker_index in range(
        resolved.execution_config.worker_process_count()
    ):
        affinity_result = preflight_cpu_affinity(
            label=f"worker-{worker_index}",
            cpus=resolved.execution_config.worker_cpu_set(worker_index),
        )
        if isinstance(affinity_result, _result.Rejected):
            return affinity_result
    load_result = load_runtime_checkpoint_state(
        path=resolved.checkpoint_path,
        model_config=resolved.model_config,
        train_config=resolved.train_config,
        execution_config=resolved.execution_config,
    )
    if isinstance(load_result, _result.Rejected):
        return load_result
    timeline_result = canonicalize_resume_timeline(
        run_dir=resolved.run_dir,
        selected_checkpoint=resolved.checkpoint_path,
    )
    if isinstance(timeline_result, _result.Rejected):
        return timeline_result
    result = run_training_coordinator(
        run_dir=resolved.run_dir,
        runtime_id=str(uuid4()),
        model_config=resolved.model_config,
        train_config=resolved.train_config,
        checkpoint_policy=resolved.checkpoint_policy,
        execution_config=resolved.execution_config,
        max_samples=resolved.max_samples,
        resume=resolved.run_dir / "checkpoints" / "latest.json",
        stop_request=stop_request,
    )
    if isinstance(result, _result.Rejected):
        return result
    value = result.value
    return _result.Ok(
        value=TrainingRunResult(
            checkpoint_path=value.checkpoint_path,
            total_rounds=value.total_rounds,
            total_samples=value.total_samples,
            total_updates=value.total_updates,
        )
    )
