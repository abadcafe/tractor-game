"""In-process model-compute implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from server.foundation import result as _result
from server.foundation.result import Rejected
from server.policy_model.actions.decoding import (
    ActionSampler,
)
from server.policy_model.network import ModelConfig
from server.training.config import TrainConfig
from server.training.device_execution import DeviceExecutionPolicy
from server.training.lifecycle.state import (
    LoadedTrainingState,
    create_model,
    create_value_model,
)
from server.training.ppo import (
    PPOTrainer,
    PPOUpdateInput,
    PPOUpdateStats,
)
from server.training.ppo.distributed import (
    PPOUpdatePartition,
    single_update_partition,
)
from server.training.rollout_inference.batch import (
    DevicePolicyRequestBatch,
)
from server.training.rollout_inference.sampler import (
    sample_policy_batch_into_arena,
)
from server.training.rollout_inference.samples import (
    CompactPolicyDecisionBatch,
    ModelRankSampleArena,
    RankTrajectoryBatch,
)
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.seeding import seed_training_rng
from server.training.runtime.state import (
    RuntimeTrainingState,
    capture_runtime_training_state,
    load_runtime_training_state,
)


@dataclass(slots=True)
class ModelReplica:
    """Own model, optimizer, and device-local PPO operations."""

    state: LoadedTrainingState
    train_config: TrainConfig
    execution_config: ExecutionConfig
    device: torch.device
    sample_arena: ModelRankSampleArena
    sampler: ActionSampler
    execution_policy: DeviceExecutionPolicy

    def load_state(
        self,
        *,
        snapshot: RuntimeTrainingState,
    ) -> None:
        """Load a CPU snapshot into this model replica."""
        load_runtime_training_state(state=self.state, snapshot=snapshot)
        self.sample_arena.clear()

    def decide_batch(
        self, requests: DevicePolicyRequestBatch
    ) -> _result.Ok[CompactPolicyDecisionBatch] | _result.Rejected:
        """Run batched policy inference on this core's device."""
        return sample_policy_batch_into_arena(
            model=self.state.model,
            device=self.device,
            requests=requests,
            sampler=self.sampler,
            sample_arena=self.sample_arena,
            execution_policy=self.execution_policy,
        )

    def update_trajectories(
        self, *, trajectories: RankTrajectoryBatch
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected:
        """Freeze critic targets, compute GAE, and apply PPO."""
        if trajectories.is_empty():
            update_input = PPOUpdateInput(
                policy_version=trajectories.policy_version,
                local_batch=None,
            )
        else:
            _ = self.state.value_model.eval()
            source_result = self.sample_arena.ppo_batch_source(
                trajectories=trajectories,
                value_evaluator=self.state.value_model.forward,
                gae_lambda=self.train_config.gae_lambda,
            )
            if isinstance(source_result, Rejected):
                return source_result
            update_input = PPOUpdateInput(
                policy_version=trajectories.policy_version,
                local_batch=source_result.value,
            )
        update_result = self.state.trainer.update(update_input)
        if isinstance(update_result, Rejected):
            return update_result
        self.sample_arena.discard_return_batch(
            trajectories=trajectories
        )
        self.sample_arena.discard_uncommitted_policy_version(
            policy_version=trajectories.policy_version
        )
        return _result.Ok(value=update_result.value)

    def snapshot(self) -> RuntimeTrainingState:
        """Return a portable CPU state snapshot."""
        return capture_runtime_training_state(
            model=self.state.model,
            value_model=self.state.value_model,
            trainer=self.state.trainer,
        )


def create_model_replica(
    *,
    model_rank_index: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    device: torch.device,
    inference_batch_capacity: int,
    update_partition: PPOUpdatePartition | None = None,
) -> ModelReplica:
    """Create a model replica on one concrete torch device."""
    assert inference_batch_capacity > 0
    seed_training_rng(train_config.seed)
    model = create_model(model_config, device)
    value_model = create_value_model(model_config, device)
    trainer = PPOTrainer(
        model=model,
        value_model=value_model,
        train_config=train_config,
        device=device,
        profile_mode=execution_config.ppo_profile,
        update_partition=(
            single_update_partition()
            if update_partition is None
            else update_partition
        ),
    )
    state = LoadedTrainingState(
        model=model,
        value_model=value_model,
        trainer=trainer,
        total_rounds=0,
        total_trainable_decisions=0,
        total_updates=0,
    )
    return ModelReplica(
        state=state,
        train_config=train_config,
        execution_config=execution_config,
        device=device,
        sample_arena=ModelRankSampleArena(
            model_rank_index=model_rank_index,
            device=device,
        ),
        sampler=ActionSampler.create(
            batch_capacity=inference_batch_capacity,
            device=device,
        ),
        execution_policy=DeviceExecutionPolicy.for_device(device),
    )
