"""In-process model-compute implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from server.foundation import result as _result
from server.foundation.result import Rejected
from server.training.config import TrainConfig
from server.training.model import ModelConfig
from server.training.policy_inference_batch import (
    DevicePolicyRequestBatch,
)
from server.training.policy_sampling import (
    CompactPolicyDecisionBatch,
    RankReturnTargets,
)
from server.training.policy_sampling.model_rank_sample_arena import (
    ModelRankSampleArena,
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
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.seeding import seed_training_rng
from server.training.runtime.state import (
    RuntimeTrainingState,
    capture_runtime_training_state,
    load_runtime_training_state,
)
from server.training.semantic_action_plan import (
    ActionSampler,
)
from server.training.torch_sampler import sample_policy_batch_into_arena
from server.training.training_state import (
    LoadedTrainingState,
    create_model,
)


@dataclass(slots=True)
class ModelReplica:
    """Own model, optimizer, and device-local PPO operations."""

    state: LoadedTrainingState
    model_config: ModelConfig
    train_config: TrainConfig
    execution_config: ExecutionConfig
    device: torch.device
    sample_arena: ModelRankSampleArena
    sampler: ActionSampler

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
            config=self.model_config,
            device=self.device,
            requests=requests,
            sampler=self.sampler,
            sample_arena=self.sample_arena,
        )

    def update_returns(
        self, *, returns: RankReturnTargets
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected:
        """Resolve return targets on-device and apply PPO update."""
        if returns.is_empty():
            update_input = PPOUpdateInput(
                policy_version=returns.policy_version,
                local_batch=None,
            )
        else:
            source_result = self.sample_arena.ppo_batch_source(
                returns=returns
            )
            if isinstance(source_result, Rejected):
                return source_result
            update_input = PPOUpdateInput(
                policy_version=returns.policy_version,
                local_batch=source_result.value,
            )
        update_result = self.state.trainer.update(update_input)
        if isinstance(update_result, Rejected):
            return update_result
        self.sample_arena.discard_return_batch(returns=returns)
        self.sample_arena.discard_uncommitted_policy_version(
            policy_version=returns.policy_version
        )
        return _result.Ok(value=update_result.value)

    def snapshot(self) -> RuntimeTrainingState:
        """Return a portable CPU state snapshot."""
        return capture_runtime_training_state(
            model=self.state.model,
            trainer=self.state.trainer,
        )


def create_model_replica(
    *,
    model_rank_index: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    device: torch.device,
    update_partition: PPOUpdatePartition | None = None,
) -> ModelReplica:
    """Create a model replica on one concrete torch device."""
    seed_training_rng(train_config.seed)
    model = create_model(model_config, device)
    trainer = PPOTrainer(
        model=model,
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
        trainer=trainer,
        total_rounds=0,
        total_samples=0,
        total_updates=0,
    )
    return ModelReplica(
        state=state,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        device=device,
        sample_arena=ModelRankSampleArena(
            model_rank_index=model_rank_index,
            device=device,
        ),
        sampler=ActionSampler.create(
            batch_capacity=execution_config.model_inference_batch_size,
            device=device,
        ),
    )
