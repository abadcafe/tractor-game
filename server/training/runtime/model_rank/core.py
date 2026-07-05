"""In-process model-compute implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from server import result as _result
from server.result import Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.policy_request_frame import PolicyRequestBatchFrame
from server.training.policy_sampling import ModelRankPolicyDecision
from server.training.policy_sampling.replay_arena import (
    ModelRankReplayArena,
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
from server.training.runtime.update_wave import SynchronizedUpdateShard
from server.training.torch_sampler import sample_policy_decisions
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
    replay_arena: ModelRankReplayArena

    def load_state(
        self,
        *,
        snapshot: RuntimeTrainingState,
    ) -> None:
        """Load a CPU snapshot into this model replica."""
        load_runtime_training_state(state=self.state, snapshot=snapshot)
        self.replay_arena.clear()

    def decide_batch(
        self, requests: PolicyRequestBatchFrame
    ) -> tuple[
        _result.Ok[ModelRankPolicyDecision] | _result.Rejected, ...
    ]:
        """Run batched policy inference on this core's device."""
        sampled = sample_policy_decisions(
            model=self.state.model,
            config=self.model_config,
            device=self.device,
            requests=requests,
        )
        records = tuple(
            sample_result.value.replay_record
            for sample_result in sampled
            if not isinstance(sample_result, Rejected)
        )
        handles = (
            self.replay_arena.store_batch(records=records)
            if records
            else ()
        )
        handle_index = 0
        decisions: list[
            _result.Ok[ModelRankPolicyDecision] | _result.Rejected
        ] = []
        for sample_result in sampled:
            if isinstance(sample_result, Rejected):
                decisions.append(sample_result)
                continue
            sample = sample_result.value
            handle = handles[handle_index]
            handle_index += 1
            decisions.append(
                _result.Ok(
                    value=ModelRankPolicyDecision(
                        trace_token_ids=sample.trace_token_ids,
                        decision_handle=handle,
                        choice_count=sample.choice_count,
                    )
                )
            )
        return tuple(decisions)

    def update_shard(
        self, *, shard: SynchronizedUpdateShard
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected:
        """Resolve committed handles on-device and apply PPO update."""
        if shard.is_empty():
            update_input = PPOUpdateInput(
                policy_version=shard.policy_version,
                local_rollout=None,
            )
        else:
            rollout_result = self.replay_arena.build_rollout(
                commit=shard.rollout_commit
            )
            if isinstance(rollout_result, Rejected):
                return rollout_result
            update_input = PPOUpdateInput(
                policy_version=shard.policy_version,
                local_rollout=rollout_result.value,
            )
        update_result = self.state.trainer.update(update_input)
        if isinstance(update_result, Rejected):
            return update_result
        self.replay_arena.discard(commit=shard.rollout_commit)
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
        total_updates=0,
    )
    return ModelReplica(
        state=state,
        model_config=model_config,
        train_config=train_config,
        execution_config=execution_config,
        device=device,
        replay_arena=ModelRankReplayArena(
            model_rank_index=model_rank_index,
            device=device,
        ),
    )
