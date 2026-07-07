"""In-process model-compute implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from server import result as _result
from server.result import Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.policy_inference_wire import (
    DevicePolicyRequestBatch,
    PolicyRequestWireBatch,
)
from server.training.policy_sampling import ModelRankPolicyDecision
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
from server.training.returns import ReturnCommit
from server.training.runtime.config import ExecutionConfig
from server.training.runtime.model_rank.staging import (
    stage_policy_request_wires,
)
from server.training.runtime.seeding import seed_training_rng
from server.training.runtime.state import (
    RuntimeTrainingState,
    capture_runtime_training_state,
    load_runtime_training_state,
)
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
    sample_arena: ModelRankSampleArena

    def load_state(
        self,
        *,
        snapshot: RuntimeTrainingState,
    ) -> None:
        """Load a CPU snapshot into this model replica."""
        load_runtime_training_state(state=self.state, snapshot=snapshot)
        self.sample_arena.clear()

    def decide_wires(
        self, requests: PolicyRequestWireBatch
    ) -> tuple[
        _result.Ok[ModelRankPolicyDecision] | _result.Rejected, ...
    ]:
        """Stage request wires and run batched policy inference."""
        staged = stage_policy_request_wires(
            requests=requests,
            max_observation_tokens=self.model_config.max_tokens,
            device=self.device,
        )
        if isinstance(staged, Rejected):
            return tuple(staged for _ in requests.requests)
        return self.decide_batch(staged.value.device_batch)

    def decide_batch(
        self, requests: DevicePolicyRequestBatch
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
            self.sample_arena.store_batch(records=records)
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

    def update_commit(
        self, *, commit: ReturnCommit
    ) -> _result.Ok[PPOUpdateStats] | _result.Rejected:
        """Resolve return targets on-device and apply PPO update."""
        if commit.is_empty():
            update_input = PPOUpdateInput(
                policy_version=commit.policy_version,
                local_batch=None,
            )
        else:
            batch_result = self.sample_arena.materialize_return_commit(
                commit=commit
            )
            if isinstance(batch_result, Rejected):
                return batch_result
            update_input = PPOUpdateInput(
                policy_version=commit.policy_version,
                local_batch=batch_result.value,
            )
        update_result = self.state.trainer.update(update_input)
        if isinstance(update_result, Rejected):
            return update_result
        self.sample_arena.discard_commit(commit=commit)
        self.sample_arena.discard_uncommitted_policy_version(
            policy_version=commit.policy_version
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
    )
