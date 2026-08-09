"""Process-level verification of globally scheduled PPO updates."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import override

import torch
import torch.distributed as dist

from server.foundation.result import Ok
from server.policy_model.actions import (
    ACTION_CHOICE_COUNT,
    CARD_CHOICE_COUNT,
)
from server.policy_model.actions.decoding import ActionSampleBatch
from server.policy_model.network import (
    EncodedObservation,
    ModelConfig,
    PolicyModel,
)
from server.policy_model.observation import CATEGORY_COUNT
from server.policy_model.observation.tensor import (
    ObservationTensorBatch,
)
from server.training.config import TrainConfig
from server.training.ppo import ObservationValueModel, PPOTrainer
from server.training.ppo.distributed import PPOUpdatePartition
from server.training.ppo.update_input import PPOUpdateInput
from server.training.rollout_inference.samples import (
    ModelRankSampleArena,
    RankTrajectoryBatch,
)
from server.training.rollout_inference.samples.arena import (
    ArenaPPOBatchSource,
)


class _RecordingPolicyModel(PolicyModel):
    def __init__(self, *, config: ModelConfig) -> None:
        super().__init__(config=config)
        self.encode_batch_sizes: list[int] = []

    @override
    def encode_observations(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation:
        self.encode_batch_sizes.append(
            int(observation.category_ids.shape[0])
        )
        return super().encode_observations(observation)


def test_distributed_ppo_uses_one_global_minibatch_size(
    tmp_path: Path,
) -> None:
    context: SpawnContext = mp.get_context("spawn")
    rendezvous_path = tmp_path / "rendezvous"
    result_paths = (tmp_path / "rank-0", tmp_path / "rank-1")
    processes: tuple[BaseProcess, ...] = tuple(
        context.Process(
            target=_run_ppo_rank,
            kwargs={
                "rank": rank,
                "rendezvous_path": rendezvous_path,
                "result_path": result_paths[rank],
            },
        )
        for rank in range(2)
    )

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60.0)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    assert tuple(process.exitcode for process in processes) == (0, 0)
    assert tuple(path.read_text() for path in result_paths) == (
        "2,1|2",
        "2|2",
    )


def _run_ppo_rank(
    *, rank: int, rendezvous_path: Path, result_path: Path
) -> None:
    assert rank in (0, 1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path.as_posix()}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(0)
        device = torch.device("cpu")
        model_config = ModelConfig(d_model=8, layers=1, heads=1)
        train_config = TrainConfig(
            ppo_epochs=1,
            minibatch_size=4,
        )
        model = _RecordingPolicyModel(config=model_config).to(device)
        trainer = PPOTrainer(
            model=model,
            value_model=ObservationValueModel(config=model_config).to(
                device
            ),
            train_config=train_config,
            device=device,
            profile_mode="off",
            update_partition=PPOUpdatePartition(
                rank=rank, world_size=2
            ),
        )
        batch = _rank_batch(
            rank=rank,
            count=3 if rank == 0 else 2,
            device=device,
        )

        update_result = trainer.update(
            PPOUpdateInput(policy_version=0, local_batch=batch)
        )

        assert isinstance(update_result, Ok)
        step_count = trainer.optimizer_state()["step_count"]
        assert isinstance(step_count, int)
        result_path.write_text(
            ",".join(str(size) for size in model.encode_batch_sizes)
            + f"|{step_count}"
        )
    finally:
        dist.destroy_process_group()


def _rank_batch(
    *, rank: int, count: int, device: torch.device
) -> ArenaPPOBatchSource:
    assert rank >= 0
    assert count > 0
    arena = ModelRankSampleArena(model_rank_index=rank, device=device)
    category_ids = torch.zeros(
        (count, 2, CATEGORY_COUNT), dtype=torch.long, device=device
    )
    category_ids[:, :, 0] = 1
    legal_masks = torch.zeros(
        (count * 2, ACTION_CHOICE_COUNT),
        dtype=torch.bool,
        device=device,
    )
    legal_masks[:, 10] = True
    legal_masks[:, 11] = True
    stored = arena.store_sampled_result(
        policy_versions=tuple(0 for _index in range(count)),
        observation_batch=ObservationTensorBatch(
            category_ids=category_ids,
            scalar_values=torch.zeros(
                (count, 2), dtype=torch.float32, device=device
            ),
            card_rule_values=torch.zeros(
                (count, 2, 2), dtype=torch.float32, device=device
            ),
            encoded_structure_coordinates=torch.zeros(
                (count, 2, 3), dtype=torch.long, device=device
            ),
            candidate_category_ids=torch.zeros(
                (count, CARD_CHOICE_COUNT, 3),
                dtype=torch.long,
                device=device,
            ),
            candidate_counts=torch.zeros(
                (count, CARD_CHOICE_COUNT),
                dtype=torch.float32,
                device=device,
            ),
            candidate_card_rule_values=torch.zeros(
                (count, CARD_CHOICE_COUNT, 2),
                dtype=torch.float32,
                device=device,
            ),
            query_indices=torch.ones(
                (count,), dtype=torch.long, device=device
            ),
        ),
        action_sample=ActionSampleBatch(
            choice_ids_padded=torch.tensor(
                ((10, 11),), dtype=torch.long, device=device
            ).expand(count, -1),
            active_sample_indices=torch.arange(
                count, dtype=torch.long, device=device
            ).repeat_interleave(2),
            active_step_indices=torch.arange(
                2, dtype=torch.long, device=device
            ).repeat(count),
            legal_choice_masks=legal_masks,
            step_counts=torch.full(
                (count,), 2, dtype=torch.long, device=device
            ),
            choice_counts=torch.full(
                (count,), 2, dtype=torch.long, device=device
            ),
            log_probabilities=torch.zeros(
                (count,), dtype=torch.float32, device=device
            ),
            error_code=torch.zeros((), dtype=torch.long, device=device),
        ),
        old_values=torch.zeros(
            (count,), dtype=torch.float32, device=device
        ),
    )
    assert isinstance(stored, Ok)
    source_result = arena.ppo_batch_source(
        trajectories=RankTrajectoryBatch(
            policy_version=0,
            model_rank_index=rank,
            row_indices=torch.arange(
                count, dtype=torch.long, device=device
            ),
            step_counts=torch.full(
                (count,), 2, dtype=torch.long, device=device
            ),
            trajectory_offsets=torch.arange(
                count, dtype=torch.long, device=device
            ),
            trajectory_lengths=torch.ones(
                (count,), dtype=torch.long, device=device
            ),
            terminal_rewards=torch.ones(
                (count,), dtype=torch.float32, device=device
            ),
            round_count=1,
            total_step_count=count * 2,
            max_step_count=2,
        ),
        gae_lambda=1.0,
    )
    assert isinstance(source_result, Ok)
    return source_result.value
