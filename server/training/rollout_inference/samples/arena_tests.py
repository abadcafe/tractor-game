"""Black-box tests for model-rank-owned fixed-choice replay."""

from __future__ import annotations

import torch
from torch import Tensor

from server.foundation.result import Ok
from server.policy_model.actions import (
    ACTION_CHOICE_COUNT,
    CARD_CHOICE_COUNT,
)
from server.policy_model.actions.decoding import ActionSampleBatch
from server.policy_model.observation import (
    CATEGORY_COUNT,
)
from server.policy_model.observation.tensor import (
    ObservationTensorBatch,
)
from server.training.rollout_inference.samples import (
    ModelRankSampleArena,
)
from server.training.rollout_inference.samples.records import (
    RankTrajectoryBatch,
)


def test_sample_arena_materializes_variable_active_steps() -> None:
    device = torch.device("cpu")
    arena = ModelRankSampleArena(model_rank_index=0, device=device)
    action_sample = _action_sample(device=device)
    stored = arena.store_sampled_result(
        policy_versions=(7, 7, 7),
        observation_batch=_observation_batch(device=device),
        action_sample=action_sample,
        old_values=torch.tensor(
            (0.25, -0.5, 1.25),
            dtype=torch.float32,
            device=device,
        ),
    )
    assert isinstance(stored, Ok)
    assert stored.value.choice_counts == (2, 3, 4)
    assert stored.value.scored_choice_step_counts == (1, 1, 1)
    assert stored.value.row_indices == (0, 1, 2)
    assert stored.value.action_choice_batch.compact_row(
        1
    ).to_tuple() == (
        20,
        21,
        22,
    )
    trajectories = RankTrajectoryBatch(
        policy_version=7,
        model_rank_index=0,
        row_indices=torch.tensor((0, 1, 2), dtype=torch.long),
        step_counts=torch.tensor((1, 3, 2), dtype=torch.long),
        trajectory_offsets=torch.tensor((0,), dtype=torch.long),
        trajectory_lengths=torch.tensor((3,), dtype=torch.long),
        terminal_rewards=torch.tensor((3.0,), dtype=torch.float32),
        round_count=1,
        total_step_count=6,
        max_step_count=3,
    )
    source_result = arena.ppo_batch_source(
        trajectories=trajectories,
        gae_lambda=1.0,
    )
    assert isinstance(source_result, Ok)
    assert torch.equal(
        source_result.value.old_values,
        torch.tensor((0.25, -0.5, 1.25), dtype=torch.float32),
    )

    minibatch = source_result.value.select_minibatch(
        indices=torch.tensor((2, 0), dtype=torch.long),
        advantages=source_result.value.raw_advantages,
        global_count=torch.tensor(2, dtype=torch.long),
    )

    replay = minibatch.replay
    assert replay is not None
    assert replay.active_step_count == 3
    assert _tensor_tuple(replay.active_sample_indices) == (0, 0, 1)
    assert _tensor_tuple(replay.active_step_indices) == (0, 1, 0)
    assert _tensor_tuple(replay.choice_ids_padded[0]) == (30, 31, 0)
    assert _tensor_tuple(replay.choice_ids_padded[1]) == (10, 0, 0)
    selected_legal = replay.legal_choice_masks.nonzero()
    assert _tensor_tuple(selected_legal[:, 1]) == (
        30,
        32,
        31,
        10,
        11,
    )


def _observation_batch(
    *, device: torch.device
) -> ObservationTensorBatch:
    return ObservationTensorBatch(
        category_ids=torch.zeros(
            (3, 2, CATEGORY_COUNT), dtype=torch.long, device=device
        ),
        scalar_values=torch.zeros(
            (3, 2), dtype=torch.float32, device=device
        ),
        card_rule_values=torch.zeros(
            (3, 2, 2), dtype=torch.float32, device=device
        ),
        encoded_structure_coordinates=torch.zeros(
            (3, 2, 3), dtype=torch.long, device=device
        ),
        candidate_category_ids=torch.zeros(
            (3, CARD_CHOICE_COUNT, 3),
            dtype=torch.long,
            device=device,
        ),
        candidate_counts=torch.zeros(
            (3, CARD_CHOICE_COUNT),
            dtype=torch.float32,
            device=device,
        ),
        candidate_card_rule_values=torch.zeros(
            (3, CARD_CHOICE_COUNT, 2),
            dtype=torch.float32,
            device=device,
        ),
        query_indices=torch.zeros(
            (3,), dtype=torch.long, device=device
        ),
    )


def _action_sample(*, device: torch.device) -> ActionSampleBatch:
    legal_masks = torch.zeros(
        (6, ACTION_CHOICE_COUNT), dtype=torch.bool, device=device
    )
    legal_masks[0, 10] = True
    legal_masks[0, 11] = True
    legal_masks[1, 20] = True
    legal_masks[1, 23] = True
    legal_masks[2, 21] = True
    legal_masks[3, 22] = True
    legal_masks[4, 30] = True
    legal_masks[4, 32] = True
    legal_masks[5, 31] = True
    return ActionSampleBatch(
        choice_ids_padded=torch.tensor(
            ((10, 0, 0), (20, 21, 22), (30, 31, 0)),
            dtype=torch.long,
            device=device,
        ),
        active_sample_indices=torch.tensor(
            (0, 1, 1, 1, 2, 2), dtype=torch.long, device=device
        ),
        active_step_indices=torch.tensor(
            (0, 0, 1, 2, 0, 1), dtype=torch.long, device=device
        ),
        legal_choice_masks=legal_masks,
        step_counts=torch.tensor(
            (1, 3, 2), dtype=torch.long, device=device
        ),
        choice_counts=torch.tensor(
            (2, 3, 4), dtype=torch.long, device=device
        ),
        log_probabilities=torch.zeros(
            (3,), dtype=torch.float32, device=device
        ),
        error_code=torch.zeros((), dtype=torch.long, device=device),
    )


def _tensor_tuple(values: Tensor) -> tuple[int, ...]:
    return tuple(int(value.item()) for value in values)
