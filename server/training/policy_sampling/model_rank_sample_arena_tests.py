"""Tests for model-rank sample arena replay materialization."""

from __future__ import annotations

import torch
from torch import Tensor

from server.result import Ok
from server.training.policy_sampling.model_rank_sample_arena import (
    ModelRankSampleArena,
)
from server.training.policy_sampling.records import RankReturnTargets
from server.training.semantic_action_plan import (
    MAX_LEGAL_CANDIDATE_COUNT,
    SemanticActionSampleBatch,
)
from server.training.tensorize import ObservationTensorBatch


def test_sample_arena_materializes_variable_active_steps() -> None:
    device = torch.device("cpu")
    arena = ModelRankSampleArena(model_rank_index=0, device=device)
    semantic_sample = _semantic_sample(device=device)
    stored = arena.store_sampled_result(
        policy_versions=(7, 7, 7),
        observation_batch=_observation_batch(device=device),
        semantic_sample=semantic_sample,
        old_values=torch.tensor(
            (0.1, 0.2, 0.3), dtype=torch.float32, device=device
        ),
    )
    assert isinstance(stored, Ok)
    assert stored.value.choice_counts == (1, 3, 2)
    assert stored.value.row_indices == (0, 1, 2)
    returns = RankReturnTargets(
        policy_version=7,
        model_rank_index=0,
        row_indices=torch.tensor((0, 1, 2), dtype=torch.long),
        step_counts=torch.tensor((1, 3, 2), dtype=torch.long),
        return_values=torch.tensor(
            (1.0, 2.0, 3.0), dtype=torch.float32
        ),
        round_count=1,
        total_step_count=6,
        max_step_count=3,
    )
    source_result = arena.ppo_batch_source(returns=returns)
    assert isinstance(source_result, Ok)

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
    assert _tensor_tuple(replay.choice_token_ids[:, 0]) == (
        30,
        31,
        10,
    )
    assert _tensor_tuple(replay.selected_token_ids_padded[0]) == (
        301,
        302,
        0,
    )
    assert _tensor_tuple(replay.selected_token_ids_padded[1]) == (
        101,
        0,
        0,
    )


def _observation_batch(
    *, device: torch.device
) -> ObservationTensorBatch:
    return ObservationTensorBatch(
        component_ids=torch.zeros(
            (3, 2, 15), dtype=torch.long, device=device
        ),
        numeric_values=torch.zeros(
            (3, 2, 25), dtype=torch.float32, device=device
        ),
        numeric_masks=torch.zeros(
            (3, 2, 25), dtype=torch.float32, device=device
        ),
    )


def _semantic_sample(
    *, device: torch.device
) -> SemanticActionSampleBatch:
    choice_token_ids = torch.zeros(
        (6, MAX_LEGAL_CANDIDATE_COUNT),
        dtype=torch.int16,
        device=device,
    )
    choice_token_ids[:, 0] = torch.tensor(
        (10, 20, 21, 22, 30, 31),
        dtype=torch.int16,
        device=device,
    )
    choice_masks = torch.zeros(
        choice_token_ids.shape, dtype=torch.bool, device=device
    )
    choice_masks[:, 0] = True
    return SemanticActionSampleBatch(
        selected_token_ids_padded=torch.tensor(
            (
                (101, 0, 0),
                (201, 202, 203),
                (301, 302, 0),
            ),
            dtype=torch.long,
            device=device,
        ),
        active_sample_indices=torch.tensor(
            (0, 1, 1, 1, 2, 2), dtype=torch.long, device=device
        ),
        active_step_indices=torch.tensor(
            (0, 0, 1, 2, 0, 1), dtype=torch.long, device=device
        ),
        choice_token_ids=choice_token_ids,
        choice_masks=choice_masks,
        selected_choice_offsets=torch.zeros(
            (6,), dtype=torch.long, device=device
        ),
        step_counts=torch.tensor(
            (1, 3, 2), dtype=torch.long, device=device
        ),
        choice_counts=torch.tensor(
            (1, 3, 2), dtype=torch.long, device=device
        ),
        log_probabilities=torch.zeros(
            (3,), dtype=torch.float32, device=device
        ),
    )


def _tensor_tuple(values: Tensor) -> tuple[int, ...]:
    return tuple(int(value.item()) for value in values)
