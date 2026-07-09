"""Black-box tests for PPO trace replay tensors."""

from __future__ import annotations

import torch
from torch import Tensor

from server.training.ppo.replay_tensors import PPOReplayTensorBatch


def test_ppo_replay_tensor_batch_records_trace_layout() -> None:
    replay = _two_sample_replay()

    assert replay.sample_count == 2
    assert replay.max_step_count == 2
    assert replay.active_step_count == 3
    assert replay.selected_token_ids_padded.dtype == torch.long
    assert replay.active_sample_indices.dtype == torch.long
    assert replay.active_step_indices.dtype == torch.long
    assert replay.choice_token_ids.dtype == torch.int16
    assert replay.choice_masks.dtype == torch.bool
    assert replay.selected_choice_offsets.dtype == torch.long
    assert replay.step_counts.dtype == torch.long


def test_ppo_replay_tensor_batch_can_rebuild_active_steps() -> None:
    replay = _two_sample_replay()
    source_active_indices = torch.tensor((1, 2, 0), dtype=torch.long)

    shuffled = _select_replay(
        replay,
        sample_indices=torch.tensor((1, 0), dtype=torch.long),
        active_source_indices=source_active_indices,
        active_sample_indices=torch.tensor((0, 0, 1), dtype=torch.long),
        active_step_indices=torch.tensor((0, 1, 0), dtype=torch.long),
    )

    assert torch.equal(
        shuffled.selected_token_ids_padded,
        torch.tensor(((20, 22), (11, 0)), dtype=torch.long),
    )
    assert torch.equal(
        shuffled.step_counts, torch.tensor((2, 1), dtype=torch.long)
    )
    assert torch.equal(
        shuffled.choice_token_ids,
        replay.choice_token_ids.index_select(0, source_active_indices),
    )


def _select_replay(
    replay: PPOReplayTensorBatch,
    *,
    sample_indices: Tensor,
    active_source_indices: Tensor,
    active_sample_indices: Tensor,
    active_step_indices: Tensor,
) -> PPOReplayTensorBatch:
    return PPOReplayTensorBatch(
        sample_count=int(sample_indices.shape[0]),
        max_step_count=replay.max_step_count,
        active_step_count=int(active_source_indices.shape[0]),
        selected_token_ids_padded=(
            replay.selected_token_ids_padded.index_select(
                0, sample_indices
            )
        ),
        active_sample_indices=active_sample_indices,
        active_step_indices=active_step_indices,
        choice_token_ids=replay.choice_token_ids.index_select(
            0, active_source_indices
        ),
        choice_masks=replay.choice_masks.index_select(
            0, active_source_indices
        ),
        selected_choice_offsets=(
            replay.selected_choice_offsets.index_select(
                0, active_source_indices
            )
        ),
        step_counts=replay.step_counts.index_select(0, sample_indices),
    )


def _two_sample_replay() -> PPOReplayTensorBatch:
    return PPOReplayTensorBatch(
        sample_count=2,
        max_step_count=2,
        active_step_count=3,
        selected_token_ids_padded=torch.tensor(
            ((11, 0), (20, 22)), dtype=torch.long
        ),
        active_sample_indices=torch.tensor((0, 1, 1), dtype=torch.long),
        active_step_indices=torch.tensor((0, 0, 1), dtype=torch.long),
        choice_token_ids=torch.stack(
            (_choices(10, 11), _choices(20), _choices(21, 22))
        ),
        choice_masks=torch.stack(
            (_choice_mask(2), _choice_mask(1), _choice_mask(2))
        ),
        selected_choice_offsets=torch.tensor(
            (1, 0, 1), dtype=torch.long
        ),
        step_counts=torch.tensor((1, 2), dtype=torch.long),
    )


def _choices(*token_ids: int) -> Tensor:
    padded = (*token_ids, *(0 for _ in range(2 - len(token_ids))))
    return torch.tensor(padded, dtype=torch.int16)


def _choice_mask(count: int) -> Tensor:
    return torch.tensor(
        tuple(index < count for index in range(2)), dtype=torch.bool
    )
