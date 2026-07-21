"""Black-box tests for fixed-choice PPO replay tensors."""

from __future__ import annotations

import torch

from server.training.ppo.replay_tensors import PPOReplayTensorBatch
from server.training.semantic_actions.choices import ACTION_CHOICE_COUNT


def test_ppo_replay_records_exact_choice_ids_and_full_legal_masks() -> (
    None
):
    legal_masks = torch.zeros(
        (3, ACTION_CHOICE_COUNT), dtype=torch.bool
    )
    legal_masks[0, (10, 11)] = True
    legal_masks[1, 20] = True
    legal_masks[2, (21, 22)] = True

    replay = PPOReplayTensorBatch(
        sample_count=2,
        max_step_count=2,
        active_step_count=3,
        choice_ids_padded=torch.tensor(
            ((11, 0), (20, 22)), dtype=torch.long
        ),
        active_sample_indices=torch.tensor((0, 1, 1), dtype=torch.long),
        active_step_indices=torch.tensor((0, 0, 1), dtype=torch.long),
        legal_choice_masks=legal_masks,
        step_counts=torch.tensor((1, 2), dtype=torch.long),
    )

    assert replay.choice_ids_padded.dtype == torch.long
    assert replay.legal_choice_masks.dtype == torch.bool
    assert replay.legal_choice_masks.shape == (3, ACTION_CHOICE_COUNT)
    assert bool(replay.legal_choice_masks[0, 11])
    assert not bool(replay.legal_choice_masks[1, 11])
