"""Black-box tests for PPO replay tensor batching."""

from __future__ import annotations

import torch
from torch import Tensor

from server.training.ppo.replay_tensors import (
    PPOReplayTensorBatch,
    replay_prefix_tensor_batch,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC


def test_replay_prefix_tensor_batch_flattens_all_prefixes() -> None:
    replay = _two_sample_replay()

    result = replay_prefix_tensor_batch(
        replay=replay,
        sample_indices=torch.tensor((1, 0), dtype=torch.long),
    )

    assert result is not None
    assert torch.equal(result.active_positions, torch.tensor((0, 0, 1)))
    assert torch.equal(result.prefix_lengths, torch.tensor((0, 1, 0)))
    assert torch.equal(
        result.prefix_batch.argument_ids,
        torch.tensor(
            (
                (SEMANTIC_CODEC.argument_bos_id, 0),
                (SEMANTIC_CODEC.argument_bos_id, 20),
                (SEMANTIC_CODEC.argument_bos_id, 0),
            ),
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        result.prefix_batch.argument_masks,
        torch.tensor(
            ((True, False), (True, True), (True, False)),
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        result.selected_token_ids,
        torch.tensor((20, 22, 11), dtype=torch.long),
    )
    assert torch.equal(
        result.legal_choice_ids,
        torch.tensor(((20, 0), (21, 22), (10, 11)), dtype=torch.int16),
    )
    assert torch.equal(
        result.legal_choice_masks,
        torch.tensor(
            ((True, False), (True, True), (True, True)),
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        result.selected_choice_offsets,
        torch.tensor((0, 1, 1), dtype=torch.long),
    )


def test_replay_prefix_tensor_batch_returns_none_for_empty_batch() -> (
    None
):
    result = replay_prefix_tensor_batch(
        replay=_two_sample_replay(),
        sample_indices=torch.empty((0,), dtype=torch.long),
    )

    assert result is None


def _two_sample_replay() -> PPOReplayTensorBatch:
    return PPOReplayTensorBatch(
        sample_count=2,
        step_count=3,
        max_step_count=2,
        selected_token_ids_padded=torch.tensor(
            ((11, 0), (20, 22)), dtype=torch.long
        ),
        legal_choice_ids_padded=torch.stack(
            (
                torch.stack((_choices(10, 11), _choices())),
                torch.stack((_choices(20), _choices(21, 22))),
            )
        ),
        legal_choice_masks_padded=torch.stack(
            (
                torch.stack((_choice_mask(2), _choice_mask(0))),
                torch.stack((_choice_mask(1), _choice_mask(2))),
            )
        ),
        selected_choice_offsets_padded=torch.tensor(
            ((1, 0), (0, 1)), dtype=torch.long
        ),
        step_mask=torch.tensor(
            ((True, False), (True, True)), dtype=torch.bool
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
