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
        result.legal_token_masks,
        torch.stack((_mask(20), _mask(21, 22), _mask(10, 11))),
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
        legal_token_masks_padded=torch.stack(
            (
                torch.stack((_mask(10, 11), _mask())),
                torch.stack((_mask(20), _mask(21, 22))),
            )
        ),
        step_mask=torch.tensor(
            ((True, False), (True, True)), dtype=torch.bool
        ),
        step_counts=torch.tensor((1, 2), dtype=torch.long),
    )


def _mask(*token_ids: int) -> Tensor:
    result = torch.zeros(
        (SEMANTIC_CODEC.argument_vocab_size,), dtype=torch.bool
    )
    for token_id in token_ids:
        result[token_id] = True
    return result
