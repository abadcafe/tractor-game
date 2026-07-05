"""Black-box tests for device-side PPO target calculation."""

from __future__ import annotations

import torch

from server.result import Ok, Rejected
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.packed_observation import (
    OBSERVATION_COMPONENT_COUNT,
)
from server.training.ppo.device_targets import device_ppo_targets
from server.training.ppo.replay_tensors import (
    PPOReplayTensorBatch,
    RolloutTensorBatch,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import ObservationTensorBatch


def test_device_ppo_targets_accepts_finite_rollout() -> None:
    result = device_ppo_targets(
        rollout=_single_transition_rollout(
            old_log_probability=0.0,
            old_value=0.25,
        ),
        gae_lambda=1.0,
    )

    assert isinstance(result, Ok)
    assert result.value.advantages.shape == (1,)
    assert result.value.return_values.shape == (1,)
    assert torch.allclose(result.value.advantages, torch.tensor([0.75]))
    assert torch.allclose(
        result.value.return_values, torch.tensor([1.0])
    )


def test_device_ppo_targets_rejects_bad_old_logprob() -> None:
    result = device_ppo_targets(
        rollout=_single_transition_rollout(
            old_log_probability=torch.inf,
            old_value=0.0,
        ),
        gae_lambda=1.0,
    )

    assert isinstance(result, Rejected)
    assert (
        result.reason == "old policy log probabilities must be finite"
    )


def test_device_ppo_targets_rejects_non_finite_old_value() -> None:
    result = device_ppo_targets(
        rollout=_single_transition_rollout(
            old_log_probability=0.0,
            old_value=torch.nan,
        ),
        gae_lambda=1.0,
    )

    assert isinstance(result, Rejected)
    assert result.reason == "old values must be finite"


def _single_transition_rollout(
    *, old_log_probability: float, old_value: float
) -> RolloutTensorBatch:
    return RolloutTensorBatch(
        policy_version=0,
        first_episode_id=0,
        episode_count=1,
        max_trajectory_length=1,
        trajectory_count=1,
        observation_batch=ObservationTensorBatch(
            component_ids=torch.zeros(
                (1, 1, OBSERVATION_COMPONENT_COUNT), dtype=torch.long
            ),
            numeric_values=torch.zeros(
                (1, 1, NUMERIC_FEATURE_COUNT), dtype=torch.float32
            ),
            numeric_masks=torch.zeros(
                (1, 1, NUMERIC_FEATURE_COUNT), dtype=torch.bool
            ),
        ),
        replay=PPOReplayTensorBatch(
            sample_count=1,
            step_count=1,
            max_step_count=1,
            selected_token_ids_padded=torch.tensor(
                [[0]], dtype=torch.long
            ),
            legal_token_masks_padded=torch.ones(
                (1, 1, SEMANTIC_CODEC.argument_vocab_size),
                dtype=torch.bool,
            ),
            step_mask=torch.tensor([[True]], dtype=torch.bool),
            step_counts=torch.tensor([1], dtype=torch.long),
        ),
        old_log_probabilities=torch.tensor(
            [old_log_probability], dtype=torch.float32
        ),
        old_values=torch.tensor([old_value], dtype=torch.float32),
        reward_after_step=torch.tensor([0.0], dtype=torch.float32),
        terminal_rewards=torch.tensor([1.0], dtype=torch.float32),
        trajectory_offsets=torch.tensor([0, 1], dtype=torch.long),
        trajectory_team_indices=torch.tensor([0], dtype=torch.long),
    )
