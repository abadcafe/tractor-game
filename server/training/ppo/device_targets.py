"""Device-side PPO target tensors for compact rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server import result as _result
from server.training.ppo.replay_tensors import RolloutTensorBatch
from server.training.tensor_finiteness import (
    NamedTensorCheck,
    reject_if_non_finite,
)


@dataclass(frozen=True, slots=True)
class DevicePPOTargets:
    """Old policy tensors plus raw GAE targets."""

    old_log_probabilities: Tensor
    old_values: Tensor
    advantages: Tensor
    return_values: Tensor


def device_ppo_targets(
    *,
    rollout: RolloutTensorBatch,
    gae_lambda: float,
) -> _result.Ok[DevicePPOTargets] | _result.Rejected:
    """Calculate raw PPO targets on the learner device."""
    assert not rollout.is_empty()
    assert 0.0 <= gae_lambda <= 1.0
    old_log_probabilities = rollout.old_log_probabilities
    old_values = rollout.old_values
    rewards = rollout.reward_after_step
    terminal_rewards = rollout.terminal_rewards
    lengths = _trajectory_lengths(rollout)
    padded_values = _padded_by_trajectory(
        values=old_values,
        rollout=rollout,
        max_length=rollout.max_trajectory_length,
    )
    padded_rewards = _padded_by_trajectory(
        values=rewards,
        rollout=rollout,
        max_length=rollout.max_trajectory_length,
    )
    mask = _trajectory_mask(
        lengths=lengths, max_length=rollout.max_trajectory_length
    )
    advantages = _flatten_valid(
        _padded_gae(
            rewards=padded_rewards,
            values=padded_values,
            terminal_rewards=terminal_rewards,
            lengths=lengths,
            mask=mask,
            gae_lambda=gae_lambda,
        ),
        mask=mask,
    )
    return_values = advantages + old_values
    finite_check = reject_if_non_finite(
        (
            NamedTensorCheck(
                tensor=old_log_probabilities,
                reason="old policy log probabilities must be finite",
            ),
            NamedTensorCheck(
                tensor=old_values,
                reason="old values must be finite",
            ),
            NamedTensorCheck(
                tensor=advantages,
                reason="PPO advantages must be finite",
            ),
            NamedTensorCheck(
                tensor=return_values,
                reason="PPO return values must be finite",
            ),
        )
    )
    if isinstance(finite_check, _result.Rejected):
        return finite_check
    return _result.Ok(
        value=DevicePPOTargets(
            old_log_probabilities=old_log_probabilities,
            old_values=old_values,
            advantages=advantages,
            return_values=return_values,
        )
    )


def shuffled_index_tensor(
    *, indices: tuple[int, ...], device: torch.device
) -> Tensor:
    """Create a learner-device tensor for deterministic sample order."""
    assert indices
    return torch.tensor(indices, dtype=torch.long, device=device)


def _trajectory_lengths(rollout: RolloutTensorBatch) -> Tensor:
    return (
        rollout.trajectory_offsets[1:] - rollout.trajectory_offsets[:-1]
    )


def _padded_by_trajectory(
    *,
    values: Tensor,
    rollout: RolloutTensorBatch,
    max_length: int,
) -> Tensor:
    lengths = _trajectory_lengths(rollout)
    result = torch.zeros(
        (rollout.trajectory_count, max_length),
        dtype=torch.float32,
        device=values.device,
    )
    row_indices = torch.repeat_interleave(
        torch.arange(
            rollout.trajectory_count,
            dtype=torch.long,
            device=values.device,
        ),
        lengths,
    )
    base_offsets = torch.repeat_interleave(
        rollout.trajectory_offsets[:-1], lengths
    )
    column_indices = (
        torch.arange(
            int(values.shape[0]), dtype=torch.long, device=values.device
        )
        - base_offsets
    )
    result[row_indices, column_indices] = values
    return result


def _trajectory_mask(*, lengths: Tensor, max_length: int) -> Tensor:
    positions = torch.arange(
        max_length, dtype=torch.long, device=lengths.device
    ).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _padded_gae(
    *,
    rewards: Tensor,
    values: Tensor,
    terminal_rewards: Tensor,
    lengths: Tensor,
    mask: Tensor,
    gae_lambda: float,
) -> Tensor:
    max_length = int(values.shape[1])
    positions = torch.arange(
        max_length, dtype=torch.long, device=values.device
    ).unsqueeze(0)
    is_last = positions == (lengths.unsqueeze(1) - 1)
    next_values = torch.cat(
        (
            values[:, 1:],
            torch.zeros(
                (int(values.shape[0]), 1),
                dtype=torch.float32,
                device=values.device,
            ),
        ),
        dim=1,
    )
    next_values = torch.where(
        is_last, torch.zeros_like(values), next_values
    )
    final_rewards = torch.where(
        is_last,
        terminal_rewards.unsqueeze(1),
        torch.zeros_like(values),
    )
    deltas = (
        rewards + final_rewards + next_values - values
    ).masked_fill(~mask, 0.0)
    discount = _discount_matrix(
        length=max_length,
        gae_lambda=gae_lambda,
        device=values.device,
    )
    return deltas.matmul(discount.transpose(0, 1)).masked_fill(
        ~mask, 0.0
    )


def _discount_matrix(
    *, length: int, gae_lambda: float, device: torch.device
) -> Tensor:
    assert length > 0
    positions = torch.arange(length, dtype=torch.float32, device=device)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)
    valid = distance >= 0.0
    if gae_lambda == 0.0:
        return torch.eye(length, dtype=torch.float32, device=device)
    return torch.where(
        valid,
        torch.pow(
            torch.full(
                (), gae_lambda, dtype=torch.float32, device=device
            ),
            distance,
        ),
        torch.zeros(
            (length, length), dtype=torch.float32, device=device
        ),
    )


def _flatten_valid(values: Tensor, *, mask: Tensor) -> Tensor:
    return values[mask]
