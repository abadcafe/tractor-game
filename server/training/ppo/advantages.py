"""Terminal-round generalized advantage estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class GAETargets:
    """Frozen rollout advantages and critic regression targets."""

    advantages: Tensor
    value_targets: Tensor

    def __post_init__(self) -> None:
        assert self.advantages.ndim == 1
        assert self.value_targets.shape == self.advantages.shape
        assert self.advantages.dtype == torch.float32
        assert self.value_targets.dtype == torch.float32


def terminal_round_gae(
    *,
    old_values: Tensor,
    trajectory_offsets: Tensor,
    trajectory_lengths: Tensor,
    terminal_rewards: Tensor,
    gae_lambda: float,
) -> GAETargets:
    """Compute gamma=1 GAE independently for each seat trajectory."""
    assert old_values.ndim == 1
    assert old_values.dtype == torch.float32
    assert trajectory_offsets.ndim == 1
    assert trajectory_offsets.dtype == torch.long
    assert trajectory_lengths.shape == trajectory_offsets.shape
    assert trajectory_lengths.dtype == torch.long
    assert terminal_rewards.shape == trajectory_offsets.shape
    assert terminal_rewards.dtype == torch.float32
    assert math.isfinite(gae_lambda)
    assert 0.0 <= gae_lambda <= 1.0
    count = int(old_values.shape[0])
    advantages = torch.zeros_like(old_values)
    covered = torch.zeros(
        (count,), dtype=torch.bool, device=old_values.device
    )
    offsets = _cpu_ints(trajectory_offsets)
    lengths = _cpu_ints(trajectory_lengths)
    for trajectory_index, (offset, length) in enumerate(
        zip(offsets, lengths, strict=True)
    ):
        assert length > 0
        assert 0 <= offset < count
        assert offset + length <= count
        assert not bool(covered[offset : offset + length].any().item())
        covered[offset : offset + length] = True
        next_value = old_values.new_zeros(())
        next_advantage = old_values.new_zeros(())
        terminal_reward = terminal_rewards[trajectory_index]
        for local_index in range(length - 1, -1, -1):
            row = offset + local_index
            reward = (
                terminal_reward
                if local_index == length - 1
                else old_values.new_zeros(())
            )
            delta = reward + next_value - old_values[row]
            next_advantage = delta + gae_lambda * next_advantage
            advantages[row] = next_advantage
            next_value = old_values[row]
    assert bool(covered.all().item())
    return GAETargets(
        advantages=advantages,
        value_targets=advantages + old_values,
    )


def explained_variance(
    *, predictions: Tensor, targets: Tensor
) -> Tensor:
    """Return explained variance with a constant-target fallback."""
    assert predictions.ndim == 1
    assert targets.shape == predictions.shape
    residual_variance = torch.var(
        targets - predictions,
        correction=0,
    )
    target_variance = torch.var(targets, correction=0)
    return torch.where(
        target_variance <= 0.000001,
        torch.zeros_like(target_variance),
        1.0 - residual_variance / target_variance,
    )


def _cpu_ints(values: Tensor) -> tuple[int, ...]:
    cpu = values.detach().cpu()
    return tuple(
        int(cpu[index].item()) for index in range(int(cpu.shape[0]))
    )


__all__ = ("GAETargets", "explained_variance", "terminal_round_gae")
