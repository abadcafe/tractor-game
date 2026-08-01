"""Pure PPO target and loss calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class PPOObjectiveConfig:
    """Scalar coefficients for clipped PPO objective calculation."""

    ppo_clip: float
    entropy_coef: float


@dataclass(frozen=True, slots=True)
class PPOObjectiveTensors:
    """Loss tensors and diagnostics for a minibatch."""

    policy_loss: Tensor
    action_value_loss: Tensor
    entropy: Tensor
    total_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


def clipped_ppo_objective(
    *,
    old_log_probabilities: Tensor,
    new_log_probabilities: Tensor,
    advantages: Tensor,
    return_values: Tensor,
    action_values: Tensor,
    entropies: Tensor,
    config: PPOObjectiveConfig,
) -> PPOObjectiveTensors:
    """Calculate clipped PPO policy/value losses and diagnostics."""
    log_ratio = new_log_probabilities - old_log_probabilities
    lower_log_ratio, upper_log_ratio = _log_ratio_bounds(
        reference=log_ratio,
        ppo_clip=config.ppo_clip,
    )
    # Select the active clipped-PPO branch before exponentiation.
    effective_log_ratio = torch.where(
        advantages >= 0.0,
        torch.minimum(log_ratio, upper_log_ratio),
        torch.maximum(log_ratio, lower_log_ratio),
    )
    policy_loss = -(torch.exp(effective_log_ratio) * advantages).mean()
    action_value_loss = nn.functional.mse_loss(
        action_values,
        return_values,
    )
    entropy = entropies.mean()
    approx_kl = -log_ratio
    clip_fraction = (
        (log_ratio < lower_log_ratio) | (log_ratio > upper_log_ratio)
    ).to(dtype=torch.float32)
    total_loss = (
        policy_loss + action_value_loss - config.entropy_coef * entropy
    )
    return PPOObjectiveTensors(
        policy_loss=policy_loss,
        action_value_loss=action_value_loss,
        entropy=entropy,
        total_loss=total_loss,
        approx_kl=approx_kl.mean(),
        clip_fraction=clip_fraction.mean(),
    )


def _log_ratio_bounds(
    *, reference: Tensor, ppo_clip: float
) -> tuple[Tensor, Tensor]:
    assert 0.0 < ppo_clip <= 1.0
    lower = (
        -math.inf if ppo_clip == 1.0 else math.log1p(-ppo_clip)
    )
    return (
        reference.new_tensor(lower),
        reference.new_tensor(math.log1p(ppo_clip)),
    )
