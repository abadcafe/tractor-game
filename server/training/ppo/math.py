"""Pure PPO target and loss calculations."""

from __future__ import annotations

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
    ratio = torch.exp(new_log_probabilities - old_log_probabilities)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - config.ppo_clip,
        1.0 + config.ppo_clip,
    )
    policy_loss = -torch.minimum(
        ratio * advantages,
        clipped_ratio * advantages,
    ).mean()
    action_value_loss = nn.functional.mse_loss(
        action_values,
        return_values,
    )
    entropy = entropies.mean()
    approx_kl = old_log_probabilities - new_log_probabilities
    clip_fraction = (
        ratio.sub(1.0).abs().gt(config.ppo_clip).to(dtype=torch.float32)
    )
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
