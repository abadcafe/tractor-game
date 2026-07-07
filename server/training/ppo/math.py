"""Pure PPO target and loss calculations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class PPOObjectiveConfig:
    """Scalar coefficients for clipped PPO objective calculation."""

    ppo_clip: float
    value_clip: float
    value_coef: float
    entropy_coef: float


@dataclass(frozen=True, slots=True)
class PPOObjectiveTensors:
    """Loss tensors and diagnostics for a minibatch."""

    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    total_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


def clipped_ppo_objective(
    *,
    old_log_probabilities: Tensor,
    new_log_probabilities: Tensor,
    advantages: Tensor,
    old_values: Tensor,
    new_values: Tensor,
    return_values: Tensor,
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
    value_clipped = old_values + torch.clamp(
        new_values - old_values,
        -config.value_clip,
        config.value_clip,
    )
    value_loss = torch.maximum(
        nn.functional.mse_loss(
            new_values,
            return_values,
            reduction="none",
        ),
        nn.functional.mse_loss(
            value_clipped,
            return_values,
            reduction="none",
        ),
    ).mean()
    entropy = entropies.mean()
    approx_kl = old_log_probabilities - new_log_probabilities
    clip_fraction = (
        ratio.sub(1.0).abs().gt(config.ppo_clip).to(dtype=torch.float32)
    )
    total_loss = (
        policy_loss
        + config.value_coef * value_loss
        - config.entropy_coef * entropy
    )
    return PPOObjectiveTensors(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        total_loss=total_loss,
        approx_kl=approx_kl.mean(),
        clip_fraction=clip_fraction.mean(),
    )
