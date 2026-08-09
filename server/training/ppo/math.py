"""Pure PPO target and loss calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


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
    entropy: Tensor
    objective_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor
    value_loss: Tensor
    value_clip_fraction: Tensor


def clipped_ppo_objective(
    *,
    old_log_probabilities: Tensor,
    new_log_probabilities: Tensor,
    advantages: Tensor,
    entropies: Tensor,
    old_values: Tensor,
    new_values: Tensor,
    value_targets: Tensor,
    config: PPOObjectiveConfig,
) -> PPOObjectiveTensors:
    """Calculate clipped PPO policy loss and diagnostics."""
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
    effective_ratio = _stable_effective_ratio(
        effective_log_ratio,
        upper_log_ratio=upper_log_ratio,
    )
    policy_loss = -(effective_ratio * advantages).mean()
    entropy = entropies.mean()
    approx_kl = -log_ratio
    clip_fraction = (
        (log_ratio < lower_log_ratio) | (log_ratio > upper_log_ratio)
    ).to(dtype=torch.float32)
    objective_loss = policy_loss - config.entropy_coef * entropy
    value_loss, value_clip_fraction = clipped_value_loss(
        old_values=old_values,
        new_values=new_values,
        value_targets=value_targets,
        value_clip=config.value_clip,
    )
    objective_loss = objective_loss + config.value_coef * value_loss
    return PPOObjectiveTensors(
        policy_loss=policy_loss,
        entropy=entropy,
        objective_loss=objective_loss,
        approx_kl=approx_kl.mean(),
        clip_fraction=clip_fraction.mean(),
        value_loss=value_loss,
        value_clip_fraction=value_clip_fraction,
    )


def clipped_value_loss(
    *,
    old_values: Tensor,
    new_values: Tensor,
    value_targets: Tensor,
    value_clip: float,
) -> tuple[Tensor, Tensor]:
    """Return standard clipped critic regression loss and clip rate."""
    assert old_values.shape == new_values.shape
    assert value_targets.shape == new_values.shape
    assert 0.0 < value_clip <= 1.0
    value_delta = new_values - old_values
    clipped_values = old_values + torch.clamp(
        value_delta,
        min=-value_clip,
        max=value_clip,
    )
    plain_errors = (new_values - value_targets).square()
    clipped_errors = (clipped_values - value_targets).square()
    loss = 0.5 * torch.maximum(plain_errors, clipped_errors).mean()
    fraction = value_delta.abs().gt(value_clip).to(torch.float32).mean()
    return loss, fraction


def _log_ratio_bounds(
    *, reference: Tensor, ppo_clip: float
) -> tuple[Tensor, Tensor]:
    assert 0.0 < ppo_clip <= 1.0
    lower = -math.inf if ppo_clip == 1.0 else math.log1p(-ppo_clip)
    return (
        reference.new_tensor(lower),
        reference.new_tensor(math.log1p(ppo_clip)),
    )


def _stable_effective_ratio(
    effective_log_ratio: Tensor, *, upper_log_ratio: Tensor
) -> Tensor:
    """Exponentiate with a C1 linear tail beyond the trust region."""
    above_upper = effective_log_ratio > upper_log_ratio
    upper_excess = torch.where(
        above_upper,
        effective_log_ratio - upper_log_ratio,
        torch.zeros_like(effective_log_ratio),
    )
    bounded_log_ratio = torch.where(
        above_upper,
        upper_log_ratio,
        effective_log_ratio,
    )
    return torch.exp(bounded_log_ratio) * (1.0 + upper_excess)
