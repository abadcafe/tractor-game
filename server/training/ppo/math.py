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
    entropy_coef: float


@dataclass(frozen=True, slots=True)
class PPOObjectiveTensors:
    """Loss tensors and diagnostics for a minibatch."""

    policy_loss: Tensor
    entropy: Tensor
    objective_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


def clipped_ppo_objective(
    *,
    old_log_probabilities: Tensor,
    new_log_probabilities: Tensor,
    advantages: Tensor,
    entropies: Tensor,
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
    return PPOObjectiveTensors(
        policy_loss=policy_loss,
        entropy=entropy,
        objective_loss=objective_loss,
        approx_kl=approx_kl.mean(),
        clip_fraction=clip_fraction.mean(),
    )


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
