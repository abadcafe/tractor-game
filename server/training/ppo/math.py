"""Pure PPO target and loss calculations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ValueStep:
    """One value estimate with the reward observed after it."""

    reward: float
    value_estimate: float


@dataclass(frozen=True, slots=True)
class AdvantageTarget:
    """GAE advantage and value-function return for one step."""

    advantage: float
    return_value: float


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


def generalized_advantage_targets(
    *,
    steps: tuple[ValueStep, ...],
    terminal_reward: float,
    gae_lambda: float,
) -> tuple[AdvantageTarget, ...]:
    """Compute generalized advantage estimates and returns."""
    assert steps
    advantages = [0.0 for _ in steps]
    gae = 0.0
    for index in range(len(steps) - 1, -1, -1):
        next_value = (
            0.0
            if index == len(steps) - 1
            else steps[index + 1].value_estimate
        )
        step = steps[index]
        final_reward = (
            terminal_reward if index == len(steps) - 1 else 0.0
        )
        delta = (
            step.reward
            + final_reward
            + next_value
            - step.value_estimate
        )
        gae = delta + gae_lambda * gae
        advantages[index] = gae
    return tuple(
        AdvantageTarget(
            advantage=advantages[index],
            return_value=advantages[index] + step.value_estimate,
        )
        for index, step in enumerate(steps)
    )


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
