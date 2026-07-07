"""Tests for PPO objective math."""

from __future__ import annotations

import torch

from server.training.ppo.math import (
    PPOObjectiveConfig,
    clipped_ppo_objective,
)


def test_clipped_ppo_objective_returns_finite_scalar_losses() -> None:
    old_log_probabilities = torch.log(
        torch.tensor([0.4, 0.6], dtype=torch.float32)
    )
    new_log_probabilities = torch.log(
        torch.tensor([0.5, 0.5], dtype=torch.float32)
    )
    old_values = torch.tensor([0.1, -0.2], dtype=torch.float32)
    new_values = torch.tensor([0.3, -0.1], dtype=torch.float32)
    return_values = torch.tensor([1.0, -1.0], dtype=torch.float32)
    advantages = torch.tensor([0.8, -0.4], dtype=torch.float32)
    entropies = torch.tensor([0.2, 0.4], dtype=torch.float32)

    objective = clipped_ppo_objective(
        old_log_probabilities=old_log_probabilities,
        new_log_probabilities=new_log_probabilities,
        advantages=advantages,
        old_values=old_values,
        new_values=new_values,
        return_values=return_values,
        entropies=entropies,
        config=PPOObjectiveConfig(
            ppo_clip=0.2,
            value_clip=0.1,
            value_coef=0.5,
            entropy_coef=0.01,
        ),
    )

    assert objective.policy_loss.shape == ()
    assert objective.value_loss.shape == ()
    assert objective.entropy.shape == ()
    assert objective.total_loss.shape == ()
    assert objective.approx_kl.shape == ()
    assert objective.clip_fraction.shape == ()
    assert torch.isfinite(objective.total_loss)


def test_clipped_ppo_objective_clips_policy_ratio() -> None:
    objective = clipped_ppo_objective(
        old_log_probabilities=torch.log(
            torch.tensor([0.5], dtype=torch.float32)
        ),
        new_log_probabilities=torch.log(
            torch.tensor([1.0], dtype=torch.float32)
        ),
        advantages=torch.tensor([1.0], dtype=torch.float32),
        old_values=torch.tensor([0.0], dtype=torch.float32),
        new_values=torch.tensor([0.0], dtype=torch.float32),
        return_values=torch.tensor([0.0], dtype=torch.float32),
        entropies=torch.tensor([0.0], dtype=torch.float32),
        config=PPOObjectiveConfig(
            ppo_clip=0.2,
            value_clip=0.2,
            value_coef=0.0,
            entropy_coef=0.0,
        ),
    )

    assert torch.allclose(
        objective.policy_loss,
        torch.tensor(-1.2, dtype=torch.float32),
    )
    assert torch.allclose(
        objective.clip_fraction,
        torch.tensor(1.0, dtype=torch.float32),
    )
