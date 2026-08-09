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
    advantages = torch.tensor([0.8, -0.4], dtype=torch.float32)
    entropies = torch.tensor([0.2, 0.4], dtype=torch.float32)

    objective = clipped_ppo_objective(
        old_log_probabilities=old_log_probabilities,
        new_log_probabilities=new_log_probabilities,
        advantages=advantages,
        entropies=entropies,
        old_values=torch.zeros(2),
        new_values=torch.zeros(2),
        value_targets=torch.zeros(2),
        config=PPOObjectiveConfig(
            ppo_clip=0.2,
            value_clip=0.2,
            value_coef=0.0,
            entropy_coef=0.01,
        ),
    )

    assert objective.policy_loss.shape == ()
    assert objective.entropy.shape == ()
    assert objective.objective_loss.shape == ()
    assert objective.approx_kl.shape == ()
    assert objective.clip_fraction.shape == ()
    assert torch.isfinite(objective.objective_loss)


def test_clipped_ppo_objective_clips_policy_ratio() -> None:
    objective = clipped_ppo_objective(
        old_log_probabilities=torch.log(
            torch.tensor([0.5], dtype=torch.float32)
        ),
        new_log_probabilities=torch.log(
            torch.tensor([1.0], dtype=torch.float32)
        ),
        advantages=torch.tensor([1.0], dtype=torch.float32),
        entropies=torch.tensor([0.0], dtype=torch.float32),
        old_values=torch.zeros(1),
        new_values=torch.zeros(1),
        value_targets=torch.zeros(1),
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


def test_clipped_ppo_objective_matches_ratio_space_gradients() -> None:
    old_log_probabilities = torch.full((6,), -1.0)
    log_ratio = torch.tensor((-1.0, -0.1, 0.1, 1.0, -1.0, 0.1))
    advantages = torch.tensor((1.0, 1.0, 1.0, 1.0, -1.0, -1.0))
    stable_new = (old_log_probabilities + log_ratio).requires_grad_()
    reference_new = stable_new.detach().clone().requires_grad_()

    objective = clipped_ppo_objective(
        old_log_probabilities=old_log_probabilities,
        new_log_probabilities=stable_new,
        advantages=advantages,
        entropies=torch.zeros(6),
        old_values=torch.zeros(6),
        new_values=torch.zeros(6),
        value_targets=torch.zeros(6),
        config=PPOObjectiveConfig(
            ppo_clip=0.2,
            value_clip=0.2,
            value_coef=0.0,
            entropy_coef=0.0,
        ),
    )
    reference_ratio = torch.exp(reference_new - old_log_probabilities)
    reference_loss = -torch.minimum(
        reference_ratio * advantages,
        torch.clamp(reference_ratio, 0.8, 1.2) * advantages,
    ).mean()

    torch.autograd.backward(objective.policy_loss)
    torch.autograd.backward(reference_loss)

    assert torch.allclose(objective.policy_loss, reference_loss)
    assert stable_new.grad is not None
    assert reference_new.grad is not None
    assert torch.allclose(stable_new.grad, reference_new.grad)


def test_clipped_ppo_objective_avoids_clipped_exp_overflow() -> None:
    new_log_probabilities = torch.zeros(1, requires_grad=True)

    objective = clipped_ppo_objective(
        old_log_probabilities=torch.tensor((-100.0,)),
        new_log_probabilities=new_log_probabilities,
        advantages=torch.tensor((1.0,)),
        entropies=torch.zeros(1),
        old_values=torch.zeros(1),
        new_values=torch.zeros(1),
        value_targets=torch.zeros(1),
        config=PPOObjectiveConfig(
            ppo_clip=0.2,
            value_clip=0.2,
            value_coef=0.0,
            entropy_coef=0.0,
        ),
    )

    assert torch.allclose(
        objective.policy_loss,
        torch.tensor(-1.2),
    )
    assert all(
        bool(torch.isfinite(value).item())
        for value in (
            objective.policy_loss,
            objective.entropy,
            objective.objective_loss,
            objective.approx_kl,
            objective.clip_fraction,
        )
    )

    torch.autograd.backward(objective.objective_loss)

    assert new_log_probabilities.grad is not None
    assert torch.equal(
        new_log_probabilities.grad,
        torch.zeros(1),
    )


def test_clipped_ppo_objective_stabilizes_adverse_exp_tail() -> None:
    new_log_probabilities = torch.zeros(1, requires_grad=True)
    ppo_clip = 0.2

    objective = clipped_ppo_objective(
        old_log_probabilities=torch.tensor((-100.0,)),
        new_log_probabilities=new_log_probabilities,
        advantages=torch.tensor((-1.0,)),
        entropies=torch.zeros(1),
        old_values=torch.zeros(1),
        new_values=torch.zeros(1),
        value_targets=torch.zeros(1),
        config=PPOObjectiveConfig(
            ppo_clip=ppo_clip,
            value_clip=0.2,
            value_coef=0.0,
            entropy_coef=0.0,
        ),
    )
    upper_log_ratio = torch.log1p(torch.tensor(ppo_clip))
    expected_loss = (1.0 + ppo_clip) * (1.0 + 100.0 - upper_log_ratio)

    torch.autograd.backward(objective.objective_loss)

    assert torch.allclose(objective.policy_loss, expected_loss)
    assert new_log_probabilities.grad is not None
    assert torch.allclose(
        new_log_probabilities.grad,
        torch.tensor((1.0 + ppo_clip,)),
    )
    assert bool(torch.isfinite(new_log_probabilities.grad).all().item())


def test_clipped_ppo_objective_handles_extreme_zero_advantage() -> None:
    new_log_probabilities = torch.tensor(
        (100.0, -100.0), requires_grad=True
    )

    objective = clipped_ppo_objective(
        old_log_probabilities=torch.zeros(2),
        new_log_probabilities=new_log_probabilities,
        advantages=torch.zeros(2),
        entropies=torch.zeros(2),
        old_values=torch.zeros(2),
        new_values=torch.zeros(2),
        value_targets=torch.zeros(2),
        config=PPOObjectiveConfig(
            ppo_clip=0.2,
            value_clip=0.2,
            value_coef=0.0,
            entropy_coef=0.0,
        ),
    )

    torch.autograd.backward(objective.objective_loss)

    assert torch.equal(objective.policy_loss, torch.tensor(0.0))
    assert new_log_probabilities.grad is not None
    assert torch.equal(
        new_log_probabilities.grad,
        torch.zeros(2),
    )


def test_clipped_ppo_objective_supports_full_clip_range() -> None:
    new_log_probabilities = torch.tensor((-100.0,), requires_grad=True)

    objective = clipped_ppo_objective(
        old_log_probabilities=torch.zeros(1),
        new_log_probabilities=new_log_probabilities,
        advantages=torch.tensor((-1.0,)),
        entropies=torch.zeros(1),
        old_values=torch.zeros(1),
        new_values=torch.zeros(1),
        value_targets=torch.zeros(1),
        config=PPOObjectiveConfig(
            ppo_clip=1.0,
            value_clip=0.2,
            value_coef=0.0,
            entropy_coef=0.0,
        ),
    )

    torch.autograd.backward(objective.objective_loss)

    assert bool(torch.isfinite(objective.objective_loss).item())
    assert new_log_probabilities.grad is not None
    assert bool(torch.isfinite(new_log_probabilities.grad).all().item())
