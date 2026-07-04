"""Black-box tests for PPO math primitives."""

from __future__ import annotations

import math

import torch

from server.training.ppo_math import (
    PPOObjectiveConfig,
    ValueStep,
    clipped_ppo_objective,
    generalized_advantage_targets,
)


def test_generalized_advantage_targets_values() -> None:
    targets = generalized_advantage_targets(
        steps=(
            ValueStep(reward=0.0, value_estimate=0.2),
            ValueStep(reward=0.0, value_estimate=0.4),
            ValueStep(reward=1.5, value_estimate=0.1),
        ),
        gamma=0.9,
        gae_lambda=0.8,
    )

    _assert_close(targets[0].advantage, 0.66256)
    _assert_close(targets[0].return_value, 0.86256)
    _assert_close(targets[1].advantage, 0.698)
    _assert_close(targets[1].return_value, 1.098)
    _assert_close(targets[2].advantage, 1.4)
    _assert_close(targets[2].return_value, 1.5)


def test_clipped_ppo_objective_values() -> None:
    objective = clipped_ppo_objective(
        old_log_probabilities=torch.log(torch.tensor([0.5, 0.5])),
        new_log_probabilities=torch.log(torch.tensor([0.8, 0.2])),
        advantages=torch.tensor([2.0, -1.0]),
        old_values=torch.tensor([0.0, 1.0]),
        new_values=torch.tensor([1.0, 0.0]),
        return_values=torch.tensor([2.0, -1.0]),
        entropies=torch.tensor([0.3, 0.7]),
        config=PPOObjectiveConfig(
            ppo_clip=0.2,
            value_clip=0.25,
            value_coef=0.5,
            entropy_coef=0.01,
        ),
    )

    _assert_tensor_close(objective.policy_loss, -0.8)
    _assert_tensor_close(objective.value_loss, 3.0625)
    _assert_tensor_close(objective.entropy, 0.5)
    _assert_tensor_close(
        objective.approx_kl,
        (math.log(0.5) - math.log(0.8) + math.log(0.5) - math.log(0.2))
        / 2.0,
    )
    _assert_tensor_close(objective.clip_fraction, 1.0)
    _assert_tensor_close(objective.total_loss, 0.72625)


def _assert_tensor_close(actual: torch.Tensor, expected: float) -> None:
    _assert_close(float(actual.detach().cpu().item()), expected)


def _assert_close(actual: float, expected: float) -> None:
    assert abs(actual - expected) < 0.000001
