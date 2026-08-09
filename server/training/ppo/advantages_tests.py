"""Black-box tests for terminal-round GAE."""

from __future__ import annotations

import torch

from server.training.ppo import explained_variance, terminal_round_gae


def test_terminal_round_gae_keeps_seat_trajectories_independent() -> (
    None
):
    targets = terminal_round_gae(
        old_values=torch.tensor((0.5, 0.25, -0.5)),
        trajectory_offsets=torch.tensor((0, 2), dtype=torch.long),
        trajectory_lengths=torch.tensor((2, 1), dtype=torch.long),
        terminal_rewards=torch.tensor((1.0, -1.0)),
        gae_lambda=0.95,
    )

    assert torch.allclose(
        targets.advantages,
        torch.tensor((0.4625, 0.75, -0.5)),
    )
    assert torch.allclose(
        targets.value_targets,
        torch.tensor((0.9625, 1.0, -1.0)),
    )


def test_terminal_round_gae_lambda_zero_uses_one_step_bootstrap() -> (
    None
):
    targets = terminal_round_gae(
        old_values=torch.tensor((0.5, 0.25)),
        trajectory_offsets=torch.tensor((0,), dtype=torch.long),
        trajectory_lengths=torch.tensor((2,), dtype=torch.long),
        terminal_rewards=torch.tensor((1.0,)),
        gae_lambda=0.0,
    )

    assert torch.allclose(
        targets.advantages,
        torch.tensor((-0.25, 0.75)),
    )
    assert torch.allclose(
        targets.value_targets,
        torch.tensor((0.25, 1.0)),
    )


def test_explained_variance_reports_exact_and_constant_targets() -> (
    None
):
    exact = explained_variance(
        predictions=torch.tensor((1.0, 2.0, 3.0)),
        targets=torch.tensor((1.0, 2.0, 3.0)),
    )
    constant = explained_variance(
        predictions=torch.tensor((0.0, 1.0)),
        targets=torch.tensor((2.0, 2.0)),
    )

    assert torch.equal(exact, torch.tensor(1.0))
    assert torch.equal(constant, torch.tensor(0.0))
