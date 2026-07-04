"""Black-box tests for the typed AdamW optimizer."""

from __future__ import annotations

import math

import torch

from server.training.adamw import AdamWState


def test_step_applies_adamw_update() -> None:
    parameter = torch.tensor([1.0, -2.0], requires_grad=True)
    parameter.grad = torch.tensor([0.2, -0.4])
    optimizer = AdamWState(
        parameters=(parameter,),
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01,
    )

    optimizer.step()

    expected = _adamw_first_step_expected(
        parameter_before=torch.tensor([1.0, -2.0]),
        gradient=torch.tensor([0.2, -0.4]),
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01,
        eps=0.00000001,
    )
    assert torch.allclose(parameter.detach(), expected, atol=0.000001)


def _adamw_first_step_expected(
    *,
    parameter_before: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
) -> torch.Tensor:
    after_weight_decay = parameter_before * (
        1.0 - learning_rate * weight_decay
    )
    exp_avg = gradient * (1.0 - beta1)
    exp_avg_sq = gradient * gradient * (1.0 - beta2)
    step_size = learning_rate * math.sqrt(1.0 - beta2) / (1.0 - beta1)
    denominator = exp_avg_sq.sqrt() + eps
    return after_weight_decay - step_size * (exp_avg / denominator)
