"""Strictly typed AdamW optimizer state."""

from __future__ import annotations

import math
from typing import TypeGuard, cast

import torch
from torch import Tensor


class AdamWState:
    """Strictly typed AdamW optimizer state."""

    def __init__(
        self,
        *,
        parameters: tuple[Tensor, ...],
        learning_rate: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        eps: float = 0.00000001,
    ) -> None:
        self._parameters = parameters
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._weight_decay = weight_decay
        self._eps = eps
        self._step_count = 0
        self._exp_avgs: list[Tensor | None] = [None for _ in parameters]
        self._exp_avg_sqs: list[Tensor | None] = [
            None for _ in parameters
        ]

    def step(self) -> None:
        """Apply one AdamW update using current parameter gradients."""
        self._step_count += 1
        with torch.no_grad():
            for index, parameter in enumerate(self._parameters):
                gradient = parameter.grad
                if gradient is None:
                    continue
                exp_avg = self._exp_avgs[index]
                exp_avg_sq = self._exp_avg_sqs[index]
                if exp_avg is None:
                    exp_avg = torch.zeros_like(parameter)
                    self._exp_avgs[index] = exp_avg
                if exp_avg_sq is None:
                    exp_avg_sq = torch.zeros_like(parameter)
                    self._exp_avg_sqs[index] = exp_avg_sq
                if self._weight_decay != 0.0:
                    parameter.mul_(
                        1.0 - self._learning_rate * self._weight_decay
                    )
                exp_avg.mul_(self._beta1).add_(
                    gradient, alpha=1.0 - self._beta1
                )
                exp_avg_sq.mul_(self._beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - self._beta2,
                )
                bias_correction1 = 1.0 - self._beta1**self._step_count
                bias_correction2 = 1.0 - self._beta2**self._step_count
                step_size = (
                    self._learning_rate
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )
                denominator = exp_avg_sq.sqrt().add_(self._eps)
                parameter.addcdiv_(
                    exp_avg, denominator, value=-step_size
                )

    def state_dict(self) -> dict[str, object]:
        """Return a torch-saveable optimizer state payload."""
        return {
            "kind": "typed_adamw",
            "step_count": self._step_count,
            "exp_avgs": list(self._exp_avgs),
            "exp_avg_sqs": list(self._exp_avg_sqs),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Load optimizer state from a checkpoint payload."""
        kind = state["kind"]
        assert kind == "typed_adamw"
        step_count = state["step_count"]
        assert isinstance(step_count, int)
        exp_avgs = state["exp_avgs"]
        exp_avg_sqs = state["exp_avg_sqs"]
        assert _is_optional_tensor_list(exp_avgs)
        assert _is_optional_tensor_list(exp_avg_sqs)
        assert len(exp_avgs) == len(self._parameters)
        assert len(exp_avg_sqs) == len(self._parameters)
        self._step_count = step_count
        self._exp_avgs = [
            _optimizer_tensor_on_parameter_device(
                value, self._parameters[index]
            )
            for index, value in enumerate(exp_avgs)
        ]
        self._exp_avg_sqs = [
            _optimizer_tensor_on_parameter_device(
                value, self._parameters[index]
            )
            for index, value in enumerate(exp_avg_sqs)
        ]


def _optimizer_tensor_on_parameter_device(
    value: Tensor | None,
    parameter: Tensor,
) -> Tensor | None:
    if value is None:
        return None
    assert value.shape == parameter.shape
    return value.to(device=parameter.device)


def _is_optional_tensor_list(
    value: object,
) -> TypeGuard[list[Tensor | None]]:
    if not isinstance(value, list):
        return False
    items = cast(list[object], value)
    for item in items:
        if not _is_optional_tensor(item):
            return False
    return True


def _is_optional_tensor(value: object) -> TypeGuard[Tensor | None]:
    return value is None or isinstance(value, Tensor)
