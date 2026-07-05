"""Canonical AdamW optimizer used by PPO trainers."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Protocol, TypeGuard, cast

import torch
from torch import Tensor

type _OptimizerState = MutableMapping[object, object]


class _TorchAdamW(Protocol):
    state: object

    def step(self) -> object | None: ...


class PPOOptimizer:
    """AdamW optimizer with portable checkpoint state."""

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
        self._optimizer = cast(
            _TorchAdamW,
            torch.optim.AdamW(
                list(parameters),
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
                foreach=True,
            ),
        )

    def step(self) -> None:
        """Apply one AdamW update."""
        self._step_count += 1
        self._optimizer.step()

    def state_dict(self) -> dict[str, object]:
        """Return a torch-saveable optimizer state payload."""
        return {
            "kind": "ppo_adamw",
            "step_count": self._step_count,
            "exp_avgs": _optimizer_tensor_values(
                parameters=self._parameters,
                optimizer_state=_torch_optimizer_state(self._optimizer),
                key="exp_avg",
            ),
            "exp_avg_sqs": _optimizer_tensor_values(
                parameters=self._parameters,
                optimizer_state=_torch_optimizer_state(self._optimizer),
                key="exp_avg_sq",
            ),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Load a canonical optimizer state payload."""
        kind = state["kind"]
        assert kind == "ppo_adamw"
        step_count = state["step_count"]
        assert isinstance(step_count, int)
        exp_avgs = state["exp_avgs"]
        exp_avg_sqs = state["exp_avg_sqs"]
        assert _is_optional_tensor_list(exp_avgs)
        assert _is_optional_tensor_list(exp_avg_sqs)
        assert len(exp_avgs) == len(self._parameters)
        assert len(exp_avg_sqs) == len(self._parameters)
        self._step_count = step_count
        optimizer_state = _torch_optimizer_state(self._optimizer)
        optimizer_state.clear()
        for index, parameter in enumerate(self._parameters):
            exp_avg = exp_avgs[index]
            exp_avg_sq = exp_avg_sqs[index]
            assert (exp_avg is None) == (exp_avg_sq is None)
            if exp_avg is None or exp_avg_sq is None:
                continue
            optimizer_state[parameter] = {
                "step": torch.tensor(float(step_count)),
                "exp_avg": _optimizer_tensor_on_parameter_device(
                    exp_avg, parameter
                ),
                "exp_avg_sq": _optimizer_tensor_on_parameter_device(
                    exp_avg_sq, parameter
                ),
            }


def _torch_optimizer_state(
    optimizer: _TorchAdamW,
) -> _OptimizerState:
    return cast(_OptimizerState, optimizer.state)


def _optimizer_tensor_values(
    *,
    parameters: tuple[Tensor, ...],
    optimizer_state: _OptimizerState,
    key: str,
) -> list[Tensor | None]:
    values: list[Tensor | None] = []
    for parameter in parameters:
        state = _parameter_state(
            optimizer_state=optimizer_state,
            parameter=parameter,
        )
        if state is None:
            values.append(None)
            continue
        value = state[key]
        assert isinstance(value, Tensor)
        values.append(value)
    return values


def _parameter_state(
    *,
    optimizer_state: _OptimizerState,
    parameter: Tensor,
) -> _OptimizerState | None:
    raw_state = optimizer_state.get(parameter)
    if raw_state is None:
        return None
    assert isinstance(raw_state, MutableMapping)
    return cast(_OptimizerState, raw_state)


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
