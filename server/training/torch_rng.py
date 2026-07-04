"""Torch training random generator state."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TypeGuard, cast

import torch
from torch import Tensor

from server import result as _result

type PythonRandomState = tuple[int, tuple[int, ...], float | None]


@dataclass(frozen=True, slots=True)
class TorchRngState:
    """Random generator states needed for exact training resume."""

    python_random_state: PythonRandomState
    torch_cpu_state: Tensor
    torch_cuda_states: tuple[Tensor, ...]


def capture_torch_rng_state() -> TorchRngState:
    """Capture Python and torch RNG states for a checkpoint."""
    python_state = random.getstate()
    assert _is_python_random_state(python_state)
    cuda_states = (
        tuple(torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    return TorchRngState(
        python_random_state=python_state,
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_states=cuda_states,
    )


def seed_training_rng(seed: int) -> None:
    """Seed Python and torch RNGs for a fresh training run."""
    assert seed >= 0
    random.seed(seed)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    torch.set_rng_state(cpu_generator.get_state())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def restore_torch_rng_state(
    state: TorchRngState,
) -> _result.Ok[None] | _result.Rejected:
    """Restore Python and torch RNG states from a checkpoint."""
    validation = _validate_torch_rng_state(state)
    if isinstance(validation, _result.Rejected):
        return validation
    random.setstate(state.python_random_state)
    torch.set_rng_state(state.torch_cpu_state)
    if state.torch_cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(state.torch_cuda_states))
    return _result.Ok(value=None)


def _validate_torch_rng_state(
    state: TorchRngState,
) -> _result.Ok[None] | _result.Rejected:
    python_validation = _validate_python_random_state(
        state.python_random_state
    )
    if isinstance(python_validation, _result.Rejected):
        return python_validation
    cpu_validation = _validate_cpu_rng_state(state.torch_cpu_state)
    if isinstance(cpu_validation, _result.Rejected):
        return cpu_validation
    cuda_validation = _validate_cuda_rng_states(state.torch_cuda_states)
    if isinstance(cuda_validation, _result.Rejected):
        return cuda_validation
    return _result.Ok(value=None)


def _validate_python_random_state(
    state: PythonRandomState,
) -> _result.Ok[None] | _result.Rejected:
    probe = random.Random()
    try:
        probe.setstate(state)
    except TypeError, ValueError:
        return _result.Rejected(
            reason="rng_state.python_random_state is invalid"
        )
    return _result.Ok(value=None)


def _validate_cpu_rng_state(
    state: Tensor,
) -> _result.Ok[None] | _result.Rejected:
    if state.device.type != "cpu":
        return _result.Rejected(
            reason="rng_state.torch_cpu_state must be on cpu"
        )
    generator = torch.Generator(device="cpu")
    try:
        generator.set_state(state)
    except RuntimeError, TypeError:
        return _result.Rejected(
            reason="rng_state.torch_cpu_state is invalid"
        )
    return _result.Ok(value=None)


def _validate_cuda_rng_states(
    states: tuple[Tensor, ...],
) -> _result.Ok[None] | _result.Rejected:
    for state in states:
        if state.device.type != "cpu":
            return _result.Rejected(
                reason="rng_state.torch_cuda_states must be on cpu"
            )
        if state.dtype != torch.uint8 or state.ndim != 1:
            return _result.Rejected(
                reason="rng_state.torch_cuda_states is invalid"
            )
    if not states or not torch.cuda.is_available():
        return _result.Ok(value=None)
    current_states = tuple(torch.cuda.get_rng_state_all())
    try:
        torch.cuda.set_rng_state_all(list(states))
    except RuntimeError, TypeError:
        return _result.Rejected(
            reason="rng_state.torch_cuda_states is invalid"
        )
    finally:
        torch.cuda.set_rng_state_all(list(current_states))
    return _result.Ok(value=None)


def _is_python_random_state(
    value: object,
) -> TypeGuard[PythonRandomState]:
    if not isinstance(value, tuple):
        return False
    items = cast(tuple[object, ...], value)
    if len(items) != 3:
        return False
    version, internal_state, gaussian = items
    return (
        isinstance(version, int)
        and _is_int_tuple(internal_state)
        and (gaussian is None or isinstance(gaussian, float))
    )


def _is_int_tuple(value: object) -> TypeGuard[tuple[int, ...]]:
    if not isinstance(value, tuple):
        return False
    items = cast(tuple[object, ...], value)
    return all(isinstance(item, int) for item in items)
