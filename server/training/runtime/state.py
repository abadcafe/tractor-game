"""Training state snapshots for process synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

import torch
from torch import Tensor

from server import result as _result
from server.training.model import TractorPolicyModel
from server.training.ppo import PPOTrainer
from server.training.training_state import LoadedTrainingState

type ModelTensorState = dict[str, Tensor]
type OptimizerPayload = dict[str, object]


@dataclass(frozen=True, slots=True)
class RuntimeTrainingState:
    """CPU-resident model and optimizer state sent between processes."""

    model_state: ModelTensorState
    optimizer_state: OptimizerPayload


def capture_runtime_training_state(
    *,
    model: TractorPolicyModel,
    trainer: PPOTrainer,
) -> RuntimeTrainingState:
    """Capture a CPU snapshot of model and trainer state."""
    model_state: Mapping[str, Tensor] = model.state_dict()
    return RuntimeTrainingState(
        model_state=_tensor_state_to_cpu(model_state),
        optimizer_state=_optimizer_state_to_cpu(
            trainer.optimizer_state()
        ),
    )


def load_runtime_training_state(
    *,
    state: LoadedTrainingState,
    snapshot: RuntimeTrainingState,
) -> None:
    """Load a runtime snapshot into an already constructed state."""
    state.model.load_state_dict(snapshot.model_state)
    state.trainer.load_optimizer_state(snapshot.optimizer_state)


def select_canonical_runtime_training_state(
    snapshots: tuple[RuntimeTrainingState, ...],
) -> _result.Ok[RuntimeTrainingState] | _result.Rejected:
    """Return rank-zero state after rank consistency checks."""
    assert snapshots
    reference = snapshots[0]
    for index, snapshot in enumerate(snapshots[1:], start=1):
        match_result = _runtime_training_states_match(
            reference=reference,
            snapshot=snapshot,
            rank=index,
        )
        if isinstance(match_result, _result.Rejected):
            return match_result
    return _result.Ok(value=reference)


def _tensor_state_to_cpu(
    state: Mapping[str, Tensor],
) -> ModelTensorState:
    return {
        key: value.detach().to(device=torch.device("cpu")).clone()
        for key, value in state.items()
    }


def _optimizer_state_to_cpu(
    state: OptimizerPayload,
) -> OptimizerPayload:
    return {
        "kind": state["kind"],
        "step_count": state["step_count"],
        "exp_avgs": _optimizer_tensor_list_to_cpu(state["exp_avgs"]),
        "exp_avg_sqs": _optimizer_tensor_list_to_cpu(
            state["exp_avg_sqs"]
        ),
    }


def _optimizer_tensor_list_to_cpu(value: object) -> list[Tensor | None]:
    assert isinstance(value, list)
    items = cast(list[object], value)
    result: list[Tensor | None] = []
    for item in items:
        assert item is None or isinstance(item, Tensor)
        if item is None:
            result.append(None)
        else:
            result.append(
                item.detach().to(device=torch.device("cpu")).clone()
            )
    return result


def _runtime_training_states_match(
    *,
    reference: RuntimeTrainingState,
    snapshot: RuntimeTrainingState,
    rank: int,
) -> _result.Ok[None] | _result.Rejected:
    model_result = _model_states_match(
        reference=reference.model_state,
        snapshot=snapshot.model_state,
        rank=rank,
    )
    if isinstance(model_result, _result.Rejected):
        return model_result
    optimizer_result = _optimizer_states_match(
        reference=reference.optimizer_state,
        snapshot=snapshot.optimizer_state,
        rank=rank,
    )
    if isinstance(optimizer_result, _result.Rejected):
        return optimizer_result
    return _result.Ok(value=None)


def _model_states_match(
    *,
    reference: ModelTensorState,
    snapshot: ModelTensorState,
    rank: int,
) -> _result.Ok[None] | _result.Rejected:
    if tuple(reference.keys()) != tuple(snapshot.keys()):
        return _result.Rejected(
            reason=f"rank-{rank} model state keys differ"
        )
    for key, value in reference.items():
        other = snapshot[key]
        if value.shape != other.shape:
            return _result.Rejected(
                reason=f"rank-{rank} model state {key} shape differs"
            )
        if value.dtype != other.dtype:
            return _result.Rejected(
                reason=f"rank-{rank} model state {key} dtype differs"
            )
        if not bool(torch.equal(value, other)):
            return _result.Rejected(
                reason=f"rank-{rank} model state {key} value differs"
            )
    return _result.Ok(value=None)


def _optimizer_states_match(
    *,
    reference: OptimizerPayload,
    snapshot: OptimizerPayload,
    rank: int,
) -> _result.Ok[None] | _result.Rejected:
    if tuple(reference.keys()) != tuple(snapshot.keys()):
        return _result.Rejected(
            reason=f"rank-{rank} optimizer state keys differ"
        )
    for key, value in reference.items():
        if not _optimizer_values_equal(value, snapshot[key]):
            return _result.Rejected(
                reason=f"rank-{rank} optimizer state {key} differs"
            )
    return _result.Ok(value=None)


def _optimizer_values_equal(left: object, right: object) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, list) and isinstance(right, list):
        left_items = cast(list[object], left)
        right_items = cast(list[object], right)
        if len(left_items) != len(right_items):
            return False
        return all(
            _optimizer_values_equal(left_item, right_item)
            for left_item, right_item in zip(
                left_items, right_items, strict=True
            )
        )
    return left == right
