"""Black-box tests for compact policy decision materialization."""

from __future__ import annotations

from typing import Never

import pytest
import torch
from torch import Tensor

from server.foundation.result import Ok, Rejected
from server.policy_model.actions.decoding import (
    ACTION_SAMPLING_UNTERMINATED,
)
from server.training.rollout_inference.samples.materializer import (
    PolicyDecisionMaterializer,
)


def _available_training_devices() -> tuple[torch.device, ...]:
    devices: list[torch.device] = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return tuple(devices)


def _reject_tensor_item(_tensor: Tensor) -> Never:
    raise AssertionError(
        "decision materialization extracted a tensor scalar"
    )


@pytest.mark.parametrize("device", _available_training_devices())
def test_materializer_keeps_trace_and_legal_choice_counts_distinct(
    device: torch.device,
) -> None:
    materializer = PolicyDecisionMaterializer(device=device)

    decision = materializer.materialize(
        model_rank_index=2,
        policy_versions=(7, 7, 7),
        row_start=4,
        choice_ids_padded=torch.tensor(
            ((10, 0, 0), (20, 21, 22), (30, 31, 0)),
            dtype=torch.long,
            device=device,
        ),
        step_counts=torch.tensor(
            (1, 3, 2), dtype=torch.long, device=device
        ),
        choice_counts=torch.tensor(
            (2, 5, 4), dtype=torch.long, device=device
        ),
        scored_choice_step_counts=torch.tensor(
            (1, 2, 1), dtype=torch.long, device=device
        ),
        error_code=torch.zeros((), dtype=torch.long, device=device),
    )

    assert isinstance(decision, Ok)
    materialized = decision.value
    assert materialized.model_rank_index == 2
    assert materialized.policy_versions == (7, 7, 7)
    assert materialized.row_indices == (4, 5, 6)
    assert materialized.choice_counts == (2, 5, 4)
    assert materialized.scored_choice_step_counts == (1, 2, 1)
    assert materialized.action_choice_batch.choice_counts == (1, 3, 2)
    assert tuple(
        materialized.action_choice_batch.compact_row(index).to_tuple()
        for index in range(3)
    ) == ((10,), (20, 21, 22), (30, 31))


def test_materializer_does_not_extract_tensor_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.Tensor, "item", _reject_tensor_item)
    materializer = PolicyDecisionMaterializer(
        device=torch.device("cpu")
    )

    decision = materializer.materialize(
        model_rank_index=0,
        policy_versions=(1, 1),
        row_start=0,
        choice_ids_padded=torch.tensor(
            ((10, 11), (20, 0)), dtype=torch.long
        ),
        step_counts=torch.tensor((2, 1), dtype=torch.long),
        choice_counts=torch.tensor((3, 4), dtype=torch.long),
        scored_choice_step_counts=torch.tensor(
            (1, 1), dtype=torch.long
        ),
        error_code=torch.zeros((), dtype=torch.long),
    )

    assert isinstance(decision, Ok)
    assert decision.value.choice_counts == (3, 4)


@pytest.mark.parametrize("device", _available_training_devices())
def test_materializer_reuses_workspace_across_batch_shapes(
    device: torch.device,
) -> None:
    materializer = PolicyDecisionMaterializer(device=device)
    _ = materializer.materialize(
        model_rank_index=0,
        policy_versions=(3, 3),
        row_start=0,
        choice_ids_padded=torch.tensor(
            ((40, 41, 42, 43), (50, 51, 52, 53)),
            dtype=torch.long,
            device=device,
        ),
        step_counts=torch.tensor(
            (4, 4), dtype=torch.long, device=device
        ),
        choice_counts=torch.tensor(
            (6, 7), dtype=torch.long, device=device
        ),
        scored_choice_step_counts=torch.tensor(
            (2, 2), dtype=torch.long, device=device
        ),
        error_code=torch.zeros((), dtype=torch.long, device=device),
    )

    decision = materializer.materialize(
        model_rank_index=0,
        policy_versions=(4,),
        row_start=9,
        choice_ids_padded=torch.tensor(
            ((60, 61),),
            dtype=torch.long,
            device=device,
        ),
        step_counts=torch.tensor((2,), dtype=torch.long, device=device),
        choice_counts=torch.tensor(
            (8,), dtype=torch.long, device=device
        ),
        scored_choice_step_counts=torch.tensor(
            (1,), dtype=torch.long, device=device
        ),
        error_code=torch.zeros((), dtype=torch.long, device=device),
    )

    assert isinstance(decision, Ok)
    materialized = decision.value
    assert materialized.policy_versions == (4,)
    assert materialized.row_indices == (9,)
    assert materialized.choice_counts == (8,)
    assert materialized.action_choice_batch.compact_row(
        0
    ).to_tuple() == (
        60,
        61,
    )


def test_materializer_rejects_device_sampling_error() -> None:
    materializer = PolicyDecisionMaterializer(
        device=torch.device("cpu")
    )

    result = materializer.materialize(
        model_rank_index=0,
        policy_versions=(1,),
        row_start=0,
        choice_ids_padded=torch.tensor(((10,),), dtype=torch.long),
        step_counts=torch.tensor((1,), dtype=torch.long),
        choice_counts=torch.tensor((2,), dtype=torch.long),
        scored_choice_step_counts=torch.tensor((1,), dtype=torch.long),
        error_code=torch.tensor(
            ACTION_SAMPLING_UNTERMINATED, dtype=torch.long
        ),
    )

    assert isinstance(result, Rejected)
    assert result.reason == "policy action did not terminate"
