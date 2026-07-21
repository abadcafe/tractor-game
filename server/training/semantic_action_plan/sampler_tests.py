"""Black-box tests for fixed-vocabulary device action sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.foundation.result import Ok
from server.training.semantic_action_plan import (
    ActionChoiceLogitDecoder,
    ActionPlanFrame,
    ActionSampleBatch,
    ActionSampler,
    action_plan_generation_step_count,
    plan_batch_to_device,
)
from server.training.semantic_action_plan.frame import (
    ACTION_KIND_DISCARD,
    ACTION_KIND_LEAD,
    ACTION_KIND_TRACE_SET,
)
from server.training.semantic_action_plan.spec import ACTION_FACE_COUNT
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    CARD_CHOICE_BASE_ID,
    FINISH_CHOICE_ID,
    PASS_CHOICE_ID,
)


@dataclass(slots=True)
class _ZeroDecoder(ActionChoiceLogitDecoder):
    batch_size: int
    device: torch.device

    def next_choice_logits(self) -> Tensor:
        return torch.zeros(
            (self.batch_size, ACTION_CHOICE_COUNT),
            dtype=torch.float32,
            device=self.device,
        )

    def advance(self, selected_choice_ids: Tensor) -> None:
        assert selected_choice_ids.shape == (self.batch_size,)


def test_trace_set_samples_direct_choice_ids() -> None:
    sample = _sample(
        _frame(
            kind=ACTION_KIND_TRACE_SET,
            traces=((PASS_CHOICE_ID,), (CARD_CHOICE_BASE_ID,)),
        ),
        thresholds=(0.75,),
    )

    assert _trace(sample) == (CARD_CHOICE_BASE_ID,)
    assert sample.legal_choice_masks.shape == (1, ACTION_CHOICE_COUNT)
    assert bool(sample.legal_choice_masks[0, PASS_CHOICE_ID])
    assert bool(sample.legal_choice_masks[0, CARD_CHOICE_BASE_ID])


def test_lead_uses_card_then_explicit_finish() -> None:
    available = [0 for _ in range(ACTION_FACE_COUNT)]
    available[0] = 1
    sample = _sample(
        _frame(
            kind=ACTION_KIND_LEAD,
            available=tuple(available),
            min_select=1,
            max_select=1,
        ),
        thresholds=(0.0, 0.0),
    )

    assert _trace(sample) == (
        CARD_CHOICE_BASE_ID,
        FINISH_CHOICE_ID,
    )
    assert not bool(sample.legal_choice_masks[0, FINISH_CHOICE_ID])
    assert bool(sample.legal_choice_masks[1, FINISH_CHOICE_ID])


def test_exact_selection_terminates_without_finish() -> None:
    available = [0 for _ in range(ACTION_FACE_COUNT)]
    available[0] = 1
    sample = _sample(
        _frame(
            kind=ACTION_KIND_DISCARD,
            available=tuple(available),
            min_select=1,
            max_select=1,
            exact_select=1,
        ),
        thresholds=(0.0,),
    )

    assert _trace(sample) == (CARD_CHOICE_BASE_ID,)
    assert _tensor_values(sample.active_sample_indices) == (0,)
    assert _tensor_values(sample.active_step_indices) == (0,)


def _sample(
    frame: ActionPlanFrame, *, thresholds: tuple[float, ...]
) -> ActionSampleBatch:
    device = torch.device("cpu")
    steps = action_plan_generation_step_count(frame)
    assert steps == len(thresholds)
    result = ActionSampler.create(
        batch_capacity=1, device=device
    ).sample(
        action_batch=plan_batch_to_device((frame,), device=device),
        generation_step_counts=torch.tensor(
            (steps,), dtype=torch.long, device=device
        ),
        sampling_thresholds=torch.tensor(
            (thresholds,), dtype=torch.float64, device=device
        ),
        padded_generation_steps=steps,
        logit_decoder=_ZeroDecoder(batch_size=1, device=device),
    )
    assert isinstance(result, Ok)
    return result.value


def _trace(sample: ActionSampleBatch) -> tuple[int, ...]:
    count = int(sample.step_counts[0].item())
    return tuple(
        int(sample.choice_ids_padded[0, index].item())
        for index in range(count)
    )


def _tensor_values(values: Tensor) -> tuple[int, ...]:
    return tuple(
        int(values[index].item()) for index in range(values.numel())
    )


def _frame(
    *,
    kind: int,
    available: tuple[int, ...] | None = None,
    min_select: int = 0,
    max_select: int = 0,
    exact_select: int = -1,
    traces: tuple[tuple[int, ...], ...] = (),
) -> ActionPlanFrame:
    empty_int = tuple(-1 for _ in range(ACTION_FACE_COUNT))
    empty_bool = tuple(False for _ in range(ACTION_FACE_COUNT))
    return ActionPlanFrame(
        kind_code=kind,
        available_counts=(
            tuple(0 for _ in range(ACTION_FACE_COUNT))
            if available is None
            else available
        ),
        effective_suits=empty_int,
        same_suit_mask=empty_bool,
        off_suit_mask=empty_bool,
        pair_face_mask=empty_bool,
        min_select=min_select,
        max_select=max_select,
        exact_select=exact_select,
        required_same_suit_count=0,
        pair_floor=0,
        has_tractor=False,
        trace_choice_ids=traces,
        pair_plan_masks=(),
    )
