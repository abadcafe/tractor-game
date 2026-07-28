"""Black-box tests for fixed-vocabulary device action sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from server.foundation.result import Ok, Rejected
from server.policy_model._schema.actions import (
    ACTION_CHOICE_COUNT,
    CARD_CHOICE_BASE_ID,
    FINISH_CHOICE_ID,
    PASS_CHOICE_ID,
)
from server.policy_model.actions.decoding import (
    ActionChoiceLogitDecoder,
    ActionPlanFrame,
    ActionSampleBatch,
    ActionSampler,
    ActionScoreBatch,
    action_plan_generation_step_count,
    plan_batch_to_device,
)
from server.policy_model.actions.decoding.frame import (
    ACTION_KIND_DISCARD,
    ACTION_KIND_LEAD,
    ACTION_KIND_TRACE_SET,
)
from server.policy_model.actions.decoding.spec import ACTION_FACE_COUNT


@dataclass(slots=True)
class _ZeroDecoder(ActionChoiceLogitDecoder):
    batch_size: int
    device: torch.device

    def next_choice_logits(
        self,
        active_rows: Tensor,
        scored_rows: Tensor,
    ) -> Tensor:
        assert active_rows.shape == (self.batch_size,)
        assert scored_rows.shape == active_rows.shape
        return torch.zeros(
            (self.batch_size, ACTION_CHOICE_COUNT),
            dtype=torch.float32,
            device=self.device,
        )

    def advance(
        self,
        selected_choice_ids: Tensor,
        active_rows: Tensor,
    ) -> None:
        assert selected_choice_ids.shape == (self.batch_size,)
        assert active_rows.shape == (self.batch_size,)


@dataclass(slots=True)
class _RecordingDecoder(ActionChoiceLogitDecoder):
    batch_size: int
    device: torch.device
    preferred_choice: int
    scored_counts: list[int]

    def next_choice_logits(
        self,
        active_rows: Tensor,
        scored_rows: Tensor,
    ) -> Tensor:
        assert active_rows.shape == (self.batch_size,)
        assert scored_rows.shape == active_rows.shape
        self.scored_counts.append(
            int(scored_rows.count_nonzero().item())
        )
        logits = torch.zeros(
            (self.batch_size, ACTION_CHOICE_COUNT),
            dtype=torch.float32,
            device=self.device,
        )
        logits[:, self.preferred_choice] = 100.0
        return logits

    def advance(
        self,
        selected_choice_ids: Tensor,
        active_rows: Tensor,
    ) -> None:
        assert selected_choice_ids.shape == (self.batch_size,)
        assert active_rows.shape == (self.batch_size,)


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


def test_forced_prefix_is_not_scored_before_ambiguous_choice() -> None:
    device = torch.device("cpu")
    first = CARD_CHOICE_BASE_ID
    second = CARD_CHOICE_BASE_ID + 1
    preferred = CARD_CHOICE_BASE_ID + 2
    frame = _frame(
        kind=ACTION_KIND_TRACE_SET,
        traces=((first, second), (first, preferred)),
    )
    decoder = _RecordingDecoder(
        batch_size=1,
        device=device,
        preferred_choice=preferred,
        scored_counts=[],
    )

    result = ActionSampler.create(
        batch_capacity=1,
        device=device,
    ).sample(
        action_batch=plan_batch_to_device((frame,), device=device),
        generation_step_counts=torch.tensor(
            (2,), dtype=torch.long, device=device
        ),
        sampling_thresholds=torch.tensor(
            ((0.5, 0.5),), dtype=torch.float64, device=device
        ),
        padded_generation_steps=2,
        logit_decoder=decoder,
    )

    assert isinstance(result, Ok)
    assert _trace(result.value) == (first, preferred)
    assert decoder.scored_counts == [0, 1]


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


def test_score_uses_the_same_exact_legal_mask_as_sampling() -> None:
    scored = _score(
        _frame(
            kind=ACTION_KIND_TRACE_SET,
            traces=((PASS_CHOICE_ID,), (CARD_CHOICE_BASE_ID,)),
        ),
        trace=(CARD_CHOICE_BASE_ID,),
    )

    assert math.isclose(
        float(scored.log_probabilities[0].item()),
        -math.log(2.0),
        rel_tol=1e-6,
    )
    assert int(scored.choice_counts[0].item()) == 2


def test_score_rejects_a_choice_outside_the_legal_trace_set() -> None:
    device = torch.device("cpu")
    frame = _frame(
        kind=ACTION_KIND_TRACE_SET,
        traces=((PASS_CHOICE_ID,),),
    )

    result = ActionSampler.create(
        batch_capacity=1,
        device=device,
    ).score(
        action_batch=plan_batch_to_device((frame,), device=device),
        selected_choice_ids=torch.tensor(
            ((FINISH_CHOICE_ID,),),
            dtype=torch.long,
            device=device,
        ),
        step_counts=torch.tensor(
            (1,),
            dtype=torch.long,
            device=device,
        ),
        padded_generation_steps=1,
        logit_decoder=_ZeroDecoder(batch_size=1, device=device),
    )

    assert isinstance(result, Rejected)


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


def _score(
    frame: ActionPlanFrame,
    *,
    trace: tuple[int, ...],
) -> ActionScoreBatch:
    device = torch.device("cpu")
    steps = action_plan_generation_step_count(frame)
    result = ActionSampler.create(
        batch_capacity=1,
        device=device,
    ).score(
        action_batch=plan_batch_to_device((frame,), device=device),
        selected_choice_ids=torch.tensor(
            (trace + (0,) * (steps - len(trace)),),
            dtype=torch.long,
            device=device,
        ),
        step_counts=torch.tensor(
            (len(trace),),
            dtype=torch.long,
            device=device,
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
