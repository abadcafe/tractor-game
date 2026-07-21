"""Device execution for compiled semantic action plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_action_plan.frame import (
    ACTION_KIND_DISCARD,
    ACTION_KIND_EMPTY,
    ACTION_KIND_FOLLOW,
    ACTION_KIND_LEAD,
    ACTION_KIND_TRACE_SET,
    ActionPlanFrame,
)
from server.training.semantic_action_plan.spec import (
    ACTION_FACE_COUNT,
)
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    CARD_CHOICE_BASE_ID,
    FINISH_CHOICE_ID,
)
from server.training.tensor_staging import staged_tensor

_KIND_EMPTY = ACTION_KIND_EMPTY
_KIND_TRACE_SET = ACTION_KIND_TRACE_SET
_KIND_DISCARD = ACTION_KIND_DISCARD
_KIND_LEAD = ACTION_KIND_LEAD
_KIND_FOLLOW = ACTION_KIND_FOLLOW


@dataclass(frozen=True, slots=True)
class DeviceActionPlanBatch:
    """Batched compiled action specs resident on one torch device."""

    kind_codes: Tensor
    available_counts: Tensor
    effective_suits: Tensor
    same_suit_mask: Tensor
    off_suit_mask: Tensor
    pair_face_mask: Tensor
    min_select: Tensor
    max_select: Tensor
    exact_select: Tensor
    required_same_suit_count: Tensor
    pair_floor: Tensor
    has_tractor: Tensor
    trace_choice_ids: Tensor
    trace_choice_mask: Tensor
    trace_lengths: Tensor
    trace_row_mask: Tensor
    pair_plan_masks: Tensor
    pair_plan_row_mask: Tensor

    def __post_init__(self) -> None:
        batch_size = int(self.kind_codes.shape[0])
        assert batch_size > 0
        assert self.kind_codes.ndim == 1
        assert self.available_counts.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )
        assert self.effective_suits.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )
        assert self.same_suit_mask.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )
        assert self.off_suit_mask.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )
        assert self.pair_face_mask.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )

    def batch_size(self) -> int:
        """Return the number of action specs."""
        return int(self.kind_codes.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the torch device hosting this batch."""
        return self.kind_codes.device


@dataclass(frozen=True, slots=True)
class DeviceActionState:
    """Current action-choice generation state on one torch device."""

    selected_counts: Tensor
    selected_choice_ids: Tensor
    step_counts: Tensor
    selected_width: Tensor
    last_face_indices: Tensor
    selected_suit_codes: Tensor
    done: Tensor
    choice_counts: Tensor

    def __post_init__(self) -> None:
        batch_size = int(self.selected_counts.shape[0])
        assert batch_size > 0
        assert self.selected_counts.shape == (
            batch_size,
            ACTION_FACE_COUNT,
        )
        assert self.selected_choice_ids.ndim == 2
        assert int(self.selected_choice_ids.shape[0]) == batch_size
        assert self.step_counts.shape == (batch_size,)
        assert self.selected_width.shape == (batch_size,)
        assert self.last_face_indices.shape == (batch_size,)
        assert self.selected_suit_codes.shape == (batch_size,)
        assert self.done.shape == (batch_size,)
        assert self.choice_counts.shape == (batch_size,)


def plan_batch_to_device(
    specs: tuple[ActionPlanFrame, ...],
    *,
    device: torch.device,
) -> DeviceActionPlanBatch:
    """Pack compiled specs into device tensors."""
    assert specs
    max_trace_count = max(_trace_count(spec) for spec in specs)
    max_trace_steps = max(_trace_steps(spec) for spec in specs)
    max_pair_plan_count = max(_pair_plan_count(spec) for spec in specs)
    return DeviceActionPlanBatch(
        kind_codes=staged_tensor(
            tuple(_kind_code(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        available_counts=staged_tensor(
            tuple(_available_counts(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        effective_suits=staged_tensor(
            tuple(_effective_suits(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        same_suit_mask=staged_tensor(
            tuple(_same_suit_mask(spec) for spec in specs),
            dtype=torch.bool,
            device=device,
        ),
        off_suit_mask=staged_tensor(
            tuple(_off_suit_mask(spec) for spec in specs),
            dtype=torch.bool,
            device=device,
        ),
        pair_face_mask=staged_tensor(
            tuple(_pair_face_mask(spec) for spec in specs),
            dtype=torch.bool,
            device=device,
        ),
        min_select=staged_tensor(
            tuple(_min_select(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        max_select=staged_tensor(
            tuple(_max_select(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        exact_select=staged_tensor(
            tuple(_exact_select(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        required_same_suit_count=staged_tensor(
            tuple(_required_same_suit_count(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        pair_floor=staged_tensor(
            tuple(_pair_floor(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        has_tractor=staged_tensor(
            tuple(_has_tractor(spec) for spec in specs),
            dtype=torch.bool,
            device=device,
        ),
        trace_choice_ids=staged_tensor(
            tuple(
                _padded_trace_choice_ids(
                    spec,
                    max_trace_count=max_trace_count,
                    max_trace_steps=max_trace_steps,
                )
                for spec in specs
            ),
            dtype=torch.long,
            device=device,
        ),
        trace_choice_mask=staged_tensor(
            tuple(
                _padded_trace_choice_mask(
                    spec,
                    max_trace_count=max_trace_count,
                    max_trace_steps=max_trace_steps,
                )
                for spec in specs
            ),
            dtype=torch.bool,
            device=device,
        ),
        trace_lengths=staged_tensor(
            tuple(
                _padded_trace_lengths(
                    spec, max_trace_count=max_trace_count
                )
                for spec in specs
            ),
            dtype=torch.long,
            device=device,
        ),
        trace_row_mask=staged_tensor(
            tuple(
                _padded_trace_row_mask(
                    spec, max_trace_count=max_trace_count
                )
                for spec in specs
            ),
            dtype=torch.bool,
            device=device,
        ),
        pair_plan_masks=staged_tensor(
            tuple(
                _padded_pair_plan_masks(
                    spec, max_pair_plan_count=max_pair_plan_count
                )
                for spec in specs
            ),
            dtype=torch.bool,
            device=device,
        ),
        pair_plan_row_mask=staged_tensor(
            tuple(
                _padded_pair_plan_row_mask(
                    spec, max_pair_plan_count=max_pair_plan_count
                )
                for spec in specs
            ),
            dtype=torch.bool,
            device=device,
        ),
    )


def pad_candidate_columns(values: Tensor, *, width: int) -> Tensor:
    assert values.ndim == 2
    current_width = int(values.shape[1])
    assert width >= current_width
    if current_width == width:
        return values
    padding = torch.zeros(
        (int(values.shape[0]), width - current_width),
        dtype=values.dtype,
        device=values.device,
    )
    return torch.cat((values, padding), dim=1)


@dataclass(frozen=True, slots=True)
class _ChoiceTables:
    is_card: Tensor
    face_indices: Tensor
    counts: Tensor
    card_choice_ids: Tensor
    card_face_indices: Tensor
    card_counts: Tensor


_CHOICE_TABLE_CACHE: dict[torch.device, _ChoiceTables] = {}


def choice_tables(device: torch.device) -> _ChoiceTables:
    cached = _CHOICE_TABLE_CACHE.get(device)
    if cached is not None:
        return cached
    face_indices = [-1 for _ in range(ACTION_CHOICE_COUNT)]
    counts = [0 for _ in range(ACTION_CHOICE_COUNT)]
    is_card = [False for _ in range(ACTION_CHOICE_COUNT)]
    card_choice_ids: list[int] = []
    card_face_indices: list[int] = []
    card_counts: list[int] = []
    for face_index in range(ACTION_FACE_COUNT):
        for count in (1, 2):
            choice_id = CARD_CHOICE_BASE_ID + face_index * 2 + count - 1
            face_indices[choice_id] = face_index
            counts[choice_id] = count
            is_card[choice_id] = True
            card_choice_ids.append(choice_id)
            card_face_indices.append(face_index)
            card_counts.append(count)
    tables = _ChoiceTables(
        is_card=staged_tensor(is_card, dtype=torch.bool, device=device),
        face_indices=staged_tensor(
            face_indices, dtype=torch.long, device=device
        ),
        counts=staged_tensor(counts, dtype=torch.long, device=device),
        card_choice_ids=staged_tensor(
            card_choice_ids, dtype=torch.long, device=device
        ),
        card_face_indices=staged_tensor(
            card_face_indices, dtype=torch.long, device=device
        ),
        card_counts=staged_tensor(
            card_counts, dtype=torch.long, device=device
        ),
    )
    _CHOICE_TABLE_CACHE[device] = tables
    return tables


def trace_set_candidates(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
) -> tuple[Tensor, Tensor]:
    batch_size = batch.batch_size()
    trace_count = int(batch.trace_choice_ids.shape[1])
    step_count = int(batch.trace_choice_ids.shape[2])
    if trace_count == 0:
        return (
            torch.zeros(
                (batch_size, 0), dtype=torch.long, device=batch.device
            ),
            torch.zeros(
                (batch_size, 0), dtype=torch.bool, device=batch.device
            ),
        )
    positions = torch.arange(
        step_count, dtype=torch.long, device=batch.device
    ).view(1, 1, step_count)
    current_steps = state.step_counts.view(batch_size, 1, 1)
    prefix_mask = positions < current_steps
    selected = state.selected_choice_ids[:, :step_count].unsqueeze(1)
    prefix_matches = (
        (batch.trace_choice_ids == selected) | ~prefix_mask
    ).all(dim=2)
    has_next = batch.trace_lengths > state.step_counts.unsqueeze(1)
    valid_traces = (
        (batch.kind_codes == _KIND_TRACE_SET).unsqueeze(1)
        & batch.trace_row_mask
        & prefix_matches
        & has_next
        & ~state.done.unsqueeze(1)
    )
    gather_index = (
        state.step_counts.clamp(max=step_count - 1)
        .view(batch_size, 1, 1)
        .expand(batch_size, trace_count, 1)
    )
    next_choices = batch.trace_choice_ids.gather(
        dim=2, index=gather_index
    ).squeeze(2)
    sentinel = torch.full_like(next_choices, ACTION_CHOICE_COUNT)
    candidates = torch.where(valid_traces, next_choices, sentinel)
    sorted_candidates = torch.sort(candidates, dim=1).values
    previous = torch.cat(
        (
            torch.full(
                (batch_size, 1),
                -1,
                dtype=torch.long,
                device=batch.device,
            ),
            sorted_candidates[:, :-1],
        ),
        dim=1,
    )
    unique_valid = (sorted_candidates != sentinel) & (
        sorted_candidates != previous
    )
    deduplicated = torch.where(
        unique_valid, sorted_candidates, sentinel
    )
    packed = torch.sort(deduplicated, dim=1).values
    packed_mask = packed != sentinel
    return torch.where(
        packed_mask, packed, torch.zeros_like(packed)
    ), packed_mask


def selection_choice_candidates(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
) -> tuple[Tensor, Tensor]:
    tables = choice_tables(batch.device)
    batch_size = batch.batch_size()
    face_indices = tables.card_face_indices.unsqueeze(0).expand(
        batch_size, -1
    )
    counts = tables.card_counts.unsqueeze(0).expand(batch_size, -1)
    available = batch.available_counts.gather(
        dim=1,
        index=face_indices,
    )
    face_suits = batch.effective_suits.gather(
        dim=1,
        index=face_indices,
    )
    already_selected = (
        state.selected_counts.gather(dim=1, index=face_indices) > 0
    )
    new_width = state.selected_width.unsqueeze(1) + counts
    generic_select = (
        torch.ones_like(counts, dtype=torch.bool)
        & ~already_selected
        & (counts <= available)
        & (face_indices > state.last_face_indices.unsqueeze(1))
        & (new_width <= batch.max_select.unsqueeze(1))
        & ~state.done.unsqueeze(1)
    )
    discard = (
        (batch.kind_codes == _KIND_DISCARD).unsqueeze(1)
        & generic_select
        & _has_remaining_capacity(
            batch=batch,
            state=state,
            candidate_faces=face_indices,
            candidate_widths=counts,
            target_width=batch.exact_select,
            allowed_face_mask=torch.ones_like(
                batch.available_counts, dtype=torch.bool
            ),
        )
    )
    lead_same_suit = (state.selected_suit_codes.unsqueeze(1) < 0) | (
        face_suits == state.selected_suit_codes.unsqueeze(1)
    )
    lead_select = (
        (batch.kind_codes == _KIND_LEAD).unsqueeze(1)
        & generic_select
        & lead_same_suit
    )
    lead_finish = (
        (batch.kind_codes == _KIND_LEAD)
        & (state.selected_width >= batch.min_select)
        & ~state.done
    )
    follow_select = (
        (batch.kind_codes == _KIND_FOLLOW).unsqueeze(1)
        & generic_select
        & _follow_can_complete_mask(
            batch=batch,
            state=state,
            candidate_faces=face_indices,
            candidate_counts=counts,
        )
    )
    selection_mask = discard | lead_select | follow_select
    finish_ids = torch.full(
        (batch_size, 1),
        FINISH_CHOICE_ID,
        dtype=torch.long,
        device=batch.device,
    )
    card_choice_ids = tables.card_choice_ids.unsqueeze(0).expand(
        batch_size, -1
    )
    return (
        torch.cat((finish_ids, card_choice_ids), dim=1),
        torch.cat((lead_finish.unsqueeze(1), selection_mask), dim=1),
    )


def _has_remaining_capacity(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    candidate_faces: Tensor,
    candidate_widths: Tensor,
    target_width: Tensor,
    allowed_face_mask: Tensor,
) -> Tensor:
    face_positions = torch.arange(
        ACTION_FACE_COUNT, dtype=torch.long, device=batch.device
    ).view(1, 1, ACTION_FACE_COUNT)
    candidate_face_positions = candidate_faces.unsqueeze(2)
    unselected = state.selected_counts.unsqueeze(1) == 0
    after_candidate = face_positions > candidate_face_positions
    available = batch.available_counts.unsqueeze(1)
    allowed = allowed_face_mask.unsqueeze(1)
    remaining_capacity = torch.where(
        unselected & after_candidate & allowed,
        available,
        torch.zeros_like(available),
    ).sum(dim=2)
    new_width = state.selected_width.unsqueeze(1) + candidate_widths
    required_remaining = target_width.unsqueeze(1) - new_width
    return remaining_capacity >= required_remaining


def _follow_can_complete_mask(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    candidate_faces: Tensor,
    candidate_counts: Tensor,
) -> Tensor:
    candidate_same = _safe_gather_bool_2d(
        batch.same_suit_mask, candidate_faces.clamp(min=0)
    )
    candidate_off = _safe_gather_bool_2d(
        batch.off_suit_mask, candidate_faces.clamp(min=0)
    )
    same_width = state.selected_counts.masked_fill(
        ~batch.same_suit_mask, 0
    ).sum(dim=1).unsqueeze(1) + torch.where(
        candidate_same, candidate_counts, 0
    )
    off_width = state.selected_counts.masked_fill(
        ~batch.off_suit_mask, 0
    ).sum(dim=1).unsqueeze(1) + torch.where(
        candidate_off, candidate_counts, 0
    )
    required_same = batch.required_same_suit_count.unsqueeze(1)
    target = batch.exact_select.unsqueeze(1)
    basic = (same_width <= required_same) & (
        off_width <= target - required_same
    )
    all_same_required = required_same == target
    same_suit_completion = (
        candidate_same
        & _follow_same_suit_can_complete(
            batch=batch,
            state=state,
            candidate_faces=candidate_faces,
            candidate_counts=candidate_counts,
        )
    )
    exhausting_completion = _follow_exhausting_suit_can_complete(
        batch=batch,
        state=state,
        candidate_faces=candidate_faces,
        candidate_counts=candidate_counts,
        same_width=same_width,
        off_width=off_width,
    )
    return basic & torch.where(
        all_same_required, same_suit_completion, exhausting_completion
    )


def _follow_same_suit_can_complete(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    candidate_faces: Tensor,
    candidate_counts: Tensor,
) -> Tensor:
    pair_can_complete = _follow_pair_can_complete(
        batch=batch,
        state=state,
        candidate_faces=candidate_faces,
        candidate_counts=candidate_counts,
    )
    capacity_can_complete = _has_remaining_capacity(
        batch=batch,
        state=state,
        candidate_faces=candidate_faces,
        candidate_widths=candidate_counts,
        target_width=batch.exact_select,
        allowed_face_mask=batch.same_suit_mask,
    )
    new_width = state.selected_width.unsqueeze(1) + candidate_counts
    complete_now = new_width == batch.exact_select.unsqueeze(1)
    return torch.where(
        complete_now,
        pair_can_complete,
        pair_can_complete & capacity_can_complete,
    )


def _follow_pair_can_complete(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    candidate_faces: Tensor,
    candidate_counts: Tensor,
) -> Tensor:
    next_counts = _candidate_selected_counts(
        state=state,
        candidate_faces=candidate_faces,
        candidate_counts=candidate_counts,
    )
    pair_selected = next_counts == 2
    single_selected = next_counts == 1
    no_pair_requirement = batch.pair_floor.unsqueeze(1) == 0
    without_tractor = _pair_without_tractor_can_complete(
        batch=batch,
        state=state,
        candidate_faces=candidate_faces,
        candidate_counts=candidate_counts,
        pair_selected=pair_selected,
    )
    with_plans = _pair_plan_can_complete(
        batch=batch,
        state=state,
        candidate_faces=candidate_faces,
        candidate_counts=candidate_counts,
        pair_selected=pair_selected,
        single_selected=single_selected,
    )
    return torch.where(
        no_pair_requirement,
        torch.ones_like(without_tractor),
        torch.where(
            batch.has_tractor.unsqueeze(1), with_plans, without_tractor
        ),
    )


def _candidate_selected_counts(
    *,
    state: DeviceActionState,
    candidate_faces: Tensor,
    candidate_counts: Tensor,
) -> Tensor:
    batch_size = int(candidate_faces.shape[0])
    vocab_size = int(candidate_faces.shape[1])
    base = state.selected_counts.unsqueeze(1).expand(
        batch_size, vocab_size, ACTION_FACE_COUNT
    )
    additions = torch.zeros_like(base).scatter(
        dim=2,
        index=candidate_faces.clamp(min=0).unsqueeze(2),
        src=candidate_counts.unsqueeze(2),
    )
    return base + additions


def _pair_without_tractor_can_complete(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    candidate_faces: Tensor,
    candidate_counts: Tensor,
    pair_selected: Tensor,
) -> Tensor:
    selected_pair_count = (
        pair_selected & batch.pair_face_mask.unsqueeze(1)
    ).sum(dim=2)
    future_pair_capacity = _future_pair_capacity(
        batch=batch,
        state=state,
        candidate_faces=candidate_faces,
    )
    single_capacity = _single_capacity_after_candidate(
        batch=batch,
        state=state,
        candidate_faces=candidate_faces,
        selected_faces=pair_selected,
    )
    new_width = state.selected_width.unsqueeze(1) + candidate_counts
    target_width = batch.exact_select.unsqueeze(1)
    minimum_pair_need = (
        batch.pair_floor.unsqueeze(1) - selected_pair_count
    )
    minimum_width_need = target_width - new_width - single_capacity
    lower_bound = torch.maximum(
        torch.zeros_like(selected_pair_count),
        torch.maximum(minimum_pair_need, minimum_width_need),
    )
    upper_bound = torch.minimum(
        future_pair_capacity,
        torch.div(
            target_width - new_width,
            2,
            rounding_mode="floor",
        ),
    )
    return lower_bound <= upper_bound


def _future_pair_capacity(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    candidate_faces: Tensor,
) -> Tensor:
    positions = torch.arange(
        ACTION_FACE_COUNT, dtype=torch.long, device=batch.device
    ).view(1, 1, ACTION_FACE_COUNT)
    selected = state.selected_counts.unsqueeze(1) > 0
    return (
        batch.pair_face_mask.unsqueeze(1)
        & ~selected
        & (positions > candidate_faces.unsqueeze(2))
    ).sum(dim=2)


def _single_capacity_after_candidate(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    candidate_faces: Tensor,
    selected_faces: Tensor,
) -> Tensor:
    positions = torch.arange(
        ACTION_FACE_COUNT, dtype=torch.long, device=batch.device
    ).view(1, 1, ACTION_FACE_COUNT)
    selected = (state.selected_counts.unsqueeze(1) > 0) | selected_faces
    return (
        batch.same_suit_mask.unsqueeze(1)
        & ~selected
        & (positions > candidate_faces.unsqueeze(2))
    ).sum(dim=2)


def _pair_plan_can_complete(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    candidate_faces: Tensor,
    candidate_counts: Tensor,
    pair_selected: Tensor,
    single_selected: Tensor,
) -> Tensor:
    if int(batch.pair_plan_masks.shape[1]) == 0:
        return torch.zeros(
            candidate_faces.shape,
            dtype=torch.bool,
            device=batch.device,
        )
    required = pair_selected.unsqueeze(2)
    forbidden = single_selected.unsqueeze(2)
    plans = batch.pair_plan_masks.unsqueeze(1)
    plan_rows = batch.pair_plan_row_mask.unsqueeze(1)
    required_subset = (~required | plans).all(dim=3)
    no_forbidden = ~(forbidden & plans).any(dim=3)
    positions = torch.arange(
        ACTION_FACE_COUNT, dtype=torch.long, device=batch.device
    ).view(1, 1, 1, ACTION_FACE_COUNT)
    future_plan_faces = plans & ~required
    respects_last = (
        ~future_plan_faces
        | (positions > candidate_faces.unsqueeze(2).unsqueeze(3))
    ).all(dim=3)
    future_pairs = future_plan_faces.sum(dim=3)
    new_width = state.selected_width.unsqueeze(1).unsqueeze(2) + (
        candidate_counts.unsqueeze(2)
    )
    fixed_width = new_width + future_pairs * 2
    selected_or_plan = (
        state.selected_counts.unsqueeze(1).unsqueeze(2) > 0
    ) | plans
    single_capacity = _single_capacity_for_plan(
        batch=batch,
        candidate_faces=candidate_faces,
        selected_or_plan=selected_or_plan,
    )
    remaining = batch.exact_select.view(-1, 1, 1) - fixed_width
    valid = (
        plan_rows
        & required_subset
        & no_forbidden
        & respects_last
        & (fixed_width <= batch.exact_select.view(-1, 1, 1))
        & (remaining <= single_capacity)
    )
    return valid.any(dim=2)


def _single_capacity_for_plan(
    *,
    batch: DeviceActionPlanBatch,
    candidate_faces: Tensor,
    selected_or_plan: Tensor,
) -> Tensor:
    positions = torch.arange(
        ACTION_FACE_COUNT, dtype=torch.long, device=batch.device
    ).view(1, 1, 1, ACTION_FACE_COUNT)
    return (
        batch.same_suit_mask.unsqueeze(1).unsqueeze(2)
        & ~selected_or_plan
        & (positions > candidate_faces.unsqueeze(2).unsqueeze(3))
    ).sum(dim=3)


def _follow_exhausting_suit_can_complete(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    candidate_faces: Tensor,
    candidate_counts: Tensor,
    same_width: Tensor,
    off_width: Tensor,
) -> Tensor:
    next_counts = _candidate_selected_counts(
        state=state,
        candidate_faces=candidate_faces,
        candidate_counts=candidate_counts,
    )
    positions = torch.arange(
        ACTION_FACE_COUNT, dtype=torch.long, device=batch.device
    ).view(1, 1, ACTION_FACE_COUNT)
    skipped_same_faces = (
        batch.same_suit_mask.unsqueeze(1)
        & (positions <= candidate_faces.unsqueeze(2))
        & (next_counts < batch.available_counts.unsqueeze(1))
    )
    same_prefix_ok = ~skipped_same_faces.any(dim=2)
    required_off = batch.exact_select - batch.required_same_suit_count
    remaining_off = required_off.unsqueeze(1) - off_width
    positions = torch.arange(
        ACTION_FACE_COUNT, dtype=torch.long, device=batch.device
    ).view(1, 1, ACTION_FACE_COUNT)
    remaining_off_capacity = (
        torch.where(
            batch.off_suit_mask.unsqueeze(1)
            & (next_counts == 0)
            & (positions > candidate_faces.unsqueeze(2)),
            batch.available_counts.unsqueeze(1),
            torch.zeros_like(batch.available_counts.unsqueeze(1)),
        ).sum(dim=2)
        >= remaining_off
    )
    same_count_ok = (
        same_width <= batch.required_same_suit_count.unsqueeze(1)
    )
    return (
        same_prefix_ok
        & same_count_ok
        & remaining_off_capacity
        & (remaining_off >= 0)
    )


def trace_done(
    *,
    batch: DeviceActionPlanBatch,
    selected_choice_ids: Tensor,
    step_counts: Tensor,
) -> Tensor:
    step_count = int(batch.trace_choice_ids.shape[2])
    positions = torch.arange(
        step_count, dtype=torch.long, device=batch.device
    ).view(1, 1, step_count)
    selected = selected_choice_ids[:, :step_count].unsqueeze(1)
    prefix_mask = positions < step_counts.view(-1, 1, 1)
    prefix_matches = (
        (batch.trace_choice_ids == selected) | ~prefix_mask
    ).all(dim=2)
    complete_length = batch.trace_lengths == step_counts.unsqueeze(1)
    return (
        (batch.kind_codes == _KIND_TRACE_SET).unsqueeze(1)
        & batch.trace_row_mask
        & prefix_matches
        & complete_length
    ).any(dim=1)


def safe_gather_2d(values: Tensor, indices: Tensor) -> Tensor:
    return values.gather(dim=1, index=indices.unsqueeze(1)).squeeze(1)


def _safe_gather_bool_2d(values: Tensor, indices: Tensor) -> Tensor:
    return values.gather(dim=1, index=indices)


def _kind_code(spec: ActionPlanFrame) -> int:
    return spec.kind_code


def _trace_count(spec: ActionPlanFrame) -> int:
    return max(len(spec.trace_choice_ids), 1)


def _trace_steps(spec: ActionPlanFrame) -> int:
    if not spec.trace_choice_ids:
        return 1
    return max(len(trace) for trace in spec.trace_choice_ids)


def _pair_plan_count(spec: ActionPlanFrame) -> int:
    return max(len(spec.pair_plan_masks), 1)


def _available_counts(spec: ActionPlanFrame) -> tuple[int, ...]:
    return spec.available_counts


def _effective_suits(spec: ActionPlanFrame) -> tuple[int, ...]:
    return spec.effective_suits


def _same_suit_mask(spec: ActionPlanFrame) -> tuple[bool, ...]:
    return spec.same_suit_mask


def _off_suit_mask(spec: ActionPlanFrame) -> tuple[bool, ...]:
    return spec.off_suit_mask


def _pair_face_mask(spec: ActionPlanFrame) -> tuple[bool, ...]:
    return spec.pair_face_mask


def _min_select(spec: ActionPlanFrame) -> int:
    return spec.min_select


def _max_select(spec: ActionPlanFrame) -> int:
    return spec.max_select


def _exact_select(spec: ActionPlanFrame) -> int:
    return spec.exact_select


def _required_same_suit_count(spec: ActionPlanFrame) -> int:
    return spec.required_same_suit_count


def _pair_floor(spec: ActionPlanFrame) -> int:
    return spec.pair_floor


def _has_tractor(spec: ActionPlanFrame) -> bool:
    return spec.has_tractor


def _padded_trace_choice_ids(
    spec: ActionPlanFrame,
    *,
    max_trace_count: int,
    max_trace_steps: int,
) -> tuple[tuple[int, ...], ...]:
    traces = spec.trace_choice_ids
    rows: list[tuple[int, ...]] = []
    for trace in traces:
        rows.append(_pad_int_row(trace, max_trace_steps))
    while len(rows) < max_trace_count:
        rows.append(_pad_int_row((), max_trace_steps))
    return tuple(rows)


def _padded_trace_choice_mask(
    spec: ActionPlanFrame,
    *,
    max_trace_count: int,
    max_trace_steps: int,
) -> tuple[tuple[bool, ...], ...]:
    traces = spec.trace_choice_ids
    rows: list[tuple[bool, ...]] = []
    for trace in traces:
        rows.append(
            tuple(
                index < len(trace) for index in range(max_trace_steps)
            )
        )
    while len(rows) < max_trace_count:
        rows.append(tuple(False for _ in range(max_trace_steps)))
    return tuple(rows)


def _padded_trace_lengths(
    spec: ActionPlanFrame, *, max_trace_count: int
) -> tuple[int, ...]:
    traces = spec.trace_choice_ids
    values = [len(trace) for trace in traces]
    while len(values) < max_trace_count:
        values.append(0)
    return tuple(values)


def _padded_trace_row_mask(
    spec: ActionPlanFrame, *, max_trace_count: int
) -> tuple[bool, ...]:
    count = len(spec.trace_choice_ids)
    return tuple(index < count for index in range(max_trace_count))


def _padded_pair_plan_masks(
    spec: ActionPlanFrame, *, max_pair_plan_count: int
) -> tuple[tuple[bool, ...], ...]:
    rows = list(spec.pair_plan_masks)
    while len(rows) < max_pair_plan_count:
        rows.append(tuple(False for _ in range(ACTION_FACE_COUNT)))
    return tuple(rows)


def _padded_pair_plan_row_mask(
    spec: ActionPlanFrame, *, max_pair_plan_count: int
) -> tuple[bool, ...]:
    count = len(spec.pair_plan_masks)
    return tuple(index < count for index in range(max_pair_plan_count))


def _pad_int_row(
    values: tuple[int, ...], length: int
) -> tuple[int, ...]:
    return (*values, *(0 for _ in range(length - len(values))))
