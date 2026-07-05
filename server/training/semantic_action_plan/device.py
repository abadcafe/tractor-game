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
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import ArgumentPrefixTensorBatch

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
    trace_tokens: Tensor
    trace_token_mask: Tensor
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
    """Current semantic token generation state on one torch device."""

    selected_counts: Tensor
    selected_token_ids: Tensor
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
        assert self.selected_token_ids.ndim == 2
        assert int(self.selected_token_ids.shape[0]) == batch_size
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
        kind_codes=torch.tensor(
            tuple(_kind_code(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        available_counts=torch.tensor(
            tuple(_available_counts(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        effective_suits=torch.tensor(
            tuple(_effective_suits(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        same_suit_mask=torch.tensor(
            tuple(_same_suit_mask(spec) for spec in specs),
            dtype=torch.bool,
            device=device,
        ),
        off_suit_mask=torch.tensor(
            tuple(_off_suit_mask(spec) for spec in specs),
            dtype=torch.bool,
            device=device,
        ),
        pair_face_mask=torch.tensor(
            tuple(_pair_face_mask(spec) for spec in specs),
            dtype=torch.bool,
            device=device,
        ),
        min_select=torch.tensor(
            tuple(_min_select(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        max_select=torch.tensor(
            tuple(_max_select(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        exact_select=torch.tensor(
            tuple(_exact_select(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        required_same_suit_count=torch.tensor(
            tuple(_required_same_suit_count(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        pair_floor=torch.tensor(
            tuple(_pair_floor(spec) for spec in specs),
            dtype=torch.long,
            device=device,
        ),
        has_tractor=torch.tensor(
            tuple(_has_tractor(spec) for spec in specs),
            dtype=torch.bool,
            device=device,
        ),
        trace_tokens=torch.tensor(
            tuple(
                _padded_trace_tokens(
                    spec,
                    max_trace_count=max_trace_count,
                    max_trace_steps=max_trace_steps,
                )
                for spec in specs
            ),
            dtype=torch.long,
            device=device,
        ),
        trace_token_mask=torch.tensor(
            tuple(
                _padded_trace_token_mask(
                    spec,
                    max_trace_count=max_trace_count,
                    max_trace_steps=max_trace_steps,
                )
                for spec in specs
            ),
            dtype=torch.bool,
            device=device,
        ),
        trace_lengths=torch.tensor(
            tuple(
                _padded_trace_lengths(
                    spec, max_trace_count=max_trace_count
                )
                for spec in specs
            ),
            dtype=torch.long,
            device=device,
        ),
        trace_row_mask=torch.tensor(
            tuple(
                _padded_trace_row_mask(
                    spec, max_trace_count=max_trace_count
                )
                for spec in specs
            ),
            dtype=torch.bool,
            device=device,
        ),
        pair_plan_masks=torch.tensor(
            tuple(
                _padded_pair_plan_masks(
                    spec, max_pair_plan_count=max_pair_plan_count
                )
                for spec in specs
            ),
            dtype=torch.bool,
            device=device,
        ),
        pair_plan_row_mask=torch.tensor(
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


def initial_action_state(
    batch: DeviceActionPlanBatch,
) -> DeviceActionState:
    """Return an empty token-generation state for the plan batch."""
    batch_size = batch.batch_size()
    return DeviceActionState(
        selected_counts=torch.zeros(
            (batch_size, ACTION_FACE_COUNT),
            dtype=torch.long,
            device=batch.device,
        ),
        selected_token_ids=torch.zeros(
            (batch_size, SEMANTIC_CODEC.max_argument_tokens),
            dtype=torch.long,
            device=batch.device,
        ),
        step_counts=torch.zeros(
            (batch_size,), dtype=torch.long, device=batch.device
        ),
        selected_width=torch.zeros(
            (batch_size,), dtype=torch.long, device=batch.device
        ),
        last_face_indices=torch.full(
            (batch_size,), -1, dtype=torch.long, device=batch.device
        ),
        selected_suit_codes=torch.full(
            (batch_size,), -1, dtype=torch.long, device=batch.device
        ),
        done=batch.kind_codes == _KIND_EMPTY,
        choice_counts=torch.zeros(
            (batch_size,), dtype=torch.long, device=batch.device
        ),
    )


def action_prefix_batch(
    state: DeviceActionState,
    *,
    generated_token_count: int,
) -> ArgumentPrefixTensorBatch:
    """Return model prefix tensors for the current action state."""
    assert generated_token_count >= 0
    assert generated_token_count < SEMANTIC_CODEC.max_argument_tokens
    safe_token_ids = state.selected_token_ids[:, :generated_token_count]
    positions = torch.arange(
        generated_token_count,
        dtype=torch.long,
        device=state.step_counts.device,
    ).unsqueeze(0)
    prefix_masks = positions < state.step_counts.unsqueeze(1)
    safe_prefix_ids = torch.where(
        prefix_masks,
        safe_token_ids,
        torch.zeros_like(safe_token_ids),
    )
    bos = torch.full(
        (int(state.step_counts.shape[0]), 1),
        SEMANTIC_CODEC.argument_bos_id,
        dtype=torch.long,
        device=state.step_counts.device,
    )
    return ArgumentPrefixTensorBatch(
        argument_ids=torch.cat((bos, safe_prefix_ids), dim=1),
        argument_masks=torch.cat(
            (
                torch.ones(
                    (int(state.step_counts.shape[0]), 1),
                    dtype=torch.bool,
                    device=state.step_counts.device,
                ),
                prefix_masks,
            ),
            dim=1,
        ),
    )


def legal_token_mask(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
) -> Tensor:
    """Return full-vocab legal token masks for the current state."""
    mask = torch.zeros(
        (batch.batch_size(), SEMANTIC_CODEC.argument_vocab_size),
        dtype=torch.bool,
        device=batch.device,
    )
    mask = mask | _trace_set_mask(batch=batch, state=state)
    mask = mask | _selection_mask(batch=batch, state=state)
    done_mask = state.done.unsqueeze(1)
    done_tokens = torch.zeros_like(mask)
    done_tokens[:, SEMANTIC_CODEC.argument_pass_id] = True
    mask = torch.where(done_mask, done_tokens, mask)
    return mask


def advance_action_state(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    selected_token_ids: Tensor,
    legal_mask: Tensor,
) -> DeviceActionState:
    """Advance device state after one sampled semantic token."""
    assert selected_token_ids.ndim == 1
    active = ~state.done
    token_tables = _token_tables(batch.device)
    token_face = token_tables.face_indices.index_select(
        dim=0, index=selected_token_ids
    )
    token_count = token_tables.counts.index_select(
        dim=0, index=selected_token_ids
    )
    token_suit = _safe_gather_2d(
        batch.effective_suits, token_face.clamp(min=0)
    )
    is_select = token_tables.is_select.index_select(
        dim=0, index=selected_token_ids
    )
    write_positions = state.step_counts.clamp(
        max=SEMANTIC_CODEC.max_argument_tokens - 1
    )
    next_tokens = state.selected_token_ids.scatter(
        dim=1,
        index=write_positions.unsqueeze(1),
        src=selected_token_ids.unsqueeze(1),
    )
    next_counts = state.selected_counts.scatter_add(
        dim=1,
        index=token_face.clamp(min=0).unsqueeze(1),
        src=torch.where(active & is_select, token_count, 0).unsqueeze(
            1
        ),
    )
    next_width = state.selected_width + torch.where(
        active & is_select, token_count, 0
    )
    next_step_counts = state.step_counts + active.to(dtype=torch.long)
    next_last_faces = torch.where(
        active & is_select, token_face, state.last_face_indices
    )
    next_suits = torch.where(
        active & is_select & (state.selected_suit_codes < 0),
        token_suit,
        state.selected_suit_codes,
    )
    terminal_token = (
        selected_token_ids == SEMANTIC_CODEC.argument_pass_id
    ) | (selected_token_ids == SEMANTIC_CODEC.argument_stop_id)
    exact_done = (
        (batch.kind_codes == _KIND_DISCARD)
        | (batch.kind_codes == _KIND_FOLLOW)
    ) & (next_width == batch.exact_select)
    trace_done = _trace_done(
        batch=batch,
        selected_token_ids=next_tokens,
        step_counts=next_step_counts,
    )
    next_done = state.done | (
        active & (terminal_token | exact_done | trace_done)
    )
    choice_add = legal_mask.sum(dim=1).to(dtype=torch.long)
    next_choice_counts = state.choice_counts + torch.where(
        active, choice_add, torch.zeros_like(choice_add)
    )
    return DeviceActionState(
        selected_counts=next_counts,
        selected_token_ids=next_tokens,
        step_counts=next_step_counts,
        selected_width=next_width,
        last_face_indices=next_last_faces,
        selected_suit_codes=next_suits,
        done=next_done,
        choice_counts=next_choice_counts,
    )


def action_trace_ids(
    state: DeviceActionState,
) -> tuple[tuple[int, ...], ...]:
    """Return completed semantic trace token ids on CPU."""
    token_rows = state.selected_token_ids.detach().cpu()
    counts = state.step_counts.detach().cpu()
    result: list[tuple[int, ...]] = []
    for row_index in range(int(token_rows.shape[0])):
        count = int(counts[row_index].item())
        result.append(
            tuple(
                int(token_rows[row_index, token_index].item())
                for token_index in range(count)
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _TokenTables:
    is_select: Tensor
    face_indices: Tensor
    counts: Tensor


_TOKEN_TABLE_CACHE: dict[torch.device, _TokenTables] = {}


def _token_tables(device: torch.device) -> _TokenTables:
    cached = _TOKEN_TABLE_CACHE.get(device)
    if cached is not None:
        return cached
    face_indices = [
        -1 for _ in range(SEMANTIC_CODEC.argument_vocab_size)
    ]
    counts = [0 for _ in range(SEMANTIC_CODEC.argument_vocab_size)]
    is_select = [
        False for _ in range(SEMANTIC_CODEC.argument_vocab_size)
    ]
    for face_index in range(ACTION_FACE_COUNT):
        for count in (1, 2):
            token_id = (
                SEMANTIC_CODEC.argument_select_base_id
                + face_index * 2
                + count
                - 1
            )
            face_indices[token_id] = face_index
            counts[token_id] = count
            is_select[token_id] = True
    tables = _TokenTables(
        is_select=torch.tensor(
            is_select, dtype=torch.bool, device=device
        ),
        face_indices=torch.tensor(
            face_indices, dtype=torch.long, device=device
        ),
        counts=torch.tensor(counts, dtype=torch.long, device=device),
    )
    _TOKEN_TABLE_CACHE[device] = tables
    return tables


def _trace_set_mask(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
) -> Tensor:
    batch_size = batch.batch_size()
    trace_count = int(batch.trace_tokens.shape[1])
    step_count = int(batch.trace_tokens.shape[2])
    if trace_count == 0:
        return torch.zeros(
            (batch_size, SEMANTIC_CODEC.argument_vocab_size),
            dtype=torch.bool,
            device=batch.device,
        )
    positions = torch.arange(
        step_count, dtype=torch.long, device=batch.device
    ).view(1, 1, step_count)
    current_steps = state.step_counts.view(batch_size, 1, 1)
    prefix_mask = positions < current_steps
    selected = state.selected_token_ids[:, :step_count].unsqueeze(1)
    prefix_matches = (
        (batch.trace_tokens == selected) | ~prefix_mask
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
    next_tokens = batch.trace_tokens.gather(
        dim=2, index=gather_index
    ).squeeze(2)
    one_hot = torch.nn.functional.one_hot(
        next_tokens.clamp(min=0),
        num_classes=SEMANTIC_CODEC.argument_vocab_size,
    ).to(dtype=torch.bool)
    return (one_hot & valid_traces.unsqueeze(2)).any(dim=1)


def _selection_mask(
    *,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
) -> Tensor:
    token_tables = _token_tables(batch.device)
    face_indices = token_tables.face_indices.clamp(min=0).unsqueeze(0)
    counts = token_tables.counts.unsqueeze(0)
    available = batch.available_counts.gather(
        dim=1,
        index=face_indices.expand(batch.batch_size(), -1),
    )
    face_suits = batch.effective_suits.gather(
        dim=1,
        index=face_indices.expand(batch.batch_size(), -1),
    )
    already_selected = (
        state.selected_counts.gather(
            dim=1,
            index=face_indices.expand(batch.batch_size(), -1),
        )
        > 0
    )
    new_width = state.selected_width.unsqueeze(1) + counts
    generic_select = (
        token_tables.is_select.unsqueeze(0)
        & ~already_selected
        & (counts <= available)
        & (
            token_tables.face_indices.unsqueeze(0)
            > state.last_face_indices.unsqueeze(1)
        )
        & (new_width <= batch.max_select.unsqueeze(1))
        & ~state.done.unsqueeze(1)
    )
    discard = (
        (batch.kind_codes == _KIND_DISCARD).unsqueeze(1)
        & generic_select
        & _has_remaining_capacity(
            batch=batch,
            state=state,
            candidate_faces=token_tables.face_indices.unsqueeze(
                0
            ).expand(batch.batch_size(), -1),
            candidate_widths=counts.expand(batch.batch_size(), -1),
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
    lead_stop = (
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
            candidate_faces=token_tables.face_indices.unsqueeze(
                0
            ).expand(batch.batch_size(), -1),
            candidate_counts=counts.expand(batch.batch_size(), -1),
        )
    )
    mask = discard | lead_select | follow_select
    mask[:, SEMANTIC_CODEC.argument_stop_id] = lead_stop
    return mask


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
    max_future = int(ACTION_FACE_COUNT)
    result = torch.zeros_like(selected_pair_count, dtype=torch.bool)
    new_width = state.selected_width.unsqueeze(1) + candidate_counts
    for future_pair_count in range(max_future + 1):
        future_pairs = torch.full_like(
            selected_pair_count, future_pair_count
        )
        final_pair_count = selected_pair_count + future_pairs
        fixed_width = new_width + future_pairs * 2
        remaining = batch.exact_select.unsqueeze(1) - fixed_width
        candidate_ok = (
            (future_pairs <= future_pair_capacity)
            & (final_pair_count >= batch.pair_floor.unsqueeze(1))
            & (fixed_width <= batch.exact_select.unsqueeze(1))
            & (remaining <= single_capacity - future_pairs)
        )
        result = result | candidate_ok
    return result


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


def _trace_done(
    *,
    batch: DeviceActionPlanBatch,
    selected_token_ids: Tensor,
    step_counts: Tensor,
) -> Tensor:
    step_count = int(batch.trace_tokens.shape[2])
    positions = torch.arange(
        step_count, dtype=torch.long, device=batch.device
    ).view(1, 1, step_count)
    selected = selected_token_ids[:, :step_count].unsqueeze(1)
    prefix_mask = positions < step_counts.view(-1, 1, 1)
    prefix_matches = (
        (batch.trace_tokens == selected) | ~prefix_mask
    ).all(dim=2)
    complete_length = batch.trace_lengths == step_counts.unsqueeze(1)
    return (
        (batch.kind_codes == _KIND_TRACE_SET).unsqueeze(1)
        & batch.trace_row_mask
        & prefix_matches
        & complete_length
    ).any(dim=1)


def _safe_gather_2d(values: Tensor, indices: Tensor) -> Tensor:
    return values.gather(dim=1, index=indices.unsqueeze(1)).squeeze(1)


def _safe_gather_bool_2d(values: Tensor, indices: Tensor) -> Tensor:
    return values.gather(dim=1, index=indices)


def _kind_code(spec: ActionPlanFrame) -> int:
    return spec.kind_code


def _trace_count(spec: ActionPlanFrame) -> int:
    return max(len(spec.trace_tokens), 1)


def _trace_steps(spec: ActionPlanFrame) -> int:
    if not spec.trace_tokens:
        return 1
    return max(len(trace) for trace in spec.trace_tokens)


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


def _padded_trace_tokens(
    spec: ActionPlanFrame,
    *,
    max_trace_count: int,
    max_trace_steps: int,
) -> tuple[tuple[int, ...], ...]:
    traces = spec.trace_tokens
    rows: list[tuple[int, ...]] = []
    for trace in traces:
        rows.append(_pad_int_row(trace, max_trace_steps))
    while len(rows) < max_trace_count:
        rows.append(_pad_int_row((), max_trace_steps))
    return tuple(rows)


def _padded_trace_token_mask(
    spec: ActionPlanFrame,
    *,
    max_trace_count: int,
    max_trace_steps: int,
) -> tuple[tuple[bool, ...], ...]:
    traces = spec.trace_tokens
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
    traces = spec.trace_tokens
    values = [len(trace) for trace in traces]
    while len(values) < max_trace_count:
        values.append(0)
    return tuple(values)


def _padded_trace_row_mask(
    spec: ActionPlanFrame, *, max_trace_count: int
) -> tuple[bool, ...]:
    count = len(spec.trace_tokens)
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
