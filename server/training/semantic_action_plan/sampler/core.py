"""Batched fixed-choice sampling with reusable device state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.semantic_action_plan.choices import (
    DeviceLegalChoiceBatch,
    legal_choice_batch,
)
from server.training.semantic_action_plan.device import (
    DeviceActionPlanBatch,
    DeviceActionState,
    choice_tables,
    safe_gather_2d,
    selection_choice_candidates,
    trace_done,
    trace_set_candidates,
)
from server.training.semantic_action_plan.frame import (
    ACTION_KIND_DISCARD,
    ACTION_KIND_EMPTY,
    ACTION_KIND_FOLLOW,
    ACTION_KIND_TRACE_SET,
)
from server.training.semantic_action_plan.sampling import (
    action_sampling_error_reason,
    sample_legal_choices,
)
from server.training.semantic_action_plan.spec import ACTION_FACE_COUNT
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    FINISH_CHOICE_ID,
    MAX_ACTION_STEPS,
    PASS_CHOICE_ID,
)

_ERROR_UNTERMINATED = 1000


class ActionChoiceLogitDecoder(Protocol):
    """Stateful fixed-vocabulary decoder consumed during sampling."""

    def next_choice_logits(self) -> Tensor:
        """Return logits for all 110 choices at the current prefix."""
        ...

    def advance(self, selected_choice_ids: Tensor) -> None:
        """Advance with the choice sampled at the current step."""
        ...


@dataclass(frozen=True, slots=True)
class ActionSampleBatch:
    """Generated traces plus the minimal legal-mask PPO replay data."""

    choice_ids_padded: Tensor
    active_sample_indices: Tensor
    active_step_indices: Tensor
    legal_choice_masks: Tensor
    step_counts: Tensor
    choice_counts: Tensor
    log_probabilities: Tensor

    def __post_init__(self) -> None:
        batch_size, max_steps = self.choice_ids_padded.shape
        active_count = int(self.active_sample_indices.shape[0])
        assert batch_size > 0
        assert 0 < max_steps <= MAX_ACTION_STEPS
        assert self.active_step_indices.shape == (active_count,)
        assert self.legal_choice_masks.shape == (
            active_count,
            ACTION_CHOICE_COUNT,
        )
        assert self.step_counts.shape == (batch_size,)
        assert self.choice_counts.shape == (batch_size,)
        assert self.log_probabilities.shape == (batch_size,)
        assert self.choice_ids_padded.dtype == torch.long
        assert self.active_sample_indices.dtype == torch.long
        assert self.active_step_indices.dtype == torch.long
        assert self.legal_choice_masks.dtype == torch.bool
        assert self.step_counts.dtype == torch.long
        assert self.choice_counts.dtype == torch.long
        device = self.choice_ids_padded.device
        assert self.active_sample_indices.device == device
        assert self.active_step_indices.device == device
        assert self.legal_choice_masks.device == device
        assert self.step_counts.device == device
        assert self.choice_counts.device == device
        assert self.log_probabilities.device == device


type ActionSamplingResult = Ok[ActionSampleBatch] | Rejected


@dataclass(frozen=True, slots=True)
class _FlatReplay:
    active_sample_indices: Tensor
    active_step_indices: Tensor
    legal_choice_masks: Tensor


@dataclass(slots=True)
class ActionSampler:
    """Own reusable fixed-vocabulary sampling state for one device."""

    workspace: ActionSamplerWorkspace

    @classmethod
    def create(
        cls, *, batch_capacity: int, device: torch.device
    ) -> ActionSampler:
        return cls(
            workspace=ActionSamplerWorkspace(
                batch_capacity=batch_capacity, device=device
            )
        )

    def sample(
        self,
        *,
        action_batch: DeviceActionPlanBatch,
        generation_step_counts: Tensor,
        sampling_thresholds: Tensor,
        padded_generation_steps: int,
        logit_decoder: ActionChoiceLogitDecoder,
    ) -> ActionSamplingResult:
        return _sample_actions(
            action_batch=action_batch,
            generation_step_counts=generation_step_counts,
            sampling_thresholds=sampling_thresholds,
            padded_generation_steps=padded_generation_steps,
            logit_decoder=logit_decoder,
            workspace=self.workspace,
        )


@dataclass(slots=True)
class ActionSamplerWorkspace:
    """Reusable mutable tensors for fixed-vocabulary generation."""

    batch_capacity: int
    device: torch.device
    selected_counts: Tensor = field(init=False)
    choice_ids: Tensor = field(init=False)
    step_counts: Tensor = field(init=False)
    selected_width: Tensor = field(init=False)
    last_face_indices: Tensor = field(init=False)
    selected_suit_codes: Tensor = field(init=False)
    done: Tensor = field(init=False)
    choice_counts: Tensor = field(init=False)
    log_probabilities: Tensor = field(init=False)
    legal_masks: Tensor = field(init=False)
    replay_legal_masks: Tensor = field(init=False)
    row_indices: Tensor = field(init=False)

    def __post_init__(self) -> None:
        assert self.batch_capacity > 0
        self.selected_counts = torch.zeros(
            (self.batch_capacity, ACTION_FACE_COUNT),
            dtype=torch.long,
            device=self.device,
        )
        self.choice_ids = torch.zeros(
            (self.batch_capacity, MAX_ACTION_STEPS),
            dtype=torch.long,
            device=self.device,
        )
        self.step_counts = torch.zeros(
            (self.batch_capacity,), dtype=torch.long, device=self.device
        )
        self.selected_width = torch.zeros_like(self.step_counts)
        self.last_face_indices = torch.full_like(self.step_counts, -1)
        self.selected_suit_codes = torch.full_like(self.step_counts, -1)
        self.done = torch.zeros(
            (self.batch_capacity,), dtype=torch.bool, device=self.device
        )
        self.choice_counts = torch.zeros_like(self.step_counts)
        self.log_probabilities = torch.zeros(
            (self.batch_capacity,),
            dtype=torch.float32,
            device=self.device,
        )
        self.legal_masks = torch.zeros(
            (self.batch_capacity, ACTION_CHOICE_COUNT),
            dtype=torch.bool,
            device=self.device,
        )
        self.replay_legal_masks = torch.zeros(
            (
                self.batch_capacity,
                MAX_ACTION_STEPS,
                ACTION_CHOICE_COUNT,
            ),
            dtype=torch.bool,
            device=self.device,
        )
        self.row_indices = torch.arange(
            self.batch_capacity, dtype=torch.long, device=self.device
        )

    def reset(
        self,
        *,
        action_batch: DeviceActionPlanBatch,
        padded_generation_steps: int,
        log_probability_dtype: torch.dtype,
    ) -> None:
        batch_size = action_batch.batch_size()
        assert batch_size <= self.batch_capacity
        assert 0 < padded_generation_steps <= MAX_ACTION_STEPS
        self.selected_counts[:batch_size].zero_()
        self.choice_ids[:batch_size, :padded_generation_steps].zero_()
        self.step_counts[:batch_size].zero_()
        self.selected_width[:batch_size].zero_()
        self.last_face_indices[:batch_size].fill_(-1)
        self.selected_suit_codes[:batch_size].fill_(-1)
        self.done[:batch_size].copy_(
            action_batch.kind_codes == ACTION_KIND_EMPTY
        )
        self.choice_counts[:batch_size].zero_()
        if self.log_probabilities.dtype != log_probability_dtype:
            self.log_probabilities = torch.zeros(
                (self.batch_capacity,),
                dtype=log_probability_dtype,
                device=self.device,
            )
        else:
            self.log_probabilities[:batch_size].zero_()
        self.legal_masks[:batch_size].zero_()
        self.replay_legal_masks[
            :batch_size, :padded_generation_steps
        ].zero_()

    def state_view(self, *, batch_size: int) -> DeviceActionState:
        assert 0 < batch_size <= self.batch_capacity
        return DeviceActionState(
            selected_counts=self.selected_counts[:batch_size],
            selected_choice_ids=self.choice_ids[:batch_size],
            step_counts=self.step_counts[:batch_size],
            selected_width=self.selected_width[:batch_size],
            last_face_indices=self.last_face_indices[:batch_size],
            selected_suit_codes=self.selected_suit_codes[:batch_size],
            done=self.done[:batch_size],
            choice_counts=self.choice_counts[:batch_size],
        )


def _sample_actions(
    *,
    action_batch: DeviceActionPlanBatch,
    generation_step_counts: Tensor,
    sampling_thresholds: Tensor,
    padded_generation_steps: int,
    logit_decoder: ActionChoiceLogitDecoder,
    workspace: ActionSamplerWorkspace,
) -> ActionSamplingResult:
    batch_size = action_batch.batch_size()
    assert batch_size <= workspace.batch_capacity
    assert generation_step_counts.shape == (batch_size,)
    assert sampling_thresholds.shape == (
        batch_size,
        padded_generation_steps,
    )
    expected_dtype = (
        torch.float32
        if action_batch.device.type == "mps"
        else torch.float64
    )
    assert sampling_thresholds.dtype == expected_dtype
    workspace.reset(
        action_batch=action_batch,
        padded_generation_steps=padded_generation_steps,
        log_probability_dtype=torch.float32,
    )
    error_code = torch.zeros(
        (), dtype=torch.long, device=action_batch.device
    )
    for step_index in range(padded_generation_steps):
        state = workspace.state_view(batch_size=batch_size)
        active_rows = (~state.done) & (
            state.step_counts < generation_step_counts
        )
        legal = _legal_choices(
            workspace=workspace,
            batch=action_batch,
            state=state,
            active_rows=active_rows,
        )
        sampled = sample_legal_choices(
            choice_logits=logit_decoder.next_choice_logits(),
            legal_choices=legal,
            thresholds=sampling_thresholds[:, step_index],
            active_rows=active_rows,
        )
        error_code = _merge_error_code(error_code, sampled.error_code)
        workspace.replay_legal_masks[:batch_size, step_index].copy_(
            legal.masks
        )
        _advance_state(
            workspace=workspace,
            batch=action_batch,
            selected_choice_ids=sampled.choice_ids,
            choice_counts=legal.choice_counts,
            active_rows=active_rows,
        )
        workspace.log_probabilities[:batch_size].add_(
            torch.where(
                active_rows,
                sampled.selected_log_probabilities,
                torch.zeros_like(sampled.selected_log_probabilities),
            )
        )
        if step_index + 1 < padded_generation_steps:
            logit_decoder.advance(sampled.choice_ids)
    final_state = workspace.state_view(batch_size=batch_size)
    error_code = _set_error_if(
        error_code, (~final_state.done).any(), _ERROR_UNTERMINATED
    )
    error_value = int(error_code.detach().cpu().item())
    if error_value != 0:
        return Rejected(reason=_error_reason(error_value))
    replay = _flat_active_replay(
        workspace=workspace,
        batch_size=batch_size,
        padded_generation_steps=padded_generation_steps,
    )
    return _result.Ok(
        value=ActionSampleBatch(
            choice_ids_padded=workspace.choice_ids[
                :batch_size, :padded_generation_steps
            ],
            active_sample_indices=replay.active_sample_indices,
            active_step_indices=replay.active_step_indices,
            legal_choice_masks=replay.legal_choice_masks,
            step_counts=workspace.step_counts[:batch_size],
            choice_counts=workspace.choice_counts[:batch_size],
            log_probabilities=workspace.log_probabilities[:batch_size],
        )
    )


def _legal_choices(
    *,
    workspace: ActionSamplerWorkspace,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    active_rows: Tensor,
) -> DeviceLegalChoiceBatch:
    batch_size = batch.batch_size()
    trace_ids, trace_mask = trace_set_candidates(
        batch=batch, state=state
    )
    selection_ids, selection_mask = selection_choice_candidates(
        batch=batch, state=state
    )
    trace_full = _scatter_choice_mask(
        ids=trace_ids, masks=trace_mask, batch_size=batch_size
    )
    selection_full = _scatter_choice_mask(
        ids=selection_ids,
        masks=selection_mask,
        batch_size=batch_size,
    )
    trace_rows = (batch.kind_codes == ACTION_KIND_TRACE_SET).unsqueeze(
        1
    )
    active_masks = torch.where(trace_rows, trace_full, selection_full)
    fallback = torch.zeros_like(active_masks)
    fallback[:, PASS_CHOICE_ID] = True
    workspace.legal_masks[:batch_size].copy_(
        torch.where(active_rows.unsqueeze(1), active_masks, fallback)
    )
    return legal_choice_batch(masks=workspace.legal_masks[:batch_size])


def _scatter_choice_mask(
    *, ids: Tensor, masks: Tensor, batch_size: int
) -> Tensor:
    counts = torch.zeros(
        (batch_size, ACTION_CHOICE_COUNT),
        dtype=torch.long,
        device=ids.device,
    )
    counts.scatter_add_(1, ids, masks.to(dtype=torch.long))
    return counts > 0


def _flat_active_replay(
    *,
    workspace: ActionSamplerWorkspace,
    batch_size: int,
    padded_generation_steps: int,
) -> _FlatReplay:
    positions = torch.arange(
        padded_generation_steps,
        dtype=torch.long,
        device=workspace.device,
    ).unsqueeze(0)
    active = positions.expand(batch_size, -1) < workspace.step_counts[
        :batch_size
    ].unsqueeze(1)
    flat_positions = torch.nonzero(
        active.reshape(-1), as_tuple=False
    ).flatten()
    return _FlatReplay(
        active_sample_indices=torch.div(
            flat_positions,
            padded_generation_steps,
            rounding_mode="floor",
        ),
        active_step_indices=flat_positions.remainder(
            padded_generation_steps
        ),
        legal_choice_masks=workspace.replay_legal_masks[
            :batch_size, :padded_generation_steps
        ]
        .reshape(
            batch_size * padded_generation_steps,
            ACTION_CHOICE_COUNT,
        )
        .index_select(0, flat_positions),
    )


def _advance_state(
    *,
    workspace: ActionSamplerWorkspace,
    batch: DeviceActionPlanBatch,
    selected_choice_ids: Tensor,
    choice_counts: Tensor,
    active_rows: Tensor,
) -> None:
    batch_size = batch.batch_size()
    state = workspace.state_view(batch_size=batch_size)
    tables = choice_tables(batch.device)
    choice_face = tables.face_indices.index_select(
        0, selected_choice_ids
    )
    choice_count = tables.counts.index_select(0, selected_choice_ids)
    choice_suit = safe_gather_2d(
        batch.effective_suits, choice_face.clamp(min=0)
    )
    is_card = tables.is_card.index_select(0, selected_choice_ids)
    positions = state.step_counts.clamp(max=MAX_ACTION_STEPS - 1)
    rows = workspace.row_indices[:batch_size]
    current = state.selected_choice_ids[rows, positions]
    state.selected_choice_ids[rows, positions] = torch.where(
        active_rows, selected_choice_ids, current
    )
    additions = torch.where(active_rows & is_card, choice_count, 0)
    state.selected_counts.scatter_add_(
        1,
        choice_face.clamp(min=0).unsqueeze(1),
        additions.unsqueeze(1),
    )
    state.selected_width.add_(additions)
    state.step_counts.add_(active_rows.to(dtype=torch.long))
    state.last_face_indices.copy_(
        torch.where(
            active_rows & is_card,
            choice_face,
            state.last_face_indices,
        )
    )
    state.selected_suit_codes.copy_(
        torch.where(
            active_rows & is_card & (state.selected_suit_codes < 0),
            choice_suit,
            state.selected_suit_codes,
        )
    )
    terminal = (selected_choice_ids == PASS_CHOICE_ID) | (
        selected_choice_ids == FINISH_CHOICE_ID
    )
    exact_done = (
        (batch.kind_codes == ACTION_KIND_DISCARD)
        | (batch.kind_codes == ACTION_KIND_FOLLOW)
    ) & (state.selected_width == batch.exact_select)
    trace_complete = trace_done(
        batch=batch,
        selected_choice_ids=state.selected_choice_ids,
        step_counts=state.step_counts,
    )
    state.done.copy_(
        state.done
        | (active_rows & (terminal | exact_done | trace_complete))
    )
    state.choice_counts.add_(
        torch.where(
            active_rows, choice_counts, torch.zeros_like(choice_counts)
        )
    )


def _set_error_if(
    error_code: Tensor, condition: Tensor, code: int
) -> Tensor:
    assert error_code.shape == ()
    assert condition.shape == ()
    return torch.where(
        (error_code == 0) & condition,
        torch.full(
            (), code, dtype=torch.long, device=error_code.device
        ),
        error_code,
    )


def _merge_error_code(current: Tensor, incoming: Tensor) -> Tensor:
    return torch.where(
        (current == 0) & (incoming != 0), incoming, current
    )


def _error_reason(error_code: int) -> str:
    if error_code == _ERROR_UNTERMINATED:
        return "policy action did not terminate"
    sampling_reason = action_sampling_error_reason(error_code)
    if sampling_reason is not None:
        return sampling_reason
    return "policy sampling failed"


__all__ = (
    "ActionChoiceLogitDecoder",
    "ActionSampleBatch",
    "ActionSampler",
)
