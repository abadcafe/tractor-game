"""Batched semantic action sampling with reusable device workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

from server import result as _result
from server.result import Ok, Rejected
from server.training.semantic_action_plan.choices import (
    DeviceLegalCandidateBatch,
    legal_candidate_batch,
)
from server.training.semantic_action_plan.device import (
    DeviceActionPlanBatch,
    DeviceActionState,
    safe_gather_2d,
    selection_candidates,
    token_tables,
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
    sample_legal_candidates,
    sample_legal_token_error_reason,
)
from server.training.semantic_action_plan.spec import (
    ACTION_FACE_COUNT,
    MAX_LEGAL_CANDIDATE_COUNT,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC

_ERROR_UNTERMINATED = 1000


class SemanticArgumentLogitDecoder(Protocol):
    """Stateful argument decoder consumed by semantic sampling."""

    def next_logits(self) -> Tensor:
        """Return logits for the current semantic argument prefix."""
        ...

    def advance(self, selected_token_ids: Tensor) -> None:
        """Advance the decoder with tokens sampled at this step."""
        ...


@dataclass(frozen=True, slots=True)
class SemanticActionSampleBatch:
    """Workspace-backed semantic traces and PPO replay tensors."""

    selected_token_ids_padded: Tensor
    active_sample_indices: Tensor
    active_step_indices: Tensor
    choice_token_ids: Tensor
    choice_masks: Tensor
    selected_choice_offsets: Tensor
    step_counts: Tensor
    choice_counts: Tensor
    log_probabilities: Tensor

    def __post_init__(self) -> None:
        batch_size = int(self.step_counts.shape[0])
        max_steps = int(self.selected_token_ids_padded.shape[1])
        assert batch_size > 0
        assert max_steps > 0
        assert max_steps <= SEMANTIC_CODEC.max_argument_tokens
        assert self.selected_token_ids_padded.shape == (
            batch_size,
            max_steps,
        )
        assert self.active_sample_indices.ndim == 1
        assert self.active_step_indices.shape == (
            int(self.active_sample_indices.shape[0]),
        )
        assert self.choice_token_ids.shape == (
            int(self.active_sample_indices.shape[0]),
            MAX_LEGAL_CANDIDATE_COUNT,
        )
        assert self.choice_masks.shape == self.choice_token_ids.shape
        assert self.selected_choice_offsets.shape == (
            int(self.active_sample_indices.shape[0]),
        )
        assert self.choice_counts.shape == (batch_size,)
        assert self.log_probabilities.shape == (batch_size,)
        assert self.selected_token_ids_padded.dtype == torch.long
        assert self.active_sample_indices.dtype == torch.long
        assert self.active_step_indices.dtype == torch.long
        assert self.choice_token_ids.dtype == torch.int16
        assert self.choice_masks.dtype == torch.bool
        assert self.selected_choice_offsets.dtype == torch.long
        assert self.step_counts.dtype == torch.long
        assert self.choice_counts.dtype == torch.long
        device = self.step_counts.device
        assert self.selected_token_ids_padded.device == device
        assert self.active_sample_indices.device == device
        assert self.active_step_indices.device == device
        assert self.choice_token_ids.device == device
        assert self.choice_masks.device == device
        assert self.selected_choice_offsets.device == device
        assert self.choice_counts.device == device
        assert self.log_probabilities.device == device


type SemanticSamplingResult = Ok[SemanticActionSampleBatch] | Rejected


@dataclass(frozen=True, slots=True)
class _FlatReplay:
    active_sample_indices: Tensor
    active_step_indices: Tensor
    choice_token_ids: Tensor
    choice_masks: Tensor
    selected_choice_offsets: Tensor


@dataclass(slots=True)
class SemanticActionSampler:
    """Own reusable semantic sampling state for one device."""

    workspace: "SemanticActionSamplerWorkspace"

    @classmethod
    def create(
        cls, *, batch_capacity: int, device: torch.device
    ) -> "SemanticActionSampler":
        """Create a sampler with reusable device-local workspace."""
        return cls(
            workspace=SemanticActionSamplerWorkspace(
                batch_capacity=batch_capacity,
                device=device,
            )
        )

    def sample(
        self,
        *,
        action_batch: DeviceActionPlanBatch,
        generation_step_counts: Tensor,
        sampling_thresholds: Tensor,
        padded_generation_steps: int,
        logit_decoder: SemanticArgumentLogitDecoder,
    ) -> SemanticSamplingResult:
        """Sample semantic argument traces for a plan batch."""
        return _sample_semantic_actions(
            action_batch=action_batch,
            generation_step_counts=generation_step_counts,
            sampling_thresholds=sampling_thresholds,
            padded_generation_steps=padded_generation_steps,
            logit_decoder=logit_decoder,
            workspace=self.workspace,
        )


@dataclass(slots=True)
class SemanticActionSamplerWorkspace:
    """Reusable mutable tensors for semantic action generation."""

    batch_capacity: int
    device: torch.device
    selected_counts: Tensor = field(init=False)
    selected_token_ids: Tensor = field(init=False)
    step_counts: Tensor = field(init=False)
    selected_width: Tensor = field(init=False)
    last_face_indices: Tensor = field(init=False)
    selected_suit_codes: Tensor = field(init=False)
    done: Tensor = field(init=False)
    choice_counts: Tensor = field(init=False)
    log_probabilities: Tensor = field(init=False)
    candidate_ids: Tensor = field(init=False)
    candidate_masks: Tensor = field(init=False)
    trace_candidate_ids: Tensor = field(init=False)
    trace_candidate_masks: Tensor = field(init=False)
    selection_candidate_ids: Tensor = field(init=False)
    selection_candidate_masks: Tensor = field(init=False)
    inactive_candidate_ids: Tensor = field(init=False)
    inactive_candidate_masks: Tensor = field(init=False)
    row_indices: Tensor = field(init=False)
    replay_choice_ids: Tensor = field(init=False)
    replay_choice_masks: Tensor = field(init=False)
    replay_selected_offsets: Tensor = field(init=False)

    def __post_init__(self) -> None:
        assert self.batch_capacity > 0
        self.selected_counts = torch.zeros(
            (self.batch_capacity, ACTION_FACE_COUNT),
            dtype=torch.long,
            device=self.device,
        )
        self.selected_token_ids = torch.zeros(
            (
                self.batch_capacity,
                SEMANTIC_CODEC.max_argument_tokens,
            ),
            dtype=torch.long,
            device=self.device,
        )
        self.step_counts = torch.zeros(
            (self.batch_capacity,), dtype=torch.long, device=self.device
        )
        self.selected_width = torch.zeros(
            (self.batch_capacity,), dtype=torch.long, device=self.device
        )
        self.last_face_indices = torch.full(
            (self.batch_capacity,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.selected_suit_codes = torch.full(
            (self.batch_capacity,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.done = torch.zeros(
            (self.batch_capacity,), dtype=torch.bool, device=self.device
        )
        self.choice_counts = torch.zeros(
            (self.batch_capacity,), dtype=torch.long, device=self.device
        )
        self.log_probabilities = torch.zeros(
            (self.batch_capacity,),
            dtype=torch.float32,
            device=self.device,
        )
        self.candidate_ids = torch.zeros(
            (self.batch_capacity, MAX_LEGAL_CANDIDATE_COUNT),
            dtype=torch.long,
            device=self.device,
        )
        self.candidate_masks = torch.zeros(
            self.candidate_ids.shape,
            dtype=torch.bool,
            device=self.device,
        )
        self.trace_candidate_ids = torch.zeros_like(self.candidate_ids)
        self.trace_candidate_masks = torch.zeros_like(
            self.candidate_masks
        )
        self.selection_candidate_ids = torch.zeros_like(
            self.candidate_ids
        )
        self.selection_candidate_masks = torch.zeros_like(
            self.candidate_masks
        )
        self.inactive_candidate_ids = torch.zeros(
            self.candidate_ids.shape,
            dtype=torch.long,
            device=self.device,
        )
        self.inactive_candidate_ids[:, 0].fill_(
            SEMANTIC_CODEC.argument_pass_id
        )
        self.inactive_candidate_masks = torch.zeros(
            self.candidate_ids.shape,
            dtype=torch.bool,
            device=self.device,
        )
        self.inactive_candidate_masks[:, 0].fill_(True)
        self.row_indices = torch.arange(
            self.batch_capacity,
            dtype=torch.long,
            device=self.device,
        )
        self.replay_choice_ids = torch.zeros(
            (
                self.batch_capacity,
                SEMANTIC_CODEC.max_argument_tokens,
                MAX_LEGAL_CANDIDATE_COUNT,
            ),
            dtype=torch.int16,
            device=self.device,
        )
        self.replay_choice_masks = torch.zeros(
            self.replay_choice_ids.shape,
            dtype=torch.bool,
            device=self.device,
        )
        self.replay_selected_offsets = torch.zeros(
            (
                self.batch_capacity,
                SEMANTIC_CODEC.max_argument_tokens,
            ),
            dtype=torch.long,
            device=self.device,
        )

    def reset(
        self,
        *,
        action_batch: DeviceActionPlanBatch,
        padded_generation_steps: int,
        log_probability_dtype: torch.dtype,
    ) -> None:
        """Reset workspace rows for one new sampling batch."""
        batch_size = action_batch.batch_size()
        assert batch_size <= self.batch_capacity
        assert padded_generation_steps > 0
        assert (
            padded_generation_steps
            <= SEMANTIC_CODEC.max_argument_tokens
        )
        self.selected_counts[:batch_size, :].zero_()
        self.selected_token_ids[
            :batch_size, :padded_generation_steps
        ].zero_()
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
        self.replay_choice_ids[
            :batch_size, :padded_generation_steps, :
        ].zero_()
        self.replay_choice_masks[
            :batch_size, :padded_generation_steps, :
        ].fill_(False)
        self.replay_selected_offsets[
            :batch_size, :padded_generation_steps
        ].zero_()

    def state_view(self, *, batch_size: int) -> DeviceActionState:
        """Return a batch-sized view of the mutable action state."""
        assert batch_size > 0
        assert batch_size <= self.batch_capacity
        return DeviceActionState(
            selected_counts=self.selected_counts[:batch_size],
            selected_token_ids=self.selected_token_ids[:batch_size],
            step_counts=self.step_counts[:batch_size],
            selected_width=self.selected_width[:batch_size],
            last_face_indices=self.last_face_indices[:batch_size],
            selected_suit_codes=self.selected_suit_codes[:batch_size],
            done=self.done[:batch_size],
            choice_counts=self.choice_counts[:batch_size],
        )


def _sample_semantic_actions(
    *,
    action_batch: DeviceActionPlanBatch,
    generation_step_counts: Tensor,
    sampling_thresholds: Tensor,
    padded_generation_steps: int,
    logit_decoder: SemanticArgumentLogitDecoder,
    workspace: SemanticActionSamplerWorkspace,
) -> SemanticSamplingResult:
    """Sample semantic argument traces for a plan batch."""
    batch_size = action_batch.batch_size()
    assert batch_size <= workspace.batch_capacity
    assert generation_step_counts.shape == (batch_size,)
    assert sampling_thresholds.shape == (
        batch_size,
        padded_generation_steps,
    )
    assert generation_step_counts.dtype == torch.long
    assert sampling_thresholds.dtype == torch.float64
    workspace.reset(
        action_batch=action_batch,
        padded_generation_steps=padded_generation_steps,
        log_probability_dtype=torch.float32,
    )
    error_code = torch.zeros(
        (), dtype=torch.long, device=action_batch.device
    )
    for argument_index in range(padded_generation_steps):
        state = workspace.state_view(batch_size=batch_size)
        active_rows = (~state.done) & (
            state.step_counts < generation_step_counts
        )
        choices = _legal_token_candidates(
            workspace=workspace,
            batch=action_batch,
            state=state,
            active_rows=active_rows,
        )
        logits = logit_decoder.next_logits()
        sampled = sample_legal_candidates(
            argument_logits=logits,
            legal_candidates=choices,
            thresholds=sampling_thresholds[:, argument_index],
            active_rows=active_rows,
        )
        error_code = _merge_error_code(error_code, sampled.error_code)
        _record_replay_step(
            workspace=workspace,
            batch_size=batch_size,
            argument_index=argument_index,
            choices=choices,
            selected_offsets=sampled.selected_choice_offsets,
        )
        _advance_workspace_state(
            workspace=workspace,
            batch=action_batch,
            selected_token_ids=sampled.token_ids,
            choice_counts=choices.choice_counts,
            active_rows=active_rows,
        )
        _record_step_outputs(
            workspace=workspace,
            batch_size=batch_size,
            argument_index=argument_index,
            active_rows=active_rows,
            selected_token_ids=sampled.token_ids,
            selected_log_probabilities=sampled.selected_log_probabilities,
        )
        if argument_index + 1 < padded_generation_steps:
            logit_decoder.advance(sampled.token_ids)
    final_state = workspace.state_view(batch_size=batch_size)
    error_code = _set_error_if(
        error_code, (~final_state.done).any(), _ERROR_UNTERMINATED
    )
    error_value = _error_code_value(error_code)
    if error_value != 0:
        return Rejected(reason=_error_reason(error_value))
    flat_replay = _flat_active_replay(
        workspace=workspace,
        batch_size=batch_size,
        padded_generation_steps=padded_generation_steps,
    )
    return _result.Ok(
        value=SemanticActionSampleBatch(
            selected_token_ids_padded=(
                workspace.selected_token_ids[
                    :batch_size, :padded_generation_steps
                ]
            ),
            active_sample_indices=flat_replay.active_sample_indices,
            active_step_indices=flat_replay.active_step_indices,
            choice_token_ids=flat_replay.choice_token_ids,
            choice_masks=flat_replay.choice_masks,
            selected_choice_offsets=flat_replay.selected_choice_offsets,
            step_counts=workspace.step_counts[:batch_size],
            choice_counts=workspace.choice_counts[:batch_size],
            log_probabilities=(
                workspace.log_probabilities[:batch_size]
            ),
        )
    )


def _flat_active_replay(
    *,
    workspace: SemanticActionSamplerWorkspace,
    batch_size: int,
    padded_generation_steps: int,
) -> _FlatReplay:
    positions = torch.arange(
        padded_generation_steps,
        dtype=torch.long,
        device=workspace.device,
    ).unsqueeze(0)
    step_positions = positions.expand(batch_size, -1)
    sample_positions = torch.arange(
        batch_size,
        dtype=torch.long,
        device=workspace.device,
    ).unsqueeze(1)
    sample_positions = sample_positions.expand(
        -1, padded_generation_steps
    )
    active_mask = step_positions < workspace.step_counts[
        :batch_size
    ].unsqueeze(1)
    return _FlatReplay(
        active_sample_indices=sample_positions[active_mask],
        active_step_indices=step_positions[active_mask],
        choice_token_ids=workspace.replay_choice_ids[
            :batch_size, :padded_generation_steps, :
        ][active_mask],
        choice_masks=workspace.replay_choice_masks[
            :batch_size, :padded_generation_steps, :
        ][active_mask],
        selected_choice_offsets=workspace.replay_selected_offsets[
            :batch_size, :padded_generation_steps
        ][active_mask],
    )


def _legal_token_candidates(
    *,
    workspace: SemanticActionSamplerWorkspace,
    batch: DeviceActionPlanBatch,
    state: DeviceActionState,
    active_rows: Tensor,
) -> DeviceLegalCandidateBatch:
    batch_size = batch.batch_size()
    trace_ids, trace_mask = trace_set_candidates(
        batch=batch, state=state
    )
    selection_ids, selection_mask = selection_candidates(
        batch=batch, state=state
    )
    trace_ids = _padded_candidate_columns(
        destination=workspace.trace_candidate_ids[:batch_size],
        values=trace_ids,
    )
    trace_mask = _padded_candidate_columns(
        destination=workspace.trace_candidate_masks[:batch_size],
        values=trace_mask,
    )
    selection_ids = _padded_candidate_columns(
        destination=workspace.selection_candidate_ids[:batch_size],
        values=selection_ids,
    )
    selection_mask = _padded_candidate_columns(
        destination=workspace.selection_candidate_masks[:batch_size],
        values=selection_mask,
    )
    trace_rows = (batch.kind_codes == ACTION_KIND_TRACE_SET).unsqueeze(
        1
    )
    candidate_ids = torch.where(trace_rows, trace_ids, selection_ids)
    candidate_masks = torch.where(
        trace_rows, trace_mask, selection_mask
    )
    active = active_rows.unsqueeze(1)
    return legal_candidate_batch(
        candidate_token_ids=torch.where(
            active,
            candidate_ids,
            workspace.inactive_candidate_ids[:batch_size],
        ),
        candidate_mask=torch.where(
            active,
            candidate_masks,
            workspace.inactive_candidate_masks[:batch_size],
        ),
    )


def _record_replay_step(
    *,
    workspace: SemanticActionSamplerWorkspace,
    batch_size: int,
    argument_index: int,
    choices: DeviceLegalCandidateBatch,
    selected_offsets: Tensor,
) -> None:
    assert argument_index >= 0
    assert argument_index < SEMANTIC_CODEC.max_argument_tokens
    workspace.replay_choice_ids[:batch_size, argument_index, :].copy_(
        choices.token_ids.to(dtype=torch.int16)
    )
    workspace.replay_choice_masks[:batch_size, argument_index, :].copy_(
        choices.masks
    )
    workspace.replay_selected_offsets[
        :batch_size, argument_index
    ].copy_(selected_offsets)


def _padded_candidate_columns(
    *, destination: Tensor, values: Tensor
) -> Tensor:
    assert destination.ndim == 2
    assert values.ndim == 2
    assert int(values.shape[0]) == int(destination.shape[0])
    assert int(values.shape[1]) <= int(destination.shape[1])
    destination.zero_()
    destination[:, : int(values.shape[1])].copy_(values)
    return destination


def _record_step_outputs(
    *,
    workspace: SemanticActionSamplerWorkspace,
    batch_size: int,
    argument_index: int,
    active_rows: Tensor,
    selected_token_ids: Tensor,
    selected_log_probabilities: Tensor,
) -> None:
    assert argument_index >= 0
    workspace.log_probabilities[:batch_size].add_(
        torch.where(
            active_rows,
            selected_log_probabilities,
            torch.zeros_like(selected_log_probabilities),
        )
    )


def _advance_workspace_state(
    *,
    workspace: SemanticActionSamplerWorkspace,
    batch: DeviceActionPlanBatch,
    selected_token_ids: Tensor,
    choice_counts: Tensor,
    active_rows: Tensor,
) -> None:
    batch_size = batch.batch_size()
    state = workspace.state_view(batch_size=batch_size)
    tables = token_tables(batch.device)
    token_face = tables.face_indices.index_select(
        dim=0, index=selected_token_ids
    )
    token_count = tables.counts.index_select(
        dim=0, index=selected_token_ids
    )
    token_suit = safe_gather_2d(
        batch.effective_suits, token_face.clamp(min=0)
    )
    is_select = tables.is_select.index_select(
        dim=0, index=selected_token_ids
    )
    write_positions = state.step_counts.clamp(
        max=SEMANTIC_CODEC.max_argument_tokens - 1
    )
    row_indices = workspace.row_indices[:batch_size]
    current_tokens = state.selected_token_ids[
        row_indices, write_positions
    ]
    state.selected_token_ids[row_indices, write_positions] = (
        torch.where(
            active_rows,
            selected_token_ids,
            current_tokens,
        )
    )
    additions = torch.where(active_rows & is_select, token_count, 0)
    state.selected_counts.scatter_add_(
        dim=1,
        index=token_face.clamp(min=0).unsqueeze(1),
        src=additions.unsqueeze(1),
    )
    state.selected_width.add_(
        torch.where(active_rows & is_select, token_count, 0)
    )
    state.step_counts.add_(active_rows.to(dtype=torch.long))
    state.last_face_indices.copy_(
        torch.where(
            active_rows & is_select,
            token_face,
            state.last_face_indices,
        )
    )
    state.selected_suit_codes.copy_(
        torch.where(
            active_rows & is_select & (state.selected_suit_codes < 0),
            token_suit,
            state.selected_suit_codes,
        )
    )
    terminal_token = (
        selected_token_ids == SEMANTIC_CODEC.argument_pass_id
    ) | (selected_token_ids == SEMANTIC_CODEC.argument_stop_id)
    exact_done = (
        (batch.kind_codes == ACTION_KIND_DISCARD)
        | (batch.kind_codes == ACTION_KIND_FOLLOW)
    ) & (state.selected_width == batch.exact_select)
    trace_done_result = trace_done(
        batch=batch,
        selected_token_ids=state.selected_token_ids,
        step_counts=state.step_counts,
    )
    state.done.copy_(
        state.done
        | (
            active_rows
            & (terminal_token | exact_done | trace_done_result)
        )
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
    assert current.shape == ()
    assert incoming.shape == ()
    return torch.where(
        (current == 0) & (incoming != 0), incoming, current
    )


def _error_code_value(error_code: Tensor) -> int:
    return int(error_code.detach().cpu().item())


def _error_reason(error_code: int) -> str:
    if error_code == _ERROR_UNTERMINATED:
        return "policy semantic action did not terminate"
    sampling_reason = sample_legal_token_error_reason(error_code)
    if sampling_reason is not None:
        return sampling_reason
    return "policy sampling failed"
