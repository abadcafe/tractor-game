"""Direct policy proposal and complete-action value improvement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from server.foundation.result import Ok, Rejected
from server.policy_model.actions.decoding import (
    ActionProposalSampler,
    ActionSampler,
    action_trace_choice_ids,
)
from server.policy_model.network import PolicyActionModel
from server.policy_model.observation.packing import (
    PackedObservation,
    pack_observation,
)
from server.policy_model.observation.tensor_batch import (
    tensorize_packed_observations,
)

from ._batches import attention_batch_groups, deduplicate_packed
from ._proposals import (
    ActionProposalRequest,
    ActionProposals,
    ProposedAction,
    propose_actions,
)
from .contracts import ActionDecision, ActionDecisionRequest


@dataclass(frozen=True, slots=True)
class _CandidateRow:
    request_index: int
    candidate: ProposedAction


def decide_actions(
    *,
    model: PolicyActionModel,
    device: torch.device,
    sampler: ActionSampler,
    proposal_sampler: ActionProposalSampler,
    requests: tuple[ActionDecisionRequest, ...],
    max_batch_rows: int,
) -> Ok[tuple[ActionDecision, ...]] | Rejected:
    """Select actions from policy proposals with batched ``Q(o, a)``."""
    assert requests
    proposals = propose_actions(
        model=model,
        device=device,
        sampler=sampler,
        proposal_sampler=proposal_sampler,
        requests=tuple(
            ActionProposalRequest(
                query=request.query,
                candidate_count=request.candidate_count,
                draw=request.draw,
            )
            for request in requests
        ),
        max_batch_rows=max_batch_rows,
    )
    if isinstance(proposals, Rejected):
        return proposals
    candidate_rows = tuple(
        _CandidateRow(
            request_index=request_index,
            candidate=candidate,
        )
        for request_index, request_proposals in enumerate(
            proposals.value
        )
        if len(request_proposals.candidates) > 1
        for candidate in request_proposals.candidates
    )
    action_values_result = _evaluate_action_values(
        model=model,
        device=device,
        requests=requests,
        rows=candidate_rows,
        max_batch_rows=max_batch_rows,
    )
    if isinstance(action_values_result, Rejected):
        return action_values_result
    return Ok(
        _select_decisions(
            requests=requests,
            proposals=proposals.value,
            candidate_rows=candidate_rows,
            action_values=action_values_result.value,
        )
    )


def _evaluate_action_values(
    *,
    model: PolicyActionModel,
    device: torch.device,
    requests: tuple[ActionDecisionRequest, ...],
    rows: tuple[_CandidateRow, ...],
    max_batch_rows: int,
) -> Ok[tuple[float, ...]] | Rejected:
    if not rows:
        return Ok(())
    packed = tuple(
        pack_observation(requests[row.request_index].query.observation)
        for row in rows
    )
    trace_ids = tuple(
        action_trace_choice_ids(row.candidate.action.trace)
        for row in rows
    )
    values: list[float | None] = [None for _row in rows]
    for group in attention_batch_groups(
        packed_rows=packed,
        indices=tuple(range(len(rows))),
        max_rows=max_batch_rows,
    ):
        evaluated = _evaluate_group(
            model=model,
            device=device,
            packed=packed,
            trace_ids=trace_ids,
            row_indices=group,
        )
        for row_index, value in zip(group, evaluated, strict=True):
            if not math.isfinite(value):
                return Rejected(
                    reason="model action value must be finite"
                )
            values[row_index] = value
    assert all(value is not None for value in values)
    return Ok(tuple(value for value in values if value is not None))


def _evaluate_group(
    *,
    model: PolicyActionModel,
    device: torch.device,
    packed: tuple[PackedObservation, ...],
    trace_ids: tuple[tuple[int, ...], ...],
    row_indices: tuple[int, ...],
) -> tuple[float, ...]:
    packed_rows = deduplicate_packed(
        tuple(packed[index] for index in row_indices)
    )
    encoding = model.encode_action_value_observations(
        tensorize_packed_observations(
            observations=packed_rows.unique,
            device=device,
        )
    )
    max_steps = max(len(trace_ids[index]) for index in row_indices)
    choice_ids_padded = torch.tensor(
        tuple(
            trace_ids[index]
            + (0,) * (max_steps - len(trace_ids[index]))
            for index in row_indices
        ),
        dtype=torch.long,
        device=device,
    )
    step_counts = torch.tensor(
        tuple(len(trace_ids[index]) for index in row_indices),
        dtype=torch.long,
        device=device,
    )
    action_values = model.action_values(
        encoding,
        source_rows=torch.tensor(
            packed_rows.source_indices,
            dtype=torch.long,
            device=device,
        ),
        choice_ids_padded=choice_ids_padded,
        step_counts=step_counts,
    )
    cpu_values = action_values.detach().cpu()
    return tuple(
        float(cpu_values[index].item())
        for index in range(int(cpu_values.shape[0]))
    )


def _select_decisions(
    *,
    requests: tuple[ActionDecisionRequest, ...],
    proposals: tuple[ActionProposals, ...],
    candidate_rows: tuple[_CandidateRow, ...],
    action_values: tuple[float, ...],
) -> tuple[ActionDecision, ...]:
    values_by_request: list[list[float]] = [[] for _request in requests]
    for row, value in zip(candidate_rows, action_values, strict=True):
        values_by_request[row.request_index].append(value)
    decisions: list[ActionDecision] = []
    for request_index, request_proposals in enumerate(proposals):
        candidates = request_proposals.candidates
        if len(candidates) == 1:
            decisions.append(
                ActionDecision(
                    action=candidates[0].action,
                    candidate_count=1,
                    selected_action_value=None,
                )
            )
            continue
        values = values_by_request[request_index]
        assert len(values) == len(candidates)
        temperature = requests[request_index].action_value_temperature
        selected_index = max(
            range(len(candidates)),
            key=lambda index: (
                candidates[index].perturbed_root_score
                + values[index] / temperature
            ),
        )
        decisions.append(
            ActionDecision(
                action=candidates[selected_index].action,
                candidate_count=len(candidates),
                selected_action_value=values[selected_index],
            )
        )
    return tuple(decisions)


__all__ = ("decide_actions",)
