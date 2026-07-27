"""Exact-length teacher-forced scoring for observed actions."""

from __future__ import annotations

import math

import torch

from server.foundation.result import Ok, Rejected
from server.training.model import TractorPolicyModel
from server.training.packed_observation import (
    PackedObservation,
    pack_observation,
)
from server.training.semantic_action_plan import (
    ActionPlanFrame,
    ActionSampler,
    action_trace_choice_ids,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.training.tensorize import tensorize_packed_observations

from ..model import ActionScoreRequest
from ._batches import deduplicate_packed
from ._forced import resolve_forced_actions


def score_actions(
    *,
    model: TractorPolicyModel,
    device: torch.device,
    sampler: ActionSampler,
    requests: tuple[ActionScoreRequest, ...],
) -> Ok[tuple[float, ...]] | Rejected:
    """Score supplied actions without value-head or padding work."""
    assert requests
    plans = tuple(
        compile_legal_action_frame(request.query.legal_actions)
        for request in requests
    )
    trace_ids = tuple(
        action_trace_choice_ids(request.action.trace)
        for request in requests
    )
    for request in requests:
        decoded = request.query.legal_actions.decode(
            request.action.trace
        )
        if isinstance(decoded, Rejected):
            return decoded
        if decoded.value != request.action:
            return Rejected(
                reason="model action does not match legal action index"
            )
    forced = resolve_forced_actions(
        plans=plans,
        legal_actions=tuple(
            request.query.legal_actions for request in requests
        ),
        sampler=sampler,
        device=device,
    )
    if isinstance(forced, Rejected):
        return forced
    results: list[float | None] = [None for _request in requests]
    unforced_indices: list[int] = []
    for request_index, forced_action in enumerate(forced.value):
        if forced_action is None:
            unforced_indices.append(request_index)
            continue
        if forced_action != requests[request_index].action:
            return Rejected(
                reason="action differs from the forced legal action"
            )
        results[request_index] = 0.0
    if unforced_indices:
        packed = tuple(
            pack_observation(request.query.observation)
            for request in requests
        )
        groups = _score_groups(
            indices=tuple(unforced_indices),
            packed=packed,
            trace_ids=trace_ids,
        )
        for group in groups:
            scored = _score_group(
                model=model,
                device=device,
                sampler=sampler,
                requests=requests,
                plans=plans,
                packed=packed,
                trace_ids=trace_ids,
                request_indices=group,
            )
            if isinstance(scored, Rejected):
                return scored
            for request_index, value in scored.value:
                assert results[request_index] is None
                results[request_index] = value
    assert all(value is not None for value in results)
    return Ok(tuple(value for value in results if value is not None))


def _score_group(
    *,
    model: TractorPolicyModel,
    device: torch.device,
    sampler: ActionSampler,
    requests: tuple[ActionScoreRequest, ...],
    plans: tuple[ActionPlanFrame, ...],
    packed: tuple[PackedObservation, ...],
    trace_ids: tuple[tuple[int, ...], ...],
    request_indices: tuple[int, ...],
) -> Ok[tuple[tuple[int, float], ...]] | Rejected:
    packed_rows = deduplicate_packed(
        tuple(packed[index] for index in request_indices)
    )
    observation_batch = tensorize_packed_observations(
        observations=packed_rows.unique,
        device=device,
    )
    encoding = model.encode_observations(observation_batch)
    source_rows = torch.tensor(
        packed_rows.source_indices,
        dtype=torch.long,
        device=device,
    )
    selected = torch.tensor(
        tuple(trace_ids[index] for index in request_indices),
        dtype=torch.long,
        device=device,
    )
    trace_length = int(selected.shape[1])
    step_counts = torch.full(
        (len(request_indices),),
        trace_length,
        dtype=torch.long,
        device=device,
    )
    decoder = model.begin_action_decode_session(
        encoding,
        source_rows=source_rows,
        max_steps=trace_length,
    )
    scored = sampler.score(
        action_batch=plan_batch_to_device(
            tuple(plans[index] for index in request_indices),
            device=device,
        ),
        selected_choice_ids=selected,
        step_counts=step_counts,
        padded_generation_steps=trace_length,
        logit_decoder=decoder,
    )
    if isinstance(scored, Rejected):
        return scored
    probabilities = _float_tuple(scored.value.log_probabilities)
    if not all(math.isfinite(value) for value in probabilities):
        return Rejected(
            reason="model action probability must be finite"
        )
    return Ok(tuple(zip(request_indices, probabilities, strict=True)))


def _score_groups(
    *,
    indices: tuple[int, ...],
    packed: tuple[PackedObservation, ...],
    trace_ids: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    groups: dict[tuple[int, int], list[int]] = {}
    for index in indices:
        key = (packed[index].token_count(), len(trace_ids[index]))
        groups.setdefault(key, []).append(index)
    return tuple(tuple(group) for group in groups.values())


def _float_tuple(values: torch.Tensor) -> tuple[float, ...]:
    assert values.ndim == 1
    cpu_values = values.detach().cpu()
    return tuple(
        float(cpu_values[index].item())
        for index in range(int(cpu_values.shape[0]))
    )


__all__ = ("score_actions",)
