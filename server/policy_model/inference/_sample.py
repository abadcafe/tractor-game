"""Exact-length, unique-prefix live action sampling."""

from __future__ import annotations

import math

import torch

from server.foundation.result import Ok, Rejected
from server.policy_model.actions.decoding import (
    ActionPlanFrame,
    ActionSampler,
    action_plan_generation_step_count,
    action_trace_from_choice_ids,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.policy_model.network import (
    EncodedObservation,
    PolicyValueModel,
)
from server.policy_model.observation.packing import (
    PackedObservation,
    pack_observation,
)
from server.policy_model.observation.tensor_batch import (
    tensorize_packed_observations,
)

from ._batches import attention_batch_groups, deduplicate_packed
from ._forced import resolve_forced_actions
from .contracts import (
    ActionSamples,
    PolicySampleRequest,
    SampledAction,
    SamplingSeed,
)
from .randomness import choice_threshold

_FLOAT32_BELOW_ONE = float.fromhex("0x1.fffffep-1")


def sample_actions(
    *,
    model: PolicyValueModel,
    device: torch.device,
    sampler: ActionSampler,
    requests: tuple[PolicySampleRequest, ...],
    max_batch_rows: int,
) -> Ok[tuple[ActionSamples, ...]] | Rejected:
    """Sample requests through padding-aware observation batches."""
    return _execute_actions(
        model=model,
        device=device,
        sampler=sampler,
        requests=requests,
        max_batch_rows=max_batch_rows,
    )


def _execute_actions(
    *,
    model: PolicyValueModel,
    device: torch.device,
    sampler: ActionSampler,
    requests: tuple[PolicySampleRequest, ...],
    max_batch_rows: int,
) -> Ok[tuple[ActionSamples, ...]] | Rejected:
    assert requests
    plans = tuple(
        compile_legal_action_frame(request.query.legal_actions)
        for request in requests
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
    results: list[ActionSamples | None] = [
        None for _request in requests
    ]
    unforced_indices: list[int] = []
    for request_index, action in enumerate(forced.value):
        if action is None:
            unforced_indices.append(request_index)
            continue
        results[request_index] = ActionSamples(
            samples=tuple(
                SampledAction(
                    action=action,
                    log_probability=0.0,
                )
                for _draw in requests[request_index].draws
            )
        )
    if unforced_indices:
        packed = tuple(
            pack_observation(request.query.observation)
            for request in requests
        )
        token_groups = attention_batch_groups(
            packed_rows=packed,
            indices=tuple(unforced_indices),
            max_rows=max_batch_rows,
        )
        for token_group in token_groups:
            sampled = _sample_token_group(
                model=model,
                device=device,
                sampler=sampler,
                requests=requests,
                plans=plans,
                packed=packed,
                request_indices=token_group,
            )
            if isinstance(sampled, Rejected):
                return sampled
            for request_index, request_samples in sampled.value:
                assert results[request_index] is None
                results[request_index] = request_samples
    assert all(item is not None for item in results)
    return Ok(tuple(item for item in results if item is not None))


def _sample_token_group(
    *,
    model: PolicyValueModel,
    device: torch.device,
    sampler: ActionSampler,
    requests: tuple[PolicySampleRequest, ...],
    plans: tuple[ActionPlanFrame, ...],
    packed: tuple[PackedObservation, ...],
    request_indices: tuple[int, ...],
) -> Ok[tuple[tuple[int, ActionSamples], ...]] | Rejected:
    packed_rows = deduplicate_packed(
        tuple(packed[index] for index in request_indices)
    )
    observation_batch = tensorize_packed_observations(
        observations=packed_rows.unique,
        device=device,
    )
    encoding = model.encode_observations(observation_batch)
    source_by_request = {
        request_index: packed_rows.source_indices[group_index]
        for group_index, request_index in enumerate(request_indices)
    }
    generation_steps = tuple(
        action_plan_generation_step_count(plans[index])
        for index in range(len(requests))
    )
    step_groups = _index_groups(
        indices=request_indices,
        values=generation_steps,
    )
    result: list[tuple[int, ActionSamples]] = []
    for step_group in step_groups:
        sampled = _sample_step_group(
            model=model,
            device=device,
            sampler=sampler,
            encoding=encoding,
            requests=requests,
            plans=plans,
            source_by_request=source_by_request,
            request_indices=step_group,
            generation_steps=generation_steps[step_group[0]],
        )
        if isinstance(sampled, Rejected):
            return sampled
        result.extend(sampled.value)
    return Ok(tuple(result))


def _sample_step_group(
    *,
    model: PolicyValueModel,
    device: torch.device,
    sampler: ActionSampler,
    encoding: EncodedObservation,
    requests: tuple[PolicySampleRequest, ...],
    plans: tuple[ActionPlanFrame, ...],
    source_by_request: dict[int, int],
    request_indices: tuple[int, ...],
    generation_steps: int,
) -> Ok[tuple[tuple[int, ActionSamples], ...]] | Rejected:
    flat_request_indices = tuple(
        request_index
        for request_index in request_indices
        for _draw in requests[request_index].draws
    )
    flat_draws = tuple(
        draw
        for request_index in request_indices
        for draw in requests[request_index].draws
    )
    source_rows = torch.tensor(
        tuple(
            source_by_request[request_index]
            for request_index in flat_request_indices
        ),
        dtype=torch.long,
        device=device,
    )
    action_batch = plan_batch_to_device(
        tuple(plans[index] for index in flat_request_indices),
        device=device,
    )
    generation_step_counts = torch.full(
        (len(flat_request_indices),),
        generation_steps,
        dtype=torch.long,
        device=device,
    )
    thresholds = _sampling_thresholds(
        requests=requests,
        request_indices=flat_request_indices,
        draws=flat_draws,
        generation_steps=generation_steps,
        device=device,
    )
    decoder = model.begin_action_decode_session(
        encoding,
        source_rows=source_rows,
        max_steps=generation_steps,
    )
    sampled = sampler.sample(
        action_batch=action_batch,
        generation_step_counts=generation_step_counts,
        sampling_thresholds=thresholds,
        padded_generation_steps=generation_steps,
        logit_decoder=decoder,
    )
    if isinstance(sampled, Rejected):
        return sampled
    rows = sampled.value
    step_counts = _int_tuple(rows.step_counts)
    choice_ids = rows.choice_ids_padded.detach().cpu().clone()
    log_probabilities = _float_tuple(rows.log_probabilities)
    if not all(math.isfinite(value) for value in log_probabilities):
        return Rejected(
            reason="model sample probability must be finite"
        )
    samples_by_request: dict[int, list[SampledAction]] = {
        index: [] for index in request_indices
    }
    for row_index, request_index in enumerate(flat_request_indices):
        trace_ids = tuple(
            int(choice_ids[row_index, step].item())
            for step in range(step_counts[row_index])
        )
        trace = action_trace_from_choice_ids(trace_ids)
        if isinstance(trace, Rejected):
            return trace
        decoded = requests[request_index].query.legal_actions.decode(
            trace.value
        )
        if isinstance(decoded, Rejected):
            return decoded
        samples_by_request[request_index].append(
            SampledAction(
                action=decoded.value,
                log_probability=log_probabilities[row_index],
            )
        )
    return Ok(
        tuple(
            (
                request_index,
                ActionSamples(
                    samples=tuple(samples_by_request[request_index])
                ),
            )
            for request_index in request_indices
        )
    )


def _sampling_thresholds(
    *,
    requests: tuple[PolicySampleRequest, ...],
    request_indices: tuple[int, ...],
    draws: tuple[SamplingSeed, ...],
    generation_steps: int,
    device: torch.device,
) -> torch.Tensor:
    rows = tuple(
        tuple(
            choice_threshold(
                sampling_seed=draw,
                seat=requests[request_index].query.seat,
                step_index=step_index,
            )
            for step_index in range(generation_steps)
        )
        for request_index, draw in zip(
            request_indices,
            draws,
            strict=True,
        )
    )
    if device.type != "mps":
        return torch.tensor(rows, dtype=torch.float64, device=device)
    return torch.tensor(
        tuple(
            tuple(min(value, _FLOAT32_BELOW_ONE) for value in row)
            for row in rows
        ),
        dtype=torch.float32,
        device=device,
    )


def _index_groups(
    *,
    indices: tuple[int, ...],
    values: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    groups: dict[int, list[int]] = {}
    for index in indices:
        groups.setdefault(values[index], []).append(index)
    return tuple(tuple(group) for group in groups.values())


def _int_tuple(values: torch.Tensor) -> tuple[int, ...]:
    assert values.ndim == 1
    cpu_values = values.detach().cpu()
    return tuple(
        int(cpu_values[index].item())
        for index in range(int(cpu_values.shape[0]))
    )


def _float_tuple(values: torch.Tensor) -> tuple[float, ...]:
    assert values.ndim == 1
    cpu_values = values.detach().cpu()
    return tuple(
        float(cpu_values[index].item())
        for index in range(int(cpu_values.shape[0]))
    )


__all__ = ("sample_actions",)
