"""Batched complete-action sampling from the learned policy."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from server.foundation.result import Ok, Rejected
from server.policy_model.actions import GeneratedAction
from server.policy_model.actions.decoding import (
    ActionPlanFrame,
    ActionSampler,
    action_plan_generation_step_count,
    action_trace_from_choice_ids,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.policy_model.network import PolicyModel
from server.policy_model.observation.packing import (
    PackedObservation,
    pack_observation,
)
from server.policy_model.observation.tensor_batch import (
    tensorize_packed_observations,
)

from ._batches import attention_batch_groups, deduplicate_packed
from ._forced import resolve_forced_actions
from .contracts import PolicyDecisionRequest, SamplingSeed

_U53_SCALE = 1.0 / float(1 << 53)


@dataclass(frozen=True, slots=True)
class _SamplingInputs:
    plans: tuple[ActionPlanFrame, ...]
    packed: tuple[PackedObservation, ...]


def sample_actions(
    *,
    model: PolicyModel,
    device: torch.device,
    sampler: ActionSampler,
    requests: tuple[PolicyDecisionRequest, ...],
    max_batch_rows: int,
) -> Ok[tuple[GeneratedAction, ...]] | Rejected:
    """Sample one exact legal action for every request."""
    assert requests
    inputs = _SamplingInputs(
        plans=tuple(
            compile_legal_action_frame(request.query.legal_actions)
            for request in requests
        ),
        packed=tuple(
            pack_observation(request.query.observation)
            for request in requests
        ),
    )
    forced = resolve_forced_actions(
        plans=inputs.plans,
        legal_actions=tuple(
            request.query.legal_actions for request in requests
        ),
        sampler=sampler,
        device=device,
    )
    if isinstance(forced, Rejected):
        return forced
    results: list[GeneratedAction | None] = list(forced.value)
    pending = tuple(
        index for index, action in enumerate(results) if action is None
    )
    if pending:
        for group in attention_batch_groups(
            packed_rows=inputs.packed,
            indices=pending,
            max_rows=max_batch_rows,
        ):
            sampled = _sample_group(
                model=model,
                device=device,
                sampler=sampler,
                requests=requests,
                inputs=inputs,
                indices=group,
            )
            if isinstance(sampled, Rejected):
                return sampled
            for request_index, action in zip(
                group, sampled.value, strict=True
            ):
                results[request_index] = action
    assert all(action is not None for action in results)
    return Ok(tuple(action for action in results if action is not None))


def _sample_group(
    *,
    model: PolicyModel,
    device: torch.device,
    sampler: ActionSampler,
    requests: tuple[PolicyDecisionRequest, ...],
    inputs: _SamplingInputs,
    indices: tuple[int, ...],
) -> Ok[tuple[GeneratedAction, ...]] | Rejected:
    packed = deduplicate_packed(
        tuple(inputs.packed[index] for index in indices)
    )
    encoding = model.encode_observations(
        tensorize_packed_observations(
            observations=packed.unique,
            device=device,
        )
    )
    plans = tuple(inputs.plans[index] for index in indices)
    generation_steps = tuple(
        action_plan_generation_step_count(plan) for plan in plans
    )
    padded_steps = max(generation_steps)
    decoder = model.begin_action_decode_session(
        encoding,
        source_rows=torch.tensor(
            packed.source_indices,
            dtype=torch.long,
            device=device,
        ),
        max_steps=padded_steps,
    )
    sampled = sampler.sample(
        action_batch=plan_batch_to_device(plans, device=device),
        generation_step_counts=torch.tensor(
            generation_steps,
            dtype=torch.long,
            device=device,
        ),
        sampling_thresholds=torch.tensor(
            tuple(
                tuple(
                    _policy_threshold(
                        draw=requests[index].draw,
                        step_index=step_index,
                    )
                    for step_index in range(padded_steps)
                )
                for index in indices
            ),
            dtype=(
                torch.float32 if device.type == "mps" else torch.float64
            ),
            device=device,
        ),
        padded_generation_steps=padded_steps,
        logit_decoder=decoder,
    )
    if isinstance(sampled, Rejected):
        return sampled
    choice_ids = sampled.value.choice_ids_padded.detach().cpu().clone()
    step_counts = sampled.value.step_counts.detach().cpu()
    actions: list[GeneratedAction] = []
    for row_index, request_index in enumerate(indices):
        trace = action_trace_from_choice_ids(
            tuple(
                int(choice_ids[row_index, step_index].item())
                for step_index in range(
                    int(step_counts[row_index].item())
                )
            )
        )
        if isinstance(trace, Rejected):
            return trace
        decoded = requests[request_index].query.legal_actions.decode(
            trace.value
        )
        if isinstance(decoded, Rejected):
            return decoded
        actions.append(decoded.value)
    return Ok(tuple(actions))


def _policy_threshold(*, draw: SamplingSeed, step_index: int) -> float:
    assert step_index >= 0
    digest = hashlib.blake2b(
        (f"policy|{draw.seed}|{draw.ordinal}|{step_index}").encode(
            "utf-8"
        ),
        digest_size=8,
        person=b"tractor",
    ).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return float(value >> 11) * _U53_SCALE


__all__ = ("sample_actions",)
