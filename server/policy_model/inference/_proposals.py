"""Structured no-replacement root action proposals."""

from __future__ import annotations

import hashlib
import math

import torch

from server.foundation.result import Ok, Rejected
from server.policy_model._schema.actions import (
    ACTION_CHOICE_COUNT,
)
from server.policy_model.actions.decoding import (
    ActionProposalSampler,
    ActionSampler,
    action_plan_generation_step_count,
    action_trace_from_choice_ids,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.policy_model.network import PolicyValueModel
from server.policy_model.observation.packing import pack_observation
from server.policy_model.observation.tensor_batch import (
    tensorize_packed_observations,
)

from ._batches import attention_batch_groups, deduplicate_packed
from ._forced import resolve_forced_actions
from .contracts import (
    ActionProposalRequest,
    ActionProposals,
    ProposedAction,
)

_U64_SCALE = 1.0 / float(1 << 64)


def propose_actions(
    *,
    model: PolicyValueModel,
    device: torch.device,
    sampler: ActionSampler,
    proposal_sampler: ActionProposalSampler,
    requests: tuple[ActionProposalRequest, ...],
    max_batch_rows: int,
) -> Ok[tuple[ActionProposals, ...]] | Rejected:
    """Generate exact structured Gumbel Top-k candidates."""
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
    results: list[ActionProposals | None] = [
        None for _request in requests
    ]
    pending: list[int] = []
    for request_index, action in enumerate(forced.value):
        if action is None:
            pending.append(request_index)
            continue
        results[request_index] = ActionProposals(
            candidates=(
                ProposedAction(
                    action=action,
                    log_probability=0.0,
                    perturbed_root_score=0.0,
                ),
            )
        )
    if pending:
        packed = tuple(
            pack_observation(request.query.observation)
            for request in requests
        )
        for group in attention_batch_groups(
            packed_rows=packed,
            indices=tuple(pending),
            max_rows=max_batch_rows,
        ):
            packed_rows = deduplicate_packed(
                tuple(packed[index] for index in group)
            )
            encoding = model.encode_observations(
                tensorize_packed_observations(
                    observations=packed_rows.unique,
                    device=device,
                )
            )
            for group_index, request_index in enumerate(group):
                request = requests[request_index]
                plan = plans[request_index]
                width = request.candidate_count
                score_dtype = (
                    torch.float32
                    if device.type == "mps"
                    else torch.float64
                )
                generation_steps = action_plan_generation_step_count(
                    plan
                )
                source_row = packed_rows.source_indices[group_index]
                decoder = model.begin_action_decode_session(
                    encoding,
                    source_rows=torch.full(
                        (width,),
                        source_row,
                        dtype=torch.long,
                        device=device,
                    ),
                    max_steps=generation_steps,
                )
                proposed = proposal_sampler.propose(
                    action_batch=plan_batch_to_device(
                        tuple(plan for _index in range(width)),
                        device=device,
                    ),
                    generation_steps=generation_steps,
                    root_gumbel=torch.tensor(
                        _gumbel(
                            request=request,
                            step_index=-1,
                            lane_index=0,
                            choice_index=0,
                        ),
                        dtype=score_dtype,
                        device=device,
                    ),
                    edge_gumbels=torch.tensor(
                        tuple(
                            tuple(
                                tuple(
                                    _gumbel(
                                        request=request,
                                        step_index=step_index,
                                        lane_index=lane_index,
                                        choice_index=choice_index,
                                    )
                                    for choice_index in range(
                                        ACTION_CHOICE_COUNT
                                    )
                                )
                                for lane_index in range(width)
                            )
                            for step_index in range(generation_steps)
                        ),
                        dtype=score_dtype,
                        device=device,
                    ),
                    logit_decoder=decoder,
                )
                if isinstance(proposed, Rejected):
                    return proposed
                decoded = _decode_proposals(
                    request=request,
                    choice_ids=(
                        proposed.value.choice_ids_padded.detach()
                        .cpu()
                        .clone()
                    ),
                    step_counts=(
                        proposed.value.step_counts.detach().cpu()
                    ),
                    log_probabilities=(
                        proposed.value.log_probabilities.detach().cpu()
                    ),
                    perturbed_scores=(
                        proposed.value.perturbed_root_scores.detach().cpu()
                    ),
                )
                if isinstance(decoded, Rejected):
                    return decoded
                results[request_index] = decoded.value
    assert all(result is not None for result in results)
    return Ok(tuple(result for result in results if result is not None))


def _decode_proposals(
    *,
    request: ActionProposalRequest,
    choice_ids: torch.Tensor,
    step_counts: torch.Tensor,
    log_probabilities: torch.Tensor,
    perturbed_scores: torch.Tensor,
) -> Ok[ActionProposals] | Rejected:
    candidates: list[ProposedAction] = []
    seen: set[tuple[int, ...]] = set()
    for row_index in range(int(step_counts.shape[0])):
        trace_ids = tuple(
            int(choice_ids[row_index, step_index].item())
            for step_index in range(int(step_counts[row_index].item()))
        )
        assert trace_ids not in seen
        seen.add(trace_ids)
        trace = action_trace_from_choice_ids(trace_ids)
        if isinstance(trace, Rejected):
            return trace
        action = request.query.legal_actions.decode(trace.value)
        if isinstance(action, Rejected):
            return action
        candidates.append(
            ProposedAction(
                action=action.value,
                log_probability=float(
                    log_probabilities[row_index].item()
                ),
                perturbed_root_score=float(
                    perturbed_scores[row_index].item()
                ),
            )
        )
    return Ok(ActionProposals(candidates=tuple(candidates)))


def _gumbel(
    *,
    request: ActionProposalRequest,
    step_index: int,
    lane_index: int,
    choice_index: int,
) -> float:
    digest = hashlib.blake2b(
        (
            f"proposal|{request.draw.seed}|{request.draw.ordinal}|"
            f"{step_index}|{lane_index}|{choice_index}"
        ).encode("utf-8"),
        digest_size=8,
        person=b"tractor",
    ).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    uniform = (float(integer) + 0.5) * _U64_SCALE
    return -math.log(-math.log(uniform))


__all__ = ("propose_actions",)
