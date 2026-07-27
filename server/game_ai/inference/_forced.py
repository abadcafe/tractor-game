"""Resolve mathematically unique legal actions without model logits."""

from __future__ import annotations

import torch

from server.foundation.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.semantic_action_plan import (
    ActionPlanFrame,
    ActionSampler,
    action_plan_generation_step_count,
    action_trace_from_choice_ids,
    plan_batch_to_device,
)
from server.training.semantic_actions import GeneratedAction


def resolve_forced_actions(
    *,
    plans: tuple[ActionPlanFrame, ...],
    legal_actions: tuple[LegalActionIndex, ...],
    sampler: ActionSampler,
    device: torch.device,
) -> Ok[tuple[GeneratedAction | None, ...]] | Rejected:
    """Return actions when the complete legal trace is unique."""
    assert plans
    assert len(plans) == len(legal_actions)
    generation_steps = tuple(
        action_plan_generation_step_count(plan) for plan in plans
    )
    padded_steps = max(generation_steps)
    resolved = sampler.resolve_forced(
        action_batch=plan_batch_to_device(plans, device=device),
        generation_step_counts=torch.tensor(
            generation_steps,
            dtype=torch.long,
            device=device,
        ),
        padded_generation_steps=padded_steps,
    )
    if isinstance(resolved, Rejected):
        return resolved
    forced_mask = _bool_tuple(resolved.value.forced_mask)
    step_counts = _int_tuple(resolved.value.step_counts)
    choice_ids = resolved.value.choice_ids_padded.detach().cpu().clone()
    actions: list[GeneratedAction | None] = []
    for row_index, is_forced in enumerate(forced_mask):
        if not is_forced:
            actions.append(None)
            continue
        trace_ids = tuple(
            int(choice_ids[row_index, step_index].item())
            for step_index in range(step_counts[row_index])
        )
        trace = action_trace_from_choice_ids(trace_ids)
        if isinstance(trace, Rejected):
            return trace
        decoded = legal_actions[row_index].decode(trace.value)
        if isinstance(decoded, Rejected):
            return decoded
        actions.append(decoded.value)
    return Ok(tuple(actions))


def _bool_tuple(values: torch.Tensor) -> tuple[bool, ...]:
    assert values.ndim == 1
    cpu_values = values.detach().cpu()
    return tuple(
        bool(cpu_values[index].item())
        for index in range(int(cpu_values.shape[0]))
    )


def _int_tuple(values: torch.Tensor) -> tuple[int, ...]:
    assert values.ndim == 1
    cpu_values = values.detach().cpu()
    return tuple(
        int(cpu_values[index].item())
        for index in range(int(cpu_values.shape[0]))
    )


__all__ = ("resolve_forced_actions",)
