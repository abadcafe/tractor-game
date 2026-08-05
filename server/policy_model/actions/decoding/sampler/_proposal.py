"""Exact structured Gumbel Top-k over the legal action prefix tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from server.foundation.result import Ok, Rejected
from server.policy_model._schema.actions import (
    ACTION_CHOICE_COUNT,
    MAX_ACTION_STEPS,
    PASS_CHOICE_ID,
)
from server.policy_model.actions.decoding.device import (
    DeviceActionPlanBatch,
)

from ._executor import (
    ActionSamplerWorkspace,
    advance_action_state,
    legal_action_choices,
)


class ActionProposalLogitDecoder(Protocol):
    """Prefix decoder that can fork arbitrary selected beam parents."""

    def next_choice_logits(
        self,
        active_rows: Tensor,
        scored_rows: Tensor,
    ) -> Tensor:
        """Return logits for ambiguous active beam rows."""
        ...

    def fork(
        self,
        *,
        parent_rows: Tensor,
        selected_choice_ids: Tensor,
        active_rows: Tensor,
    ) -> None:
        """Replace beam lanes with selected child prefixes."""
        ...


@dataclass(frozen=True, slots=True)
class ActionProposalBatch:
    """Complete unique traces ordered by perturbed root score."""

    choice_ids_padded: Tensor
    step_counts: Tensor
    perturbed_root_scores: Tensor

    def __post_init__(self) -> None:
        row_count, step_capacity = self.choice_ids_padded.shape
        assert row_count > 0
        assert 0 < step_capacity <= MAX_ACTION_STEPS
        assert self.step_counts.shape == (row_count,)
        assert self.perturbed_root_scores.shape == (row_count,)
        assert self.choice_ids_padded.dtype == torch.long
        assert self.step_counts.dtype == torch.long


@dataclass(slots=True)
class ActionProposalSampler:
    """Own reusable grammar state for stochastic beam search."""

    workspace: ActionSamplerWorkspace

    @classmethod
    def create(
        cls,
        *,
        candidate_capacity: int,
        device: torch.device,
    ) -> ActionProposalSampler:
        assert candidate_capacity > 0
        return cls(
            workspace=ActionSamplerWorkspace(
                batch_capacity=candidate_capacity,
                device=device,
            )
        )

    def propose(
        self,
        *,
        action_batch: DeviceActionPlanBatch,
        generation_steps: int,
        root_gumbel: Tensor,
        edge_gumbels: Tensor,
        logit_decoder: ActionProposalLogitDecoder,
    ) -> Ok[ActionProposalBatch] | Rejected:
        """Return exact Gumbel Top-k leaves without replacement."""
        width = action_batch.batch_size()
        assert width <= self.workspace.batch_capacity
        assert 0 < generation_steps <= MAX_ACTION_STEPS
        assert root_gumbel.shape == ()
        assert edge_gumbels.shape == (
            generation_steps,
            width,
            ACTION_CHOICE_COUNT,
        )
        self.workspace.reset(
            action_batch=action_batch,
            padded_generation_steps=generation_steps,
            log_probability_dtype=edge_gumbels.dtype,
        )
        prefix_log_probabilities = torch.full(
            (width,),
            -torch.inf,
            dtype=edge_gumbels.dtype,
            device=action_batch.device,
        )
        prefix_scores = torch.full_like(
            prefix_log_probabilities, -torch.inf
        )
        prefix_log_probabilities[0] = 0.0
        prefix_scores[0] = root_gumbel
        beam_count = 1
        rows = torch.arange(
            width,
            dtype=torch.long,
            device=action_batch.device,
        )
        for step_index in range(generation_steps):
            state = self.workspace.state_view(batch_size=width)
            in_beam = rows < beam_count
            active_rows = in_beam & ~state.done
            completed_rows = in_beam & state.done
            if not bool(active_rows.any().item()):
                break
            legal = legal_action_choices(
                workspace=self.workspace,
                batch=action_batch,
                state=state,
                active_rows=active_rows,
            )
            if bool(
                (active_rows & legal.choice_counts.eq(0)).any().item()
            ):
                return Rejected(
                    reason="action proposal prefix has no legal choice"
                )
            scored_rows = active_rows & legal.choice_counts.gt(1)
            logits = logit_decoder.next_choice_logits(
                active_rows,
                scored_rows,
            )
            valid_logits = logits[
                legal.masks & active_rows.unsqueeze(1)
            ]
            if bool((~torch.isfinite(valid_logits)).any().item()):
                return Rejected(
                    reason="policy proposal logits must be finite"
                )
            masked_logits = logits.masked_fill(~legal.masks, -torch.inf)
            choice_log_probabilities = torch.log_softmax(
                masked_logits, dim=1
            )
            child_log_probabilities = (
                prefix_log_probabilities.unsqueeze(1)
                + choice_log_probabilities
            )
            raw_child_scores = (
                child_log_probabilities + edge_gumbels[step_index]
            ).masked_fill(
                ~(legal.masks & active_rows.unsqueeze(1)),
                -torch.inf,
            )
            child_scores = _condition_child_gumbels(
                raw_scores=raw_child_scores,
                legal_masks=(legal.masks & active_rows.unsqueeze(1)),
                parent_scores=prefix_scores,
            )
            flattened_child_scores = child_scores.reshape(-1)
            carried_scores = torch.where(
                completed_rows,
                prefix_scores,
                torch.full_like(prefix_scores, -torch.inf),
            )
            option_scores = torch.cat(
                (flattened_child_scores, carried_scores)
            )
            valid_option_count = int(
                torch.isfinite(option_scores).sum().item()
            )
            if valid_option_count == 0:
                return Rejected(
                    reason="action proposal produced no complete path"
                )
            next_count = min(width, valid_option_count)
            selected = torch.topk(
                option_scores,
                k=next_count,
                largest=True,
                sorted=True,
            )
            child_option_count = width * ACTION_CHOICE_COUNT
            selected_children = selected.indices < child_option_count
            child_indices = selected.indices.clamp(
                max=child_option_count - 1
            )
            child_parent_rows = torch.div(
                child_indices,
                ACTION_CHOICE_COUNT,
                rounding_mode="floor",
            )
            child_choice_ids = child_indices.remainder(
                ACTION_CHOICE_COUNT
            )
            carry_parent_rows = (
                selected.indices - child_option_count
            ).clamp(min=0, max=width - 1)
            parent_rows = torch.where(
                selected_children,
                child_parent_rows,
                carry_parent_rows,
            )
            selected_choice_ids = torch.where(
                selected_children,
                child_choice_ids,
                torch.full_like(child_choice_ids, PASS_CHOICE_ID),
            )
            selected_log_probabilities = torch.where(
                selected_children,
                child_log_probabilities[
                    child_parent_rows, child_choice_ids
                ],
                prefix_log_probabilities[carry_parent_rows],
            )
            selected_choice_counts = torch.where(
                selected_children,
                legal.choice_counts.index_select(0, child_parent_rows),
                torch.zeros_like(child_parent_rows),
            )
            padded_parents = torch.zeros(
                (width,),
                dtype=torch.long,
                device=action_batch.device,
            )
            padded_choices = torch.full(
                (width,),
                PASS_CHOICE_ID,
                dtype=torch.long,
                device=action_batch.device,
            )
            child_rows = rows < next_count
            padded_parents[:next_count] = parent_rows
            padded_choices[:next_count] = selected_choice_ids
            self.workspace.fork_state(
                parent_rows=parent_rows,
                row_count=next_count,
                padded_generation_steps=generation_steps,
            )
            advance_action_state(
                workspace=self.workspace,
                batch=action_batch,
                selected_choice_ids=padded_choices,
                choice_counts=torch.cat(
                    (
                        selected_choice_counts,
                        torch.zeros(
                            (width - next_count,),
                            dtype=torch.long,
                            device=action_batch.device,
                        ),
                    )
                ),
                active_rows=child_rows
                & torch.cat(
                    (
                        selected_children,
                        torch.zeros(
                            (width - next_count,),
                            dtype=torch.bool,
                            device=action_batch.device,
                        ),
                    )
                ),
            )
            active_selected = child_rows & torch.cat(
                (
                    selected_children,
                    torch.zeros(
                        (width - next_count,),
                        dtype=torch.bool,
                        device=action_batch.device,
                    ),
                )
            )
            next_state = self.workspace.state_view(batch_size=width)
            active_selected = active_selected & ~next_state.done
            if step_index + 1 < generation_steps:
                logit_decoder.fork(
                    parent_rows=padded_parents,
                    selected_choice_ids=padded_choices,
                    active_rows=active_selected,
                )
            else:
                assert not bool(active_selected.any().item())
            _ = prefix_log_probabilities.fill_(-torch.inf)
            _ = prefix_scores.fill_(-torch.inf)
            prefix_log_probabilities[:next_count] = (
                selected_log_probabilities
            )
            prefix_scores[:next_count] = selected.values
            beam_count = next_count
        final_state = self.workspace.state_view(batch_size=width)
        if bool((~final_state.done[:beam_count]).any().item()):
            return Rejected(
                reason="action proposal exceeded generation steps"
            )
        return Ok(
            ActionProposalBatch(
                choice_ids_padded=self.workspace.choice_ids[
                    :beam_count, :generation_steps
                ].clone(),
                step_counts=final_state.step_counts[
                    :beam_count
                ].clone(),
                perturbed_root_scores=prefix_scores[
                    :beam_count
                ].clone(),
            )
        )


def _condition_child_gumbels(
    *,
    raw_scores: Tensor,
    legal_masks: Tensor,
    parent_scores: Tensor,
) -> Tensor:
    """Condition every sibling maximum to its parent's Gumbel."""
    assert raw_scores.shape == legal_masks.shape
    assert parent_scores.shape == (int(raw_scores.shape[0]),)
    safe_raw = torch.where(
        legal_masks,
        raw_scores,
        torch.zeros_like(raw_scores),
    )
    maxima = raw_scores.max(dim=1).values
    deltas = safe_raw - maxima.unsqueeze(1)
    strictly_below = deltas < 0.0
    tail_logs = torch.where(
        strictly_below,
        -safe_raw + torch.log1p(-torch.exp(deltas)),
        torch.full_like(safe_raw, -torch.inf),
    )
    conditioned = -torch.logaddexp(
        -parent_scores.unsqueeze(1),
        tail_logs,
    )
    return conditioned.masked_fill(~legal_masks, -torch.inf)


__all__ = (
    "ActionProposalBatch",
    "ActionProposalLogitDecoder",
    "ActionProposalSampler",
)
