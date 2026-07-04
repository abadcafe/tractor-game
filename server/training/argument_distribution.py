"""Categorical distributions over legal semantic arguments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server import result as _result
from server.training.choice_trace import SemanticChoiceStep
from server.training.semantic_actions.arguments import SemanticArgument
from server.training.semantic_actions.codec import semantic_argument_id


@dataclass(frozen=True, slots=True)
class ArgumentDistribution:
    """Model scores restricted to one legal semantic-argument set."""

    logits: Tensor
    probabilities: Tensor
    log_probabilities: Tensor
    entropy: Tensor


@dataclass(frozen=True, slots=True)
class BatchedArgumentDistribution:
    """Selected log-probabilities and entropies for choices."""

    selected_log_probabilities: Tensor
    entropies: Tensor


def argument_distribution(
    *,
    argument_logits: Tensor,
    choices: tuple[SemanticArgument, ...],
) -> _result.Ok[ArgumentDistribution] | _result.Rejected:
    """Build the masked categorical distribution for legal arguments."""
    logits = argument_logits_for_choices(
        argument_logits=argument_logits,
        choices=choices,
    )
    if not _all_finite(logits):
        return _result.Rejected(
            reason="policy argument logits must be finite"
        )
    probabilities = torch.softmax(logits, dim=0)
    log_probabilities = torch.log_softmax(logits, dim=0)
    entropy = -(probabilities * log_probabilities).sum()
    if (
        not _all_finite(probabilities)
        or not _all_finite(log_probabilities)
        or not _all_finite(entropy)
    ):
        return _result.Rejected(
            reason="policy argument distribution must be finite"
        )
    return _result.Ok(
        value=ArgumentDistribution(
            logits=logits,
            probabilities=probabilities,
            log_probabilities=log_probabilities,
            entropy=entropy,
        )
    )


def batched_argument_distribution(
    *,
    argument_logits: Tensor,
    choice_steps: tuple[SemanticChoiceStep, ...],
) -> _result.Ok[BatchedArgumentDistribution] | _result.Rejected:
    """Build masked categorical distributions for selected choices."""
    assert argument_logits.ndim == 2
    assert choice_steps
    assert int(argument_logits.shape[0]) == len(choice_steps)
    choice_ids, choice_masks, selected_offsets = _choice_step_tensors(
        choice_steps=choice_steps,
        device=argument_logits.device,
    )
    choice_logits = argument_logits.gather(dim=1, index=choice_ids)
    valid_logits = choice_logits[choice_masks]
    if not _all_finite(valid_logits):
        return _result.Rejected(
            reason="policy argument logits must be finite"
        )
    masked_logits = choice_logits.masked_fill(~choice_masks, -torch.inf)
    probabilities = torch.softmax(masked_logits, dim=1).masked_fill(
        ~choice_masks, 0.0
    )
    log_probabilities = torch.log_softmax(
        masked_logits, dim=1
    ).masked_fill(~choice_masks, 0.0)
    selected_log_probabilities = log_probabilities.gather(
        dim=1, index=selected_offsets.unsqueeze(1)
    ).squeeze(1)
    entropies = -(probabilities * log_probabilities).sum(dim=1)
    if not _all_finite(selected_log_probabilities) or not _all_finite(
        entropies
    ):
        return _result.Rejected(
            reason="policy argument distribution must be finite"
        )
    return _result.Ok(
        value=BatchedArgumentDistribution(
            selected_log_probabilities=selected_log_probabilities,
            entropies=entropies,
        )
    )


def argument_logits_for_choices(
    *,
    argument_logits: Tensor,
    choices: tuple[SemanticArgument, ...],
) -> Tensor:
    """Select one logit per legal argument from one model output row."""
    assert choices
    ids = tuple(semantic_argument_id(argument) for argument in choices)
    index = torch.tensor(
        ids,
        dtype=torch.long,
        device=argument_logits.device,
    )
    return argument_logits.index_select(dim=0, index=index)


def _choice_step_tensors(
    *,
    choice_steps: tuple[SemanticChoiceStep, ...],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    max_choice_count = max(
        len(step.allowed_argument_ids) for step in choice_steps
    )
    choice_id_rows: list[tuple[int, ...]] = []
    choice_mask_rows: list[tuple[bool, ...]] = []
    selected_offsets: list[int] = []
    for step in choice_steps:
        pad_count = max_choice_count - len(step.allowed_argument_ids)
        choice_id_rows.append(
            (*step.allowed_argument_ids, *(0 for _ in range(pad_count)))
        )
        choice_mask_rows.append(
            (
                *(True for _ in step.allowed_argument_ids),
                *(False for _ in range(pad_count)),
            )
        )
        selected_offsets.append(step.selected_argument_offset)
    return (
        torch.tensor(choice_id_rows, dtype=torch.long, device=device),
        torch.tensor(choice_mask_rows, dtype=torch.bool, device=device),
        torch.tensor(selected_offsets, dtype=torch.long, device=device),
    )


def _all_finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu().item())
