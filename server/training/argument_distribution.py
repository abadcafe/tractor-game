"""Categorical distributions over legal semantic arguments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server import result as _result
from server.training.semantic_actions.arguments import SemanticArgument
from server.training.semantic_actions.codec import semantic_argument_id


@dataclass(frozen=True, slots=True)
class ArgumentDistribution:
    """Model scores restricted to one legal semantic-argument set."""

    logits: Tensor
    probabilities: Tensor
    log_probabilities: Tensor
    entropy: Tensor


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


def _all_finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu().item())
