"""Torch helpers for scoring selection choices."""

from __future__ import annotations

import torch
from torch import Tensor

from server.training.model import (
    SelectionHeadOutput,
    TractorPolicyModel,
)
from server.training.selection_actions import (
    ActionQuery,
    SelectionChoice,
)
from server.training.tensorize import (
    ObservationTensorBatch,
    SelectionStateTensorBatch,
)


def forward_selection_head(
    *,
    model: TractorPolicyModel,
    query: ActionQuery,
    observation: ObservationTensorBatch,
    selection: SelectionStateTensorBatch,
) -> SelectionHeadOutput:
    """Run the model head matching the current decision kind."""
    if query.kind == "bid":
        return model.forward_bid(observation, selection)
    if query.kind == "stir":
        return model.forward_stir(observation, selection)
    if query.kind == "discard":
        return model.forward_discard(observation, selection)
    if query.kind == "lead_play":
        return model.forward_lead_play(observation, selection)
    if query.kind == "follow_play":
        return model.forward_follow_play(observation, selection)
    assert False


def logits_for_choices(
    output: SelectionHeadOutput,
    choices: tuple[SelectionChoice, ...],
) -> Tensor:
    """Return one logit per legal selection choice for batch size 1."""
    assert choices
    logits: list[Tensor] = []
    for choice in choices:
        if choice.kind == "select_card":
            assert choice.slot is not None
            logits.append(output.card_logits[0, choice.slot])
        elif choice.kind == "pass":
            assert output.pass_logits is not None
            logits.append(output.pass_logits[0])
        else:
            assert choice.kind == "stop"
            assert output.stop_logits is not None
            logits.append(output.stop_logits[0])
    return torch.stack(logits)
