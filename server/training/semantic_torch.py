"""Torch helpers for scoring semantic argument choices."""

from __future__ import annotations

import torch
from torch import Tensor

from server.training.model import ArgumentHeadOutput, TractorPolicyModel
from server.training.semantic_actions import (
    SemanticArgument,
    semantic_argument_id,
)
from server.training.tensorize import (
    ArgumentPrefixTensorBatch,
    ObservationTensorBatch,
)


def forward_argument_head(
    *,
    model: TractorPolicyModel,
    observation: ObservationTensorBatch,
    prefix: ArgumentPrefixTensorBatch,
) -> ArgumentHeadOutput:
    """Run the model semantic-argument head."""
    return model.forward_argument(observation, prefix)


def logits_for_arguments(
    output: ArgumentHeadOutput,
    choices: tuple[SemanticArgument, ...],
) -> Tensor:
    """Return one logit per legal semantic argument for batch size 1."""
    assert choices
    ids = [semantic_argument_id(argument) for argument in choices]
    index = torch.tensor(
        ids,
        dtype=torch.long,
        device=output.argument_logits.device,
    )
    return output.argument_logits[0].index_select(dim=0, index=index)
