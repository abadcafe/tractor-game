"""Torch helpers for scoring semantic argument choices."""

from __future__ import annotations

from server.training.model import ArgumentHeadOutput, TractorPolicyModel
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
