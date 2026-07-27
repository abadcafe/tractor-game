"""Exact-length value inference over unique observations."""

from __future__ import annotations

import math

import torch

from server.foundation.result import Ok, Rejected
from server.training.model import TractorPolicyModel
from server.training.observation import Observation
from server.training.packed_observation import (
    PackedObservation,
    pack_observation,
)
from server.training.tensorize import tensorize_packed_observations

from ._batches import deduplicate_packed, exact_length_groups


def estimate_values(
    *,
    model: TractorPolicyModel,
    device: torch.device,
    observations: tuple[Observation, ...],
) -> Ok[tuple[float, ...]] | Rejected:
    """Estimate values without padding or duplicate rows."""
    assert observations
    packed = tuple(pack_observation(item) for item in observations)
    results: list[float | None] = [None for _item in observations]
    for indices in exact_length_groups(packed):
        estimated = _estimate_group(
            model=model,
            device=device,
            packed=packed,
            indices=indices,
        )
        for index, value in estimated:
            results[index] = value
    assert all(value is not None for value in results)
    values = tuple(value for value in results if value is not None)
    if not all(math.isfinite(value) for value in values):
        return Rejected(reason="model value estimate must be finite")
    return Ok(values)


def _estimate_group(
    *,
    model: TractorPolicyModel,
    device: torch.device,
    packed: tuple[PackedObservation, ...],
    indices: tuple[int, ...],
) -> tuple[tuple[int, float], ...]:
    packed_rows = deduplicate_packed(
        tuple(packed[index] for index in indices)
    )
    batch = tensorize_packed_observations(
        observations=packed_rows.unique,
        device=device,
    )
    encoding = model.encode_observations(batch)
    unique_values = _float_tuple(model.value_estimates(encoding))
    return tuple(
        (
            index,
            unique_values[packed_rows.source_indices[group_index]],
        )
        for group_index, index in enumerate(indices)
    )


def _float_tuple(values: torch.Tensor) -> tuple[float, ...]:
    assert values.ndim == 1
    cpu_values = values.detach().cpu()
    return tuple(
        float(cpu_values[index].item())
        for index in range(int(cpu_values.shape[0]))
    )


__all__ = ("estimate_values",)
