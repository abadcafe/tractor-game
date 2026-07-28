"""Public model-observation API."""

from __future__ import annotations

from server.policy_model._schema.encoding import CATEGORY_COUNT
from server.policy_model.observation.history import (
    ObservationMemory,
    ObservationMemoryView,
    ObservedBidAction,
)
from server.policy_model.observation.interface import (
    Observation,
    build_observation,
)
from server.policy_model.observation.packing import (
    MAX_LOSSLESS_OBSERVATION_TOKENS,
    PackedObservation,
    pack_observation,
    padded_packed_observation,
)
from server.policy_model.observation.structure import (
    STRUCTURE_AXIS_COUNT,
    StructureAxis,
)

__all__ = (
    "Observation",
    "ObservationMemory",
    "ObservationMemoryView",
    "ObservedBidAction",
    "PackedObservation",
    "CATEGORY_COUNT",
    "MAX_LOSSLESS_OBSERVATION_TOKENS",
    "STRUCTURE_AXIS_COUNT",
    "StructureAxis",
    "build_observation",
    "pack_observation",
    "padded_packed_observation",
)
