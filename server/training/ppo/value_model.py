"""Independent player-visible observation critic."""

from __future__ import annotations

from typing import Protocol, final, override

from torch import Tensor, nn

from server.policy_model.network import (
    EncodedObservation,
    ModelConfig,
    ObservationBackbone,
)
from server.policy_model.observation.tensor import (
    ObservationTensorBatch,
)


class _BackboneCall(Protocol):
    def __call__(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation: ...


def _encode(
    backbone: _BackboneCall,
    observation: ObservationTensorBatch,
) -> EncodedObservation:
    return backbone(observation)


@final
class ObservationValueModel(nn.Module):
    """Estimate terminal round return from one player's observation."""

    def __init__(self, *, config: ModelConfig) -> None:
        super().__init__()
        self._observation_backbone = ObservationBackbone(
            d_model=config.d_model,
            layers=config.layers,
            heads=config.heads,
        )
        self._value_norm = nn.LayerNorm(config.d_model)
        self._value_projection = nn.Linear(config.d_model, 1)

    @override
    def forward(self, observation: ObservationTensorBatch) -> Tensor:
        """Return one scalar value for each observation row."""
        encoding = _encode(self._observation_backbone, observation)
        normalized: Tensor = self._value_norm.forward(
            encoding.observation_context()
        )
        values: Tensor = self._value_projection.forward(normalized)
        assert values.ndim == 2
        assert int(values.shape[1]) == 1
        return values[:, 0]


__all__ = ("ObservationValueModel",)
