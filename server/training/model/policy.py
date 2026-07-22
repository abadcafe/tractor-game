"""Complete Tractor policy/value model composition."""

from __future__ import annotations

from torch import Tensor, nn

from server.training.tensorize import ObservationTensorBatch

from .action_decoder import (
    ActionDecoder,
    ActionDecodeSession,
    ActionTraceScores,
)
from .observation_encoder import EncodedObservation, ObservationEncoder


class TractorPolicyModel(nn.Module):
    """Compose observation, action, and value modules."""

    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        assert d_model > 0
        assert d_model % heads == 0
        self._observation_encoder = ObservationEncoder(
            d_model=d_model,
            layers=layers,
            heads=heads,
        )
        self._action_decoder = ActionDecoder(
            d_model=d_model,
            heads=heads,
        )
        self._value_head = nn.Linear(d_model, 1)

    def encode_observations(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation:
        """Encode observations once for both model heads."""
        return self._observation_encoder(observation)

    def value_estimates(self, encoding: EncodedObservation) -> Tensor:
        """Estimate values from contextual query state."""
        return self._value_head(encoding.value_context()).squeeze(-1)

    def score_action_traces(
        self,
        encoding: EncodedObservation,
        *,
        choice_ids_padded: Tensor,
        step_counts: Tensor,
    ) -> ActionTraceScores:
        """Score action traces through the model-owned decoder."""
        return self._action_decoder.score_action_traces(
            encoding,
            choice_ids_padded=choice_ids_padded,
            step_counts=step_counts,
        )

    def begin_action_decode_session(
        self,
        encoding: EncodedObservation,
        *,
        max_steps: int,
    ) -> ActionDecodeSession:
        """Begin incremental action decoding for live inference."""
        return self._action_decoder.begin_decode_session(
            encoding,
            max_steps=max_steps,
        )


__all__ = ("TractorPolicyModel",)
