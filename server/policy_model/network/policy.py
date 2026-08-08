"""Complete Tractor policy model composition."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor, nn

from server.policy_model.observation.tensor_batch import (
    ObservationTensorBatch,
)

from .action_decoder import (
    ActionDecoder,
    ActionDecodeSession,
    ActionTraceScores,
)
from .config import ModelConfig
from .observation_encoder import EncodedObservation, ObservationEncoder


class _ObservationEncoderCall(Protocol):
    def __call__(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation: ...


def _call_observation_encoder(
    encoder: _ObservationEncoderCall,
    observation: ObservationTensorBatch,
) -> EncodedObservation:
    return encoder(observation)


class PolicyModel(nn.Module):
    """Encode observations and autoregressively decode actions."""

    def __init__(
        self,
        *,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self._observation_encoder: ObservationEncoder = (
            ObservationEncoder(
                d_model=config.d_model,
                layers=config.layers,
                heads=config.heads,
            )
        )
        self._action_decoder: ActionDecoder = ActionDecoder(
            d_model=config.d_model,
            heads=config.heads,
        )

    def encode_observations(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation:
        """Encode observations for autoregressive policy decoding."""
        return _call_observation_encoder(
            self._observation_encoder, observation
        )

    def score_action_traces(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: Tensor,
        choice_ids_padded: Tensor,
        step_counts: Tensor,
    ) -> ActionTraceScores:
        """Score action traces through the model-owned decoder."""
        return self._action_decoder.score_action_traces(
            encoding,
            source_rows=source_rows,
            choice_ids_padded=choice_ids_padded,
            step_counts=step_counts,
        )

    def begin_action_decode_session(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: Tensor,
        max_steps: int,
    ) -> ActionDecodeSession:
        """Begin incremental action decoding for live inference."""
        return self._action_decoder.begin_decode_session(
            encoding,
            source_rows=source_rows,
            max_steps=max_steps,
        )


__all__ = ("PolicyModel",)
