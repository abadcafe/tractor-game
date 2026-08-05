"""Complete Tractor policy and action-value model composition."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor, nn

from server.policy_model.observation.tensor_batch import (
    ObservationTensorBatch,
)

from ._module_call import call_tensor
from .action_decoder import (
    ActionDecoder,
    ActionDecodeSession,
    ActionTraceScores,
)
from .action_value import CompleteActionEncoder
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


class PolicyActionModel(nn.Module):
    """Compose independent policy and complete-action value branches."""

    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
        action_value_layers: int = 2,
    ) -> None:
        super().__init__()
        assert d_model > 0
        assert d_model % heads == 0
        assert action_value_layers > 0
        self._policy_observation_encoder: ObservationEncoder = (
            ObservationEncoder(
                d_model=d_model,
                layers=layers,
                heads=heads,
            )
        )
        self._action_decoder: ActionDecoder = ActionDecoder(
            d_model=d_model,
            heads=heads,
        )
        self._action_value_observation_encoder: ObservationEncoder = (
            ObservationEncoder(
                d_model=d_model,
                layers=layers,
                heads=heads,
            )
        )
        self._complete_action_encoder: CompleteActionEncoder = (
            CompleteActionEncoder(
                d_model=d_model,
                heads=heads,
                layers=action_value_layers,
            )
        )
        self._action_value_head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 1),
        )

    def encode_policy_observations(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation:
        """Encode observations for autoregressive policy decoding."""
        return _call_observation_encoder(
            self._policy_observation_encoder, observation
        )

    def encode_action_value_observations(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation:
        """Encode observations through the independent value branch."""
        return _call_observation_encoder(
            self._action_value_observation_encoder, observation
        )

    def action_values(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: Tensor,
        choice_ids_padded: Tensor,
        step_counts: Tensor,
    ) -> Tensor:
        """Evaluate complete semantic action traces."""
        action_features = call_tensor(
            self._complete_action_encoder,
            encoding,
            source_rows=source_rows,
            choice_ids_padded=choice_ids_padded,
            step_counts=step_counts,
        )
        return call_tensor(
            self._action_value_head,
            action_features,
        ).squeeze(-1)

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

    def policy_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return exactly the parameters owned by the policy branch."""
        return (
            *tuple(self._policy_observation_encoder.parameters()),
            *tuple(self._action_decoder.parameters()),
        )

    def action_value_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return parameters owned by the complete-action branch."""
        return (
            *tuple(self._action_value_observation_encoder.parameters()),
            *tuple(self._complete_action_encoder.parameters()),
            *tuple(self._action_value_head.parameters()),
        )


__all__ = ("PolicyActionModel",)
