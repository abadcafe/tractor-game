"""Torch-backed TrainingPolicy implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from server.result import Rejected
from server.training.argument_distribution import argument_distribution
from server.training.config import ModelConfig
from server.training.legal_actions import LegalActionIndex
from server.training.model import TractorPolicyModel
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
)
from server.training.semantic_codec import SEMANTIC_CODEC
from server.training.semantic_torch import (
    forward_argument_head,
)
from server.training.tensorize import (
    tensorize_argument_prefix,
    tensorize_observation,
)


class TorchTrainingPolicy:
    """Sample semantic argument traces from a torch model."""

    def __init__(
        self,
        *,
        model: TractorPolicyModel,
        config: ModelConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
    ) -> PolicyDecision:
        self.model.eval()
        with torch.no_grad():
            observation_batch = tensorize_observation(
                observation=observation,
                max_observation_tokens=self.config.max_tokens,
                device=self.device,
            )
            prefix = SemanticArgumentPrefix(arguments=())
            arguments: list[SemanticArgument] = []
            log_probability = 0.0
            entropy = 0.0
            value_estimate = 0.0
            choice_count = 0
            for _ in range(SEMANTIC_CODEC.max_argument_tokens):
                prefix_batch = tensorize_argument_prefix(
                    prefix=prefix,
                    device=self.device,
                )
                output = forward_argument_head(
                    model=self.model,
                    observation=observation_batch,
                    prefix=prefix_batch,
                )
                value_estimate = float(
                    output.values[0].detach().cpu().item()
                )
                allowed = legal_actions.allowed_next(prefix)
                if not allowed:
                    break
                choice_count += len(allowed)
                distribution = argument_distribution(
                    argument_logits=output.argument_logits[0],
                    choices=allowed,
                )
                sampled = torch.multinomial(
                    distribution.probabilities, num_samples=1
                )
                argument_index = int(sampled[0].detach().cpu().item())
                argument = allowed[argument_index]
                log_probability += _float_tensor(
                    distribution.log_probabilities[argument_index]
                )
                entropy += _float_tensor(distribution.entropy)
                arguments.append(argument)
                if argument.kind in ("pass", "stop"):
                    break
                assert argument.kind == "select_face_count"
                prefix = SemanticArgumentPrefix(
                    arguments=(*prefix.arguments, argument)
                )
            else:
                assert False
        trace = SemanticArgumentTrace(arguments=tuple(arguments))
        decoded = legal_actions.decode(trace)
        if isinstance(decoded, Rejected):
            raise AssertionError(
                "invalid semantic trace: "
                f"query={legal_actions.query!r}, "
                f"trace={trace!r}, reason={decoded.reason}"
            )
        return PolicyDecision(
            action=decoded.value,
            log_probability=log_probability,
            value_estimate=value_estimate,
            entropy=entropy,
            choice_count=choice_count,
        )


def _float_tensor(value: Tensor) -> float:
    return float(value.detach().cpu().item())
