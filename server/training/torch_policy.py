"""Torch-backed TrainingPolicy implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from server.result import Ok
from server.training.action_tokens import (
    ACTION_TOKEN_VOCAB_SIZE,
    BEGIN_TOKEN_ID,
    MAX_ACTION_TOKENS,
    STOP_TOKEN_ID,
    ActionQuery,
    decode_action_tokens,
    valid_next_token_ids,
)
from server.training.config import ModelConfig
from server.training.model import TractorPolicyModel
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.tensorize import (
    tensorize_action_prefix,
    tensorize_observation,
)


class TorchTrainingPolicy:
    """Sample action-token sequences from a torch policy/value model."""

    def __init__(
        self,
        *,
        model: TractorPolicyModel,
        config: ModelConfig,
        device: torch.device,
        temperature: float = 1.0,
    ) -> None:
        assert temperature > 0.0
        self.model = model
        self.config = config
        self.device = device
        self.temperature = temperature

    def decide(
        self,
        observation: Observation,
        query: ActionQuery,
    ) -> PolicyDecision:
        self.model.eval()
        with torch.no_grad():
            observation_ids = tensorize_observation(
                observation=observation,
                max_observation_tokens=self.config.max_tokens,
                device=self.device,
            )
            prefix = (BEGIN_TOKEN_ID,)
            log_probability = 0.0
            entropy = 0.0
            value_estimate = 0.0
            for _ in range(MAX_ACTION_TOKENS - 1):
                action_prefix_ids = tensorize_action_prefix(
                    prefix=prefix,
                    device=self.device,
                )
                logits, values = self.model.forward_action(
                    observation_ids,
                    action_prefix_ids,
                )
                value_estimate = float(values[0].detach().cpu().item())
                allowed = valid_next_token_ids(query, prefix)
                assert allowed
                masked_logits = _masked_logits(
                    logits[0] / self.temperature,
                    allowed,
                    device=self.device,
                )
                probabilities = torch.softmax(masked_logits, dim=0)
                log_probabilities = torch.log_softmax(
                    masked_logits, dim=0
                )
                sampled = torch.multinomial(
                    probabilities, num_samples=1
                )
                token_id = int(sampled[0].detach().cpu().item())
                log_probability += float(
                    log_probabilities[token_id].detach().cpu().item()
                )
                entropy += float(
                    (-(probabilities * log_probabilities).sum())
                    .detach()
                    .cpu()
                    .item()
                )
                prefix = (*prefix, token_id)
                if token_id == STOP_TOKEN_ID:
                    break
        decoded = decode_action_tokens(query, prefix)
        assert isinstance(decoded, Ok)
        return PolicyDecision(
            action=decoded.value,
            log_probability=log_probability,
            value_estimate=value_estimate,
            entropy=entropy,
            token_count=len(decoded.value.token_ids),
        )


def _masked_logits(
    logits: Tensor,
    allowed_token_ids: tuple[int, ...],
    *,
    device: torch.device,
) -> Tensor:
    mask = torch.full(
        (ACTION_TOKEN_VOCAB_SIZE,),
        float("-inf"),
        dtype=logits.dtype,
        device=device,
    )
    index = torch.tensor(
        list(allowed_token_ids),
        dtype=torch.long,
        device=device,
    )
    mask[index] = logits[index]
    return mask
