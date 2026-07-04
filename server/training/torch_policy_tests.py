"""Tests for torch-backed training policy sampling."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from server.player.test_helpers import card, make_snapshot
from server.result import Ok, Rejected
from server.training.argument_distribution import argument_distribution
from server.training.config import ModelConfig
from server.training.legal_actions import LegalActionIndex
from server.training.model import ArgumentHeadOutput, TractorPolicyModel
from server.training.observation import build_observation
from server.training.semantic_actions import (
    ActionQuery,
    GeneratedAction,
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.codec import (
    SEMANTIC_CODEC,
    semantic_argument_id,
)
from server.training.tensorize import (
    ArgumentPrefixTensorBatch,
    ObservationTensorBatch,
)
from server.training.torch_policy import TorchTrainingPolicy


def test_decide_scores_sampled_argument_with_shared_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_argument = SemanticArgument("pass")
    stop_argument = SemanticArgument("stop")
    logits = torch.zeros(
        (SEMANTIC_CODEC.argument_vocab_size,), dtype=torch.float32
    )
    logits[semantic_argument_id(pass_argument)] = 1.0
    logits[semantic_argument_id(stop_argument)] = 3.0
    model = _FixedArgumentModel(argument_logits=logits)
    model_config = ModelConfig(d_model=4, layers=1, heads=1)
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2", 1)],
        trump_rank="2",
    )
    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )
    legal_actions = _FixedLegalActionIndex(
        query=observation.action_query,
        allowed=(pass_argument, stop_argument),
    )

    def choose_first(
        input: Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
        out: Tensor | None = None,
    ) -> Tensor:
        assert num_samples == 1
        assert replacement is False
        assert generator is None
        assert out is None
        assert input.shape == (2,)
        return torch.tensor([0], dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", choose_first)

    decision_result = TorchTrainingPolicy(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
    ).decide(observation, legal_actions)
    assert isinstance(decision_result, Ok)
    decision = decision_result.value
    expected_result = argument_distribution(
        argument_logits=logits,
        choices=(pass_argument, stop_argument),
    )
    assert isinstance(expected_result, Ok)
    expected = expected_result.value

    expected_log_probability = float(
        expected.log_probabilities[0].detach().cpu().item()
    )
    expected_entropy = float(expected.entropy.detach().cpu().item())
    assert abs(decision.log_probability - expected_log_probability) < (
        0.000001
    )
    assert abs(decision.entropy - expected_entropy) < 0.000001


def test_decide_rejects_non_finite_argument_logits() -> None:
    pass_argument = SemanticArgument("pass")
    logits = torch.zeros(
        (SEMANTIC_CODEC.argument_vocab_size,), dtype=torch.float32
    )
    logits[semantic_argument_id(pass_argument)] = torch.nan
    model = _FixedArgumentModel(argument_logits=logits)
    model_config = ModelConfig(d_model=4, layers=1, heads=1)
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2", 1)],
        trump_rank="2",
    )
    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )
    legal_actions = _FixedLegalActionIndex(
        query=observation.action_query,
        allowed=(pass_argument,),
    )

    result = TorchTrainingPolicy(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
    ).decide(observation, legal_actions)

    assert isinstance(result, Rejected)
    assert "logits must be finite" in result.reason


def test_decide_rejects_sampling_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_argument = SemanticArgument("pass")
    logits = torch.zeros(
        (SEMANTIC_CODEC.argument_vocab_size,), dtype=torch.float32
    )
    model = _FixedArgumentModel(argument_logits=logits)
    model_config = ModelConfig(d_model=4, layers=1, heads=1)
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2", 1)],
        trump_rank="2",
    )
    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )
    legal_actions = _FixedLegalActionIndex(
        query=observation.action_query,
        allowed=(pass_argument,),
    )

    def fail_multinomial(
        input: Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
        out: Tensor | None = None,
    ) -> Tensor:
        assert input.shape == (1,)
        assert num_samples == 1
        assert replacement is False
        assert generator is None
        assert out is None
        raise RuntimeError("invalid multinomial distribution")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)

    result = TorchTrainingPolicy(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
    ).decide(observation, legal_actions)

    assert isinstance(result, Rejected)
    assert "policy sampling failed" in result.reason


class _FixedArgumentModel(TractorPolicyModel):
    def __init__(self, *, argument_logits: Tensor) -> None:
        super().__init__(d_model=4, layers=1, heads=1)
        self._fixed_argument_logits = argument_logits

    def forward_argument(
        self,
        observation: ObservationTensorBatch,
        prefix: ArgumentPrefixTensorBatch,
    ) -> ArgumentHeadOutput:
        batch_size = int(observation.token_type_ids.shape[0])
        return ArgumentHeadOutput(
            argument_logits=self._fixed_argument_logits.repeat(
                batch_size, 1
            ),
            values=torch.zeros((batch_size,), dtype=torch.float32),
        )


class _FixedLegalActionIndex(LegalActionIndex):
    def __init__(
        self,
        *,
        query: ActionQuery,
        allowed: tuple[SemanticArgument, ...],
    ) -> None:
        self._query = query
        self._allowed = allowed

    @property
    def query(self) -> ActionQuery:
        return self._query

    def allowed_next(
        self, prefix: SemanticArgumentPrefix
    ) -> tuple[SemanticArgument, ...]:
        assert prefix.arguments == ()
        return self._allowed

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        assert trace.arguments == (SemanticArgument("pass"),)
        return Ok(
            value=GeneratedAction(
                action_kind="pass",
                message_type="bid",
                face_counts=(),
                semantic_trace=trace,
                is_pass=True,
            )
        )
