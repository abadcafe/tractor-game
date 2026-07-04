"""Tests for torch-backed training policy sampling."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from server.player.test_helpers import card, make_snapshot
from server.result import Ok, Rejected
from server.training.argument_distribution import argument_distribution
from server.training.config import ModelConfig
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.model import (
    ArgumentPrefixScores,
    ObservationEncoding,
    TractorPolicyModel,
)
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
    assert len(decision.choice_trace.steps) == 1
    assert decision.choice_trace.steps[
        0
    ].selected_argument_id == semantic_argument_id(pass_argument)
    assert decision.choice_trace.steps[0].selected_argument_offset == 0


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


def test_decide_reuses_observation_encoding_for_full_trace() -> None:
    model = _FixedArgumentModel(
        argument_logits=torch.zeros(
            (SEMANTIC_CODEC.argument_vocab_size,), dtype=torch.float32
        )
    )
    model_config = ModelConfig(d_model=4, layers=1, heads=1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[card("spades", "A", 1)],
    )
    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )

    decision_result = TorchTrainingPolicy(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
    ).decide(observation, legal_actions)

    assert isinstance(decision_result, Ok)
    decision = decision_result.value
    assert model.encode_calls == 1
    assert model.score_batch_sizes == [1, 1]
    assert model.score_prefix_widths == [1, 2]
    assert len(decision.choice_trace.steps) == len(
        decision.action.semantic_trace.arguments
    )


class _FixedArgumentModel(TractorPolicyModel):
    def __init__(self, *, argument_logits: Tensor) -> None:
        super().__init__(d_model=4, layers=1, heads=1)
        self._fixed_argument_logits = argument_logits
        self.encode_calls = 0
        self.score_batch_sizes: list[int] = []
        self.score_prefix_widths: list[int] = []

    def encode_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> ObservationEncoding:
        self.encode_calls += 1
        return super().encode_observations(observation)

    def score_argument_prefixes(
        self,
        encoding: ObservationEncoding,
        prefix: ArgumentPrefixTensorBatch,
    ) -> ArgumentPrefixScores:
        batch_size = int(prefix.argument_ids.shape[0])
        self.score_batch_sizes.append(batch_size)
        self.score_prefix_widths.append(
            int(prefix.argument_ids.shape[1])
        )
        logits = self._fixed_argument_logits.to(
            prefix.argument_ids.device
        )
        return ArgumentPrefixScores(
            argument_logits=logits.repeat(batch_size, 1),
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
