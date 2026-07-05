"""Tests for torch-backed training policy sampling."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor

from server.player.test_helpers import card, make_snapshot
from server.result import Ok, Rejected
from server.rules.card_faces import FaceCount, card_face
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
from server.training.observation import Observation, build_observation
from server.training.policy_request_frame import (
    PolicyRequestBatchFrame,
    PolicyRequestFrame,
    build_policy_request_frame,
)
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_actions import (
    SemanticArgument,
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
from server.training.torch_sampler import sample_policy_decisions


def test_decide_scores_sampled_argument_with_distribution() -> None:
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
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )
    bid_card = card("hearts", "2", 1)
    bid_argument = SemanticArgument(
        "select_face_count",
        FaceCount(face=card_face(bid_card), count=1),
    )
    logits[semantic_argument_id(bid_argument)] = 3.0
    logits[semantic_argument_id(stop_argument)] = 0.0

    sample_results = sample_policy_decisions(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
        requests=PolicyRequestBatchFrame(
            frames=(_request_frame(observation, legal_actions),)
        ),
    )
    assert isinstance(sample_results[0], Ok)
    sample = sample_results[0].value
    first_token_id = int(sample.replay_record.selected_token_ids[0])
    first_legal_token_ids = _legal_token_ids(
        sample.replay_record.legal_token_masks[0]
    )
    selected_offset = first_legal_token_ids.index(first_token_id)
    expected_first_log_probabilities = torch.log_softmax(
        torch.tensor([1.0, 3.0], dtype=torch.float32), dim=0
    )
    expected_log_probability = float(
        expected_first_log_probabilities[selected_offset]
        .detach()
        .cpu()
        .item()
    )
    actual_log_probability = float(
        sample.replay_record.old_log_probability.detach().cpu().item()
    )
    assert (
        abs(actual_log_probability - expected_log_probability)
        < 0.000001
    )
    assert first_legal_token_ids == (
        semantic_argument_id(pass_argument),
        semantic_argument_id(bid_argument),
    )
    assert first_token_id == semantic_argument_id(bid_argument)
    assert tuple(
        int(token_id)
        for token_id in sample.replay_record.selected_token_ids
    ) == (
        semantic_argument_id(bid_argument),
        semantic_argument_id(stop_argument),
    )


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
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )

    result = TorchTrainingPolicy(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
    ).decide(observation, legal_actions, _decision_key())

    assert isinstance(result, Rejected)
    assert "logits must be finite" in result.reason


def test_sample_policy_decisions_rejects_empty_legal_token_mask() -> (
    None
):
    model = _FixedArgumentModel(
        argument_logits=torch.zeros(
            (SEMANTIC_CODEC.argument_vocab_size,), dtype=torch.float32
        )
    )
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
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )
    request = _request_frame(observation, legal_actions)
    malformed_request = PolicyRequestFrame(
        component_rows=request.component_rows,
        numeric_value_rows=request.numeric_value_rows,
        numeric_mask_rows=request.numeric_mask_rows,
        action_plan=replace(request.action_plan, trace_tokens=()),
        decision_key=request.decision_key,
    )

    results = sample_policy_decisions(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
        requests=PolicyRequestBatchFrame(frames=(malformed_request,)),
    )

    assert isinstance(results[0], Rejected)
    assert (
        results[0].reason == "policy action has no legal semantic token"
    )


def test_decide_does_not_use_torch_multinomial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )

    def fail_multinomial(*args: object, **kwargs: object) -> Tensor:
        assert not args
        assert not kwargs
        raise AssertionError("torch.multinomial must not be used")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)

    result = TorchTrainingPolicy(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
    ).decide(observation, legal_actions, _decision_key())

    assert isinstance(result, Ok)


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
    ).decide(observation, legal_actions, _decision_key())

    assert isinstance(decision_result, Ok)
    decision = decision_result.value
    assert model.encode_calls == 1
    assert model.score_batch_sizes == [1, 1]
    assert model.score_prefix_widths == [1, 2]
    assert decision.decision_handle.slot_index == 0
    assert decision.decision_handle.slot_generation == 1
    assert decision.choice_count == len(
        decision.action.semantic_trace.arguments
    )


def test_sample_policy_decisions_batches_observation_encoding() -> None:
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
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )

    results = sample_policy_decisions(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
        requests=PolicyRequestBatchFrame(
            frames=(
                _request_frame(observation, legal_actions),
                _request_frame(observation, legal_actions),
            )
        ),
    )

    assert len(results) == 2
    assert isinstance(results[0], Ok)
    assert isinstance(results[1], Ok)
    assert model.encode_calls == 1
    assert model.score_batch_sizes == [2]
    assert model.score_prefix_widths == [1]


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


def _decision_key() -> PolicyDecisionKey:
    return PolicyDecisionKey(
        base_seed=0,
        policy_version=0,
        episode_id=0,
        player_index=0,
        decision_index=0,
    )


def _request_frame(
    observation: Observation,
    legal_actions: LegalActionIndex,
) -> PolicyRequestFrame:
    frame_result = build_policy_request_frame(
        observation=observation,
        legal_actions=legal_actions,
        decision_key=_decision_key(),
    )
    assert isinstance(frame_result, Ok)
    return frame_result.value


def _legal_token_ids(mask: Tensor) -> tuple[int, ...]:
    positions = torch.nonzero(
        mask.detach().cpu(), as_tuple=False
    ).squeeze(1)
    return tuple(
        int(positions[index].item())
        for index in range(int(positions.shape[0]))
    )
