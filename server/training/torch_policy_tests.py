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
from server.training.policy_inference_wire import (
    DevicePolicyRequestBatch,
    PolicyRequestWire,
    PolicyRequestWireBatch,
    build_policy_request_wire,
)
from server.training.runtime.model_rank.staging import (
    stage_policy_request_wires,
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
from server.training.torch_sampler import sample_policy_batch


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

    sample_result = sample_policy_batch(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
        requests=_request_batch(observation, legal_actions),
    )
    assert isinstance(sample_result, Ok)
    sample = sample_result.value
    first_token_id = int(sample.selected_token_ids_padded[0, 0])
    first_legal_token_ids = _legal_token_ids(
        sample.legal_token_masks_padded[0, 0]
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
        sample.old_log_probabilities[0].detach().cpu().item()
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
        int(sample.selected_token_ids_padded[0, index])
        for index in range(int(sample.step_counts[0].item()))
    ) == (
        semantic_argument_id(bid_argument),
        semantic_argument_id(stop_argument),
    )


@pytest.mark.asyncio
async def test_decide_rejects_non_finite_argument_logits() -> None:
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

    result = await TorchTrainingPolicy(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
    ).decide(observation, legal_actions, _decision_key())

    assert isinstance(result, Rejected)
    assert "logits must be finite" in result.reason


def test_sample_policy_batch_rejects_empty_legal_token_mask() -> None:
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
    request = _request_batch(observation, legal_actions)
    malformed_action_plan = replace(
        request.action_plan_batch,
        trace_tokens=torch.zeros((1, 1, 1), dtype=torch.long),
        trace_token_mask=torch.zeros((1, 1, 1), dtype=torch.bool),
        trace_lengths=torch.zeros((1, 1), dtype=torch.long),
        trace_row_mask=torch.zeros((1, 1), dtype=torch.bool),
    )
    malformed_request = DevicePolicyRequestBatch(
        observation_batch=request.observation_batch,
        action_plan_batch=malformed_action_plan,
        sampling_thresholds=request.sampling_thresholds,
        policy_versions=request.policy_versions,
    )

    result = sample_policy_batch(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
        requests=malformed_request,
    )

    assert isinstance(result, Rejected)
    assert result.reason == "policy action has no legal semantic token"


@pytest.mark.asyncio
async def test_decide_does_not_use_torch_multinomial(
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

    result = await TorchTrainingPolicy(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
    ).decide(observation, legal_actions, _decision_key())

    assert isinstance(result, Ok)


@pytest.mark.asyncio
async def test_decide_reuses_observation_encoding_for_full_trace() -> (
    None
):
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

    decision_result = await TorchTrainingPolicy(
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


def test_sample_policy_batch_batches_observation_encoding() -> None:
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

    result = sample_policy_batch(
        model=model,
        config=model_config,
        device=torch.device("cpu"),
        requests=_request_batch(
            observation,
            legal_actions,
            batch_size=2,
        ),
    )

    assert isinstance(result, Ok)
    assert len(result.value.policy_versions) == 2
    assert int(result.value.selected_token_ids_padded.shape[0]) == 2
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


def _decision_key(*, decision_index: int = 0) -> PolicyDecisionKey:
    return PolicyDecisionKey(
        base_seed=0,
        policy_version=0,
        episode_id=0,
        player_index=0,
        decision_index=decision_index,
    )


def _request_batch(
    observation: Observation,
    legal_actions: LegalActionIndex,
    *,
    batch_size: int = 1,
) -> DevicePolicyRequestBatch:
    wires: list[PolicyRequestWire] = []
    for index in range(batch_size):
        wire_result = build_policy_request_wire(
            max_observation_tokens=512,
            worker_index=0,
            request_id=index,
            observation=observation,
            legal_actions=legal_actions,
            decision_key=_decision_key(decision_index=index),
        )
        assert isinstance(wire_result, Ok)
        wires.append(wire_result.value)
    staged_result = stage_policy_request_wires(
        requests=PolicyRequestWireBatch(requests=tuple(wires)),
        max_observation_tokens=512,
        device=torch.device("cpu"),
    )
    assert isinstance(staged_result, Ok)
    return staged_result.value.device_batch


def _legal_token_ids(mask: Tensor) -> tuple[int, ...]:
    positions = torch.nonzero(
        mask.detach().cpu(), as_tuple=False
    ).squeeze(1)
    return tuple(
        int(positions[index].item())
        for index in range(int(positions.shape[0]))
    )
