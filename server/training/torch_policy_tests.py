"""Black-box tests for torch-backed fixed-choice policy sampling."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from server.foundation.result import Ok, Rejected
from server.game.players.test_helpers import card, make_snapshot
from server.game.rules.card_faces import FaceCount, card_face
from server.training.config import ModelConfig
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.model import (
    ActionDecodeSession,
    ObservationEncoding,
    TractorPolicyModel,
)
from server.training.observation import Observation, build_observation
from server.training.observation_memory import ObservationMemoryView
from server.training.policy_inference_batch import (
    DevicePolicyRequestBatch,
    PolicyRequestCompiler,
    PolicyRequestInput,
    PolicyRequestRoute,
    materialize_borrowed_policy_request_batch,
)
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_action_plan import ActionSampler
from server.training.semantic_actions import ActionChoice
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    PASS_CHOICE_ID,
    action_choice_id,
)
from server.training.tensorize import ObservationTensorBatch
from server.training.torch_policy import TorchTrainingPolicy
from server.training.torch_sampler import sample_policy_batch


def test_batch_sampling_scores_the_fixed_legal_choice_mask() -> None:
    observation, legal, selected_choice_id = _bid_fixture()
    logits = torch.full((ACTION_CHOICE_COUNT,), -100.0)
    logits[PASS_CHOICE_ID] = 1.0
    logits[selected_choice_id] = 100.0
    model = _FixedChoiceModel(choice_logits=logits)

    result = sample_policy_batch(
        model=model,
        config=_model_config(),
        device=torch.device("cpu"),
        requests=_request_batch(observation, legal),
        sampler=_sampler(batch_size=1),
    )

    assert isinstance(result, Ok)
    sample = result.value
    assert int(sample.choice_ids_padded[0, 0]) == selected_choice_id
    assert sample.legal_choice_masks.shape == (
        1,
        ACTION_CHOICE_COUNT,
    )
    assert bool(sample.legal_choice_masks[0, PASS_CHOICE_ID])
    assert bool(sample.legal_choice_masks[0, selected_choice_id])
    assert torch.isfinite(sample.old_log_probabilities).all()


@pytest.mark.asyncio
async def test_policy_rejects_non_finite_choice_logits() -> None:
    observation, legal, _selected_choice_id = _bid_fixture()
    logits = torch.zeros((ACTION_CHOICE_COUNT,), dtype=torch.float32)
    logits[PASS_CHOICE_ID] = torch.nan

    result = await TorchTrainingPolicy(
        model=_FixedChoiceModel(choice_logits=logits),
        config=_model_config(),
        device=torch.device("cpu"),
    ).decide(observation, legal, _decision_key())

    assert isinstance(result, Rejected)
    assert "logits must be finite" in result.reason


@pytest.mark.asyncio
async def test_policy_uses_one_observation_encoding_for_the_trace() -> (
    None
):
    observation, legal, selected_choice_id = _bid_fixture()
    logits = torch.full((ACTION_CHOICE_COUNT,), -100.0)
    logits[selected_choice_id] = 100.0
    model = _FixedChoiceModel(choice_logits=logits)

    decision = await TorchTrainingPolicy(
        model=model,
        config=_model_config(),
        device=torch.device("cpu"),
    ).decide(observation, legal, _decision_key())

    assert isinstance(decision, Ok)
    assert model.encode_calls == 1
    assert model.score_batch_sizes == [1]
    assert decision.value.action.trace.choices == (
        ActionChoice(
            "card",
            FaceCount(face=card_face(card("hearts", "2", 1)), count=1),
        ),
    )
    assert decision.value.choice_count == 2


@pytest.mark.asyncio
async def test_policy_does_not_use_torch_multinomial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation, legal, _selected_choice_id = _bid_fixture()

    def fail_multinomial(*args: object, **kwargs: object) -> Tensor:
        assert not args
        assert not kwargs
        raise AssertionError("torch.multinomial must not be used")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)

    result = await TorchTrainingPolicy(
        model=_FixedChoiceModel(
            choice_logits=torch.zeros(ACTION_CHOICE_COUNT)
        ),
        config=_model_config(),
        device=torch.device("cpu"),
    ).decide(observation, legal, _decision_key())

    assert isinstance(result, Ok)


def test_batch_sampling_encodes_multiple_observations_together() -> (
    None
):
    observation, legal, _selected_choice_id = _bid_fixture()
    model = _FixedChoiceModel(
        choice_logits=torch.zeros(ACTION_CHOICE_COUNT)
    )

    result = sample_policy_batch(
        model=model,
        config=_model_config(),
        device=torch.device("cpu"),
        requests=_request_batch(observation, legal, batch_size=2),
        sampler=_sampler(batch_size=2),
    )

    assert isinstance(result, Ok)
    assert result.value.policy_versions == (0, 0)
    assert result.value.choice_ids_padded.shape[0] == 2
    assert model.encode_calls == 1
    assert model.score_batch_sizes == [2]


class _FixedChoiceModel(TractorPolicyModel):
    def __init__(self, *, choice_logits: Tensor) -> None:
        super().__init__(d_model=8, layers=1, heads=1)
        assert choice_logits.shape == (ACTION_CHOICE_COUNT,)
        self._fixed_choice_logits = choice_logits
        self.encode_calls = 0
        self.score_batch_sizes: list[int] = []

    def encode_observations(
        self, observation: ObservationTensorBatch
    ) -> ObservationEncoding:
        self.encode_calls += 1
        return super().encode_observations(observation)

    def begin_action_decode_session(
        self,
        encoding: ObservationEncoding,
        *,
        max_steps: int,
    ) -> ActionDecodeSession:
        return _FixedChoiceSession(
            choice_logits=self._fixed_choice_logits,
            batch_size=int(encoding.memory.shape[0]),
            device=encoding.memory.device,
            score_batch_sizes=self.score_batch_sizes,
            max_steps=max_steps,
        )


class _FixedChoiceSession(ActionDecodeSession):
    def __init__(
        self,
        *,
        choice_logits: Tensor,
        batch_size: int,
        device: torch.device,
        score_batch_sizes: list[int],
        max_steps: int,
    ) -> None:
        self._choice_logits = choice_logits
        self._batch_size = batch_size
        self._device = device
        self._score_batch_sizes = score_batch_sizes
        self._max_steps = max_steps
        self._step_index = 0

    def next_choice_logits(self) -> Tensor:
        self._score_batch_sizes.append(self._batch_size)
        return self._choice_logits.to(self._device).repeat(
            self._batch_size, 1
        )

    def advance(self, selected_choice_ids: Tensor) -> None:
        assert selected_choice_ids.shape == (self._batch_size,)
        self._step_index += 1
        assert self._step_index <= self._max_steps


def _bid_fixture() -> tuple[Observation, LegalActionIndex, int]:
    revealed = card("hearts", "2", 1)
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[revealed],
        player_hand_counts=[1, 0, 0, 0],
        trump_rank="2",
    )
    observation = build_observation(
        viewer=0,
        snapshot=snapshot,
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )
    legal = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
        query=observation.action_query,
    )
    selected = action_choice_id(
        ActionChoice(
            "card",
            FaceCount(face=card_face(revealed), count=1),
        )
    )
    return (observation, legal, selected)


def _decision_key(*, decision_index: int = 0) -> PolicyDecisionKey:
    return PolicyDecisionKey(
        base_seed=0,
        policy_version=0,
        rollout_id="torch-policy-test",
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
    compiled = PolicyRequestCompiler(
        batch_capacity=batch_size
    ).compile_batch(
        tuple(
            PolicyRequestInput(
                route=PolicyRequestRoute(
                    worker_index=0, request_id=index
                ),
                observation=observation,
                legal_actions=legal_actions,
                decision_key=_decision_key(decision_index=index),
            )
            for index in range(batch_size)
        )
    )
    assert isinstance(compiled, Ok)
    result = materialize_borrowed_policy_request_batch(
        batch=compiled.value, device=torch.device("cpu")
    )
    assert isinstance(result, Ok)
    return result.value


def _sampler(*, batch_size: int) -> ActionSampler:
    return ActionSampler.create(
        batch_capacity=batch_size, device=torch.device("cpu")
    )


def _model_config() -> ModelConfig:
    return ModelConfig(d_model=8, layers=1, heads=1)
