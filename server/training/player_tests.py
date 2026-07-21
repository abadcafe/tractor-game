"""Tests for TrainingPlayer behavior through the Player interface."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
import torch

from server.foundation.result import Ok, Rejected
from server.game.players.test_helpers import (
    card,
    make_game,
    make_snapshot,
    make_state_message,
)
from server.game.protocol import ScoringSnapshot
from server.game.rules.card_faces import CardFace, FaceCount
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.observation_memory import ObservationMemory
from server.training.player import TrainingPlayer
from server.training.policy import PolicyDecision
from server.training.policy_sampling import DecisionHandle
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_action_plan import (
    ActionChoiceLogitDecoder,
    ActionSampler,
    action_plan_generation_step_count,
    action_trace_from_choice_ids,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.training.semantic_actions import ActionTrace
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
)
from server.training.tokenization import GlobalToken, RoundToken
from server.training.tokenization.payloads import RoundField
from server.training.trajectory import TrajectoryRecorder


@dataclass(slots=True)
class _ZeroChoiceDecoder:
    batch_size: int
    device: torch.device

    def next_choice_logits(self) -> torch.Tensor:
        return torch.zeros(
            (self.batch_size, ACTION_CHOICE_COUNT),
            dtype=torch.float32,
            device=self.device,
        )

    def advance(self, selected_choice_ids: torch.Tensor) -> None:
        assert selected_choice_ids.shape == (self.batch_size,)


class FirstCardPlayPolicy:
    """Deterministic test policy that plays the first hand face."""

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        assert decision_key.base_seed >= 0
        assert decision_key.policy_version >= 0
        assert decision_key.episode_id >= 0
        trace_result = _first_legal_trace(legal_actions)
        assert isinstance(trace_result, Ok)
        decoded = legal_actions.decode(trace_result.value)
        assert isinstance(decoded, Ok)
        return Ok(
            value=PolicyDecision(
                action=decoded.value,
                decision_handle=DecisionHandle(
                    model_rank_index=0,
                    policy_version=decision_key.policy_version,
                    row_index=decision_key.decision_index,
                ),
                choice_count=len(decoded.value.trace.choices),
            )
        )


def _first_legal_trace(
    legal_actions: LegalActionIndex,
) -> Ok[ActionTrace] | Rejected:
    device = torch.device("cpu")
    action_plan = compile_legal_action_frame(legal_actions)
    generation_steps = action_plan_generation_step_count(action_plan)
    batch = plan_batch_to_device((action_plan,), device=device)

    logit_decoder: ActionChoiceLogitDecoder = _ZeroChoiceDecoder(
        batch_size=1, device=device
    )
    sampler = ActionSampler.create(batch_capacity=1, device=device)
    sample_result = sampler.sample(
        action_batch=batch,
        generation_step_counts=torch.tensor(
            (generation_steps,), dtype=torch.long, device=device
        ),
        sampling_thresholds=torch.zeros(
            (1, generation_steps), dtype=torch.float64, device=device
        ),
        padded_generation_steps=generation_steps,
        logit_decoder=logit_decoder,
    )
    if isinstance(sample_result, Rejected):
        return sample_result
    sample = sample_result.value
    step_count = int(sample.step_counts[0].item())
    trace_ids = tuple(
        int(sample.choice_ids_padded[0, index].item())
        for index in range(step_count)
    )
    return action_trace_from_choice_ids(trace_ids)


class CapturingFirstCardPlayPolicy(FirstCardPlayPolicy):
    """Test policy that stores the observation it receives."""

    def __init__(self) -> None:
        self.observation: Observation | None = None

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        self.observation = observation
        return await super().decide(
            observation, legal_actions, decision_key
        )


class RejectingPolicy:
    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        assert decision_key.base_seed >= 0
        assert observation.action_query == legal_actions.query
        return Rejected(reason="policy sampling failed")


@pytest.mark.asyncio
async def test_training_player_submits_action_without_hints() -> None:
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
        action_hints=[],
    )
    game = make_game(snapshot)
    recorder = TrajectoryRecorder()
    player = TrainingPlayer(
        index=0,
        policy=FirstCardPlayPolicy(),
        recorder=recorder,
        observation_memory=_observation_memory(last_seq=0),
    )

    await player.on_state(game, make_state_message(snapshot, seq=1))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    message = game.receive.call_args[0][1]
    assert message.raw == {"type": "play", "cards": [test_card.id]}
    assert recorder.steps() == ()


@pytest.mark.asyncio
async def test_training_player_records_action_after_acceptance() -> (
    None
):
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    next_snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action=None,
        player_hand=[],
    )
    game = make_game(snapshot)
    recorder = TrajectoryRecorder()
    player = TrainingPlayer(
        index=0,
        policy=FirstCardPlayPolicy(),
        recorder=recorder,
        observation_memory=_observation_memory(last_seq=0),
    )

    await player.on_state(game, make_state_message(snapshot, seq=1))
    await asyncio.sleep(0.05)
    await player.on_state(
        game, make_state_message(next_snapshot, seq=2)
    )

    steps = recorder.steps()
    assert len(steps) == 1
    assert steps[0].player_index == 0
    assert steps[0].action.face_counts == (
        FaceCount(CardFace(test_card.suit, test_card.rank), 1),
    )
    assert steps[0].choice_count == 2


@pytest.mark.asyncio
async def test_training_player_returns_policy_rejection() -> None:
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    game = make_game(snapshot)
    player = TrainingPlayer(
        index=0,
        policy=RejectingPolicy(),
        observation_memory=_observation_memory(last_seq=0),
    )

    await player.on_state(game, make_state_message(snapshot, seq=1))

    result = player.raise_background_errors()
    assert isinstance(result, Rejected)
    assert "policy sampling failed" in result.reason
    game.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_acceptance_contract_requires_seq_advance() -> None:
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    accepted_snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action=None,
        player_hand=[],
    )
    game = make_game(snapshot)
    recorder = TrajectoryRecorder()
    player = TrainingPlayer(
        index=0,
        policy=FirstCardPlayPolicy(),
        recorder=recorder,
        observation_memory=_observation_memory(last_seq=6),
    )

    await player.on_state(game, make_state_message(snapshot, seq=7))
    await asyncio.sleep(0.05)
    await player.on_state(game, make_state_message(snapshot, seq=7))
    assert recorder.steps() == ()

    await player.on_state(
        game, make_state_message(accepted_snapshot, seq=8)
    )

    assert len(recorder.steps()) == 1


@pytest.mark.asyncio
async def test_training_player_submits_initial_next_round() -> None:
    snapshot = make_snapshot(
        phase="WAITING", awaiting_action="next_round"
    )
    game = make_game(snapshot)
    player = TrainingPlayer(
        index=2,
        policy=FirstCardPlayPolicy(),
    )

    await player.on_state(game, make_state_message(snapshot))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    message = game.receive.call_args[0][1]
    assert message.raw == {"type": "next_round"}


@pytest.mark.asyncio
async def test_training_player_holds_scoring_next_round() -> None:
    snapshot = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        scoring=ScoringSnapshot(
            round_winning_team=1,
            defender_points=80,
            total_defender_points=80,
            bottom_card_bonus=0,
            bottom_cards=[],
        ),
    )
    game = make_game(snapshot)
    player = TrainingPlayer(
        index=2,
        policy=FirstCardPlayPolicy(),
    )

    await player.on_state(game, make_state_message(snapshot, seq=9))
    game.receive.assert_not_awaited()

    confirmed = await player.confirm_held_scoring_next_round(game)

    assert confirmed
    game.receive.assert_awaited()
    message = game.receive.call_args[0][1]
    assert message.raw == {"type": "next_round"}
    assert message.seq == 9


@pytest.mark.asyncio
async def test_training_player_observation_uses_rules_plan() -> None:
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
        team0_level="2",
        team1_level="2",
    )
    game = make_game(snapshot)
    policy = CapturingFirstCardPlayPolicy()
    player = TrainingPlayer(
        index=0,
        policy=policy,
        observation_memory=_observation_memory(last_seq=0),
    )

    await player.on_state(game, make_state_message(snapshot, seq=1))
    await asyncio.sleep(0.05)

    observation = policy.observation
    assert observation is not None
    payloads = tuple(node.payload for node in observation.tokens)
    assert any(isinstance(payload, GlobalToken) for payload in payloads)
    assert any(
        isinstance(payload, RoundToken)
        and payload.field == RoundField.OWN_TARGET
        for payload in payloads
    )


def _observation_memory(*, last_seq: int) -> ObservationMemory:
    memory = ObservationMemory()
    result = memory.observe(
        make_state_message(
            make_snapshot(phase="WAITING", awaiting_action=None),
            seq=last_seq,
        )
    )
    assert isinstance(result, Ok)
    return memory
