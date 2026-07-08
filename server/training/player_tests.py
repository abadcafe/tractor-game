"""Tests for TrainingPlayer behavior through the Player interface."""

from __future__ import annotations

import asyncio

import pytest
import torch

from server.player.test_helpers import (
    card,
    make_game,
    make_snapshot,
    make_state_message,
)
from server.protocol import ScoringSnapshot
from server.result import Ok, Rejected
from server.rules.card_faces import CardFace, FaceCount
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.player import TrainingPlayer
from server.training.policy import PolicyDecision
from server.training.policy_sampling import DecisionHandle
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_action_plan import (
    advance_action_state,
    compile_legal_action_frame,
    initial_action_state,
    legal_token_choices,
    plan_batch_to_device,
    semantic_trace_from_token_ids,
)
from server.training.semantic_actions import (
    SemanticArgumentTrace,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tokens import GlobalFieldToken, RoundFieldToken
from server.training.trajectory import TrajectoryRecorder


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
                    slot_index=decision_key.decision_index,
                    slot_generation=0,
                ),
                choice_count=len(
                    decoded.value.semantic_trace.arguments
                ),
            )
        )


def _first_legal_trace(
    legal_actions: LegalActionIndex,
) -> Ok[SemanticArgumentTrace] | Rejected:
    device = torch.device("cpu")
    batch = plan_batch_to_device(
        (compile_legal_action_frame(legal_actions),), device=device
    )
    state = initial_action_state(batch)
    for _ in range(SEMANTIC_CODEC.max_argument_tokens):
        if bool(state.done[0].item()):
            trace_ids = tuple(
                int(state.selected_token_ids[0, index].item())
                for index in range(int(state.step_counts[0].item()))
            )
            return semantic_trace_from_token_ids(trace_ids)
        choices = legal_token_choices(batch=batch, state=state)
        assert int(choices.choice_counts[0].item()) > 0
        state = advance_action_state(
            batch=batch,
            state=state,
            selected_token_ids=choices.token_ids[0].view(1),
            choice_counts=choices.choice_counts,
        )
    assert False


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
    )

    await player.on_state(game, make_state_message(snapshot, seq=7))
    await asyncio.sleep(0.05)
    await player.on_state(
        game, make_state_message(accepted_snapshot, seq=7)
    )
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
    )

    await player.on_state(game, make_state_message(snapshot, seq=1))
    await asyncio.sleep(0.05)

    observation = policy.observation
    assert observation is not None
    assert GlobalFieldToken("required_level", "A") in observation.tokens
    assert (
        RoundFieldToken("self_team_required_level", "A")
        in observation.tokens
    )
