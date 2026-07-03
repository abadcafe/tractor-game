"""Tests for TrainingPlayer behavior through the Player interface."""

from __future__ import annotations

import asyncio

import pytest

from server.player.test_helpers import (
    card,
    make_game,
    make_snapshot,
    make_state_message,
)
from server.protocol import ScoringSnapshot
from server.result import Ok
from server.rules.card_faces import CardFace, FaceCount
from server.rules.cards import Rank
from server.sm.required_progress import (
    DEFAULT_REQUIRED_LEVEL_PLAN,
    RequiredLevelPlan,
)
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.player import TrainingPlayer
from server.training.policy import PolicyDecision
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
)
from server.training.tokens import GlobalFieldToken, RoundFieldToken
from server.training.trajectory import TrajectoryRecorder


class FirstCardPlayPolicy:
    """Deterministic test policy that plays the first hand face."""

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
    ) -> PolicyDecision:
        first_choices = legal_actions.allowed_next(
            SemanticArgumentPrefix(arguments=())
        )
        assert first_choices
        first_argument = first_choices[0]
        prefix = SemanticArgumentPrefix(arguments=(first_argument,))
        trace_args: list[SemanticArgument] = [first_argument]
        second_choices = legal_actions.allowed_next(prefix)
        if second_choices:
            trace_args.append(second_choices[0])
        decoded = legal_actions.decode(
            SemanticArgumentTrace(arguments=tuple(trace_args))
        )
        assert isinstance(decoded, Ok)
        return PolicyDecision(
            action=decoded.value,
            log_probability=0.0,
            value_estimate=0.0,
            entropy=0.0,
            choice_count=len(decoded.value.semantic_trace.arguments),
        )


class CapturingFirstCardPlayPolicy(FirstCardPlayPolicy):
    """Test policy that stores the observation it receives."""

    def __init__(self) -> None:
        self.observation: Observation | None = None

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
    ) -> PolicyDecision:
        self.observation = observation
        return super().decide(observation, legal_actions)


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
        required_level_plan=DEFAULT_REQUIRED_LEVEL_PLAN,
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
        required_level_plan=DEFAULT_REQUIRED_LEVEL_PLAN,
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
async def test_training_player_submits_initial_next_round() -> None:
    snapshot = make_snapshot(
        phase="WAITING", awaiting_action="next_round"
    )
    game = make_game(snapshot)
    player = TrainingPlayer(
        index=2,
        policy=FirstCardPlayPolicy(),
        required_level_plan=DEFAULT_REQUIRED_LEVEL_PLAN,
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
        required_level_plan=DEFAULT_REQUIRED_LEVEL_PLAN,
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
async def test_training_player_observation_uses_custom_plan() -> None:
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
        required_level_plan=RequiredLevelPlan(
            required_levels=(Rank.JACK, Rank.ACE)
        ),
    )

    await player.on_state(game, make_state_message(snapshot, seq=1))
    await asyncio.sleep(0.05)

    observation = policy.observation
    assert observation is not None
    assert GlobalFieldToken("required_level", "J") in observation.tokens
    assert GlobalFieldToken("required_level", "A") in observation.tokens
    assert (
        RoundFieldToken("self_team_required_level", "J")
        in observation.tokens
    )
