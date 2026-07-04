"""Black-box tests for terminal reward rollout construction."""

from __future__ import annotations

import torch

from server.player.test_helpers import card, make_snapshot
from server.result import Ok
from server.rules.card_faces import CardFace, FaceCount
from server.training.choice_trace import (
    SemanticChoiceStep,
    SemanticChoiceTrace,
    semantic_choice_step_from_argument,
)
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.observation import build_observation
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
)
from server.training.tensorize import tensorize_observation
from server.training.terminal_rewards import terminal_reward_rollout
from server.training.trajectory import DecisionStep


def test_terminal_reward_rollout_groups_team_transitions() -> None:
    steps = tuple(
        _decision_step(player_index) for player_index in (0, 1, 2, 3, 0)
    )

    batch = terminal_reward_rollout(
        steps=steps,
        team0_reward=2.0,
        team1_reward=-2.0,
    )

    assert batch.transition_count() == 5
    assert not batch.is_empty()
    assert len(batch.trajectories) == 2
    team0 = batch.trajectories[0]
    team1 = batch.trajectories[1]
    assert team0.team_index == 0
    assert team1.team_index == 1
    assert [
        transition.decision.player_index
        for transition in team0.transitions
    ] == [0, 2, 0]
    assert [
        transition.decision.player_index
        for transition in team1.transitions
    ] == [1, 3]


def test_terminal_reward_rollout_separates_terminal_reward() -> None:
    steps = tuple(
        _decision_step(player_index) for player_index in (0, 2, 1, 3)
    )

    batch = terminal_reward_rollout(
        steps=steps,
        team0_reward=1.0,
        team1_reward=-1.0,
    )

    team0 = batch.trajectories[0]
    team1 = batch.trajectories[1]
    assert team0.terminal_reward == 1.0
    assert team1.terminal_reward == -1.0
    assert [
        transition.reward_after_step for transition in team0.transitions
    ] == [0.0, 0.0]
    assert [
        transition.reward_after_step for transition in team1.transitions
    ] == [0.0, 0.0]


def _decision_step(player_index: int) -> DecisionStep:
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    observation = build_observation(
        player_index=player_index,
        snapshot=snapshot,
        history=(),
    )
    legal_actions = build_legal_action_index(
        player_index=player_index,
        snapshot=snapshot,
        query=observation.action_query,
    )
    trace = SemanticArgumentTrace(
        arguments=(
            SemanticArgument(
                "select_face_count",
                FaceCount(CardFace(test_card.suit, test_card.rank), 1),
            ),
            SemanticArgument("stop"),
        )
    )
    decoded = legal_actions.decode(trace)
    assert isinstance(decoded, Ok)
    return DecisionStep(
        player_index=player_index,
        seq=1,
        observation_batch=tensorize_observation(
            observation=observation,
            max_observation_tokens=64,
            device=torch.device("cpu"),
        ),
        choice_trace=_choice_trace_for(
            legal_actions=legal_actions,
            trace=trace,
        ),
        action=decoded.value,
        log_probability=0.0,
        value_estimate=0.0,
        entropy=0.0,
        choice_count=2,
    )


def _choice_trace_for(
    *,
    legal_actions: LegalActionIndex,
    trace: SemanticArgumentTrace,
) -> SemanticChoiceTrace:
    prefix = SemanticArgumentPrefix(arguments=())
    steps: list[SemanticChoiceStep] = []
    for argument in trace.arguments:
        allowed = legal_actions.allowed_next(prefix)
        steps.append(
            semantic_choice_step_from_argument(
                allowed=allowed,
                selected_argument=argument,
            )
        )
        if argument.kind in ("pass", "stop"):
            continue
        prefix = SemanticArgumentPrefix(
            arguments=(*prefix.arguments, argument)
        )
    return SemanticChoiceTrace(steps=tuple(steps))
