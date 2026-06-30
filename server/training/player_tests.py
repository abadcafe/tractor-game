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
from server.training.action_tokens import (
    ACTION_PLAY_TOKEN_ID,
    BEGIN_TOKEN_ID,
    FIRST_CARD_TOKEN_ID,
    STOP_TOKEN_ID,
    ActionQuery,
    decode_action_tokens,
)
from server.training.observation import Observation
from server.training.player import TrainingPlayer
from server.training.policy import PolicyDecision
from server.training.trajectory import TrajectoryRecorder


class FirstCardPlayPolicy:
    """Deterministic test policy that plays the first hand slot."""

    def decide(
        self,
        observation: Observation,
        query: ActionQuery,
    ) -> PolicyDecision:
        decoded = decode_action_tokens(
            query,
            (
                BEGIN_TOKEN_ID,
                ACTION_PLAY_TOKEN_ID,
                FIRST_CARD_TOKEN_ID,
                STOP_TOKEN_ID,
            ),
        )
        assert isinstance(decoded, Ok)
        return PolicyDecision(
            action=decoded.value,
            log_probability=0.0,
            value_estimate=0.0,
            entropy=0.0,
            token_count=len(decoded.value.token_ids),
        )


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
    assert steps[0].action.card_ids == (test_card.id,)
    assert steps[0].token_count == 4


@pytest.mark.asyncio
async def test_training_player_submits_initial_next_round() -> None:
    snapshot = make_snapshot(
        phase="WAITING", awaiting_action="next_round"
    )
    game = make_game(snapshot)
    player = TrainingPlayer(index=2, policy=FirstCardPlayPolicy())

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
            declarer_team=0,
            defender_points=80,
            total_defender_points=80,
            bottom_card_bonus=0,
            bottom_cards=[],
        ),
    )
    game = make_game(snapshot)
    player = TrainingPlayer(index=2, policy=FirstCardPlayPolicy())

    await player.on_state(game, make_state_message(snapshot, seq=9))
    game.receive.assert_not_awaited()

    confirmed = await player.confirm_held_scoring_next_round(game)

    assert confirmed
    game.receive.assert_awaited()
    message = game.receive.call_args[0][1]
    assert message.raw == {"type": "next_round"}
    assert message.seq == 9
