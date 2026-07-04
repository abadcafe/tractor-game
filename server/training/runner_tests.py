"""Tests for training round reward extraction from snapshots."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from server.player.base import Player
from server.player.test_helpers import make_snapshot
from server.protocol import (
    PlayerMessage,
    ScoringSnapshot,
    StateMessage,
    StateSnapshot,
)
from server.result import Ok, Rejected
from server.training import runner
from server.training.config import ModelConfig
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.policy import PolicyDecision, RandomTrainingPolicy
from server.training.runner import SelfPlaySession, round_rewards
from server.training.tokens import FaceCountToken, TrickResultFieldToken


class _ScriptedBoundaryGame:
    """Fake Game exposing stale waiting state before next round."""

    def __init__(
        self,
        players: Sequence[Player],
    ) -> None:
        self._players = tuple(players)
        self._confirmations = 0
        self._first_completed = False
        self._second_started = False
        self._second_snapshot_count = 0
        self._first_before = make_snapshot(
            phase="PLAYING",
            awaiting_action=None,
            declarer_team=0,
            team0_level="A",
            team1_level="K",
        )
        self._first_final = make_snapshot(
            phase="WAITING",
            awaiting_action="next_round",
            declarer_team=0,
            scoring=ScoringSnapshot(
                round_winning_team=1,
                defender_points=120,
                total_defender_points=120,
                bottom_card_bonus=0,
                bottom_cards=[],
            ),
            team0_level="A",
            team1_level="A",
        )
        self._second_before = make_snapshot(
            phase="DEAL_BID",
            awaiting_action=None,
            declarer_team=1,
            team0_level="A",
            team1_level="A",
        )
        self._second_final = make_snapshot(
            phase="WAITING",
            awaiting_action="next_round",
            declarer_team=1,
            scoring=ScoringSnapshot(
                round_winning_team=1,
                defender_points=0,
                total_defender_points=0,
                bottom_card_bonus=0,
                bottom_cards=[],
            ),
            winning_team=1,
            team0_level="A",
            team1_level="A",
        )

    async def receive(
        self,
        player_index: int,
        message: PlayerMessage,
    ) -> None:
        if message.seq == 0:
            self._first_completed = True
            await self._send(player_index, self._first_final, seq=10)
            return
        self._confirmations += 1
        if self._confirmations < 4:
            await self._send(player_index, self._first_final, seq=10)
            return
        self._second_started = True
        for index in range(len(self._players)):
            await self._send(index, self._second_before, seq=11)

    def snapshot(self, for_player: int) -> StateSnapshot:
        assert for_player in (0, 1, 2, 3)
        if not self._first_completed:
            return self._first_before
        if not self._second_started:
            return self._first_final
        if self._second_snapshot_count == 0:
            self._second_snapshot_count += 1
            return self._second_before
        return self._second_final

    async def _send(
        self,
        player_index: int,
        snapshot: StateSnapshot,
        *,
        seq: int,
    ) -> None:
        await self._players[player_index].on_state(
            self,
            StateMessage(seq=seq, state=snapshot),
        )


class _RecordingPolicy:
    """Random policy wrapper that records all observations."""

    def __init__(self, seed: int) -> None:
        self._delegate = _random_policy(seed=seed)
        self.observations: list[Observation] = []

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
    ) -> Ok[PolicyDecision] | Rejected:
        self.observations.append(observation)
        return self._delegate.decide(observation, legal_actions)


@pytest.mark.asyncio
async def test_self_play_session_clears_round_history() -> None:
    policy = _RecordingPolicy(seed=7)
    session = SelfPlaySession(policy=policy)

    first_result = await session.play_round(max_seconds=120.0)
    assert isinstance(first_result, Ok)
    first = first_result.value
    first_decision_count = len(policy.observations)
    second_result = await session.play_round(max_seconds=120.0)
    assert isinstance(second_result, Ok)
    second = second_result.value
    second_round_bid = _first_bid_observation(
        tuple(policy.observations[first_decision_count:])
    )

    assert not first.rollout.is_empty()
    assert not second.rollout.is_empty()
    assert first.generated_action_count > 0
    assert second.generated_action_count > 0
    assert first.accepted_action_count > 0
    assert second.accepted_action_count > 0
    assert _play_record_token_count(second_round_bid) == 0
    assert _trick_result_token_count(second_round_bid) == 0


@pytest.mark.asyncio
async def test_self_play_session_uses_new_round_boundary_for_rewards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "Game", _ScriptedBoundaryGame)
    session = SelfPlaySession(policy=_random_policy(seed=7))

    first = await session.play_round(max_seconds=120.0)
    assert isinstance(first, Ok)
    second_result = await session.play_round(max_seconds=120.0)
    assert isinstance(second_result, Ok)
    second = second_result.value

    assert second.team0_reward == -1.0
    assert second.team1_reward == 1.0
    assert second.game_over is True


@pytest.mark.asyncio
async def test_self_play_session_rejects_round_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "Game", _ScriptedBoundaryGame)
    session = SelfPlaySession(policy=_random_policy(seed=7))

    result = await session.play_round(max_seconds=0.0)

    assert isinstance(result, Rejected)
    assert "training round timed out" in result.reason


def test_round_rewards_uses_scoring_round_winning_team() -> None:
    before = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer_team=1,
        team0_level="10",
        team1_level="10",
    )
    after = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=1,
        scoring=ScoringSnapshot(
            round_winning_team=0,
            defender_points=80,
            total_defender_points=80,
            bottom_card_bonus=0,
            bottom_cards=[],
        ),
        team0_level="10",
        team1_level="10",
    )

    team0_reward, team1_reward = round_rewards(
        before=before,
        after=after,
    )

    assert team0_reward == 1.0
    assert team1_reward == -1.0


def _random_policy(*, seed: int) -> RandomTrainingPolicy:
    return RandomTrainingPolicy(
        model_config=ModelConfig(max_tokens=512),
        device=torch.device("cpu"),
        seed=seed,
    )


def test_round_rewards_counts_defender_taking_stage_control() -> None:
    before = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer_team=0,
        team0_level="10",
        team1_level="10",
    )
    after = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=0,
        scoring=ScoringSnapshot(
            round_winning_team=1,
            defender_points=120,
            total_defender_points=120,
            bottom_card_bonus=0,
            bottom_cards=[],
        ),
        team0_level="10",
        team1_level="J",
    )

    team0_reward, team1_reward = round_rewards(
        before=before,
        after=after,
    )

    assert team0_reward == -2.0
    assert team1_reward == 2.0


def test_round_rewards_uses_completed_round_declarer() -> None:
    before = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer_team=0,
        team0_level="10",
        team1_level="10",
    )
    after = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=0,
        scoring=ScoringSnapshot(
            round_winning_team=1,
            defender_points=135,
            total_defender_points=135,
            bottom_card_bonus=0,
            bottom_cards=[],
        ),
        team0_level="10",
        team1_level="J",
    )

    team0_reward, team1_reward = round_rewards(
        before=before,
        after=after,
    )

    _assert_close(team0_reward, -2.375)
    _assert_close(team1_reward, 2.375)


def test_round_rewards_uses_after_declarer_if_before_unset() -> None:
    before = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=None,
        team0_level="10",
        team1_level="10",
    )
    after = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=0,
        scoring=ScoringSnapshot(
            round_winning_team=1,
            defender_points=135,
            total_defender_points=135,
            bottom_card_bonus=0,
            bottom_cards=[],
        ),
        team0_level="10",
        team1_level="J",
    )

    team0_reward, team1_reward = round_rewards(
        before=before,
        after=after,
    )

    _assert_close(team0_reward, -2.375)
    _assert_close(team1_reward, 2.375)


def test_round_rewards_maps_winning_team_to_terminal_progress() -> None:
    before = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer_team=0,
        team0_level="A",
        team1_level="A",
    )
    after = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=0,
        scoring=ScoringSnapshot(
            round_winning_team=0,
            defender_points=0,
            total_defender_points=0,
            bottom_card_bonus=0,
            bottom_cards=[],
        ),
        winning_team=0,
        team0_level="A",
        team1_level="A",
    )

    team0_reward, team1_reward = round_rewards(
        before=before,
        after=after,
    )

    assert team0_reward == 1.0
    assert team1_reward == -1.0


def test_round_rewards_counts_team1_terminal_win() -> None:
    before = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer_team=1,
        team0_level="A",
        team1_level="A",
    )
    after = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=1,
        scoring=ScoringSnapshot(
            round_winning_team=1,
            defender_points=0,
            total_defender_points=0,
            bottom_card_bonus=0,
            bottom_cards=[],
        ),
        winning_team=1,
        team0_level="A",
        team1_level="A",
    )

    team0_reward, team1_reward = round_rewards(
        before=before,
        after=after,
    )

    assert team0_reward == -1.0
    assert team1_reward == 1.0


def _first_bid_observation(
    observations: tuple[Observation, ...],
) -> Observation:
    for observation in observations:
        if observation.action_query.kind == "bid":
            return observation
    assert False


def _play_record_token_count(observation: Observation) -> int:
    return sum(
        1
        for token in observation.tokens
        if isinstance(token, FaceCountToken)
        and token.segment == "play_record"
    )


def _trick_result_token_count(observation: Observation) -> int:
    return sum(
        1
        for token in observation.tokens
        if isinstance(token, TrickResultFieldToken)
    )


def _assert_close(actual: float, expected: float) -> None:
    assert abs(actual - expected) <= 0.000000000001
