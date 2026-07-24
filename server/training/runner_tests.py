"""Black-box tests for direct pure-game self-play."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from server.foundation.result import Ok, Rejected
from server.game.snapshots.review import ScoringSnapshot
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.policy import (
    PolicyDecision,
    RandomTrainingPolicy,
)
from server.training.runner import SelfPlaySession, round_rewards
from server.training.sampling import PolicyDecisionKey
from server.training.tokenization import ActionToken
from tests.support import snapshot


def _observations() -> list[Observation]:
    return []


@dataclass(slots=True)
class _RecordingPolicy:
    delegate: RandomTrainingPolicy = field(
        default_factory=RandomTrainingPolicy
    )
    observations: list[Observation] = field(
        default_factory=_observations
    )

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        self.observations.append(observation)
        return await self.delegate.decide(
            observation,
            legal_actions,
            decision_key,
        )


class _NeverDecidesPolicy:
    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        assert observation.action_query == legal_actions.query
        assert decision_key.episode_id >= 0
        await asyncio.Event().wait()
        assert False


async def test_session_plays_consecutive_rounds_without_runtime() -> (
    None
):
    policy = _RecordingPolicy()
    session = SelfPlaySession(policy=policy)

    first = await session.play_round(
        base_seed=7,
        policy_version=1,
        rollout_id="rollout",
        episode_id=1,
        max_seconds=60.0,
    )
    assert isinstance(first, Ok)
    assert first.value.generated_action_count > 0
    assert (
        first.value.accepted_action_count
        == first.value.generated_action_count
    )
    first_observation_count = len(policy.observations)

    second = await session.play_round(
        base_seed=7,
        policy_version=1,
        rollout_id="rollout",
        episode_id=2,
        max_seconds=60.0,
    )
    assert isinstance(second, Ok)
    assert len(policy.observations) > first_observation_count
    second_round = policy.observations[first_observation_count:]
    first_bid = next(
        observation
        for observation in second_round
        if observation.action_query.kind == "bid"
    )
    assert not any(
        isinstance(node.payload, ActionToken)
        and node.payload.kind == "play"
        and node.payload.occurrence == "fact"
        for node in first_bid.tokens
    )


def test_round_rewards_use_completed_round_facts() -> None:
    before = snapshot(
        declarer_team=0,
        team0_level="10",
        team1_level="10",
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=0,
        scoring=ScoringSnapshot(
            round_winning_team=1,
            defender_points=135,
            total_defender_points=135,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        team0_level="10",
        team1_level="J",
    )

    team0_reward, team1_reward = round_rewards(
        before=before,
        after=after,
    )

    assert abs(team0_reward - -2.375) < 1e-9
    assert abs(team1_reward - 2.375) < 1e-9


def test_round_rewards_map_game_winner_to_terminal_progress() -> None:
    before = snapshot(
        declarer_team=1,
        team0_level="A",
        team1_level="A",
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action=None,
        declarer_team=1,
        scoring=ScoringSnapshot(
            round_winning_team=1,
            defender_points=0,
            total_defender_points=0,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        winning_team=1,
        team0_level="A",
        team1_level="A",
    )

    assert round_rewards(before=before, after=after) == (-1.0, 1.0)


async def test_session_rejects_policy_timeout() -> None:
    session = SelfPlaySession(policy=_NeverDecidesPolicy())

    result = await session.play_round(
        base_seed=13,
        policy_version=0,
        rollout_id="rollout",
        episode_id=0,
        max_seconds=0.001,
    )

    assert isinstance(result, Rejected)
    assert "training round timed out" in result.reason


def test_round_rewards_use_scoring_winning_team_at_threshold() -> None:
    before = snapshot(
        declarer_team=1,
        team0_level="10",
        team1_level="10",
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=1,
        scoring=ScoringSnapshot(
            round_winning_team=0,
            defender_points=80,
            total_defender_points=80,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        team0_level="10",
        team1_level="10",
    )

    assert round_rewards(before=before, after=after) == (1.0, -1.0)


def test_round_rewards_count_defender_stage_progress() -> None:
    before = snapshot(
        declarer_team=0,
        team0_level="10",
        team1_level="10",
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=0,
        scoring=ScoringSnapshot(
            round_winning_team=1,
            defender_points=120,
            total_defender_points=120,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        team0_level="10",
        team1_level="J",
    )

    assert round_rewards(before=before, after=after) == (-2.0, 2.0)


def test_round_rewards_use_after_declarer_when_before_is_unset() -> (
    None
):
    before = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=None,
        team0_level="10",
        team1_level="10",
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer_team=0,
        scoring=ScoringSnapshot(
            round_winning_team=1,
            defender_points=135,
            total_defender_points=135,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        team0_level="10",
        team1_level="J",
    )

    team0, team1 = round_rewards(before=before, after=after)
    assert abs(team0 - -2.375) < 1e-9
    assert abs(team1 - 2.375) < 1e-9


def test_round_rewards_map_team0_terminal_win() -> None:
    before = snapshot(
        declarer_team=0,
        team0_level="A",
        team1_level="A",
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action=None,
        declarer_team=0,
        scoring=ScoringSnapshot(
            round_winning_team=0,
            defender_points=0,
            total_defender_points=0,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        winning_team=0,
        team0_level="A",
        team1_level="A",
    )

    assert round_rewards(before=before, after=after) == (1.0, -1.0)
