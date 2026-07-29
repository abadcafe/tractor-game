"""Black-box tests for direct pure-game self-play."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from server.foundation.result import Ok, Rejected
from server.game import Partnership, Seat
from server.game.snapshots.review import ScoringSnapshot
from server.policy_model.actions import LegalActionSpace
from server.policy_model.observation import Observation
from server.policy_model.observation.tokenization import ActionToken
from server.policy_model.return_target import round_return_targets
from server.training.rollout_inference.randomness import (
    TrainingDecisionKey,
)
from server.training.self_play.policy import (
    PolicyDecision,
    RandomTrainingPolicy,
)
from server.training.self_play.session import (
    SelfPlaySession,
)
from tests.support import partnership_levels, snapshot


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
        legal_actions: LegalActionSpace,
        decision_key: TrainingDecisionKey,
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
        legal_actions: LegalActionSpace,
        decision_key: TrainingDecisionKey,
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


def test_round_return_targets_use_completed_round_facts() -> None:
    before = snapshot(
        declarer=Seat.A,
        partnership_levels=partnership_levels("10", "10"),
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer=Seat.A,
        scoring=ScoringSnapshot(
            winning_partnership=Partnership.SECOND,
            defender_points=135,
            total_defender_points=135,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        partnership_levels=partnership_levels("10", "J"),
    )

    first_partnership_reward, second_partnership_reward = (
        round_return_targets(
            before=before,
            after=after,
        )
    )

    assert abs(first_partnership_reward - -2.375) < 1e-9
    assert abs(second_partnership_reward - 2.375) < 1e-9


def test_round_return_targets_map_winner_to_terminal_progress() -> None:
    before = snapshot(
        declarer=Seat.B,
        partnership_levels=partnership_levels("A", "A"),
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action=None,
        declarer=Seat.B,
        scoring=ScoringSnapshot(
            winning_partnership=Partnership.SECOND,
            defender_points=0,
            total_defender_points=0,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        winning_partnership=Partnership.SECOND,
        partnership_levels=partnership_levels("A", "A"),
    )

    assert round_return_targets(before=before, after=after) == (
        -1.0,
        1.0,
    )


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


def test_round_return_targets_use_scoring_winner_at_threshold() -> None:
    before = snapshot(
        declarer=Seat.B,
        partnership_levels=partnership_levels("10", "10"),
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer=Seat.B,
        scoring=ScoringSnapshot(
            winning_partnership=Partnership.FIRST,
            defender_points=80,
            total_defender_points=80,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        partnership_levels=partnership_levels("10", "10"),
    )

    assert round_return_targets(before=before, after=after) == (
        1.0,
        -1.0,
    )


def test_round_return_targets_count_defender_stage_progress() -> None:
    before = snapshot(
        declarer=Seat.A,
        partnership_levels=partnership_levels("10", "10"),
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer=Seat.A,
        scoring=ScoringSnapshot(
            winning_partnership=Partnership.SECOND,
            defender_points=120,
            total_defender_points=120,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        partnership_levels=partnership_levels("10", "J"),
    )

    assert round_return_targets(before=before, after=after) == (
        -2.0,
        2.0,
    )


def test_return_target_uses_after_declarer_when_before_is_unset() -> (
    None
):
    before = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer=None,
        partnership_levels=partnership_levels("10", "10"),
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action="next_round",
        declarer=Seat.A,
        scoring=ScoringSnapshot(
            winning_partnership=Partnership.SECOND,
            defender_points=135,
            total_defender_points=135,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        partnership_levels=partnership_levels("10", "J"),
    )

    first_partnership, second_partnership = round_return_targets(
        before=before, after=after
    )
    assert abs(first_partnership - -2.375) < 1e-9
    assert abs(second_partnership - 2.375) < 1e-9


def test_round_return_targets_map_first_partnership_terminal_win() -> (
    None
):
    before = snapshot(
        declarer=Seat.A,
        partnership_levels=partnership_levels("A", "A"),
    )
    after = snapshot(
        phase="WAITING",
        awaiting_action=None,
        declarer=Seat.A,
        scoring=ScoringSnapshot(
            winning_partnership=Partnership.FIRST,
            defender_points=0,
            total_defender_points=0,
            bottom_card_bonus=0,
            bottom_cards=(),
        ),
        winning_partnership=Partnership.FIRST,
        partnership_levels=partnership_levels("A", "A"),
    )

    assert round_return_targets(before=before, after=after) == (
        1.0,
        -1.0,
    )
