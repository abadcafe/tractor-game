"""Black-box tests for the training policy decision engine."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game import commands
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.player import TrainingPlayer
from server.training.policy import PolicyDecision, RandomTrainingPolicy
from server.training.sampling import PolicyDecisionKey
from tests.support import card, snapshot


class _RejectingPolicy:
    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        assert observation.action_query == legal_actions.query
        assert decision_key.decision_index == 0
        return Rejected("policy sampling failed")


async def test_player_records_only_accepted_typed_decisions() -> None:
    player = TrainingPlayer(
        index=0,
        policy=RandomTrainingPolicy(),
    )
    player.reset_round_tracking(
        base_seed=7,
        policy_version=3,
        rollout_id="rollout",
        episode_id=11,
    )
    current = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2")],
        player_hand_counts=[1, 0, 0, 0],
        trump_rank="2",
    )
    observed = player.observe(seq=4, snapshot=current)
    assert isinstance(observed, Ok)

    decision = await player.decide(seq=4, snapshot=current)
    assert isinstance(decision, Ok)
    assert isinstance(
        decision.value.command,
        commands.PassBid | commands.RevealBid,
    )
    assert player.stats().generated_action_count == 1
    assert player.stats().accepted_action_count == 0
    assert player.recorder().steps() == ()

    player.accept(decision.value)

    assert player.stats().accepted_action_count == 1
    assert player.recorder().steps() == (decision.value.step,)


def test_player_rejects_non_contiguous_observation_history() -> None:
    player = TrainingPlayer(
        index=0,
        policy=RandomTrainingPolicy(),
    )
    player.reset_round_tracking(
        base_seed=0,
        policy_version=0,
        rollout_id="rollout",
        episode_id=0,
    )
    initial = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2")],
        player_hand_counts=[1, 0, 0, 0],
    )
    assert isinstance(player.observe(seq=8, snapshot=initial), Ok)

    missed = player.observe(seq=10, snapshot=initial)

    assert isinstance(missed, Rejected)
    assert "missed state sequence 9" in missed.reason


async def test_decide_propagates_policy_rejection() -> None:
    player = TrainingPlayer(index=0, policy=_RejectingPolicy())
    player.reset_round_tracking(
        base_seed=5,
        policy_version=2,
        rollout_id="rollout",
        episode_id=3,
    )
    current = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2")],
        player_hand_counts=[1, 0, 0, 0],
    )
    assert isinstance(player.observe(seq=1, snapshot=current), Ok)

    result = await player.decide(seq=1, snapshot=current)

    assert isinstance(result, Rejected)
    assert result.reason == "policy sampling failed"
    assert player.stats().generated_action_count == 0
    assert player.stats().accepted_action_count == 0
    assert player.recorder().steps() == ()


def test_player_reset_forgets_previous_observation_sequence() -> None:
    player = TrainingPlayer(
        index=0,
        policy=RandomTrainingPolicy(),
    )
    initial = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[card("hearts", "2")],
        player_hand_counts=[1, 0, 0, 0],
    )
    player.reset_round_tracking(
        base_seed=1,
        policy_version=0,
        rollout_id="first",
        episode_id=1,
    )
    assert isinstance(player.observe(seq=20, snapshot=initial), Ok)

    player.reset_round_tracking(
        base_seed=2,
        policy_version=1,
        rollout_id="second",
        episode_id=2,
    )
    restarted = player.observe(seq=1, snapshot=initial)

    assert isinstance(restarted, Ok)
    assert player.stats().generated_action_count == 0
    assert player.stats().accepted_action_count == 0
