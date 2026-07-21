"""Black-box tests for the typed policy model."""

from __future__ import annotations

import torch

from server.game.players.test_helpers import card, make_snapshot
from server.training.model import TractorPolicyModel
from server.training.observation import Observation, build_observation
from server.training.observation_memory import ObservationMemoryView
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    CARD_CHOICE_BASE_ID,
    PASS_CHOICE_ID,
)
from server.training.tensorize import tensorize_observation


def test_model_scores_exactly_110_choices_and_one_value() -> None:
    device = torch.device("cpu")
    model = TractorPolicyModel(d_model=16, layers=1, heads=2)
    batch = tensorize_observation(
        observation=_bid_observation(), device=device
    )

    encoding = model.encode_observations(batch)
    scores = model.score_action_traces(
        encoding,
        choice_ids_padded=torch.tensor(
            ((PASS_CHOICE_ID,),), dtype=torch.long, device=device
        ),
        step_counts=torch.tensor((1,), dtype=torch.long, device=device),
    )

    assert scores.choice_logits.shape == (1, 1, ACTION_CHOICE_COUNT)
    assert model.value_estimates(encoding).shape == (1,)


def test_live_query_seed_matches_teacher_forced_scoring() -> None:
    device = torch.device("cpu")
    model = TractorPolicyModel(d_model=16, layers=1, heads=2)
    model.eval()
    batch = tensorize_observation(
        observation=_bid_observation(), device=device
    )
    choices = torch.tensor(
        ((CARD_CHOICE_BASE_ID, PASS_CHOICE_ID),),
        dtype=torch.long,
        device=device,
    )
    steps = torch.tensor((2,), dtype=torch.long, device=device)

    with torch.no_grad():
        encoding = model.encode_observations(batch)
        scored = model.score_action_traces(
            encoding,
            choice_ids_padded=choices,
            step_counts=steps,
        )
        session = model.begin_action_decode_session(
            encoding, max_steps=2
        )
        first = session.next_choice_logits()
        session.advance(choices[:, 0])
        second = session.next_choice_logits()

    torch.testing.assert_close(first, scored.choice_logits[:, 0])
    torch.testing.assert_close(second, scored.choice_logits[:, 1])


def _bid_observation() -> Observation:
    return build_observation(
        viewer=0,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            player_hand=[card("hearts", "2")],
            player_hand_counts=[1, 0, 0, 0],
            trump_rank="2",
        ),
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )
