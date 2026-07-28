"""Black-box tests for the complete policy model package."""

from __future__ import annotations

import torch

from server.game import Seat
from server.policy_model import network as model_api
from server.policy_model._schema.actions import (
    ACTION_CHOICE_COUNT,
    CARD_CHOICE_BASE_ID,
    PASS_CHOICE_ID,
)
from server.policy_model.network import PolicyValueModel
from server.policy_model.observation import (
    Observation,
    build_observation,
)
from server.policy_model.observation.history import (
    ObservationMemoryView,
)
from server.policy_model.observation.tensor_batch import (
    tensorize_observation,
)
from tests.support import card, seat_values
from tests.support import snapshot as make_snapshot


def test_model_package_exposes_only_complete_model_api() -> None:
    assert model_api.__all__ == (
        "ActionDecodeSession",
        "ActionTraceScores",
        "EncodedObservation",
        "MIN_ATTENTION_HEAD_DIMENSION",
        "ModelConfig",
        "PolicyValueModel",
    )


def test_model_scores_exactly_110_choices_and_one_value() -> None:
    device = torch.device("cpu")
    model = PolicyValueModel(d_model=16, layers=1, heads=2)
    batch = tensorize_observation(
        observation=_bid_observation(), device=device
    )

    encoding = model.encode_observations(batch)
    scores = model.score_action_traces(
        encoding,
        source_rows=torch.tensor((0,), dtype=torch.long, device=device),
        choice_ids_padded=torch.tensor(
            ((PASS_CHOICE_ID,),), dtype=torch.long, device=device
        ),
        step_counts=torch.tensor((1,), dtype=torch.long, device=device),
    )

    assert scores.choice_logits.shape == (1, 1, ACTION_CHOICE_COUNT)
    assert model.value_estimates(encoding).shape == (1,)


def test_live_query_seed_matches_teacher_forced_scoring() -> None:
    device = torch.device("cpu")
    model = PolicyValueModel(d_model=16, layers=1, heads=2)
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
            source_rows=torch.tensor(
                (0,), dtype=torch.long, device=device
            ),
            choice_ids_padded=choices,
            step_counts=steps,
        )
        session = model.begin_action_decode_session(
            encoding,
            source_rows=torch.tensor(
                (0,), dtype=torch.long, device=device
            ),
            max_steps=2,
        )
        active = torch.tensor((True,), dtype=torch.bool, device=device)
        first = session.next_choice_logits(active, active)
        session.advance(choices[:, 0], active)
        second = session.next_choice_logits(active, active)

    torch.testing.assert_close(first, scored.choice_logits[:, 0])
    torch.testing.assert_close(second, scored.choice_logits[:, 1])


def _bid_observation() -> Observation:
    return build_observation(
        viewer=Seat.A,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            hand=[card("hearts", "2")],
            remaining_cards=seat_values(1, 0, 0, 0),
            trump_rank="2",
        ),
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )
