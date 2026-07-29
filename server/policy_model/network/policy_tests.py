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
from server.policy_model.network import PolicyActionModel
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
        "PolicyActionModel",
    )


def test_model_scores_policy_choices_and_complete_action_value() -> (
    None
):
    device = torch.device("cpu")
    model = PolicyActionModel(d_model=16, layers=1, heads=2)
    batch = tensorize_observation(
        observation=_bid_observation(), device=device
    )

    encoding = model.encode_policy_observations(batch)
    scores = model.score_action_traces(
        encoding,
        source_rows=torch.tensor((0,), dtype=torch.long, device=device),
        choice_ids_padded=torch.tensor(
            ((PASS_CHOICE_ID,),), dtype=torch.long, device=device
        ),
        step_counts=torch.tensor((1,), dtype=torch.long, device=device),
    )

    value_encoding = model.encode_action_value_observations(batch)
    action_values = model.action_values(
        value_encoding,
        source_rows=torch.tensor((0,), dtype=torch.long, device=device),
        choice_ids_padded=torch.tensor(
            ((PASS_CHOICE_ID,),),
            dtype=torch.long,
            device=device,
        ),
        step_counts=torch.tensor((1,), dtype=torch.long, device=device),
    )

    assert scores.choice_logits.shape == (1, 1, ACTION_CHOICE_COUNT)
    assert action_values.shape == (1,)


def test_policy_and_action_value_parameters_are_disjoint() -> None:
    model = PolicyActionModel(d_model=16, layers=1, heads=2)

    policy_parameters = model.policy_parameters()
    action_value_parameters = model.action_value_parameters()
    policy_ids = {id(parameter) for parameter in policy_parameters}
    action_value_ids = {
        id(parameter) for parameter in action_value_parameters
    }

    assert policy_ids
    assert action_value_ids
    assert policy_ids.isdisjoint(action_value_ids)
    assert policy_ids | action_value_ids == {
        id(parameter) for parameter in model.parameters()
    }


def test_each_loss_reaches_only_its_owned_branch() -> None:
    device = torch.device("cpu")
    model = PolicyActionModel(d_model=16, layers=1, heads=2)
    batch = tensorize_observation(
        observation=_bid_observation(), device=device
    )
    source_rows = torch.tensor((0,), dtype=torch.long, device=device)
    choice_ids = torch.tensor(
        ((PASS_CHOICE_ID,),),
        dtype=torch.long,
        device=device,
    )
    step_counts = torch.tensor((1,), dtype=torch.long, device=device)

    policy_encoding = model.encode_policy_observations(batch)
    policy_scores = model.score_action_traces(
        policy_encoding,
        source_rows=source_rows,
        choice_ids_padded=choice_ids,
        step_counts=step_counts,
    )
    torch.autograd.backward(policy_scores.choice_logits.sum())

    assert any(
        parameter.grad is not None
        for parameter in model.policy_parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.action_value_parameters()
    )

    model.zero_grad(set_to_none=True)
    action_value_encoding = model.encode_action_value_observations(
        batch
    )
    torch.autograd.backward(
        model.action_values(
            action_value_encoding,
            source_rows=source_rows,
            choice_ids_padded=choice_ids,
            step_counts=step_counts,
        ).sum()
    )

    assert all(
        parameter.grad is None
        for parameter in model.policy_parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.action_value_parameters()
    )


def test_action_value_reads_last_choice_and_ignores_padding() -> None:
    device = torch.device("cpu")
    model = PolicyActionModel(d_model=16, layers=1, heads=2)
    model.eval()
    batch = tensorize_observation(
        observation=_bid_observation(), device=device
    )
    encoding = model.encode_action_value_observations(batch)
    source_rows = torch.tensor(
        (0, 0, 0),
        dtype=torch.long,
        device=device,
    )
    choices = torch.tensor(
        (
            (CARD_CHOICE_BASE_ID, PASS_CHOICE_ID, PASS_CHOICE_ID),
            (CARD_CHOICE_BASE_ID, PASS_CHOICE_ID, CARD_CHOICE_BASE_ID),
            (
                CARD_CHOICE_BASE_ID,
                CARD_CHOICE_BASE_ID,
                PASS_CHOICE_ID,
            ),
        ),
        dtype=torch.long,
        device=device,
    )
    values = model.action_values(
        encoding,
        source_rows=source_rows,
        choice_ids_padded=choices,
        step_counts=torch.tensor(
            (2, 2, 2),
            dtype=torch.long,
            device=device,
        ),
    )

    torch.testing.assert_close(values[0], values[1])
    assert not torch.allclose(values[0], values[2])


def test_live_query_seed_matches_teacher_forced_scoring() -> None:
    device = torch.device("cpu")
    model = PolicyActionModel(d_model=16, layers=1, heads=2)
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
        encoding = model.encode_policy_observations(batch)
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
