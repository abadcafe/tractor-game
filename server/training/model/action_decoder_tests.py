"""Black-box tests for teacher-forced and cached action decoding."""

import torch

from server.game.players.test_helpers import card, make_snapshot
from server.training.observation import build_observation
from server.training.observation_memory import ObservationMemoryView
from server.training.tensorize import tensorize_observation

from .action_decoder import ActionDecoder
from .observation_encoder import EncodedObservation, ObservationEncoder


def test_cached_decode_exactly_matches_causal_teacher_forcing() -> None:
    decoder = ActionDecoder(d_model=8, heads=1)
    decoder.eval()
    encoding = _encoding()
    choices = torch.tensor(((2, 7, 1),), dtype=torch.long)
    step_counts = torch.tensor((3,), dtype=torch.long)

    with torch.no_grad():
        teacher = decoder.score_action_traces(
            encoding,
            choice_ids_padded=choices,
            step_counts=step_counts,
        ).choice_logits
        session = decoder.begin_decode_session(encoding, max_steps=3)
        live_steps = [session.next_choice_logits()]
        session.advance(choices[:, 0])
        live_steps.append(session.next_choice_logits())
        session.advance(choices[:, 1])
        live_steps.append(session.next_choice_logits())

    torch.testing.assert_close(
        torch.stack(live_steps, dim=1),
        teacher,
    )


def test_teacher_forcing_cannot_see_future_choices() -> None:
    decoder = ActionDecoder(d_model=8, heads=1)
    decoder.eval()
    encoding = _encoding()
    first = torch.tensor(((2, 7, 1),), dtype=torch.long)
    changed_future = torch.tensor(((2, 31, 44),), dtype=torch.long)
    step_counts = torch.tensor((3,), dtype=torch.long)

    with torch.no_grad():
        first_scores = decoder.score_action_traces(
            encoding,
            choice_ids_padded=first,
            step_counts=step_counts,
        ).choice_logits
        changed_scores = decoder.score_action_traces(
            encoding,
            choice_ids_padded=changed_future,
            step_counts=step_counts,
        ).choice_logits

    torch.testing.assert_close(
        first_scores[:, :2],
        changed_scores[:, :2],
    )


def test_action_decoder_parameters_receive_finite_gradients() -> None:
    decoder = ActionDecoder(d_model=8, heads=1)
    scores = decoder.score_action_traces(
        _encoding(),
        choice_ids_padded=torch.tensor(((2, 7, 1),), dtype=torch.long),
        step_counts=torch.tensor((3,), dtype=torch.long),
    )

    torch.autograd.backward(scores.choice_logits.square().mean())

    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in decoder.parameters()
    )


def _encoding() -> EncodedObservation:
    observation = build_observation(
        viewer=0,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            player_hand=[card("hearts", "2")],
            player_hand_counts=[1, 0, 0, 0],
            trump_rank="2",
        ),
        memory=ObservationMemoryView(
            bid_actions=(),
            completed_tricks=(),
        ),
    )
    return ObservationEncoder(d_model=8, layers=1, heads=1)(
        tensorize_observation(
            observation=observation,
            device=torch.device("cpu"),
        )
    )
