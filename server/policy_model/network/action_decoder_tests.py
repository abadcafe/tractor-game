"""Black-box tests for teacher-forced and cached action decoding."""

from typing import NoReturn

import pytest
import torch
from torch import Tensor

from server.game import Seat
from server.policy_model.observation import build_observation
from server.policy_model.observation.history import (
    ObservationMemoryView,
)
from server.policy_model.observation.tensor_batch import (
    tensorize_observation,
)
from tests.support import card, seat_values
from tests.support import snapshot as make_snapshot

from .action_decoder import ActionDecoder
from .observation_backbone import (
    EncodedObservation,
    ObservationBackbone,
)


def _available_test_devices() -> tuple[torch.device, ...]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps:0"))
    return tuple(devices)


def test_cached_decode_exactly_matches_causal_teacher_forcing() -> None:
    decoder = ActionDecoder(d_model=8, heads=1)
    _ = decoder.eval()
    encoding = _encoding()
    choices = torch.tensor(((2, 7, 1),), dtype=torch.long)
    step_counts = torch.tensor((3,), dtype=torch.long)

    with torch.no_grad():
        teacher = decoder.score_action_traces(
            encoding,
            source_rows=torch.tensor((0,), dtype=torch.long),
            choice_ids_padded=choices,
            step_counts=step_counts,
        ).choice_logits
        session = decoder.begin_decode_session(
            encoding,
            source_rows=torch.tensor((0,), dtype=torch.long),
            max_steps=3,
        )
        active = torch.tensor((True,), dtype=torch.bool)
        live_steps = [session.next_choice_logits(active, active)]
        session.advance(choices[:, 0], active)
        live_steps.append(session.next_choice_logits(active, active))
        session.advance(choices[:, 1], active)
        live_steps.append(session.next_choice_logits(active, active))

    torch.testing.assert_close(
        torch.stack(live_steps, dim=1),
        teacher,
    )


@pytest.mark.parametrize("device", _available_test_devices())
def test_cached_decode_never_reads_device_scalars(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = ActionDecoder(d_model=8, heads=1)
    _ = decoder.to(device)
    _ = decoder.eval()
    encoding = _encoding(device=device)
    source_rows = torch.tensor((0, 0), dtype=torch.long, device=device)
    active = torch.tensor(
        (True, False), dtype=torch.bool, device=device
    )
    scored = torch.tensor(
        (True, False), dtype=torch.bool, device=device
    )
    choices = torch.tensor((2, 0), dtype=torch.long, device=device)
    monkeypatch.setattr(Tensor, "item", _reject_tensor_item)

    with torch.no_grad():
        session = decoder.begin_decode_session(
            encoding,
            source_rows=source_rows,
            max_steps=2,
        )
        first = session.next_choice_logits(active, scored)
        session.advance(choices, active)
        second = session.next_choice_logits(active, scored)

    assert first.shape == (2, 110)
    assert second.shape == first.shape


def test_teacher_forcing_cannot_see_future_choices() -> None:
    decoder = ActionDecoder(d_model=8, heads=1)
    _ = decoder.eval()
    encoding = _encoding()
    first = torch.tensor(((2, 7, 1),), dtype=torch.long)
    changed_future = torch.tensor(((2, 31, 44),), dtype=torch.long)
    step_counts = torch.tensor((3,), dtype=torch.long)

    with torch.no_grad():
        first_scores = decoder.score_action_traces(
            encoding,
            source_rows=torch.tensor((0,), dtype=torch.long),
            choice_ids_padded=first,
            step_counts=step_counts,
        ).choice_logits
        changed_scores = decoder.score_action_traces(
            encoding,
            source_rows=torch.tensor((0,), dtype=torch.long),
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
        source_rows=torch.tensor((0,), dtype=torch.long),
        choice_ids_padded=torch.tensor(((2, 7, 1),), dtype=torch.long),
        step_counts=torch.tensor((3,), dtype=torch.long),
    )

    torch.autograd.backward(scores.choice_logits.square().mean())

    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in decoder.parameters()
    )


def _encoding(
    *, device: torch.device = torch.device("cpu")
) -> EncodedObservation:
    observation = build_observation(
        viewer=Seat.A,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            hand=[card("hearts", "2")],
            remaining_cards=seat_values(1, 0, 0, 0),
            trump_rank="2",
        ),
        memory=ObservationMemoryView(
            bid_actions=(),
            completed_tricks=(),
        ),
    )
    backbone = ObservationBackbone(d_model=8, layers=1, heads=1)
    _ = backbone.to(device)
    return backbone.forward(
        tensorize_observation(
            observation=observation,
            device=device,
        )
    )


def _reject_tensor_item(_tensor: Tensor) -> NoReturn:
    raise AssertionError("device scalar read is forbidden")
