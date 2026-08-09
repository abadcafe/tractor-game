"""Black-box tests for reusable observation encoding."""

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
    tensorize_observations,
)
from tests.support import card, seat_values
from tests.support import snapshot as make_snapshot

from .observation_backbone import ObservationBackbone


def _available_test_devices() -> tuple[torch.device, ...]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps:0"))
    return tuple(devices)


def test_encoder_returns_query_and_all_card_candidates() -> None:
    encoder = ObservationBackbone(d_model=16, layers=1, heads=2)
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

    encoded = encoder.forward(
        tensorize_observation(
            observation=observation,
            device=torch.device("cpu"),
        )
    )

    assert encoded.batch_size == 1
    assert encoded.device == torch.device("cpu")
    inputs = encoded.action_decoder_inputs()
    assert inputs.observation_context.shape == (1, 16)
    assert inputs.card_choice_embeddings.shape == (
        1,
        108,
        16,
    )


@pytest.mark.parametrize("device", _available_test_devices())
def test_encoder_never_reads_device_scalars(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    batch = tensorize_observations(
        observations=(observation, observation),
        device=device,
    )
    encoder = ObservationBackbone(d_model=16, layers=1, heads=2)
    _ = encoder.to(device)
    monkeypatch.setattr(Tensor, "item", _reject_tensor_item)

    encoded = encoder.forward(batch)

    assert encoded.batch_size == 2
    assert (
        encoded.action_decoder_inputs().card_choice_embeddings.shape
        == (
            2,
            108,
            16,
        )
    )


def _reject_tensor_item(_tensor: Tensor) -> NoReturn:
    raise AssertionError("device scalar read is forbidden")
