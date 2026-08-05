"""Black-box tests for reusable observation encoding."""

import torch

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

from .observation_encoder import ObservationEncoder


def test_encoder_returns_query_and_all_card_candidates() -> None:
    encoder = ObservationEncoder(d_model=16, layers=1, heads=2)
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
