"""Black-box tests for reusable observation encoding."""

import torch

from server.game.players.test_helpers import card, make_snapshot
from server.training.observation import build_observation
from server.training.observation_memory import ObservationMemoryView
from server.training.tensorize import tensorize_observation

from .observation_encoder import ObservationEncoder


def test_encoder_returns_query_and_all_card_candidates() -> None:
    encoder = ObservationEncoder(d_model=16, layers=1, heads=2)
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

    encoded = encoder(
        tensorize_observation(
            observation=observation,
            device=torch.device("cpu"),
        )
    )

    assert encoded.batch_size == 1
    assert encoded.device == torch.device("cpu")
    assert encoded.value_context().shape == (1, 16)
    assert (
        encoded.action_decoder_inputs().card_choice_embeddings.shape
        == (
            1,
            108,
            16,
        )
    )
