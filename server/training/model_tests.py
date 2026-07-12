"""Tests for torch policy model forward pass."""

from __future__ import annotations

import torch

from server.game.players.test_helpers import card, make_snapshot
from server.training.model import TractorPolicyModel
from server.training.observation import build_observation
from server.training.semantic_actions import SemanticArgument
from server.training.semantic_actions.codec import semantic_argument_id
from server.training.tensorize import tensorize_observation


def test_tractor_policy_model_scores_trace_shapes() -> None:
    device = torch.device("cpu")
    model = TractorPolicyModel(
        d_model=8,
        layers=1,
        heads=2,
    )
    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action="play",
            player_hand=[card("spades", "A", 1)],
        ),
        history=(),
    )
    observation_batch = tensorize_observation(
        observation=observation,
        max_observation_tokens=64,
        device=device,
    )

    encoding = model.encode_observations(observation_batch)
    scores = model.score_argument_traces(
        encoding,
        selected_token_ids_padded=torch.zeros(
            (1, 2), dtype=torch.long, device=device
        ),
        step_counts=torch.tensor((2,), dtype=torch.long, device=device),
    )
    values = model.value_estimates(encoding)

    assert scores.argument_logits.shape[:2] == (1, 2)
    assert values.shape == (1,)


def test_live_argument_cache_matches_teacher_forced_trace_scores() -> (
    None
):
    device = torch.device("cpu")
    model = TractorPolicyModel(
        d_model=8,
        layers=1,
        heads=2,
    )
    model.eval()
    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            player_hand=[card("hearts", "2", 1)],
            trump_rank="2",
        ),
        history=(),
    )
    observation_batch = tensorize_observation(
        observation=observation,
        max_observation_tokens=64,
        device=device,
    )
    selected_ids = torch.tensor(
        (
            (
                semantic_argument_id(SemanticArgument("pass")),
                semantic_argument_id(SemanticArgument("stop")),
            ),
        ),
        dtype=torch.long,
        device=device,
    )
    step_counts = torch.tensor((2,), dtype=torch.long, device=device)

    with torch.no_grad():
        encoding = model.encode_observations(observation_batch)
        trace_scores = model.score_argument_traces(
            encoding,
            selected_token_ids_padded=selected_ids,
            step_counts=step_counts,
        )
        session = model.begin_argument_decode_session(
            encoding, max_steps=2
        )
        first_logits = session.next_logits()
        session.advance(selected_ids[:, 0])
        second_logits = session.next_logits()

    torch.testing.assert_close(
        first_logits, trace_scores.argument_logits[:, 0, :]
    )
    torch.testing.assert_close(
        second_logits, trace_scores.argument_logits[:, 1, :]
    )
