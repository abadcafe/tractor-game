"""Black-box tests for the strict typed policy request wire format."""

from __future__ import annotations

import torch

from server.foundation.result import Ok, Rejected
from server.game.players.test_helpers import card, make_snapshot
from server.game.rules.cards import Card
from server.training.legal_actions import build_legal_action_index
from server.training.observation import Observation, build_observation
from server.training.observation_memory import ObservationMemoryView
from server.training.policy_inference_batch import (
    PolicyRequestCompiler,
    PolicyRequestInput,
    PolicyRequestRoute,
    materialize_borrowed_policy_request_batch,
)
from server.training.policy_inference_batch.device import (
    materialize_policy_request_batch_frame,
)
from server.training.policy_inference_batch.schema import I64
from server.training.policy_inference_batch.types import (
    PolicyRequestWireFrame,
)
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_actions.choices import CARD_CHOICE_COUNT
from server.training.tokenization.encoding_schema import CATEGORY_COUNT


def test_request_round_trip_preserves_typed_observation_columns() -> (
    None
):
    hand = [card("spades", "A", 1), card("spades", "A", 2)]
    observation = _observation(hand)
    compiled = PolicyRequestCompiler(batch_capacity=2).compile_batch(
        (_request(observation, hand=hand, request_id=9),)
    )
    assert isinstance(compiled, Ok)

    materialized = materialize_borrowed_policy_request_batch(
        batch=compiled.value,
        device=torch.device("cpu"),
    )

    assert isinstance(materialized, Ok)
    batch = materialized.value
    token_count = len(observation.tokens)
    assert batch.observation_batch.category_ids.shape == (
        1,
        token_count,
        CATEGORY_COUNT,
    )
    assert batch.observation_batch.candidate_counts.shape == (
        1,
        CARD_CHOICE_COUNT,
    )
    counts = batch.observation_batch.candidate_counts[0]
    assert {
        float(counts[index].item())
        for index in range(int(counts.shape[0]))
    } == {1.0, 2.0}
    assert batch.policy_versions == (4,)
    assert batch.action_plan_batch.batch_size() == 1


def test_request_uses_batch_local_padding_without_limit() -> None:
    short = _observation([card("spades", "A", 1)])
    long = _observation(
        [
            card("spades", "A", 1),
            card("hearts", "K", 1),
            card("clubs", "3", 1),
        ]
    )
    compiled = PolicyRequestCompiler(batch_capacity=2).compile_batch(
        (
            _request(
                short,
                hand=[card("spades", "A", 1)],
                request_id=1,
            ),
            _request(
                long,
                hand=[
                    card("spades", "A", 1),
                    card("hearts", "K", 1),
                    card("clubs", "3", 1),
                ],
                request_id=2,
            ),
        )
    )

    assert isinstance(compiled, Ok)
    assert compiled.value.observation_token_capacity == len(long.tokens)
    assert compiled.value.row_count() == 2
    assert compiled.value.routes == (
        PolicyRequestRoute(worker_index=3, request_id=1),
        PolicyRequestRoute(worker_index=3, request_id=2),
    )


def test_request_wire_rejects_previous_schema_magic() -> None:
    hand = [card("spades", "A", 1)]
    observation = _observation(hand)
    compiled = PolicyRequestCompiler(batch_capacity=1).compile_batch(
        (_request(observation, hand=hand, request_id=1),)
    )
    assert isinstance(compiled, Ok)
    stale = bytearray(compiled.value.frame.view())
    I64.pack_into(stale, 0, 0x5452504F4C495144)

    result = materialize_policy_request_batch_frame(
        frame=PolicyRequestWireFrame(
            buffer=stale, byte_count=len(stale)
        ),
        device=torch.device("cpu"),
    )

    assert isinstance(result, Rejected)
    assert result.reason == "policy request frame schema is invalid"


def _observation(hand: list[Card]) -> Observation:
    return build_observation(
        viewer=0,
        snapshot=make_snapshot(player_hand=hand),
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )


def _request(
    observation: Observation, *, hand: list[Card], request_id: int
) -> PolicyRequestInput:
    return PolicyRequestInput(
        route=PolicyRequestRoute(worker_index=3, request_id=request_id),
        observation=observation,
        legal_actions=build_legal_action_index(
            player_index=0,
            snapshot=make_snapshot(player_hand=hand),
            query=observation.action_query,
        ),
        decision_key=PolicyDecisionKey(
            base_seed=11,
            policy_version=4,
            rollout_id="request-wire-test",
            episode_id=2,
            player_index=0,
            decision_index=request_id,
        ),
    )
