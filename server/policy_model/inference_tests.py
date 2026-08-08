"""Black-box tests for the production decision runtime."""

from __future__ import annotations

import asyncio
import time
from typing import override

import torch

from server.foundation.result import Ok
from server.game import Seat
from server.game.snapshots import PlayerSnapshot
from server.policy_model.actions.legality import (
    build_legal_action_space,
)
from server.policy_model.inference import (
    PolicyDecisionRequest,
    PolicyQuery,
    SamplingSeed,
)
from server.policy_model.inference.runtime import InferenceRuntime
from server.policy_model.network import (
    EncodedObservation,
    ModelConfig,
    PolicyModel,
)
from server.policy_model.observation import (
    Observation,
    build_observation,
)
from server.policy_model.observation.history import (
    ObservationMemoryView,
)
from server.policy_model.observation.tensor_batch import (
    ObservationTensorBatch,
)
from tests.support import card, seat_values
from tests.support import snapshot as make_snapshot


class _CountingModel(PolicyModel):
    encoded_row_counts: list[int]

    def __init__(self) -> None:
        super().__init__(
            config=ModelConfig(d_model=16, layers=1, heads=2)
        )
        self.encoded_row_counts = []

    @override
    def encode_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> EncodedObservation:
        self.encoded_row_counts.append(
            int(observation.category_ids.shape[0])
        )
        return super().encode_observations(observation)


class _SlowModel(_CountingModel):
    delay_next_encode: bool

    def __init__(self) -> None:
        super().__init__()
        self.delay_next_encode = True

    @override
    def encode_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> EncodedObservation:
        if self.delay_next_encode:
            self.delay_next_encode = False
            time.sleep(0.05)
        return super().encode_observations(observation)


class _BrokenModel(_CountingModel):
    @override
    def encode_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> EncodedObservation:
        del observation
        raise AssertionError("broken inference model")


async def test_decide_returns_one_legal_policy_action() -> None:
    model = _CountingModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)

    result = await runtime.decide(
        request=_request(query=query, seed=42)
    )
    await runtime.close()

    assert isinstance(result, Ok)
    decoded = query.legal_actions.decode(result.value.trace)
    assert isinstance(decoded, Ok)
    assert decoded.value == result.value
    assert model.encoded_row_counts == [1]


async def test_decide_same_seed_produces_same_action() -> None:
    runtime = InferenceRuntime.create(
        model=_CountingModel(),
        device=torch.device("cpu"),
    )
    request = _request(query=_play_query(), seed=73)

    first = await runtime.decide(request=request)
    second = await runtime.decide(request=request)
    await runtime.close()

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value == second.value


async def test_decide_forced_action_skips_model() -> None:
    model = _CountingModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )

    result = await runtime.decide(
        request=_request(
            query=_bid_query(reveal_is_legal=False),
            seed=44,
        )
    )
    await runtime.close()

    assert isinstance(result, Ok)
    assert result.value.is_pass
    assert model.encoded_row_counts == []


async def test_decide_concurrent_calls_share_observation_encoding() -> (
    None
):
    model = _CountingModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)

    first, second = await asyncio.gather(
        runtime.decide(request=_request(query=query, seed=91)),
        runtime.decide(request=_request(query=query, seed=92)),
    )
    await runtime.close()

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert model.encoded_row_counts == [1]


async def test_decide_is_independent_of_concurrent_request_order() -> (
    None
):
    query = _play_query()
    first_request = _request(query=query, seed=19)
    second_request = _request(query=query, seed=23)
    first_model = _CountingModel()
    second_model = _CountingModel()
    _ = second_model.load_state_dict(first_model.state_dict())
    runtime = InferenceRuntime.create(
        model=first_model, device=torch.device("cpu")
    )
    first_order = await asyncio.gather(
        runtime.decide(request=first_request),
        runtime.decide(request=second_request),
    )
    await runtime.close()
    reversed_runtime = InferenceRuntime.create(
        model=second_model, device=torch.device("cpu")
    )
    second_order = await asyncio.gather(
        reversed_runtime.decide(request=second_request),
        reversed_runtime.decide(request=first_request),
    )
    await reversed_runtime.close()

    assert isinstance(first_order[0], Ok)
    assert isinstance(first_order[1], Ok)
    assert isinstance(second_order[0], Ok)
    assert isinstance(second_order[1], Ok)
    assert first_order[0].value == second_order[1].value
    assert first_order[1].value == second_order[0].value


async def test_decide_programming_failure_reaches_every_waiter() -> (
    None
):
    runtime = InferenceRuntime.create(
        model=_BrokenModel(),
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)

    outcomes = await asyncio.gather(
        runtime.decide(request=_request(query=query, seed=1)),
        runtime.decide(request=_request(query=query, seed=2)),
        return_exceptions=True,
    )
    closing = await asyncio.gather(
        runtime.close(),
        return_exceptions=True,
    )

    assert all(
        isinstance(outcome, AssertionError) for outcome in outcomes
    )
    assert isinstance(closing[0], AssertionError)


async def test_decide_cancelled_caller_does_not_break_runtime() -> None:
    runtime = InferenceRuntime.create(
        model=_SlowModel(),
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)
    pending = asyncio.create_task(
        runtime.decide(request=_request(query=query, seed=7))
    )
    await asyncio.sleep(0)
    _ = pending.cancel()

    cancellation = await asyncio.gather(
        pending,
        return_exceptions=True,
    )
    resumed = await runtime.decide(
        request=_request(query=query, seed=8)
    )
    await runtime.close()

    assert isinstance(cancellation[0], asyncio.CancelledError)
    assert isinstance(resumed, Ok)


def _request(*, query: PolicyQuery, seed: int) -> PolicyDecisionRequest:
    return PolicyDecisionRequest(
        query=query,
        draw=SamplingSeed(seed=seed, ordinal=0),
    )


def _bid_query(*, reveal_is_legal: bool) -> PolicyQuery:
    observation = _bid_observation(reveal_is_legal=reveal_is_legal)
    snapshot = _bid_snapshot(reveal_is_legal=reveal_is_legal)
    return PolicyQuery(
        observation=observation,
        legal_actions=build_legal_action_space(
            viewer=Seat.A,
            snapshot=snapshot,
            query=observation.action_query,
        ),
    )


def _play_query() -> PolicyQuery:
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        hand=[
            card("hearts", "3"),
            card("hearts", "4"),
            card("spades", "5"),
            card("clubs", "6"),
        ],
        trump_rank="2",
    )
    observation = build_observation(
        viewer=Seat.A,
        snapshot=snapshot,
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )
    return PolicyQuery(
        observation=observation,
        legal_actions=build_legal_action_space(
            viewer=Seat.A,
            snapshot=snapshot,
            query=observation.action_query,
        ),
    )


def _bid_observation(*, reveal_is_legal: bool) -> Observation:
    return build_observation(
        viewer=Seat.A,
        snapshot=_bid_snapshot(reveal_is_legal=reveal_is_legal),
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )


def _bid_snapshot(*, reveal_is_legal: bool) -> PlayerSnapshot:
    return make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=(
            [card("hearts", "2")]
            if reveal_is_legal
            else [card("spades", "3")]
        ),
        remaining_cards=seat_values(1, 0, 0, 0),
        trump_rank="2",
    )
