"""Black-box tests for the production decision runtime."""

from __future__ import annotations

import asyncio
import time

import torch

from server.foundation.result import Ok
from server.game import Seat
from server.game.snapshots import PlayerSnapshot
from server.policy_model._schema.actions import PASS_CHOICE_ID
from server.policy_model.actions.legality import (
    build_legal_action_space,
)
from server.policy_model.inference import (
    ActionDecisionRequest,
    PolicyQuery,
    SamplingSeed,
)
from server.policy_model.inference.runtime import InferenceRuntime
from server.policy_model.network import (
    EncodedObservation,
    PolicyActionModel,
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


class _CountingModel(PolicyActionModel):
    policy_row_counts: list[int]
    action_value_row_counts: list[int]
    action_counts: list[int]

    def __init__(self) -> None:
        super().__init__(
            d_model=16,
            layers=1,
            heads=2,
            action_value_layers=1,
        )
        self.policy_row_counts = []
        self.action_value_row_counts = []
        self.action_counts = []

    def encode_policy_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> EncodedObservation:
        self.policy_row_counts.append(
            int(observation.category_ids.shape[0])
        )
        return super().encode_policy_observations(observation)

    def encode_action_value_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> EncodedObservation:
        self.action_value_row_counts.append(
            int(observation.category_ids.shape[0])
        )
        return super().encode_action_value_observations(observation)

    def action_values(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: torch.Tensor,
        choice_ids_padded: torch.Tensor,
        step_counts: torch.Tensor,
    ) -> torch.Tensor:
        self.action_counts.append(int(step_counts.shape[0]))
        return super().action_values(
            encoding,
            source_rows=source_rows,
            choice_ids_padded=choice_ids_padded,
            step_counts=step_counts,
        )


class _SlowModel(_CountingModel):
    delay_next_encode: bool

    def __init__(self) -> None:
        super().__init__()
        self.delay_next_encode = True

    def encode_policy_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> EncodedObservation:
        if self.delay_next_encode:
            self.delay_next_encode = False
            time.sleep(0.05)
        return super().encode_policy_observations(observation)


class _BrokenModel(_CountingModel):
    def encode_policy_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> EncodedObservation:
        del observation
        raise AssertionError("broken inference model")


class _PreferRevealModel(_CountingModel):
    def action_values(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: torch.Tensor,
        choice_ids_padded: torch.Tensor,
        step_counts: torch.Tensor,
    ) -> torch.Tensor:
        del encoding, source_rows, step_counts
        return (
            choice_ids_padded[:, 0]
            .ne(PASS_CHOICE_ID)
            .to(dtype=torch.float32)
            .mul(1000.0)
        )


async def test_decision_batches_policy_and_action_value_work() -> None:
    model = _CountingModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)

    result = await runtime.decide(
        requests=(_request(query=query, seed=42),)
    )
    await runtime.close()

    assert isinstance(result, Ok)
    assert len(result.value) == 1
    decision = result.value[0]
    decoded = query.legal_actions.decode(decision.action.trace)
    assert isinstance(decoded, Ok)
    assert decoded.value == decision.action
    assert decision.candidate_count == 2
    assert decision.selected_action_value is not None
    assert model.policy_row_counts == [1]
    assert model.action_value_row_counts == [1]
    assert model.action_counts == [2]


async def test_same_seed_produces_same_decision() -> None:
    model = _CountingModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    request = _request(
        query=_play_query(),
        seed=73,
        candidate_count=8,
    )

    first = await runtime.decide(requests=(request,))
    second = await runtime.decide(requests=(request,))
    await runtime.close()

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value == second.value
    assert 1 < first.value[0].candidate_count <= 8


async def test_action_value_can_improve_over_policy_perturbation() -> (
    None
):
    runtime = InferenceRuntime.create(
        model=_PreferRevealModel(),
        device=torch.device("cpu"),
    )

    result = await runtime.decide(
        requests=(
            _request(
                query=_bid_query(reveal_is_legal=True),
                seed=81,
            ),
        )
    )
    await runtime.close()

    assert isinstance(result, Ok)
    assert not result.value[0].action.is_pass
    assert result.value[0].selected_action_value == 1000.0


async def test_forced_decision_skips_both_model_branches() -> None:
    model = _CountingModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )

    result = await runtime.decide(
        requests=(
            _request(
                query=_bid_query(reveal_is_legal=False),
                seed=44,
            ),
        )
    )
    await runtime.close()

    assert isinstance(result, Ok)
    assert result.value[0].candidate_count == 1
    assert result.value[0].selected_action_value is None
    assert model.policy_row_counts == []
    assert model.action_value_row_counts == []
    assert model.action_counts == []


async def test_concurrent_decisions_share_observation_batches() -> None:
    model = _CountingModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)

    first, second = await asyncio.gather(
        runtime.decide(requests=(_request(query=query, seed=91),)),
        runtime.decide(requests=(_request(query=query, seed=92),)),
    )
    await runtime.close()

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert model.policy_row_counts == [1]
    assert model.action_value_row_counts == [1]
    assert model.action_counts == [4]


async def test_programming_failure_reaches_every_waiter() -> None:
    runtime = InferenceRuntime.create(
        model=_BrokenModel(),
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)

    outcomes = await asyncio.gather(
        runtime.decide(requests=(_request(query=query, seed=1),)),
        runtime.decide(requests=(_request(query=query, seed=2),)),
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


async def test_cancelled_caller_does_not_break_runtime() -> None:
    model = _SlowModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)
    pending = asyncio.create_task(
        runtime.decide(requests=(_request(query=query, seed=7),))
    )
    await asyncio.sleep(0)
    pending.cancel()

    cancellation = await asyncio.gather(
        pending,
        return_exceptions=True,
    )
    resumed = await runtime.decide(
        requests=(_request(query=query, seed=8),)
    )
    await runtime.close()

    assert isinstance(cancellation[0], asyncio.CancelledError)
    assert isinstance(resumed, Ok)


def _request(
    *,
    query: PolicyQuery,
    seed: int,
    candidate_count: int = 8,
) -> ActionDecisionRequest:
    return ActionDecisionRequest(
        query=query,
        candidate_count=candidate_count,
        action_value_temperature=1.0,
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
            bid_actions=(),
            completed_tricks=(),
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
            bid_actions=(),
            completed_tricks=(),
        ),
    )


def _bid_snapshot(*, reveal_is_legal: bool) -> PlayerSnapshot:
    return make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[
            card(
                "hearts",
                "2" if reveal_is_legal else "3",
            )
        ],
        remaining_cards=seat_values(1, 0, 0, 0),
        trump_rank="2",
    )
