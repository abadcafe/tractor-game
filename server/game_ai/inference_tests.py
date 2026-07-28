"""Black-box tests for the asynchronous inference runtime."""

from __future__ import annotations

import asyncio
import math
import time

import torch

from server.foundation.result import Ok
from server.game import Seat
from server.game.snapshots import PlayerSnapshot
from server.game_ai import (
    ActionProposalRequest,
    InferenceDrawKey,
    ModelQuery,
    PolicySampleRequest,
    PolicyScoreRequest,
)
from server.game_ai.inference import InferenceRuntime
from server.training.legal_actions import build_legal_action_index
from server.training.model import (
    ActionDecodeSession,
    EncodedObservation,
    TractorPolicyModel,
)
from server.training.observation import Observation, build_observation
from server.training.observation_memory import ObservationMemoryView
from server.training.semantic_action_plan import action_trace_choice_ids
from server.training.tensorize import ObservationTensorBatch
from tests.support import card, seat_values
from tests.support import snapshot as make_snapshot


class _CountingPolicyModel(TractorPolicyModel):
    encoded_row_counts: list[int]
    decode_root_counts: list[int]
    value_call_count: int

    def __init__(self) -> None:
        super().__init__(d_model=16, layers=1, heads=2)
        self.encoded_row_counts = []
        self.decode_root_counts = []
        self.value_call_count = 0

    def encode_observations(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation:
        self.encoded_row_counts.append(
            int(observation.category_ids.shape[0])
        )
        return super().encode_observations(observation)

    def begin_action_decode_session(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: torch.Tensor,
        max_steps: int,
    ) -> ActionDecodeSession:
        session = super().begin_action_decode_session(
            encoding,
            source_rows=source_rows,
            max_steps=max_steps,
        )
        self.decode_root_counts.append(session.unique_prefix_count)
        return session

    def value_estimates(
        self, encoding: EncodedObservation
    ) -> torch.Tensor:
        self.value_call_count += 1
        return super().value_estimates(encoding)


class _SlowPolicyModel(_CountingPolicyModel):
    delay_next_encode: bool

    def __init__(self) -> None:
        super().__init__()
        self.delay_next_encode = True

    def encode_observations(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation:
        if self.delay_next_encode:
            self.delay_next_encode = False
            time.sleep(0.05)
        return super().encode_observations(observation)


class _BrokenPolicyModel(_CountingPolicyModel):
    def encode_observations(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation:
        del observation
        raise AssertionError("broken inference model")


async def test_sample_draws_share_encoder_and_root_prefix() -> None:
    model = _CountingPolicyModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)

    result = await runtime.sample(
        requests=(
            PolicySampleRequest(
                query=query,
                draws=_draws(seed=42, count=8),
            ),
        )
    )
    await runtime.close()

    assert isinstance(result, Ok)
    assert len(result.value) == 1
    assert len(result.value[0].samples) == 8
    assert model.encoded_row_counts == [1]
    assert model.decode_root_counts == [1]
    assert model.value_call_count == 0


async def test_structured_proposals_are_unique_and_reproducible() -> (
    None
):
    model = _CountingPolicyModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)
    request = ActionProposalRequest(
        query=query,
        candidate_count=8,
        draw=InferenceDrawKey(seed=43, ordinal=0),
    )
    first = await runtime.propose(requests=(request,))
    second = await runtime.propose(requests=(request,))
    await runtime.close()

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    first_candidates = first.value[0].candidates
    second_candidates = second.value[0].candidates
    assert len(first_candidates) == 2
    assert tuple(item.action for item in first_candidates) == tuple(
        item.action for item in second_candidates
    )
    assert (
        len(
            {
                tuple(
                    choice.kind for choice in item.action.trace.choices
                )
                for item in first_candidates
            }
        )
        == 2
    )
    assert model.encoded_row_counts == [1, 1]
    assert model.decode_root_counts == [1, 1]
    assert model.value_call_count == 0


async def test_structured_play_proposals_are_unique() -> None:
    model = _CountingPolicyModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    result = await runtime.propose(
        requests=(
            ActionProposalRequest(
                query=_play_query(),
                candidate_count=16,
                draw=InferenceDrawKey(seed=45, ordinal=0),
            ),
        )
    )
    await runtime.close()

    assert isinstance(result, Ok)
    candidates = result.value[0].candidates
    trace_ids = tuple(
        action_trace_choice_ids(candidate.action.trace)
        for candidate in candidates
    )
    assert 2 < len(candidates) <= 16
    assert len(set(trace_ids)) == len(trace_ids)
    assert model.encoded_row_counts == [1]


async def test_forced_proposal_skips_encoder() -> None:
    model = _CountingPolicyModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    result = await runtime.propose(
        requests=(
            ActionProposalRequest(
                query=_bid_query(reveal_is_legal=False),
                candidate_count=8,
                draw=InferenceDrawKey(seed=44, ordinal=0),
            ),
        )
    )
    await runtime.close()

    assert isinstance(result, Ok)
    assert len(result.value) == 1
    assert len(result.value[0].candidates) == 1
    assert model.encoded_row_counts == []


async def test_batch_reordering_preserves_draw_results() -> None:
    model = _CountingPolicyModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)
    draws = _draws(seed=91, count=8)

    grouped = await runtime.sample(
        requests=(PolicySampleRequest(query=query, draws=draws),)
    )
    individual = await runtime.sample(
        requests=tuple(
            PolicySampleRequest(query=query, draws=(draw,))
            for draw in draws
        )
    )
    await runtime.close()

    assert isinstance(grouped, Ok)
    assert isinstance(individual, Ok)
    grouped_samples = grouped.value[0].samples
    individual_samples = tuple(
        group.samples[0] for group in individual.value
    )
    assert tuple(item.action for item in grouped_samples) == tuple(
        item.action for item in individual_samples
    )
    for grouped_item, individual_item in zip(
        grouped_samples,
        individual_samples,
        strict=True,
    ):
        assert math.isclose(
            grouped_item.log_probability,
            individual_item.log_probability,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )


async def test_forced_action_skips_encoder_and_value_head() -> None:
    model = _CountingPolicyModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )

    sampled = await runtime.sample(
        requests=(
            PolicySampleRequest(
                query=_bid_query(reveal_is_legal=False),
                draws=_draws(seed=17, count=8),
            ),
        )
    )
    await runtime.close()

    assert isinstance(sampled, Ok)
    assert all(
        item.log_probability == 0.0 for item in sampled.value[0].samples
    )
    assert model.encoded_row_counts == []
    assert model.value_call_count == 0


async def test_teacher_forced_score_matches_live_sample() -> None:
    model = _CountingPolicyModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    query = _bid_query(reveal_is_legal=True)
    sampled = await runtime.sample(
        requests=(
            PolicySampleRequest(
                query=query,
                draws=(InferenceDrawKey(seed=23, ordinal=0),),
            ),
        )
    )
    assert isinstance(sampled, Ok)

    scored = await runtime.score(
        requests=(
            PolicyScoreRequest(
                query=query,
                action=sampled.value[0].samples[0].action,
            ),
        )
    )
    await runtime.close()

    assert isinstance(scored, Ok)
    assert math.isclose(
        scored.value[0],
        sampled.value[0].samples[0].log_probability,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    assert model.value_call_count == 0


async def test_concurrent_values_share_one_runtime_batch() -> None:
    model = _CountingPolicyModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    observation = _bid_observation(reveal_is_legal=True)

    first, second = await asyncio.gather(
        runtime.values(observations=(observation,)),
        runtime.values(observations=(observation,)),
    )
    await runtime.close()

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert model.encoded_row_counts == [1]
    assert model.value_call_count == 1


async def test_cpu_inference_splits_rows_at_throughput_knee() -> None:
    model = _CountingPolicyModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    base = _bid_snapshot(reveal_is_legal=True)
    observations = tuple(
        build_observation(
            viewer=Seat.A,
            snapshot=base.model_copy(update={"defender_points": index}),
            memory=ObservationMemoryView(
                bid_actions=(),
                completed_tricks=(),
            ),
        )
        for index in range(40)
    )

    result = await runtime.values(observations=observations)
    await runtime.close()

    assert isinstance(result, Ok)
    assert len(result.value) == 40
    assert model.encoded_row_counts == [32, 8]
    assert model.value_call_count == 2


async def test_inference_programming_failure_reaches_every_waiter() -> (
    None
):
    runtime = InferenceRuntime.create(
        model=_BrokenPolicyModel(),
        device=torch.device("cpu"),
    )
    observation = _bid_observation(reveal_is_legal=True)

    outcomes = await asyncio.gather(
        runtime.values(observations=(observation,)),
        runtime.values(observations=(observation,)),
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
    model = _SlowPolicyModel()
    runtime = InferenceRuntime.create(
        model=model,
        device=torch.device("cpu"),
    )
    observation = _bid_observation(reveal_is_legal=True)
    pending = asyncio.create_task(
        runtime.values(observations=(observation,))
    )
    await asyncio.sleep(0)
    pending.cancel()

    cancellation = await asyncio.gather(
        pending,
        return_exceptions=True,
    )
    resumed = await runtime.values(observations=(observation,))
    await runtime.close()

    assert isinstance(cancellation[0], asyncio.CancelledError)
    assert isinstance(resumed, Ok)


def _draws(*, seed: int, count: int) -> tuple[InferenceDrawKey, ...]:
    return tuple(
        InferenceDrawKey(seed=seed, ordinal=index)
        for index in range(count)
    )


def _bid_query(*, reveal_is_legal: bool) -> ModelQuery:
    observation = _bid_observation(reveal_is_legal=reveal_is_legal)
    snapshot = _bid_snapshot(reveal_is_legal=reveal_is_legal)
    return ModelQuery(
        observation=observation,
        legal_actions=build_legal_action_index(
            viewer=Seat.A,
            snapshot=snapshot,
            query=observation.action_query,
        ),
        seat=Seat.A,
    )


def _play_query() -> ModelQuery:
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
    return ModelQuery(
        observation=observation,
        legal_actions=build_legal_action_index(
            viewer=Seat.A,
            snapshot=snapshot,
            query=observation.action_query,
        ),
        seat=Seat.A,
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
