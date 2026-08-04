"""Black-box tests for the trained-model AI controller."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import torch
from torch import Tensor

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game_ai.controller import AIController
from server.policy_model._schema.actions import ACTION_CHOICE_COUNT
from server.policy_model.actions import GeneratedAction
from server.policy_model.actions.decoding import (
    ActionChoiceLogitDecoder,
    ActionSampler,
    action_plan_generation_step_count,
    action_trace_from_choice_ids,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.policy_model.inference import (
    ActionDecision,
    ActionDecisionRequest,
    PolicyQuery,
)
from tests.support import card
from tests.support import snapshot as make_snapshot


class _RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@dataclass(slots=True)
class _ZeroDecoder(ActionChoiceLogitDecoder):
    batch_size: int

    def next_choice_logits(
        self,
        active_rows: Tensor,
        scored_rows: Tensor,
    ) -> Tensor:
        assert active_rows.shape == (self.batch_size,)
        assert scored_rows.shape == active_rows.shape
        return torch.zeros(
            (self.batch_size, ACTION_CHOICE_COUNT),
            dtype=torch.float32,
        )

    def advance(
        self,
        selected_choice_ids: Tensor,
        active_rows: Tensor,
    ) -> None:
        assert selected_choice_ids.shape == (self.batch_size,)
        assert active_rows.shape == (self.batch_size,)


def _requests() -> list[ActionDecisionRequest]:
    return []


@dataclass(slots=True)
class _DecisionModel:
    requests: list[ActionDecisionRequest] = field(
        default_factory=_requests
    )

    async def decide(
        self,
        *,
        requests: tuple[ActionDecisionRequest, ...],
    ) -> Ok[tuple[ActionDecision, ...]] | Rejected:
        self.requests.extend(requests)
        decisions: list[ActionDecision] = []
        for request in requests:
            action = _first_legal_action(request.query)
            if isinstance(action, Rejected):
                return action
            decisions.append(
                ActionDecision(
                    action=action.value,
                    candidate_count=request.candidate_count,
                    selected_action_value=0.25,
                )
            )
        return Ok(tuple(decisions))


async def test_ai_controller_builds_one_direct_model_decision() -> None:
    model = _DecisionModel()
    controller = AIController(
        seat=Seat.A,
        model=model,
        candidate_count=12,
        action_value_temperature=0.75,
        random_source=random.Random(7),
    )
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[
            card("hearts", "2"),
        ],
        trump_rank="2",
    )
    observed = controller.observe(
        seq=0,
        snapshot=snapshot,
        error=None,
    )
    assert isinstance(observed, Ok)

    decided = await controller.decide(seq=0, snapshot=snapshot)

    assert isinstance(decided, Ok)
    assert isinstance(
        decided.value,
        commands.PassBid | commands.RevealBid,
    )
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request.candidate_count == 12
    assert request.action_value_temperature == 0.75
    assert request.draw.ordinal == 0


async def test_ai_controller_logs_successful_decision_at_debug() -> (
    None
):
    controller = AIController(
        seat=Seat.A,
        model=_DecisionModel(),
        candidate_count=12,
        action_value_temperature=0.75,
        random_source=random.Random(7),
    )
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[card("hearts", "2")],
        trump_rank="2",
    )
    observed = controller.observe(
        seq=0,
        snapshot=snapshot,
        error=None,
    )
    assert isinstance(observed, Ok)
    logger = logging.getLogger("server.game_ai.controller")
    previous_level = logger.level
    handler = _RecordHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        decided = await controller.decide(seq=0, snapshot=snapshot)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    assert isinstance(decided, Ok)
    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.levelno == logging.DEBUG
    assert record.getMessage().startswith("ai.decision kind=bid ")


async def test_ai_controller_rejects_non_strategic_view() -> None:
    controller = AIController(
        seat=Seat.A,
        model=_DecisionModel(),
        candidate_count=4,
        action_value_temperature=1.0,
        random_source=random.Random(3),
    )
    snapshot = make_snapshot(
        phase="WAITING",
        awaiting_action="next_round",
    )

    decided = await controller.decide(seq=0, snapshot=snapshot)

    assert isinstance(decided, Rejected)
    assert decided.reason == "AI has no strategic action to decide"


def test_ai_controller_requires_contiguous_observations() -> None:
    controller = AIController(
        seat=Seat.A,
        model=_DecisionModel(),
        candidate_count=4,
        action_value_temperature=1.0,
        random_source=random.Random(3),
    )
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=[card("hearts", "2")],
        trump_rank="2",
    )
    first = controller.observe(
        seq=0,
        snapshot=snapshot,
        error=None,
    )
    skipped = controller.observe(
        seq=2,
        snapshot=snapshot,
        error=None,
    )

    assert isinstance(first, Ok)
    assert isinstance(skipped, Rejected)


def _first_legal_action(
    query: PolicyQuery,
) -> Ok[GeneratedAction] | Rejected:
    frame = compile_legal_action_frame(query.legal_actions)
    steps = action_plan_generation_step_count(frame)
    device = torch.device("cpu")
    sampled = ActionSampler.create(
        batch_capacity=1,
        device=device,
    ).sample(
        action_batch=plan_batch_to_device((frame,), device=device),
        generation_step_counts=torch.tensor(
            (steps,),
            dtype=torch.long,
            device=device,
        ),
        sampling_thresholds=torch.zeros(
            (1, steps),
            dtype=torch.float64,
            device=device,
        ),
        padded_generation_steps=steps,
        logit_decoder=_ZeroDecoder(batch_size=1),
    )
    if isinstance(sampled, Rejected):
        return sampled
    count = int(sampled.value.step_counts[0].item())
    choice_ids = tuple(
        int(sampled.value.choice_ids_padded[0, index].item())
        for index in range(count)
    )
    trace = action_trace_from_choice_ids(choice_ids)
    if isinstance(trace, Rejected):
        return trace
    return query.legal_actions.decode(trace.value)
