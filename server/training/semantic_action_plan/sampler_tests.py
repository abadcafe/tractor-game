"""Black-box tests for batched semantic action sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.foundation.result import Ok, Rejected
from server.game.rules.card_faces import CardFace, FaceCount
from server.game.rules.cards import Rank, Suit
from server.training.semantic_action_plan import (
    ACTION_FACE_COUNT,
    ActionPlanFrame,
    SemanticActionSampleBatch,
    SemanticActionSampler,
    SemanticArgumentLogitDecoder,
    action_plan_generation_step_count,
    plan_batch_to_device,
)
from server.training.semantic_action_plan.frame import (
    ACTION_KIND_TRACE_SET,
)
from server.training.semantic_actions import SemanticArgument
from server.training.semantic_actions.codec import (
    SEMANTIC_CODEC,
    semantic_argument_id,
)


@dataclass(slots=True)
class _ConstantLogitDecoder:
    logits: Tensor
    batch_size: int = 1
    advance_count: int = 0

    def next_logits(self) -> Tensor:
        return self.logits.repeat(self.batch_size, 1)

    def advance(self, selected_token_ids: Tensor) -> None:
        assert selected_token_ids.shape == (self.batch_size,)
        self.advance_count += 1


def test_semantic_action_sampler_zero_threshold_selects_first() -> None:
    pass_id = semantic_argument_id(SemanticArgument("pass"))
    stop_id = semantic_argument_id(SemanticArgument("stop"))

    result = _sample_trace_set(
        traces=((pass_id,), (stop_id,)),
        thresholds=(0.0,),
        logit_decoder=_zero_decoder(),
    )

    assert isinstance(result, Ok)
    assert _token_ids(result.value) == (pass_id,)
    assert int(result.value.selected_choice_offsets[0].item()) == 0


def test_semantic_action_sampler_ignores_invalid_vocab_logit() -> None:
    pass_id = semantic_argument_id(SemanticArgument("pass"))
    stop_id = semantic_argument_id(SemanticArgument("stop"))
    logits = torch.zeros(
        (SEMANTIC_CODEC.argument_vocab_size,), dtype=torch.float32
    )
    logits[0] = 100.0
    logits[stop_id] = 10.0

    result = _sample_trace_set(
        traces=((pass_id,), (stop_id,)),
        thresholds=(0.5,),
        logit_decoder=_constant_decoder(logits),
    )

    assert isinstance(result, Ok)
    assert _token_ids(result.value) == (stop_id,)


def test_semantic_action_sampler_empty_choices_return_error() -> None:
    result = _sample_trace_set(
        traces=(),
        thresholds=(0.5,),
        logit_decoder=_zero_decoder(),
        padded_generation_steps=1,
    )

    assert isinstance(result, Rejected)
    assert result.reason == "policy action has no legal semantic token"


def test_semantic_action_sampler_reuses_workspace() -> None:
    pass_id = semantic_argument_id(SemanticArgument("pass"))
    stop_id = semantic_argument_id(SemanticArgument("stop"))
    sampler = SemanticActionSampler.create(
        batch_capacity=1, device=torch.device("cpu")
    )

    first = _sample_trace_set(
        traces=((pass_id,),),
        thresholds=(0.0,),
        logit_decoder=_zero_decoder(),
        sampler=sampler,
    )
    assert isinstance(first, Ok)
    first_token_ids = _token_ids(first.value)
    second = _sample_trace_set(
        traces=((stop_id,),),
        thresholds=(0.0,),
        logit_decoder=_zero_decoder(),
        sampler=sampler,
    )

    assert isinstance(second, Ok)
    assert first_token_ids == (pass_id,)
    assert _token_ids(second.value) == (stop_id,)
    assert int(second.value.choice_masks[0].sum().item()) == 1


def test_semantic_action_sampler_flattens_variable_active_steps() -> (
    None
):
    first_id = _select_token_id(Suit.HEARTS, Rank.THREE, 1)
    second_id = _select_token_id(Suit.HEARTS, Rank.FOUR, 1)
    third_id = _select_token_id(Suit.HEARTS, Rank.FIVE, 1)
    action_batch = plan_batch_to_device(
        (
            _trace_set_plan(((first_id,),)),
            _trace_set_plan(((first_id, second_id, third_id),)),
            _trace_set_plan(((second_id, third_id),)),
        ),
        device=torch.device("cpu"),
    )
    sampler = SemanticActionSampler.create(
        batch_capacity=3, device=torch.device("cpu")
    )

    result = sampler.sample(
        action_batch=action_batch,
        generation_step_counts=torch.tensor(
            (1, 3, 2), dtype=torch.long
        ),
        sampling_thresholds=torch.zeros((3, 3), dtype=torch.float64),
        padded_generation_steps=3,
        logit_decoder=_ConstantLogitDecoder(
            logits=torch.zeros(
                (SEMANTIC_CODEC.argument_vocab_size,),
                dtype=torch.float32,
            ),
            batch_size=3,
        ),
    )

    assert isinstance(result, Ok)
    assert _tensor_tuple(result.value.active_sample_indices) == (
        0,
        1,
        1,
        1,
        2,
        2,
    )
    assert _tensor_tuple(result.value.active_step_indices) == (
        0,
        0,
        1,
        2,
        0,
        1,
    )
    assert _tensor_tuple(result.value.selected_choice_offsets) == (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert int(result.value.choice_token_ids.shape[0]) == 6


def _sample_trace_set(
    *,
    traces: tuple[tuple[int, ...], ...],
    thresholds: tuple[float, ...],
    logit_decoder: SemanticArgumentLogitDecoder,
    padded_generation_steps: int | None = None,
    sampler: SemanticActionSampler | None = None,
) -> Ok[SemanticActionSampleBatch] | Rejected:
    action_plan = _trace_set_plan(traces)
    generation_steps = (
        action_plan_generation_step_count(action_plan)
        if padded_generation_steps is None
        else padded_generation_steps
    )
    action_batch = plan_batch_to_device(
        (action_plan,), device=torch.device("cpu")
    )
    threshold_tensor = torch.tensor(
        (_pad_thresholds(thresholds, generation_steps),),
        dtype=torch.float64,
    )
    active_sampler = (
        SemanticActionSampler.create(
            batch_capacity=1, device=torch.device("cpu")
        )
        if sampler is None
        else sampler
    )
    return active_sampler.sample(
        action_batch=action_batch,
        generation_step_counts=torch.tensor(
            (generation_steps,), dtype=torch.long
        ),
        sampling_thresholds=threshold_tensor,
        padded_generation_steps=generation_steps,
        logit_decoder=logit_decoder,
    )


def _trace_set_plan(
    traces: tuple[tuple[int, ...], ...],
) -> ActionPlanFrame:
    return ActionPlanFrame(
        kind_code=ACTION_KIND_TRACE_SET,
        available_counts=tuple(0 for _ in range(ACTION_FACE_COUNT)),
        effective_suits=tuple(-1 for _ in range(ACTION_FACE_COUNT)),
        same_suit_mask=tuple(False for _ in range(ACTION_FACE_COUNT)),
        off_suit_mask=tuple(False for _ in range(ACTION_FACE_COUNT)),
        pair_face_mask=tuple(False for _ in range(ACTION_FACE_COUNT)),
        min_select=0,
        max_select=0,
        exact_select=-1,
        required_same_suit_count=0,
        pair_floor=0,
        has_tractor=False,
        trace_tokens=traces,
        pair_plan_masks=(),
    )


def _zero_decoder() -> SemanticArgumentLogitDecoder:
    return _constant_decoder(
        torch.zeros(
            (SEMANTIC_CODEC.argument_vocab_size,), dtype=torch.float32
        )
    )


def _constant_decoder(logits: Tensor) -> SemanticArgumentLogitDecoder:
    return _ConstantLogitDecoder(logits=logits)


def _select_token_id(suit: Suit, rank: Rank, count: int) -> int:
    return semantic_argument_id(
        SemanticArgument(
            "select_face_count",
            FaceCount(face=CardFace(suit, rank), count=count),
        )
    )


def _pad_thresholds(
    thresholds: tuple[float, ...], generation_steps: int
) -> tuple[float, ...]:
    assert generation_steps > 0
    assert len(thresholds) <= generation_steps
    return (
        *thresholds,
        *(0.0 for _ in range(generation_steps - len(thresholds))),
    )


def _token_ids(sample: SemanticActionSampleBatch) -> tuple[int, ...]:
    step_count = int(sample.step_counts[0].item())
    return tuple(
        int(sample.selected_token_ids_padded[0, index].item())
        for index in range(step_count)
    )


def _tensor_tuple(values: Tensor) -> tuple[int, ...]:
    return tuple(int(value.item()) for value in values)
