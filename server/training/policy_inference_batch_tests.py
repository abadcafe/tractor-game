"""Tests for columnar policy inference request batches."""

from __future__ import annotations

import torch

from server.player.test_helpers import card, make_snapshot
from server.protocol import TrickSlotSnapshot, TrickSnapshot
from server.result import Ok, Rejected
from server.rules.cards import Card
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.numeric_features import (
    PAD_NUMERIC_FEATURES,
    numeric_feature_values,
)
from server.training.observation import Observation, build_observation
from server.training.policy_inference_batch import (
    BorrowedPolicyRequestBatch,
    PolicyRequestCompiler,
    PolicyRequestInput,
    PolicyRequestRoute,
    materialize_borrowed_policy_request_batch,
)
from server.training.policy_inference_batch.frame import (
    decode_policy_request_frame_metadata,
)
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_action_plan.frame import (
    compile_legal_action_frame,
)
from server.training.tensorize import OBSERVATION_COMPONENT_COUNT
from server.training.vocab import component_ids
from server.training.vocab_schema import (
    PAD_COMPONENT_IDS,
    VOCAB_SCHEMA,
    TokenComponentIds,
)


def test_compile_request_batch_materializes_device_batch() -> None:
    batch_result = _request_batch((_request_input(),))
    assert isinstance(batch_result, Ok)

    device_result = materialize_borrowed_policy_request_batch(
        batch=batch_result.value,
        device=torch.device("cpu"),
    )

    assert isinstance(device_result, Ok)
    batch = device_result.value
    assert batch_result.value.routes == (
        PolicyRequestRoute(worker_index=2, request_id=5),
    )
    assert batch.observation_batch.component_ids.shape == (
        1,
        batch_result.value.max_observation_tokens,
        OBSERVATION_COMPONENT_COUNT,
    )
    assert batch.observation_batch.numeric_values.shape == (
        1,
        batch_result.value.max_observation_tokens,
        NUMERIC_FEATURE_COUNT,
    )
    assert batch.policy_versions == (_decision_key().policy_version,)
    assert (
        batch.padded_generation_steps
        == batch_result.value.padded_generation_steps
    )
    assert tuple(
        int(value) for value in batch.generation_step_counts
    ) == (batch.padded_generation_steps,)


def test_compile_request_batch_accepts_mixed_generation_widths() -> (
    None
):
    batch_result = _request_batch(
        (_request_input(), _play_request_input())
    )

    assert isinstance(batch_result, Ok)
    batch = batch_result.value
    assert batch.row_count() == 2
    assert (
        batch.generation_step_counts[0]
        < (batch.generation_step_counts[1])
    )
    assert batch.padded_generation_steps == max(
        batch.generation_step_counts
    )


def test_request_frame_metadata_carries_canonical_layout() -> None:
    batch_result = _request_batch((_request_input(),))
    assert isinstance(batch_result, Ok)
    metadata = batch_result.value.metadata

    decoded_result = decode_policy_request_frame_metadata(
        batch_result.value.frame.view()
    )

    assert isinstance(decoded_result, Ok)
    decoded = decoded_result.value
    assert metadata.layout.total_bytes == metadata.byte_count
    assert metadata.layout.batch_capacity == metadata.batch_capacity
    assert metadata.layout.max_observation_tokens == (
        metadata.max_observation_tokens
    )
    assert metadata.layout.padded_generation_steps == (
        metadata.padded_generation_steps
    )
    assert decoded.layout == metadata.layout


def test_compile_request_batch_rejects_token_over_budget() -> None:
    compiler = PolicyRequestCompiler(
        batch_capacity=4,
        max_observation_tokens=1,
    )
    result = compiler.compile_batch((_request_input(),))

    assert isinstance(result, Rejected)
    assert (
        result.reason
        == "policy request observation exceeds token budget"
    )


def test_compile_request_batch_leaves_padding_columns_zero() -> None:
    observation = _observation()
    batch_result = _request_batch((_request_input(),))
    assert isinstance(batch_result, Ok)

    device_result = materialize_borrowed_policy_request_batch(
        batch=batch_result.value,
        device=torch.device("cpu"),
    )

    assert isinstance(device_result, Ok)
    token_count = len(observation.tokens)
    component_padding = (
        device_result.value.observation_batch.component_ids[
            0, token_count:
        ]
    )
    numeric_value_padding = (
        device_result.value.observation_batch.numeric_values[
            0, token_count:
        ]
    )
    numeric_mask_padding = (
        device_result.value.observation_batch.numeric_masks[
            0, token_count:
        ]
    )
    assert int(torch.count_nonzero(component_padding).item()) == 0
    assert int(torch.count_nonzero(numeric_value_padding).item()) == 0
    assert int(torch.count_nonzero(numeric_mask_padding).item()) == 0


def test_compile_request_batch_reuses_compiler_workspace() -> None:
    compiler = PolicyRequestCompiler(
        batch_capacity=4,
        max_observation_tokens=512,
    )
    first_result = compiler.compile_batch(
        (_play_request_input(), _play_request_input())
    )
    assert isinstance(first_result, Ok)
    first_buffer = first_result.value.frame.view().obj

    second_result = compiler.compile_batch((_request_input(),))

    assert isinstance(second_result, Ok)
    assert second_result.value.frame.view().obj is first_buffer


def test_reused_request_workspace_clears_padding_columns() -> None:
    compiler = PolicyRequestCompiler(
        batch_capacity=4,
        max_observation_tokens=512,
    )
    wide_result = compiler.compile_batch((_play_request_input(),))
    assert isinstance(wide_result, Ok)

    narrow_result = compiler.compile_batch((_request_input(),))
    assert isinstance(narrow_result, Ok)
    device_result = materialize_borrowed_policy_request_batch(
        batch=narrow_result.value,
        device=torch.device("cpu"),
    )

    assert isinstance(device_result, Ok)
    token_count = len(_observation().tokens)
    component_padding = (
        device_result.value.observation_batch.component_ids[
            0, token_count:
        ]
    )
    numeric_value_padding = (
        device_result.value.observation_batch.numeric_values[
            0, token_count:
        ]
    )
    numeric_mask_padding = (
        device_result.value.observation_batch.numeric_masks[
            0, token_count:
        ]
    )
    assert int(torch.count_nonzero(component_padding).item()) == 0
    assert int(torch.count_nonzero(numeric_value_padding).item()) == 0
    assert int(torch.count_nonzero(numeric_mask_padding).item()) == 0


def test_compile_request_batch_preserves_fixed_width_columns() -> None:
    request = _follow_pair_request_input()
    action_plan = compile_legal_action_frame(request.legal_actions)
    batch_result = _request_batch((request,))
    assert isinstance(batch_result, Ok)

    device_result = materialize_borrowed_policy_request_batch(
        batch=batch_result.value,
        device=torch.device("cpu"),
    )

    assert isinstance(device_result, Ok)
    device_batch = device_result.value
    token = request.observation.tokens[0]
    numeric = numeric_feature_values(token)
    action_batch = device_batch.action_plan_batch
    assert _tensor_int_tuple(
        device_batch.observation_batch.component_ids[0, 0]
    ) == _component_tuple(component_ids(token))
    assert torch.allclose(
        device_batch.observation_batch.numeric_values[0, 0],
        torch.tensor(numeric.values, dtype=torch.float32),
    )
    assert torch.allclose(
        device_batch.observation_batch.numeric_masks[0, 0],
        torch.tensor(numeric.masks, dtype=torch.float32),
    )
    assert (
        _tensor_int_tuple(action_batch.available_counts[0])
        == action_plan.available_counts
    )
    assert (
        _tensor_int_tuple(action_batch.effective_suits[0])
        == action_plan.effective_suits
    )
    assert (
        _tensor_bool_tuple(action_batch.same_suit_mask[0])
        == action_plan.same_suit_mask
    )
    assert (
        _tensor_bool_tuple(action_batch.off_suit_mask[0])
        == action_plan.off_suit_mask
    )
    assert (
        _tensor_bool_tuple(action_batch.pair_face_mask[0])
        == action_plan.pair_face_mask
    )

    pair_plan_count = len(action_plan.pair_plan_masks)
    assert pair_plan_count > 0
    for pair_plan_index, expected_mask in enumerate(
        action_plan.pair_plan_masks
    ):
        assert (
            _tensor_bool_tuple(
                action_batch.pair_plan_masks[0, pair_plan_index]
            )
            == expected_mask
        )
    assert _tensor_bool_tuple(
        action_batch.pair_plan_row_mask[0, :pair_plan_count]
    ) == tuple(True for _ in action_plan.pair_plan_masks)
    assert not bool(
        action_batch.pair_plan_row_mask[0, pair_plan_count].item()
    )


def test_padding_token_schema_is_zero_initialized() -> None:
    assert VOCAB_SCHEMA.obs_pad_id == 0
    assert PAD_COMPONENT_IDS.token_type == 0
    assert PAD_COMPONENT_IDS.segment == 0
    assert PAD_COMPONENT_IDS.field == 0
    assert PAD_COMPONENT_IDS.value == 0
    assert PAD_COMPONENT_IDS.suit == 0
    assert PAD_COMPONENT_IDS.rank == 0
    assert PAD_COMPONENT_IDS.points == 0
    assert PAD_COMPONENT_IDS.color == 0
    assert PAD_COMPONENT_IDS.role == 0
    assert PAD_COMPONENT_IDS.trick_age == 0
    assert PAD_COMPONENT_IDS.trick_state == 0
    assert PAD_COMPONENT_IDS.play_order == 0
    assert PAD_COMPONENT_IDS.count == 0
    assert PAD_COMPONENT_IDS.play_width == 0
    assert PAD_COMPONENT_IDS.event_age == 0
    assert all(value == 0.0 for value in PAD_NUMERIC_FEATURES.values)
    assert all(mask == 0.0 for mask in PAD_NUMERIC_FEATURES.masks)


def _request_batch(
    requests: tuple[PolicyRequestInput, ...],
) -> Ok[BorrowedPolicyRequestBatch] | Rejected:
    compiler = PolicyRequestCompiler(
        batch_capacity=4,
        max_observation_tokens=512,
    )
    return compiler.compile_batch(requests)


def _request_input() -> PolicyRequestInput:
    observation = _observation()
    return PolicyRequestInput(
        route=PolicyRequestRoute(worker_index=2, request_id=5),
        observation=observation,
        legal_actions=_legal_actions(observation),
        decision_key=_decision_key(),
    )


def _play_request_input() -> PolicyRequestInput:
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[
            card("hearts", "2", 1),
            card("hearts", "3", 1),
            card("hearts", "4", 1),
        ],
        trump_rank="2",
    )
    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )
    return PolicyRequestInput(
        route=PolicyRequestRoute(worker_index=2, request_id=6),
        observation=observation,
        legal_actions=build_legal_action_index(
            player_index=0,
            snapshot=snapshot,
            query=observation.action_query,
        ),
        decision_key=PolicyDecisionKey(
            base_seed=0,
            policy_version=3,
            episode_id=0,
            player_index=0,
            decision_index=1,
        ),
    )


def _follow_pair_request_input() -> PolicyRequestInput:
    lead_cards = [card("hearts", "A", 1), card("hearts", "A", 2)]
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[
            card("hearts", "K", 1),
            card("hearts", "K", 2),
            card("hearts", "Q", 1),
            card("hearts", "Q", 2),
            card("spades", "3", 1),
        ],
        trump_rank="2",
        trick=_trick(
            lead_player=1,
            current_player=2,
            lead_cards=lead_cards,
        ),
    )
    observation = build_observation(
        player_index=2,
        snapshot=snapshot,
        history=(),
    )
    return PolicyRequestInput(
        route=PolicyRequestRoute(worker_index=2, request_id=7),
        observation=observation,
        legal_actions=build_legal_action_index(
            player_index=2,
            snapshot=snapshot,
            query=observation.action_query,
        ),
        decision_key=PolicyDecisionKey(
            base_seed=0,
            policy_version=3,
            episode_id=0,
            player_index=2,
            decision_index=2,
        ),
    )


def _observation() -> Observation:
    return build_observation(
        player_index=0,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            player_hand=[card("hearts", "2", 1)],
            trump_rank="2",
        ),
        history=(),
    )


def _legal_actions(observation: Observation) -> LegalActionIndex:
    return build_legal_action_index(
        player_index=0,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            player_hand=[card("hearts", "2", 1)],
            trump_rank="2",
        ),
        query=observation.action_query,
    )


def _decision_key() -> PolicyDecisionKey:
    return PolicyDecisionKey(
        base_seed=0,
        policy_version=3,
        episode_id=0,
        player_index=0,
        decision_index=0,
    )


def _trick(
    *,
    lead_player: int,
    current_player: int,
    lead_cards: list[Card],
) -> TrickSnapshot:
    return TrickSnapshot(
        lead_player=lead_player,
        current_player=current_player,
        slots=[
            TrickSlotSnapshot(player=0, cards=[]),
            TrickSlotSnapshot(
                player=lead_player,
                cards=list(lead_cards),
            ),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
    )


def _component_tuple(
    values: TokenComponentIds,
) -> tuple[int, ...]:
    return (
        values.token_type,
        values.segment,
        values.field,
        values.value,
        values.suit,
        values.rank,
        values.points,
        values.color,
        values.role,
        values.trick_age,
        values.trick_state,
        values.play_order,
        values.count,
        values.play_width,
        values.event_age,
    )


def _tensor_int_tuple(values: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value.item()) for value in values)


def _tensor_bool_tuple(values: torch.Tensor) -> tuple[bool, ...]:
    return tuple(bool(value.item()) for value in values)
