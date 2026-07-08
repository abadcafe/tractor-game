"""Tests for PPO trainer updates."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from server.player.test_helpers import card, make_snapshot
from server.result import Ok, Rejected
from server.rules.card_faces import CardFace, FaceCount
from server.training.config import ModelConfig, TrainConfig
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.model import (
    ArgumentPrefixScores,
    ObservationEncoding,
    TractorPolicyModel,
)
from server.training.observation import build_observation
from server.training.policy_sampling import (
    DecisionHandle,
    RankReturnBatch,
    SampledPolicyBatch,
)
from server.training.policy_sampling.model_rank_sample_arena import (
    ModelRankSampleArena,
)
from server.training.ppo import (
    PPOTrainer,
    PPOUpdateInput,
    PPOUpdateProfile,
)
from server.training.ppo.distributed import PPOUpdatePartition
from server.training.ppo.replay_tensors import (
    ReadyPPOBatch,
)
from server.training.semantic_action_plan import (
    advance_action_state,
    compile_legal_action_frame,
    initial_action_state,
    legal_token_choices,
    plan_batch_to_device,
)
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.codec import (
    SEMANTIC_CODEC,
    semantic_argument_id,
)
from server.training.tensorize import (
    ArgumentPrefixTensorBatch,
    ObservationTensorBatch,
    tensorize_observations,
)


class CountingTractorPolicyModel(TractorPolicyModel):
    """Policy model that records forward batch sizes for tests."""

    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__(
            d_model=d_model,
            layers=layers,
            heads=heads,
        )
        self.encode_batch_sizes: list[int] = []
        self.score_batch_sizes: list[int] = []
        self.score_prefix_widths: list[int] = []
        self.training_modes: list[bool] = []

    def encode_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> ObservationEncoding:
        self.training_modes.append(self.training)
        self.encode_batch_sizes.append(
            int(observation.component_ids.shape[0])
        )
        return super().encode_observations(observation)

    def score_argument_prefixes(
        self,
        encoding: ObservationEncoding,
        prefix: ArgumentPrefixTensorBatch,
    ) -> ArgumentPrefixScores:
        self.training_modes.append(self.training)
        self.score_batch_sizes.append(int(prefix.argument_ids.shape[0]))
        self.score_prefix_widths.append(
            int(prefix.argument_ids.shape[1])
        )
        return super().score_argument_prefixes(encoding, prefix)


class NonFiniteValueModel(TractorPolicyModel):
    """Policy model that produces an infinite value loss."""

    def value_estimates(
        self,
        encoding: ObservationEncoding,
    ) -> Tensor:
        values = super().value_estimates(encoding)
        return torch.full_like(values, torch.inf)


def test_update_returns_stats_and_adamw_state() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=64,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=1,
    )
    model = TractorPolicyModel(
        d_model=model_config.d_model,
        layers=model_config.layers,
        heads=model_config.heads,
    ).to(device)
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode="off",
    )
    stats_result = trainer.update(_single_card_update_input(count=1))
    assert isinstance(stats_result, Ok)
    stats = stats_result.value
    state = trainer.optimizer_state()

    assert stats.total_loss >= 0.0
    assert state["kind"] == "ppo_adamw"
    assert state["step_count"] == 1


def test_update_batches_minibatch_model_forwards() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=64,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = CountingTractorPolicyModel(
        d_model=model_config.d_model,
        layers=model_config.layers,
        heads=model_config.heads,
    ).to(device)
    model.eval()
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode="detailed",
    )

    update_result = trainer.update(_single_card_update_input(count=4))
    assert isinstance(update_result, Ok)
    profile = update_result.value.profile

    assert model.encode_batch_sizes == [4]
    assert model.score_batch_sizes == [8]
    assert model.score_prefix_widths == [2]
    assert profile.update_seconds > 0.0
    assert profile.argument_decode_seconds >= 0.0
    assert 0.0 <= profile.argument_decode_fraction <= 1.0
    assert profile.argument_prefix_batch_count == 1
    assert profile.argument_prefix_row_count == 8
    assert profile.argument_prefix_token_count == 16
    assert profile.argument_prefix_valid_token_count == 12
    assert profile.argument_prefix_padding_token_count == 4
    assert model.training_modes
    assert all(training is True for training in model.training_modes)
    assert model.training is True


def test_update_rejects_ddp_without_process_group() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=64,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = TractorPolicyModel(
        d_model=model_config.d_model,
        layers=model_config.layers,
        heads=model_config.heads,
    ).to(device)
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode="off",
        update_partition=PPOUpdatePartition(rank=0, world_size=2),
    )

    result = trainer.update(_single_card_update_input(count=4))

    assert isinstance(result, Rejected)
    assert "requires initialized process group" in result.reason


def test_update_uses_configured_single_rank_partition() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=64,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = CountingTractorPolicyModel(
        d_model=model_config.d_model,
        layers=model_config.layers,
        heads=model_config.heads,
    ).to(device)
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode="detailed",
        update_partition=PPOUpdatePartition(rank=0, world_size=1),
    )

    result = trainer.update(_single_card_update_input(count=4))

    assert isinstance(result, Ok)
    assert model.encode_batch_sizes == [4]
    assert model.score_batch_sizes == [8]


def test_update_rejects_empty_single_rank_input() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=64,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = TractorPolicyModel(
        d_model=model_config.d_model,
        layers=model_config.layers,
        heads=model_config.heads,
    ).to(device)
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode="off",
    )

    result = trainer.update(
        PPOUpdateInput(policy_version=0, local_batch=None)
    )

    assert isinstance(result, Rejected)
    assert (
        result.reason == "single-rank PPO update requires local batch"
    )


def test_update_disables_profile_by_default() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=64,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = TractorPolicyModel(
        d_model=model_config.d_model,
        layers=model_config.layers,
        heads=model_config.heads,
    ).to(device)
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode="off",
    )

    update_result = trainer.update(_single_card_update_input(count=4))

    assert isinstance(update_result, Ok)
    _assert_profile_zero(update_result.value.profile)


def test_update_basic_profile_records_only_update_seconds() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=64,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = TractorPolicyModel(
        d_model=model_config.d_model,
        layers=model_config.layers,
        heads=model_config.heads,
    ).to(device)
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode="basic",
    )

    update_result = trainer.update(_single_card_update_input(count=4))

    assert isinstance(update_result, Ok)
    profile = update_result.value.profile
    assert profile.update_seconds > 0.0
    assert profile.minibatch_loss_seconds == 0.0
    assert profile.observation_batch_seconds == 0.0
    assert profile.observation_encode_seconds == 0.0
    assert profile.value_head_seconds == 0.0
    assert profile.argument_select_seconds == 0.0
    assert profile.argument_prefix_tensorize_seconds == 0.0
    assert profile.argument_decode_seconds == 0.0
    assert profile.argument_distribution_seconds == 0.0
    assert profile.backward_seconds == 0.0
    assert profile.optimizer_step_seconds == 0.0
    assert profile.argument_decode_fraction == 0.0
    assert profile.argument_prefix_batch_count == 0
    assert profile.argument_prefix_row_count == 0
    assert profile.argument_prefix_token_count == 0
    assert profile.argument_prefix_valid_token_count == 0
    assert profile.argument_prefix_padding_token_count == 0


def test_update_rejects_non_finite_loss_before_optimizer_step() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=64,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=1,
    )
    model = NonFiniteValueModel(
        d_model=model_config.d_model,
        layers=model_config.layers,
        heads=model_config.heads,
    ).to(device)
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode="off",
    )
    before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )

    result = trainer.update(_single_card_update_input(count=1))

    assert isinstance(result, Rejected)
    assert "PPO value_loss must be finite" in result.reason
    assert trainer.optimizer_state()["step_count"] == 0
    for index, parameter in enumerate(model.parameters()):
        assert torch.equal(parameter.detach(), before[index])


def test_update_rejects_non_finite_gradients_before_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=64,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=1,
    )
    model = TractorPolicyModel(
        d_model=model_config.d_model,
        layers=model_config.layers,
        heads=model_config.heads,
    ).to(device)
    trainer = PPOTrainer(
        model=model,
        train_config=train_config,
        device=device,
        profile_mode="off",
    )
    before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )

    def write_nan_gradients(*args: object, **kwargs: object) -> None:
        assert args
        assert not kwargs
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, torch.nan)

    monkeypatch.setattr(torch.autograd, "backward", write_nan_gradients)

    result = trainer.update(_single_card_update_input(count=1))

    assert isinstance(result, Rejected)
    assert "PPO gradients must be finite" in result.reason
    assert trainer.optimizer_state()["step_count"] == 0
    for index, parameter in enumerate(model.parameters()):
        assert torch.equal(parameter.detach(), before[index])


def _single_card_batch(*, count: int) -> ReadyPPOBatch:
    assert count > 0
    device = torch.device("cpu")
    store = ModelRankSampleArena(model_rank_index=0, device=device)
    handles: list[DecisionHandle] = []
    return_values: list[float] = []
    for index in range(count):
        player_index = index % 4
        handle = _store_single_card_decision(
            store=store,
            device=device,
            player_index=player_index,
        )
        handles.append(handle)
        return_values.append(1.0 if player_index in (0, 2) else -1.0)
    returns = RankReturnBatch(
        policy_version=0,
        model_rank_index=0,
        slot_indices=torch.tensor(
            tuple(handle.slot_index for handle in handles),
            dtype=torch.long,
        ),
        slot_generations=torch.tensor(
            tuple(handle.slot_generation for handle in handles),
            dtype=torch.long,
        ),
        return_values=torch.tensor(return_values, dtype=torch.float32),
        round_count=1,
    )
    batch_result = store.materialize_return_batch(returns=returns)
    assert isinstance(batch_result, Ok)
    return batch_result.value


def _single_card_update_input(*, count: int) -> PPOUpdateInput:
    batch = _single_card_batch(count=count)
    return PPOUpdateInput(
        policy_version=batch.policy_version,
        local_batch=batch,
    )


def _assert_profile_zero(profile: PPOUpdateProfile) -> None:
    assert profile.update_seconds == 0.0
    assert profile.minibatch_loss_seconds == 0.0
    assert profile.observation_batch_seconds == 0.0
    assert profile.observation_encode_seconds == 0.0
    assert profile.value_head_seconds == 0.0
    assert profile.argument_select_seconds == 0.0
    assert profile.argument_prefix_tensorize_seconds == 0.0
    assert profile.argument_decode_seconds == 0.0
    assert profile.argument_distribution_seconds == 0.0
    assert profile.backward_seconds == 0.0
    assert profile.optimizer_step_seconds == 0.0
    assert profile.argument_decode_fraction == 0.0
    assert profile.argument_prefix_batch_count == 0
    assert profile.argument_prefix_row_count == 0
    assert profile.argument_prefix_token_count == 0
    assert profile.argument_prefix_valid_token_count == 0
    assert profile.argument_prefix_padding_token_count == 0


def _store_single_card_decision(
    *,
    store: ModelRankSampleArena,
    device: torch.device,
    player_index: int,
) -> DecisionHandle:
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    observation = build_observation(
        player_index=player_index,
        snapshot=snapshot,
        history=(),
    )
    legal_actions = build_legal_action_index(
        player_index=player_index,
        snapshot=snapshot,
        query=observation.action_query,
    )
    trace = SemanticArgumentTrace(
        arguments=(
            SemanticArgument(
                "select_face_count",
                FaceCount(CardFace(test_card.suit, test_card.rank), 1),
            ),
            SemanticArgument("stop"),
        )
    )
    replay_trace = _replay_trace_for(
        legal_actions=legal_actions, trace=trace, device=device
    )
    sample_batch = _sampled_policy_batch(
        observation_batch=tensorize_observations(
            observations=(observation,),
            max_observation_tokens=64,
            device=device,
        ),
        replay_trace=replay_trace,
        device=device,
    )
    stored = store.store_sampled_batch(batch=sample_batch)
    assert len(stored) == 1
    stored_decision = stored[0]
    assert isinstance(stored_decision, Ok)
    return stored_decision.value.decision_handle


def _sampled_policy_batch(
    *,
    observation_batch: ObservationTensorBatch,
    replay_trace: _ReplayTrace,
    device: torch.device,
) -> SampledPolicyBatch:
    selected = torch.zeros(
        (1, SEMANTIC_CODEC.max_argument_tokens),
        dtype=torch.long,
        device=device,
    )
    step_count = int(replay_trace.selected_token_ids.shape[0])
    selected[0, :step_count] = replay_trace.selected_token_ids
    choice_width = int(replay_trace.legal_choice_ids.shape[1])
    choice_ids = torch.zeros(
        (
            1,
            SEMANTIC_CODEC.max_argument_tokens,
            choice_width,
        ),
        dtype=torch.int16,
        device=device,
    )
    choice_masks = torch.zeros(
        choice_ids.shape,
        dtype=torch.bool,
        device=device,
    )
    selected_offsets = torch.zeros(
        (1, SEMANTIC_CODEC.max_argument_tokens),
        dtype=torch.long,
        device=device,
    )
    choice_ids[0, :step_count, :] = replay_trace.legal_choice_ids
    choice_masks[0, :step_count, :] = replay_trace.legal_choice_masks
    selected_offsets[0, :step_count] = (
        replay_trace.selected_choice_offsets
    )
    return SampledPolicyBatch(
        policy_versions=(0,),
        status_codes=torch.zeros((1,), dtype=torch.long, device=device),
        observation_batch=observation_batch,
        selected_token_ids_padded=selected,
        legal_choice_ids_padded=choice_ids,
        legal_choice_masks_padded=choice_masks,
        selected_choice_offsets_padded=selected_offsets,
        step_counts=torch.tensor(
            (step_count,), dtype=torch.long, device=device
        ),
        choice_counts=torch.tensor(
            (int(replay_trace.legal_choice_masks.sum().item()),),
            dtype=torch.long,
            device=device,
        ),
        old_log_probabilities=torch.zeros(
            (1,), dtype=torch.float32, device=device
        ),
        old_values=torch.zeros(
            (1,), dtype=torch.float32, device=device
        ),
    )


@dataclass(frozen=True, slots=True)
class _ReplayTrace:
    selected_token_ids: Tensor
    legal_choice_ids: Tensor
    legal_choice_masks: Tensor
    selected_choice_offsets: Tensor


def _replay_trace_for(
    *,
    legal_actions: LegalActionIndex,
    trace: SemanticArgumentTrace,
    device: torch.device,
) -> _ReplayTrace:
    batch = plan_batch_to_device(
        (compile_legal_action_frame(legal_actions),), device=device
    )
    state = initial_action_state(batch)
    legal_choice_ids: list[Tensor] = []
    legal_choice_masks: list[Tensor] = []
    selected_choice_offsets: list[int] = []
    selected_token_ids: list[int] = []
    for argument in trace.arguments:
        choices = legal_token_choices(batch=batch, state=state)
        token_id = semantic_argument_id(argument)
        row_width = int(choices.choice_counts[0].item())
        row_tokens = choices.token_ids[:row_width]
        legal_choice_ids.append(row_tokens.to(dtype=torch.int16))
        legal_choice_masks.append(
            torch.ones((row_width,), dtype=torch.bool, device=device)
        )
        selected_choice_offsets.append(
            _selected_choice_offset(
                token_ids=row_tokens, selected_token_id=token_id
            )
        )
        selected_token_ids.append(token_id)
        state = advance_action_state(
            batch=batch,
            state=state,
            selected_token_ids=torch.tensor(
                (token_id,), dtype=torch.long, device=device
            ),
            choice_counts=choices.choice_counts,
        )
    max_choice_count = max(
        int(row.shape[0]) for row in legal_choice_ids
    )
    return _ReplayTrace(
        selected_token_ids=torch.tensor(
            tuple(selected_token_ids), dtype=torch.long, device=device
        ),
        legal_choice_ids=torch.stack(
            [
                _pad_choice_row(row, width=max_choice_count)
                for row in legal_choice_ids
            ]
        ),
        legal_choice_masks=torch.stack(
            [
                _pad_choice_row(row, width=max_choice_count)
                for row in legal_choice_masks
            ]
        ),
        selected_choice_offsets=torch.tensor(
            tuple(selected_choice_offsets),
            dtype=torch.long,
            device=device,
        ),
    )


def _selected_choice_offset(
    *, token_ids: Tensor, selected_token_id: int
) -> int:
    cpu_tokens = token_ids.detach().cpu()
    for index in range(int(cpu_tokens.shape[0])):
        if int(cpu_tokens[index].item()) == selected_token_id:
            return index
    assert False


def _pad_choice_row(row: Tensor, *, width: int) -> Tensor:
    if int(row.shape[0]) == width:
        return row
    padding = torch.zeros(
        (width - int(row.shape[0]),),
        dtype=row.dtype,
        device=row.device,
    )
    return torch.cat((row, padding), dim=0)
