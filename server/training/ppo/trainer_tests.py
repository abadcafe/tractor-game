"""Tests for PPO trainer updates."""

from __future__ import annotations

from itertools import cycle, islice
from typing import final, override

import pytest
import torch
from torch import Tensor

from server.foundation.result import Ok, Rejected
from server.game import Seat, seats
from server.game.rules.cards.faces import CardFace, FaceCount
from server.policy_model.actions import (
    ACTION_CHOICE_COUNT,
    ActionChoice,
    ActionTrace,
    LegalActionSpace,
    action_choice_id,
    build_legal_action_space,
)
from server.policy_model.actions.decoding import (
    ActionChoiceLogitDecoder,
    ActionSampleBatch,
    ActionSampler,
    action_plan_generation_step_count,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.policy_model.network import (
    ActionTraceScores,
    EncodedObservation,
    ModelConfig,
    PolicyActionModel,
)
from server.policy_model.observation import (
    ObservationMemoryView,
    build_observation,
)
from server.policy_model.observation.tensor import (
    ObservationTensorBatch,
    tensorize_observations,
)
from server.training.config import TrainConfig
from server.training.ppo import (
    PPOTrainer,
    PPOUpdateInput,
    PPOUpdateProfile,
)
from server.training.ppo.distributed import PPOUpdatePartition
from server.training.rollout_inference.samples import (
    DecisionHandle,
    ModelRankSampleArena,
    RankReturnTargets,
)
from server.training.rollout_inference.samples.arena import (
    ArenaPPOBatchSource,
)
from tests.support import card
from tests.support import snapshot as make_snapshot


class CountingPolicyActionModel(PolicyActionModel):
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

    @override
    def encode_policy_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> EncodedObservation:
        self.training_modes.append(self.training)
        self.encode_batch_sizes.append(
            int(observation.category_ids.shape[0])
        )
        return super().encode_policy_observations(observation)

    @override
    def score_action_traces(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: Tensor,
        choice_ids_padded: Tensor,
        step_counts: Tensor,
    ) -> ActionTraceScores:
        self.training_modes.append(self.training)
        self.score_batch_sizes.append(int(choice_ids_padded.shape[0]))
        self.score_prefix_widths.append(int(choice_ids_padded.shape[1]))
        return super().score_action_traces(
            encoding,
            source_rows=source_rows,
            choice_ids_padded=choice_ids_padded,
            step_counts=step_counts,
        )


class NonFiniteValueModel(PolicyActionModel):
    """Policy model that produces an infinite action-value loss."""

    @override
    def action_values(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: Tensor,
        choice_ids_padded: Tensor,
        step_counts: Tensor,
    ) -> Tensor:
        values = super().action_values(
            encoding,
            source_rows=source_rows,
            choice_ids_padded=choice_ids_padded,
            step_counts=step_counts,
        )
        return torch.full_like(values, torch.inf)


@final
class _TraceChoiceDecoder:
    def __init__(
        self,
        *,
        target_choice_ids: tuple[int, ...],
        batch_size: int,
        device: torch.device,
    ) -> None:
        self._target_choice_ids = target_choice_ids
        self._batch_size = batch_size
        self._device = device
        self._step_index = 0

    def next_choice_logits(
        self,
        active_rows: Tensor,
        scored_rows: Tensor,
    ) -> Tensor:
        assert active_rows.shape == (self._batch_size,)
        assert scored_rows.shape == active_rows.shape
        logits = torch.zeros(
            (self._batch_size, ACTION_CHOICE_COUNT),
            dtype=torch.float32,
            device=self._device,
        )
        if self._step_index < len(self._target_choice_ids):
            logits[:, self._target_choice_ids[self._step_index]] = 100.0
        return logits

    def advance(
        self,
        selected_choice_ids: Tensor,
        active_rows: Tensor,
    ) -> None:
        assert selected_choice_ids.shape == (self._batch_size,)
        assert active_rows.shape == (self._batch_size,)
        self._step_index += 1


def test_update_returns_stats_and_adamw_state() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=1,
    )
    model = PolicyActionModel(
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


def test_update_handles_clipped_policy_ratio_above_float32_range() -> (
    None
):
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=2,
    )
    model = PolicyActionModel(
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
    batch = _single_card_batch(count=2)
    _ = batch.old_log_probabilities.fill_(-100.0)

    update_result = trainer.update(
        PPOUpdateInput(policy_version=0, local_batch=batch)
    )

    assert isinstance(update_result, Ok)
    assert trainer.optimizer_state()["step_count"] == 1
    assert all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in model.parameters()
    )


def test_update_batches_minibatch_model_forwards() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = CountingPolicyActionModel(
        d_model=model_config.d_model,
        layers=model_config.layers,
        heads=model_config.heads,
    ).to(device)
    _ = model.eval()
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
    assert model.score_batch_sizes == [4]
    assert model.score_prefix_widths == [2]
    assert profile.update_seconds > 0.0
    assert profile.action_decode_seconds >= 0.0
    assert 0.0 <= profile.action_decode_fraction <= 1.0
    assert profile.action_trace_batch_count == 1
    assert profile.action_trace_row_count == 4
    assert profile.action_trace_choice_count == 8
    assert profile.action_trace_valid_choice_count == 8
    assert profile.action_trace_padding_choice_count == 0
    assert model.training_modes
    assert all(training is True for training in model.training_modes)
    assert model.training is True


def test_update_rejects_ddp_without_process_group() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = PolicyActionModel(
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
        heads=1,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = CountingPolicyActionModel(
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
    assert model.score_batch_sizes == [4]


def test_update_rejects_empty_single_rank_input() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = PolicyActionModel(
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
        heads=1,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = PolicyActionModel(
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
        heads=1,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=4,
    )
    model = PolicyActionModel(
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
    assert profile.policy_observation_encode_seconds == 0.0
    assert profile.action_value_observation_encode_seconds == 0.0
    assert profile.action_value_decode_seconds == 0.0
    assert profile.action_decode_seconds == 0.0
    assert profile.action_distribution_seconds == 0.0
    assert profile.backward_seconds == 0.0
    assert profile.optimizer_step_seconds == 0.0
    assert profile.action_decode_fraction == 0.0
    assert profile.action_trace_batch_count == 0
    assert profile.action_trace_row_count == 0
    assert profile.action_trace_choice_count == 0
    assert profile.action_trace_valid_choice_count == 0
    assert profile.action_trace_padding_choice_count == 0


def test_update_rejects_non_finite_loss_before_optimizer_step() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
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
    assert "PPO action_value_loss must be finite" in result.reason
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
        heads=1,
    )
    train_config = TrainConfig(
        learning_rate=0.0003,
        ppo_epochs=1,
        minibatch_size=1,
    )
    model = PolicyActionModel(
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
    assert (
        "PPO gradients must be finite before clipping" in result.reason
    )
    assert "parameter=_policy_observation_encoder" in result.reason
    assert "nan_count=" in result.reason
    assert "parameter_state=finite" in result.reason
    assert "policy_loss=" in result.reason
    assert "action_value_loss=" in result.reason
    assert "entropy=" in result.reason
    assert "policy_version=0" in result.reason
    assert "epoch=0" in result.reason
    assert "minibatch_step=0" in result.reason
    assert trainer.optimizer_state()["step_count"] == 0
    for index, parameter in enumerate(model.parameters()):
        assert torch.equal(parameter.detach(), before[index])


def test_arena_minibatch_selects_exact_fixed_choice_replay() -> None:
    batch = _single_card_batch(count=3)
    first = batch.select_minibatch(
        indices=torch.tensor((0, 1), dtype=torch.long),
        advantages=batch.raw_advantages,
        global_count=torch.tensor(2, dtype=torch.long),
    )
    assert first.replay is not None
    assert first.replay.active_step_count == 4
    assert first.replay.choice_ids_padded.shape == (2, 2)
    assert first.replay.legal_choice_masks.shape == (
        4,
        ACTION_CHOICE_COUNT,
    )

    second = batch.select_minibatch(
        indices=torch.tensor((2,), dtype=torch.long),
        advantages=batch.raw_advantages,
        global_count=torch.tensor(1, dtype=torch.long),
    )

    assert second.replay is not None
    assert second.replay.active_step_count == 2
    assert second.replay.choice_ids_padded.shape == (1, 2)
    assert second.replay.legal_choice_masks.shape == (
        2,
        ACTION_CHOICE_COUNT,
    )


def _single_card_batch(*, count: int) -> ArenaPPOBatchSource:
    assert count > 0
    device = torch.device("cpu")
    store = ModelRankSampleArena(model_rank_index=0, device=device)
    handles: list[DecisionHandle] = []
    return_values: list[float] = []
    for seat in islice(cycle(seats()), count):
        handle = _store_single_card_decision(
            store=store,
            device=device,
            seat=seat,
        )
        handles.append(handle)
        return_values.append(1.0 if seat in (Seat.A, Seat.C) else -1.0)
    returns = RankReturnTargets(
        policy_version=0,
        model_rank_index=0,
        row_indices=torch.tensor(
            tuple(handle.row_index for handle in handles),
            dtype=torch.long,
        ),
        step_counts=torch.full((len(handles),), 2, dtype=torch.long),
        return_values=torch.tensor(return_values, dtype=torch.float32),
        round_count=1,
        total_step_count=len(handles) * 2,
        max_step_count=2,
    )
    batch_result = store.ppo_batch_source(returns=returns)
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
    assert profile.policy_observation_encode_seconds == 0.0
    assert profile.action_value_observation_encode_seconds == 0.0
    assert profile.action_value_decode_seconds == 0.0
    assert profile.action_decode_seconds == 0.0
    assert profile.action_distribution_seconds == 0.0
    assert profile.backward_seconds == 0.0
    assert profile.optimizer_step_seconds == 0.0
    assert profile.action_decode_fraction == 0.0
    assert profile.action_trace_batch_count == 0
    assert profile.action_trace_row_count == 0
    assert profile.action_trace_choice_count == 0
    assert profile.action_trace_valid_choice_count == 0
    assert profile.action_trace_padding_choice_count == 0


def _store_single_card_decision(
    *,
    store: ModelRankSampleArena,
    device: torch.device,
    seat: Seat,
) -> DecisionHandle:
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        hand=[test_card],
    )
    observation = build_observation(
        viewer=seat,
        snapshot=snapshot,
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )
    legal_actions = build_legal_action_space(
        viewer=seat,
        snapshot=snapshot,
        query=observation.action_query,
    )
    trace = ActionTrace(
        choices=(
            ActionChoice(
                "card",
                FaceCount(CardFace(test_card.suit, test_card.rank), 1),
            ),
            ActionChoice("finish"),
        )
    )
    action_sample = _action_sample_for_trace(
        legal_actions=legal_actions, trace=trace, device=device
    )
    observation_batch = tensorize_observations(
        observations=(observation,),
        device=device,
    )
    stored = store.store_sampled_result(
        policy_versions=(0,),
        observation_batch=observation_batch,
        action_sample=action_sample,
    )
    assert isinstance(stored, Ok)
    assert stored.value.row_count() == 1
    return DecisionHandle(
        model_rank_index=stored.value.model_rank_index,
        policy_version=stored.value.policy_versions[0],
        row_index=stored.value.row_indices[0],
    )


def _action_sample_for_trace(
    *,
    legal_actions: LegalActionSpace,
    trace: ActionTrace,
    device: torch.device,
) -> ActionSampleBatch:
    action_plan = compile_legal_action_frame(legal_actions)
    generation_steps = action_plan_generation_step_count(action_plan)
    batch = plan_batch_to_device((action_plan,), device=device)
    target_choice_ids = tuple(
        action_choice_id(choice) for choice in trace.choices
    )

    logit_decoder: ActionChoiceLogitDecoder = _TraceChoiceDecoder(
        target_choice_ids=target_choice_ids,
        batch_size=1,
        device=device,
    )
    sampler = ActionSampler.create(batch_capacity=1, device=device)
    sample = sampler.sample(
        action_batch=batch,
        generation_step_counts=torch.tensor(
            (generation_steps,), dtype=torch.long, device=device
        ),
        sampling_thresholds=torch.full(
            (1, generation_steps),
            0.5,
            dtype=torch.float64,
            device=device,
        ),
        padded_generation_steps=generation_steps,
        logit_decoder=logit_decoder,
    )
    assert isinstance(sample, Ok)
    return sample.value
