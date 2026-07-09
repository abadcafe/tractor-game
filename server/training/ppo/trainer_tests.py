"""Tests for PPO trainer updates."""

from __future__ import annotations

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
    ArgumentTraceScores,
    ObservationEncoding,
    TractorPolicyModel,
)
from server.training.observation import build_observation
from server.training.policy_sampling import (
    DecisionHandle,
    RankReturnTargets,
    SampledPolicyBatch,
)
from server.training.policy_sampling.model_rank_sample_arena import (
    ModelRankSampleArena,
)
from server.training.ppo import (
    PPOBatchSource,
    PPOTrainer,
    PPOUpdateInput,
    PPOUpdateProfile,
)
from server.training.ppo.distributed import PPOUpdatePartition
from server.training.semantic_action_plan import (
    SemanticActionSampleBatch,
    SemanticActionSampler,
    SemanticArgumentLogitDecoder,
    action_plan_generation_step_count,
    compile_legal_action_frame,
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

    def score_argument_traces(
        self,
        encoding: ObservationEncoding,
        *,
        selected_token_ids_padded: Tensor,
        step_counts: Tensor,
    ) -> ArgumentTraceScores:
        self.training_modes.append(self.training)
        self.score_batch_sizes.append(
            int(selected_token_ids_padded.shape[0])
        )
        self.score_prefix_widths.append(
            int(selected_token_ids_padded.shape[1])
        )
        return super().score_argument_traces(
            encoding,
            selected_token_ids_padded=selected_token_ids_padded,
            step_counts=step_counts,
        )


class NonFiniteValueModel(TractorPolicyModel):
    """Policy model that produces an infinite value loss."""

    def value_estimates(
        self,
        encoding: ObservationEncoding,
    ) -> Tensor:
        values = super().value_estimates(encoding)
        return torch.full_like(values, torch.inf)


class _TraceTokenDecoder:
    def __init__(
        self,
        *,
        target_token_ids: tuple[int, ...],
        batch_size: int,
        device: torch.device,
    ) -> None:
        self._target_token_ids = target_token_ids
        self._batch_size = batch_size
        self._device = device
        self._step_index = 0

    def next_logits(self) -> Tensor:
        logits = torch.zeros(
            (self._batch_size, SEMANTIC_CODEC.argument_vocab_size),
            dtype=torch.float32,
            device=self._device,
        )
        if self._step_index < len(self._target_token_ids):
            logits[:, self._target_token_ids[self._step_index]] = 100.0
        return logits

    def advance(self, selected_token_ids: Tensor) -> None:
        assert selected_token_ids.shape == (self._batch_size,)
        self._step_index += 1


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
    assert model.score_batch_sizes == [4]
    assert model.score_prefix_widths == [2]
    assert profile.update_seconds > 0.0
    assert profile.argument_decode_seconds >= 0.0
    assert 0.0 <= profile.argument_decode_fraction <= 1.0
    assert profile.argument_trace_batch_count == 1
    assert profile.argument_trace_row_count == 4
    assert profile.argument_trace_token_count == 8
    assert profile.argument_trace_valid_token_count == 8
    assert profile.argument_trace_padding_token_count == 0
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
    assert model.score_batch_sizes == [4]


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
    assert profile.argument_decode_seconds == 0.0
    assert profile.argument_distribution_seconds == 0.0
    assert profile.backward_seconds == 0.0
    assert profile.optimizer_step_seconds == 0.0
    assert profile.argument_decode_fraction == 0.0
    assert profile.argument_trace_batch_count == 0
    assert profile.argument_trace_row_count == 0
    assert profile.argument_trace_token_count == 0
    assert profile.argument_trace_valid_token_count == 0
    assert profile.argument_trace_padding_token_count == 0


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


def _single_card_batch(*, count: int) -> PPOBatchSource:
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
    assert profile.observation_encode_seconds == 0.0
    assert profile.value_head_seconds == 0.0
    assert profile.argument_select_seconds == 0.0
    assert profile.argument_decode_seconds == 0.0
    assert profile.argument_distribution_seconds == 0.0
    assert profile.backward_seconds == 0.0
    assert profile.optimizer_step_seconds == 0.0
    assert profile.argument_decode_fraction == 0.0
    assert profile.argument_trace_batch_count == 0
    assert profile.argument_trace_row_count == 0
    assert profile.argument_trace_token_count == 0
    assert profile.argument_trace_valid_token_count == 0
    assert profile.argument_trace_padding_token_count == 0


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
    semantic_sample = _semantic_sample_for_trace(
        legal_actions=legal_actions, trace=trace, device=device
    )
    sample_batch = _sampled_policy_batch(
        observation_batch=tensorize_observations(
            observations=(observation,),
            max_observation_tokens=64,
            device=device,
        ),
        semantic_sample=semantic_sample,
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
    semantic_sample: SemanticActionSampleBatch,
    device: torch.device,
) -> SampledPolicyBatch:
    return SampledPolicyBatch(
        policy_versions=(0,),
        status_codes=torch.zeros((1,), dtype=torch.long, device=device),
        observation_batch=observation_batch,
        selected_token_ids_padded=(
            semantic_sample.selected_token_ids_padded
        ),
        choice_token_ids=semantic_sample.choice_token_ids,
        choice_masks=semantic_sample.choice_masks,
        selected_choice_offsets=semantic_sample.selected_choice_offsets,
        step_counts=semantic_sample.step_counts,
        choice_counts=semantic_sample.choice_counts,
        old_log_probabilities=torch.zeros(
            (1,), dtype=torch.float32, device=device
        ),
        old_values=torch.zeros(
            (1,), dtype=torch.float32, device=device
        ),
    )


def _semantic_sample_for_trace(
    *,
    legal_actions: LegalActionIndex,
    trace: SemanticArgumentTrace,
    device: torch.device,
) -> SemanticActionSampleBatch:
    action_plan = compile_legal_action_frame(legal_actions)
    generation_steps = action_plan_generation_step_count(action_plan)
    batch = plan_batch_to_device((action_plan,), device=device)
    target_token_ids = tuple(
        semantic_argument_id(argument) for argument in trace.arguments
    )

    logit_decoder: SemanticArgumentLogitDecoder = _TraceTokenDecoder(
        target_token_ids=target_token_ids,
        batch_size=1,
        device=device,
    )
    sampler = SemanticActionSampler.create(
        batch_capacity=1, device=device
    )
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
