"""Tests for PPO trainer updates."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from server.player.test_helpers import card, make_snapshot
from server.result import Ok, Rejected
from server.rules.card_faces import CardFace, FaceCount
from server.sm.constants import get_team_index
from server.training.choice_trace import (
    SemanticChoiceStep,
    SemanticChoiceTrace,
    semantic_choice_step_from_argument,
)
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
from server.training.ppo import PPOTrainer, PPOUpdateProfile
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
)
from server.training.tensorize import (
    ArgumentPrefixTensorBatch,
    ObservationTensorBatch,
    tensorize_observation,
)
from server.training.trajectory import (
    DecisionStep,
    DecisionTransition,
    RolloutBatch,
    TeamTrajectory,
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
        device="cpu",
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
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    test_card = card("spades", "A", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    observation = build_observation(
        player_index=0,
        snapshot=snapshot,
        history=(),
    )
    legal_actions = build_legal_action_index(
        player_index=0,
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
    decoded = legal_actions.decode(trace)
    assert isinstance(decoded, Ok)

    stats_result = trainer.update(
        RolloutBatch(
            trajectories=(
                TeamTrajectory(
                    team_index=0,
                    terminal_reward=1.0,
                    transitions=(
                        DecisionTransition(
                            decision=DecisionStep(
                                player_index=0,
                                seq=1,
                                observation_batch=tensorize_observation(
                                    observation=observation,
                                    max_observation_tokens=(
                                        model_config.max_tokens
                                    ),
                                    device=device,
                                ),
                                choice_trace=_choice_trace_for(
                                    legal_actions=legal_actions,
                                    trace=trace,
                                ),
                                action=decoded.value,
                                log_probability=0.0,
                                value_estimate=0.0,
                                entropy=0.0,
                                choice_count=2,
                            ),
                            reward_after_step=0.0,
                        ),
                    ),
                ),
            ),
        )
    )
    assert isinstance(stats_result, Ok)
    stats = stats_result.value
    state = trainer.optimizer_state()

    assert stats.total_loss >= 0.0
    assert state["kind"] == "typed_adamw"
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
        device="cpu",
        ppo_profile="detailed",
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
        model_config=model_config,
        train_config=train_config,
        device=device,
    )

    update_result = trainer.update(_single_card_rollout_batch(count=4))
    assert isinstance(update_result, Ok)
    profile = update_result.value.profile

    assert model.encode_batch_sizes == [4]
    assert model.score_batch_sizes == [4, 4]
    assert model.score_prefix_widths == [1, 2]
    assert profile.update_seconds > 0.0
    assert profile.argument_decode_seconds >= 0.0
    assert 0.0 <= profile.argument_decode_fraction <= 1.0
    assert profile.argument_prefix_batch_count == 2
    assert profile.argument_prefix_row_count == 8
    assert profile.argument_prefix_token_count == 12
    assert profile.argument_prefix_valid_token_count == 12
    assert profile.argument_prefix_padding_token_count == 0
    assert model.training_modes
    assert all(training is True for training in model.training_modes)
    assert model.training is True


def test_update_disables_profile_by_default() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        max_tokens=64,
    )
    train_config = TrainConfig(
        device="cpu",
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
        model_config=model_config,
        train_config=train_config,
        device=device,
    )

    update_result = trainer.update(_single_card_rollout_batch(count=4))

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
        device="cpu",
        ppo_profile="basic",
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
        model_config=model_config,
        train_config=train_config,
        device=device,
    )

    update_result = trainer.update(_single_card_rollout_batch(count=4))

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
        device="cpu",
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
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )

    result = trainer.update(_single_card_rollout_batch(count=1))

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
        device="cpu",
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
        model_config=model_config,
        train_config=train_config,
        device=device,
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

    result = trainer.update(_single_card_rollout_batch(count=1))

    assert isinstance(result, Rejected)
    assert "PPO gradients must be finite" in result.reason
    assert trainer.optimizer_state()["step_count"] == 0
    for index, parameter in enumerate(model.parameters()):
        assert torch.equal(parameter.detach(), before[index])


def _single_card_rollout_batch(*, count: int) -> RolloutBatch:
    assert count > 0
    transitions = tuple(
        _single_card_transition(
            player_index=index % 4,
        )
        for index in range(count)
    )
    team_trajectories: list[TeamTrajectory] = []
    for team_index in (0, 1):
        team_transitions = tuple(
            transition
            for transition in transitions
            if get_team_index(transition.decision.player_index)
            == team_index
        )
        if team_transitions:
            team_trajectories.append(
                TeamTrajectory(
                    team_index=team_index,
                    terminal_reward=1.0 if team_index == 0 else -1.0,
                    transitions=team_transitions,
                )
            )
    return RolloutBatch(trajectories=tuple(team_trajectories))


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


def _single_card_transition(
    player_index: int,
) -> DecisionTransition:
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
    decoded = legal_actions.decode(trace)
    assert isinstance(decoded, Ok)
    return DecisionTransition(
        decision=DecisionStep(
            player_index=player_index,
            seq=1,
            observation_batch=tensorize_observation(
                observation=observation,
                max_observation_tokens=64,
                device=torch.device("cpu"),
            ),
            choice_trace=_choice_trace_for(
                legal_actions=legal_actions,
                trace=trace,
            ),
            action=decoded.value,
            log_probability=0.0,
            value_estimate=0.0,
            entropy=0.0,
            choice_count=2,
        ),
        reward_after_step=0.0,
    )


def _choice_trace_for(
    *,
    legal_actions: LegalActionIndex,
    trace: SemanticArgumentTrace,
) -> SemanticChoiceTrace:
    prefix = SemanticArgumentPrefix(arguments=())
    steps: list[SemanticChoiceStep] = []
    for argument in trace.arguments:
        allowed = legal_actions.allowed_next(prefix)
        steps.append(
            semantic_choice_step_from_argument(
                allowed=allowed,
                selected_argument=argument,
            )
        )
        if argument.kind in ("pass", "stop"):
            continue
        prefix = SemanticArgumentPrefix(
            arguments=(*prefix.arguments, argument)
        )
    return SemanticChoiceTrace(steps=tuple(steps))
