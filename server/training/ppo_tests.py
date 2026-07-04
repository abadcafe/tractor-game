"""Tests for PPO trainer updates."""

from __future__ import annotations

import pytest
import torch

from server.player.test_helpers import card, make_snapshot
from server.result import Ok, Rejected
from server.rules.card_faces import CardFace, FaceCount
from server.training.config import ModelConfig, TrainConfig
from server.training.legal_actions import build_legal_action_index
from server.training.model import ArgumentHeadOutput, TractorPolicyModel
from server.training.observation import build_observation
from server.training.ppo import PPOTrainer
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.tensorize import (
    ArgumentPrefixTensorBatch,
    ObservationTensorBatch,
)
from server.training.trajectory import (
    DecisionStep,
    RewardedDecisionStep,
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
        self.batch_sizes: list[int] = []
        self.training_modes: list[bool] = []

    def forward_argument(
        self,
        observation: ObservationTensorBatch,
        prefix: ArgumentPrefixTensorBatch,
    ) -> ArgumentHeadOutput:
        self.training_modes.append(self.training)
        self.batch_sizes.append(
            int(observation.token_type_ids.shape[0])
        )
        return super().forward_argument(observation, prefix)


class NonFiniteValueModel(TractorPolicyModel):
    """Policy model that produces an infinite value loss."""

    def forward_argument(
        self,
        observation: ObservationTensorBatch,
        prefix: ArgumentPrefixTensorBatch,
    ) -> ArgumentHeadOutput:
        output = super().forward_argument(observation, prefix)
        return ArgumentHeadOutput(
            argument_logits=output.argument_logits,
            values=torch.full_like(output.values, torch.inf),
        )


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
        (
            RewardedDecisionStep(
                step=DecisionStep(
                    player_index=0,
                    seq=1,
                    observation=observation,
                    action_query=observation.action_query,
                    legal_actions=legal_actions,
                    action=decoded.value,
                    log_probability=0.0,
                    value_estimate=0.0,
                    entropy=0.0,
                    choice_count=2,
                ),
                reward=1.0,
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

    update_result = trainer.update(
        tuple(
            _single_card_rewarded_step(player_index=index % 4)
            for index in range(4)
        )
    )
    assert isinstance(update_result, Ok)

    assert max(model.batch_sizes) == 4
    assert model.training_modes
    assert all(training is True for training in model.training_modes)
    assert model.training is True


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

    result = trainer.update((_single_card_rewarded_step(0),))

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

    result = trainer.update((_single_card_rewarded_step(0),))

    assert isinstance(result, Rejected)
    assert "PPO gradients must be finite" in result.reason
    assert trainer.optimizer_state()["step_count"] == 0
    for index, parameter in enumerate(model.parameters()):
        assert torch.equal(parameter.detach(), before[index])


def _single_card_rewarded_step(
    player_index: int,
) -> RewardedDecisionStep:
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
    return RewardedDecisionStep(
        step=DecisionStep(
            player_index=player_index,
            seq=1,
            observation=observation,
            action_query=observation.action_query,
            legal_actions=legal_actions,
            action=decoded.value,
            log_probability=0.0,
            value_estimate=0.0,
            entropy=0.0,
            choice_count=2,
        ),
        reward=1.0,
    )
