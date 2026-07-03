"""Tests for PPO trainer updates."""

from __future__ import annotations

import torch

from server.player.test_helpers import card, make_snapshot
from server.result import Ok
from server.rules.card_faces import CardFace, FaceCount
from server.training.config import ModelConfig, TrainConfig
from server.training.legal_actions import build_legal_action_index
from server.training.model import TractorPolicyModel
from server.training.observation import build_observation
from server.training.ppo import PPOTrainer
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.trajectory import (
    DecisionStep,
    RewardedDecisionStep,
)


def test_update_returns_stats_and_adamw_state() -> None:
    device = torch.device("cpu")
    model_config = ModelConfig(
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
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
        dropout=model_config.dropout,
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

    stats = trainer.update(
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
    state = trainer.optimizer_state()

    assert stats.total_loss >= 0.0
    assert state["kind"] == "typed_adamw"
    assert state["step_count"] == 1
