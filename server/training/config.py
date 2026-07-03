"""Training and model configuration records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.training.json_types import JsonObject

type TrainingDevice = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Transformer shape configuration shared by checkpoints."""

    d_model: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    max_tokens: int = 768

    def to_json(self) -> JsonObject:
        return {
            "d_model": self.d_model,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_json(cls, data: JsonObject) -> ModelConfig:
        return cls(
            d_model=_int_json_field(data, "d_model"),
            layers=_int_json_field(data, "layers"),
            heads=_int_json_field(data, "heads"),
            dropout=_float_json_field(data, "dropout"),
            max_tokens=_int_json_field(data, "max_tokens"),
        )


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Run configuration that can move between machines."""

    device: TrainingDevice = "cpu"
    learning_rate: float = 0.0003
    checkpoint_every_updates: int = 50
    max_round_seconds: float = 120.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 64
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0

    def to_json(self) -> JsonObject:
        return {
            "device": self.device,
            "learning_rate": self.learning_rate,
            "checkpoint_every_updates": self.checkpoint_every_updates,
            "max_round_seconds": self.max_round_seconds,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ppo_clip": self.ppo_clip,
            "value_clip": self.value_clip,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_size": self.minibatch_size,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "weight_decay": self.weight_decay,
        }

    @classmethod
    def from_json(cls, data: JsonObject) -> TrainConfig:
        return cls(
            device=_device_json_field(data, "device"),
            learning_rate=_float_json_field(data, "learning_rate"),
            checkpoint_every_updates=_int_json_field(
                data, "checkpoint_every_updates"
            ),
            max_round_seconds=_float_json_field(
                data, "max_round_seconds"
            ),
            gamma=_float_json_field(data, "gamma"),
            gae_lambda=_float_json_field(data, "gae_lambda"),
            ppo_clip=_float_json_field(data, "ppo_clip"),
            value_clip=_float_json_field(data, "value_clip"),
            entropy_coef=_float_json_field(data, "entropy_coef"),
            value_coef=_float_json_field(data, "value_coef"),
            max_grad_norm=_float_json_field(data, "max_grad_norm"),
            ppo_epochs=_int_json_field(data, "ppo_epochs"),
            minibatch_size=_int_json_field(data, "minibatch_size"),
            adam_beta1=_float_json_field(data, "adam_beta1"),
            adam_beta2=_float_json_field(data, "adam_beta2"),
            weight_decay=_float_json_field(data, "weight_decay"),
        )


def _int_json_field(data: JsonObject, field: str) -> int:
    value = data[field]
    assert isinstance(value, int)
    return value


def _float_json_field(data: JsonObject, field: str) -> float:
    value = data[field]
    assert isinstance(value, int | float)
    return float(value)


def _device_json_field(data: JsonObject, field: str) -> TrainingDevice:
    value = data[field]
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        return "cuda"
    assert False
