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
    max_round_seconds: float = 30.0

    def to_json(self) -> JsonObject:
        return {
            "device": self.device,
            "learning_rate": self.learning_rate,
            "checkpoint_every_updates": self.checkpoint_every_updates,
            "max_round_seconds": self.max_round_seconds,
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
