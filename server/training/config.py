"""Optimizer and checkpoint scheduling configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from server.foundation.json_value import JsonObject


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Run configuration that can move between machines."""

    seed: int = 0
    learning_rate: float = 0.0003
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

    def __post_init__(self) -> None:
        assert self.seed >= 0
        assert _is_finite(self.learning_rate)
        assert self.learning_rate > 0.0
        assert _is_finite(self.ppo_clip)
        assert 0.0 < self.ppo_clip <= 1.0
        assert _is_finite(self.value_clip)
        assert self.value_clip > 0.0
        assert _is_finite(self.entropy_coef)
        assert self.entropy_coef >= 0.0
        assert _is_finite(self.value_coef)
        assert self.value_coef >= 0.0
        assert _is_finite(self.max_grad_norm)
        assert self.max_grad_norm >= 0.0
        assert self.ppo_epochs > 0
        assert self.minibatch_size > 0
        assert _is_finite(self.adam_beta1)
        assert 0.0 <= self.adam_beta1 < 1.0
        assert _is_finite(self.adam_beta2)
        assert 0.0 <= self.adam_beta2 < 1.0
        assert _is_finite(self.weight_decay)
        assert self.weight_decay >= 0.0

    def to_json(self) -> JsonObject:
        return {
            "seed": self.seed,
            "learning_rate": self.learning_rate,
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
            seed=_int_json_field(data, "seed"),
            learning_rate=_float_json_field(data, "learning_rate"),
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


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Checkpoint cadence and retention for one resume process."""

    every_updates: int
    retention_updates: int = 5

    def __post_init__(self) -> None:
        assert self.every_updates > 0
        assert self.retention_updates >= 0


def _int_json_field(data: JsonObject, field: str) -> int:
    value = data[field]
    assert isinstance(value, int)
    return value


def _float_json_field(data: JsonObject, field: str) -> float:
    value = data[field]
    assert isinstance(value, int | float)
    return float(value)


def _is_finite(value: float) -> bool:
    return math.isfinite(value)
