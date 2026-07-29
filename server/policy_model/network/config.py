"""Shape configuration for the Tractor policy model."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.json_value import JsonObject

MIN_ATTENTION_HEAD_DIMENSION: int = 8


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Transformer shape configuration persisted by checkpoints."""

    d_model: int = 128
    layers: int = 3
    heads: int = 4
    action_value_layers: int = 2

    def __post_init__(self) -> None:
        assert self.d_model > 0
        assert self.layers > 0
        assert self.heads > 0
        assert self.action_value_layers > 0
        assert self.d_model % self.heads == 0
        assert (
            self.d_model // self.heads >= MIN_ATTENTION_HEAD_DIMENSION
        )

    def to_json(self) -> JsonObject:
        return {
            "d_model": self.d_model,
            "layers": self.layers,
            "heads": self.heads,
            "action_value_layers": self.action_value_layers,
        }

    @classmethod
    def from_json(cls, data: JsonObject) -> ModelConfig:
        assert set(data) == {
            "d_model",
            "layers",
            "heads",
            "action_value_layers",
        }
        return cls(
            d_model=_int_json_field(data, "d_model"),
            layers=_int_json_field(data, "layers"),
            heads=_int_json_field(data, "heads"),
            action_value_layers=_int_json_field(
                data, "action_value_layers"
            ),
        )


def _int_json_field(data: JsonObject, field: str) -> int:
    value = data[field]
    assert isinstance(value, int)
    return value


__all__ = ("MIN_ATTENTION_HEAD_DIMENSION", "ModelConfig")
