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

    def __post_init__(self) -> None:
        assert self.d_model > 0
        assert self.layers > 0
        assert self.heads > 0
        assert self.d_model % self.heads == 0
        assert (
            self.d_model // self.heads >= MIN_ATTENTION_HEAD_DIMENSION
        )

    def to_json(self) -> JsonObject:
        return {
            "d_model": self.d_model,
            "layers": self.layers,
            "heads": self.heads,
        }

    @classmethod
    def from_json(cls, data: JsonObject) -> ModelConfig:
        assert set(data) == {"d_model", "layers", "heads"}
        return cls(
            d_model=_int_json_field(data, "d_model"),
            layers=_int_json_field(data, "layers"),
            heads=_int_json_field(data, "heads"),
        )


def _int_json_field(data: JsonObject, field: str) -> int:
    value = data[field]
    assert isinstance(value, int)
    return value


__all__ = ("MIN_ATTENTION_HEAD_DIMENSION", "ModelConfig")
