"""Exact persisted and wire envelope for current training events."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from server.foundation.json_value import JsonObject
from server.training_events.contract import (
    EventName,
    EventSeat,
    ProcessKind,
)

type TrainingEventSchemaVersion = Literal[4]
TRAINING_EVENT_SCHEMA_VERSION: TrainingEventSchemaVersion = 4


def _is_none(value: object) -> bool:
    return value is None


class TrainingEventProcess(BaseModel):
    """Exact process identity persisted with one event."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    kind: ProcessKind
    index: int | None = Field(ge=0)
    pid: int = Field(gt=0)


class TrainingEventContext(BaseModel):
    """Exact optional correlation dimensions persisted with an event."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    policy_version: int | None = Field(
        default=None,
        ge=0,
        exclude_if=_is_none,
    )
    rollout_id: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=_is_none,
    )
    worker_index: int | None = Field(
        default=None,
        ge=0,
        exclude_if=_is_none,
    )
    model_rank_index: int | None = Field(
        default=None,
        ge=0,
        exclude_if=_is_none,
    )
    game_env_index: int | None = Field(
        default=None,
        ge=0,
        exclude_if=_is_none,
    )
    round_id: int | None = Field(
        default=None,
        ge=0,
        exclude_if=_is_none,
    )
    seat: EventSeat | None = Field(
        default=None,
        exclude_if=_is_none,
    )
    decision_index: int | None = Field(
        default=None,
        ge=0,
        exclude_if=_is_none,
    )
    request_id: int | None = Field(
        default=None,
        ge=0,
        exclude_if=_is_none,
    )
    batch_id: int | None = Field(
        default=None,
        ge=0,
        exclude_if=_is_none,
    )

    @field_validator("rollout_id")
    @classmethod
    def _validate_rollout_id(cls, value: str | None) -> str | None:
        if value is not None and value.strip() != value:
            raise ValueError(
                "rollout_id must not have outer whitespace"
            )
        return value


class TrainingEvent(BaseModel):
    """One exact current-version event shared by storage and HTTP."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    schema_version: TrainingEventSchemaVersion
    event: EventName
    recorded_at_ms: int = Field(ge=0)
    process: TrainingEventProcess
    context: TrainingEventContext
    fields: JsonObject
    error: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=_is_none,
    )

    @field_validator("error")
    @classmethod
    def _validate_error(cls, value: str | None) -> str | None:
        if value is not None and value.strip() != value:
            raise ValueError("error must not have outer whitespace")
        return value


__all__ = (
    "TRAINING_EVENT_SCHEMA_VERSION",
    "TrainingEvent",
    "TrainingEventContext",
    "TrainingEventProcess",
    "TrainingEventSchemaVersion",
)
