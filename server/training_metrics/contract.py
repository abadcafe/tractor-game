"""Exact current wire contract for projected training metrics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from server.foundation.json_value import JsonObject

type TrainingMetricsSchemaVersion = Literal[4]
TRAINING_METRICS_SCHEMA_VERSION: TrainingMetricsSchemaVersion = 4


class MetricPoint(BaseModel):
    """Chart point keyed by committed log sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(gt=0)
    update: int | None = Field(default=None, ge=0)
    elapsed_seconds: float = Field(ge=0.0)
    recorded_at_ms: int = Field(ge=0)
    values: JsonObject


class MetricDatasets(BaseModel):
    """Chart-oriented metric series."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    throughput: tuple[MetricPoint, ...]
    optimization: tuple[MetricPoint, ...]
    ppo_timing: tuple[MetricPoint, ...]
    rollout: tuple[MetricPoint, ...]
    rewards: tuple[MetricPoint, ...]
    inference: tuple[MetricPoint, ...]
    processes: tuple[MetricPoint, ...]


class TrainingMetrics(BaseModel):
    """Consistent full-run read snapshot consumed by Metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: TrainingMetricsSchemaVersion
    store_id: str | None
    through_sequence: int = Field(ge=0)
    complete: bool
    dropped_event_count: int = Field(ge=0)
    totals: JsonObject
    datasets: MetricDatasets


class TrainingMetricSummary(BaseModel):
    """Latest scalar metrics consumed by one-shot status commands."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    store_id: str | None
    through_sequence: int = Field(ge=0)
    complete: bool
    dropped_event_count: int = Field(ge=0)
    totals: JsonObject


class MetricsCursor(BaseModel):
    """Store-aware cursor for Metrics-relevant persisted events."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    store_id: str | None
    through_sequence: int = Field(ge=0)


__all__ = (
    "TRAINING_METRICS_SCHEMA_VERSION",
    "MetricDatasets",
    "MetricPoint",
    "MetricsCursor",
    "TrainingMetricSummary",
    "TrainingMetrics",
    "TrainingMetricsSchemaVersion",
)
