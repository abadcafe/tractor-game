"""Black-box tests for the current Metrics wire contract."""

from pydantic import ValidationError

from server.training_metrics import (
    TRAINING_METRICS_SCHEMA_VERSION,
    MetricDatasets,
    TrainingMetrics,
)


def test_training_metrics_owns_independent_current_schema() -> None:
    snapshot = TrainingMetrics(
        schema_version=TRAINING_METRICS_SCHEMA_VERSION,
        store_id=None,
        through_sequence=0,
        complete=True,
        dropped_event_count=0,
        totals={},
        datasets=MetricDatasets(
            throughput=(),
            optimization=(),
            ppo_timing=(),
            rounds=(),
            rewards=(),
            inference=(),
            processes=(),
        ),
    )

    assert TRAINING_METRICS_SCHEMA_VERSION == 6
    assert snapshot.schema_version == 6


def test_training_metrics_rejects_missing_or_non_current_schema() -> (
    None
):
    base: dict[str, object] = {
        "schema_version": 6,
        "store_id": None,
        "through_sequence": 0,
        "complete": True,
        "dropped_event_count": 0,
        "totals": {},
        "datasets": {
            "throughput": [],
            "optimization": [],
            "ppo_timing": [],
            "rounds": [],
            "rewards": [],
            "inference": [],
            "processes": [],
        },
    }
    payloads: tuple[dict[str, object], ...] = (
        {
            key: value
            for key, value in base.items()
            if key != "schema_version"
        },
        {**base, "schema_version": 5},
        {**base, "schema_version": 7},
    )
    for payload in payloads:
        rejected = False
        try:
            _ = TrainingMetrics.model_validate(payload)
        except ValidationError:
            rejected = True
        assert rejected
