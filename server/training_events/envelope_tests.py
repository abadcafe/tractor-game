"""Black-box tests for the persisted training event envelope."""

from __future__ import annotations

from pydantic import ValidationError

from server.training_events import (
    TRAINING_EVENT_SCHEMA_VERSION,
    TrainingEvent,
    TrainingEventContext,
    TrainingEventProcess,
)


def test_training_event_serializes_exact_current_envelope() -> None:
    event = TrainingEvent(
        schema_version=TRAINING_EVENT_SCHEMA_VERSION,
        event="update",
        recorded_at_ms=11,
        process=TrainingEventProcess(
            kind="coordinator",
            index=None,
            pid=7,
        ),
        context=TrainingEventContext(policy_version=3),
        fields={"total_updates": 4},
    )

    assert TRAINING_EVENT_SCHEMA_VERSION == 3
    assert event.model_dump(mode="json") == {
        "schema_version": 3,
        "event": "update",
        "recorded_at_ms": 11,
        "process": {
            "kind": "coordinator",
            "index": None,
            "pid": 7,
        },
        "context": {"policy_version": 3},
        "fields": {"total_updates": 4},
    }


def test_training_event_rejects_non_current_schema() -> None:
    for schema_version in (2, 4):
        rejected = False
        try:
            TrainingEvent.model_validate(
                {
                    "schema_version": schema_version,
                    "event": "update",
                    "recorded_at_ms": 11,
                    "process": {
                        "kind": "coordinator",
                        "index": None,
                        "pid": 7,
                    },
                    "context": {},
                    "fields": {},
                }
            )
        except ValidationError:
            rejected = True
        assert rejected


def test_training_event_rejects_unknown_envelope_field() -> None:
    rejected = False
    try:
        TrainingEvent.model_validate(
            {
                "schema_version": 3,
                "event": "update",
                "recorded_at_ms": 11,
                "process": {
                    "kind": "coordinator",
                    "index": None,
                    "pid": 7,
                },
                "context": {},
                "fields": {},
                "outcome": "finished",
            }
        )
    except ValidationError:
        rejected = True
    assert rejected


def test_training_event_rejects_missing_required_envelope_fields() -> (
    None
):
    base_process: dict[str, object] = {
        "kind": "coordinator",
        "index": None,
        "pid": 7,
    }
    base: dict[str, object] = {
        "schema_version": 3,
        "event": "update",
        "recorded_at_ms": 11,
        "process": base_process,
        "context": {},
        "fields": {},
    }
    for root_field, process_field in (
        ("schema_version", None),
        (None, "index"),
    ):
        payload = dict(base)
        if root_field is not None:
            del payload[root_field]
        if process_field is not None:
            process = dict(base_process)
            del process[process_field]
            payload["process"] = process
        rejected = False
        try:
            TrainingEvent.model_validate(payload)
        except ValidationError:
            rejected = True
        assert rejected
