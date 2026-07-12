"""Black-box tests for portable training run initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import TypeAdapter

from server.foundation.json_value import JsonObject
from server.foundation.result import Ok, Rejected
from server.training.config import ModelConfig, TrainConfig
from server.training.persistence.schema import database_path
from server.training.run_setup import initialize_training_run
from server.training.torch_checkpoints.load import (
    read_torch_checkpoint_metadata,
)

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def test_initialize_training_run_writes_zero_update_state(
    tmp_path: Path,
) -> None:
    prepared = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(),
    )

    assert isinstance(prepared, Ok)
    checkpoint_path = tmp_path / "checkpoints" / "latest.json"
    assert prepared.value.checkpoint_path == checkpoint_path
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert isinstance(metadata, Ok)
    assert metadata.value.model_config == ModelConfig(d_model=128)
    assert metadata.value.train_config == TrainConfig()
    assert metadata.value.total_rounds == 0
    assert metadata.value.total_updates == 0
    assert [item["event"] for item in _event_documents(tmp_path)] == [
        "run.initialized"
    ]


def test_initialize_training_run_rejects_existing_run(
    tmp_path: Path,
) -> None:
    first = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(seed=1),
    )
    assert isinstance(first, Ok)

    second = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=64),
        train_config=TrainConfig(seed=2),
    )

    assert isinstance(second, Rejected)
    assert "training run directory is not empty" in second.reason
    metadata = read_torch_checkpoint_metadata(
        first.value.checkpoint_path
    )
    assert isinstance(metadata, Ok)
    assert metadata.value.model_config == ModelConfig(d_model=128)


def test_initialize_training_run_replace_existing_rebuilds_state(
    tmp_path: Path,
) -> None:
    first = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(seed=1),
    )
    assert isinstance(first, Ok)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    (tmp_path / "stdout.log").write_text("old stdout", encoding="utf-8")
    (tmp_path / "stderr.log").write_text("old stderr", encoding="utf-8")
    (tmp_path / "training.pid").write_text("123\n", encoding="ascii")
    stale_directory = tmp_path / "runtime" / "nested"
    stale_directory.mkdir(parents=True)
    (stale_directory / "state").write_text("old", encoding="utf-8")
    (tmp_path / "outside-link").symlink_to(outside)

    replaced = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=64),
        train_config=TrainConfig(seed=2),
        replace_existing=True,
    )

    assert isinstance(replaced, Ok)
    metadata = read_torch_checkpoint_metadata(
        replaced.value.checkpoint_path
    )
    assert isinstance(metadata, Ok)
    assert metadata.value.model_config == ModelConfig(d_model=64)
    assert metadata.value.train_config == TrainConfig(seed=2)
    assert len(_event_documents(tmp_path)) == 1
    assert outside.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "stdout.log").exists()
    assert not (tmp_path / "stderr.log").exists()
    assert not (tmp_path / "training.pid").exists()
    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path / "outside-link").exists()


def test_initialize_training_run_rejects_unrelated_directory_contents(
    tmp_path: Path,
) -> None:
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("preserve", encoding="utf-8")

    prepared = initialize_training_run(
        run_dir=tmp_path,
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(),
    )

    assert isinstance(prepared, Rejected)
    assert "training run directory is not empty" in prepared.reason
    assert unrelated.read_text(encoding="utf-8") == "preserve"


def _event_documents(run_dir: Path) -> tuple[JsonObject, ...]:
    with sqlite3.connect(database_path(run_dir)) as connection:
        rows = connection.execute(
            "SELECT event_json FROM training_logs ORDER BY sequence"
        ).fetchall()
    documents: list[JsonObject] = []
    for row in rows:
        value = row[0]
        assert isinstance(value, str)
        decoded = _JSON_OBJECT_ADAPTER.validate_json(value)
        documents.append(decoded)
    return tuple(documents)
