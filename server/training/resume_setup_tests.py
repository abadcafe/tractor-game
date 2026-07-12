"""Black-box tests for canonical resume timeline preparation."""

from __future__ import annotations

from pathlib import Path

import torch

from server.foundation.result import Ok
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import append_metric, read_metrics
from server.training.resume_setup import canonicalize_resume_timeline
from server.training.run_setup import initialize_training_run
from server.training.runtime.config import ExecutionConfig
from server.training.telemetry import (
    SqliteTelemetrySink,
    TelemetryEvent,
    read_telemetry_records,
)
from server.training.torch_checkpoints.load import (
    load_torch_checkpoint,
    read_torch_checkpoint_metadata,
)
from server.training.torch_checkpoints.save import save_torch_checkpoint


def test_canonicalize_resume_timeline_removes_future_state(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        d_model=2, layers=1, heads=1, max_tokens=512
    )
    train_config = TrainConfig()
    initialized = initialize_training_run(
        run_dir=tmp_path,
        model_config=model_config,
        train_config=train_config,
    )
    assert isinstance(initialized, Ok)
    loaded = load_torch_checkpoint(
        path=initialized.value.checkpoint_path,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
        device=torch.device("cpu"),
    )
    assert isinstance(loaded, Ok)
    update_one = tmp_path / "checkpoints" / "update-1.json"
    update_two = tmp_path / "checkpoints" / "update-2.json"
    for update, manifest_path in ((1, update_one), (2, update_two)):
        saved = save_torch_checkpoint(
            manifest_paths=(
                manifest_path,
                tmp_path / "checkpoints" / "latest.json",
            ),
            model=loaded.value.model,
            trainer=loaded.value.trainer,
            model_config=model_config,
            train_config=train_config,
            total_rounds=update * 10,
            total_samples=update * 100,
            total_updates=update,
            retained_update_count=5,
        )
        assert isinstance(saved, Ok)
    metrics = read_metrics(tmp_path)
    assert isinstance(metrics, Ok)
    initial_metric = metrics.value[0]
    for update in (1, 2):
        appended = append_metric(
            tmp_path,
            initial_metric.model_copy(
                update={
                    "total_games": update * 10,
                    "total_samples": update * 100,
                    "total_updates": update,
                }
            ),
        )
        assert isinstance(appended, Ok)
    telemetry = SqliteTelemetrySink(tmp_path).append(
        TelemetryEvent(
            process_label="coordinator",
            stage="rollout",
            total_rounds=20,
            total_updates=2,
            progress_numerator=1,
            progress_denominator=1,
            unix_seconds=1.0,
        )
    )
    assert isinstance(telemetry, Ok)

    result = canonicalize_resume_timeline(
        run_dir=tmp_path,
        selected_checkpoint=update_one,
    )

    assert isinstance(result, Ok)
    latest = read_torch_checkpoint_metadata(
        tmp_path / "checkpoints" / "latest.json"
    )
    assert isinstance(latest, Ok)
    assert latest.value.total_updates == 1
    assert update_one.exists()
    assert not update_two.exists()
    reconciled_metrics = read_metrics(tmp_path)
    assert isinstance(reconciled_metrics, Ok)
    assert [
        item.total_updates for item in reconciled_metrics.value
    ] == [
        0,
        1,
    ]
    reconciled_telemetry = read_telemetry_records(tmp_path)
    assert isinstance(reconciled_telemetry, Ok)
    assert reconciled_telemetry.value == ()
