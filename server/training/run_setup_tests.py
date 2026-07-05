"""Tests for training run initialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from server.result import Ok, Rejected
from server.training import run_setup
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import read_metrics
from server.training.run_setup import (
    initialize_training_run,
    prepare_training_run,
)
from server.training.runtime import (
    CpuAffinityStatus,
    CpuSet,
    ExecutionConfig,
    ModelRankPlacement,
)
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata,
)


def test_prepare_training_run_writes_dashboard_only(
    tmp_path: Path,
) -> None:
    prepared_result = prepare_training_run(
        run_dir=tmp_path,
    )
    assert isinstance(prepared_result, Ok)
    prepared = prepared_result.value

    assert prepared.dashboard_path.exists()
    assert read_metrics(tmp_path) == ()


def test_initialize_training_run_writes_torch_checkpoint_and_metrics(
    tmp_path: Path,
) -> None:
    prepared = initialize_training_run(
        run_dir=tmp_path,
        run_id="run-setup-test",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(),
        execution_config=ExecutionConfig(),
    )

    assert isinstance(prepared, Ok)
    initialized = prepared.value
    assert initialized.dashboard_path.exists()
    assert initialized.checkpoint_path.exists()
    assert (
        initialized.checkpoint_path
        == tmp_path / "checkpoints" / "latest.json"
    )
    metadata = read_torch_checkpoint_metadata(
        initialized.checkpoint_path
    )
    assert isinstance(metadata, Ok)
    assert metadata.value.model_config == ModelConfig(d_model=128)
    assert metadata.value.train_config == TrainConfig()
    assert metadata.value.total_rounds == 0
    assert metadata.value.total_updates == 0
    metrics = read_metrics(tmp_path)
    assert len(metrics) == 1
    assert metrics[0].checkpoint_path == str(
        initialized.checkpoint_path
    )


def test_initialize_training_run_rejects_unavailable_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    prepared = initialize_training_run(
        run_dir=tmp_path,
        run_id="run-setup-test",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(),
        execution_config=ExecutionConfig(
            model_ranks=ModelRankPlacement(
                kind="cuda", devices=("cuda:0",)
            )
        ),
    )

    assert isinstance(prepared, Rejected)
    assert "--model-ranks cuda is unavailable" in prepared.reason
    assert read_metrics(tmp_path) == ()
    assert not (tmp_path / "checkpoints").exists()


def test_initialize_training_run_rejects_dashboard_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write_text = Path.write_text

    def fail_index_write(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if self.name == "index.html":
            raise OSError("disk full")
        return original_write_text(
            self,
            data,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", fail_index_write)

    result = initialize_training_run(
        run_dir=tmp_path,
        run_id="run-setup-test",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(),
        execution_config=ExecutionConfig(),
    )

    assert isinstance(result, Rejected)
    assert "dashboard write failed" in result.reason
    assert read_metrics(tmp_path) == ()
    assert not (tmp_path / "checkpoints").exists()


def test_initialize_training_run_force_cuda_failure_keeps_old_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = initialize_training_run(
        run_dir=tmp_path,
        run_id="first-run",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(seed=1),
        execution_config=ExecutionConfig(),
    )
    assert isinstance(first, Ok)
    metrics_before = read_metrics(tmp_path)
    checkpoint_path = first.value.checkpoint_path
    assert checkpoint_path.exists()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    second = initialize_training_run(
        run_dir=tmp_path,
        run_id="forced-run",
        model_config=ModelConfig(d_model=64),
        train_config=TrainConfig(seed=2),
        execution_config=ExecutionConfig(
            model_ranks=ModelRankPlacement(
                kind="cuda", devices=("cuda:0",)
            )
        ),
        force_new_run=True,
    )

    assert isinstance(second, Rejected)
    assert "--model-ranks cuda is unavailable" in second.reason
    assert checkpoint_path.exists()
    assert read_metrics(tmp_path) == metrics_before
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert isinstance(metadata, Ok)
    assert metadata.value.model_config == ModelConfig(d_model=128)
    assert metadata.value.train_config == TrainConfig(seed=1)


def test_initialize_training_run_force_bad_cuda_index_keeps_old_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = initialize_training_run(
        run_dir=tmp_path,
        run_id="first-run",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(seed=1),
        execution_config=ExecutionConfig(),
    )
    assert isinstance(first, Ok)
    checkpoint_path = first.value.checkpoint_path
    metrics_before = read_metrics(tmp_path)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    second = initialize_training_run(
        run_dir=tmp_path,
        run_id="forced-run",
        model_config=ModelConfig(d_model=64),
        train_config=TrainConfig(seed=2),
        execution_config=ExecutionConfig(
            model_ranks=ModelRankPlacement(
                kind="cuda", devices=("cuda:9",)
            )
        ),
        force_new_run=True,
    )

    assert isinstance(second, Rejected)
    assert "CUDA model rank is unavailable: cuda:9" in second.reason
    assert checkpoint_path.exists()
    assert read_metrics(tmp_path) == metrics_before


def test_initialize_training_run_force_bad_worker_cpu_keeps_old_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = initialize_training_run(
        run_dir=tmp_path,
        run_id="first-run",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(seed=1),
        execution_config=ExecutionConfig(),
    )
    assert isinstance(first, Ok)
    checkpoint_path = first.value.checkpoint_path
    metrics_before = read_metrics(tmp_path)

    def reject_worker_cpu(
        *, label: str, cpus: CpuSet
    ) -> Ok[CpuAffinityStatus] | Rejected:
        assert label == "worker-0"
        assert cpus == (999,)
        return Rejected(
            reason="CPU affinity preflight failed for worker-0: (999,)"
        )

    monkeypatch.setattr(
        run_setup, "preflight_cpu_affinity", reject_worker_cpu
    )

    second = initialize_training_run(
        run_dir=tmp_path,
        run_id="forced-run",
        model_config=ModelConfig(d_model=64),
        train_config=TrainConfig(seed=2),
        execution_config=ExecutionConfig(worker_cpus=(999,)),
        force_new_run=True,
    )

    assert isinstance(second, Rejected)
    assert "CPU affinity preflight failed" in second.reason
    assert checkpoint_path.exists()
    assert read_metrics(tmp_path) == metrics_before


def test_initialize_training_run_force_cleanup_failure_keeps_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = initialize_training_run(
        run_dir=tmp_path,
        run_id="first-run",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(seed=1),
        execution_config=ExecutionConfig(),
    )
    assert isinstance(first, Ok)
    original_unlink = Path.unlink

    def fail_metric_unlink(
        self: Path,
        missing_ok: bool = False,
    ) -> None:
        if self.name == "metrics.jsonl":
            raise OSError("busy")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_metric_unlink)

    result = initialize_training_run(
        run_dir=tmp_path,
        run_id="forced-run",
        model_config=ModelConfig(d_model=64),
        train_config=TrainConfig(seed=2),
        execution_config=ExecutionConfig(),
        force_new_run=True,
    )

    assert isinstance(result, Rejected)
    assert "training artifact cleanup failed" in result.reason
    assert first.value.checkpoint_path.exists()
    metrics = read_metrics(tmp_path)
    assert len(metrics) == 1
    assert metrics[0].run_id == "first-run"


def test_initialize_training_run_rejects_existing_training_artifacts(
    tmp_path: Path,
) -> None:
    first = initialize_training_run(
        run_dir=tmp_path,
        run_id="first-run",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(seed=1),
        execution_config=ExecutionConfig(),
    )
    assert isinstance(first, Ok)

    second = initialize_training_run(
        run_dir=tmp_path,
        run_id="second-run",
        model_config=ModelConfig(d_model=64),
        train_config=TrainConfig(seed=2),
        execution_config=ExecutionConfig(),
    )

    assert isinstance(second, Rejected)
    assert "training run already exists" in second.reason
    metadata = read_torch_checkpoint_metadata(
        first.value.checkpoint_path
    )
    assert isinstance(metadata, Ok)
    assert metadata.value.model_config == ModelConfig(d_model=128)
    assert metadata.value.train_config == TrainConfig(seed=1)
    metrics = read_metrics(tmp_path)
    assert len(metrics) == 1
    assert metrics[0].run_id == "first-run"


def test_initialize_training_run_force_new_run_replaces_artifacts(
    tmp_path: Path,
) -> None:
    first = initialize_training_run(
        run_dir=tmp_path,
        run_id="first-run",
        model_config=ModelConfig(d_model=128),
        train_config=TrainConfig(seed=1),
        execution_config=ExecutionConfig(),
    )
    assert isinstance(first, Ok)

    forced = initialize_training_run(
        run_dir=tmp_path,
        run_id="forced-run",
        model_config=ModelConfig(d_model=64),
        train_config=TrainConfig(seed=2),
        execution_config=ExecutionConfig(),
        force_new_run=True,
    )

    assert isinstance(forced, Ok)
    metadata = read_torch_checkpoint_metadata(
        forced.value.checkpoint_path
    )
    assert isinstance(metadata, Ok)
    assert metadata.value.model_config == ModelConfig(d_model=64)
    assert metadata.value.train_config == TrainConfig(seed=2)
    metrics = read_metrics(tmp_path)
    assert len(metrics) == 1
    assert metrics[0].run_id == "forced-run"
