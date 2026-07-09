"""Tests for the training CLI entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from server.result import Ok, Rejected
from server.training import train as train_module
from server.training.config import ModelConfig, TrainConfig
from server.training.metrics import read_metrics
from server.training.run_setup import (
    InitializedTrainingRun,
)
from server.training.run_setup import (
    initialize_training_run as _initialize_training_run,
)
from server.training.runtime import ExecutionConfig
from server.training.runtime.result import TrainingLoopResult
from server.training.torch_checkpoints import (
    TorchCheckpointMetadata,
)
from server.training.torch_checkpoints import (
    read_torch_checkpoint_metadata as _read_torch_checkpoint_metadata,
)
from server.training.train import (
    MIN_CLI_MAX_TOKENS,
    ExecutionConfigOverrides,
    main,
    resolve_execution_config,
)


def test_init_only_prints_resumable_torch_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "latest.json"

    main(
        (
            "--run-dir",
            str(tmp_path),
            "--init-only",
            "--d-model",
            "4",
            "--layers",
            "1",
            "--heads",
            "1",
            "--max-tokens",
            "512",
            "--seed",
            "123",
        )
    )

    output = capsys.readouterr().out
    assert f"checkpoint: {checkpoint_path}" in output
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert metadata.model_config == ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    assert metadata.train_config == TrainConfig(seed=123)
    assert metadata.total_rounds == 0
    assert metadata.total_updates == 0


def test_resolve_execution_config_sets_inference_batching() -> None:
    result = resolve_execution_config(
        ExecutionConfigOverrides(
            model_inference_batch_size=32,
        )
    )

    assert isinstance(result, Ok)
    assert result.value.model_inference_batch_size == 32


def test_execution_config_rejects_bad_inference_batch_size() -> None:
    result = resolve_execution_config(
        ExecutionConfigOverrides(
            model_inference_batch_size=0,
        )
    )

    assert isinstance(result, Rejected)
    assert "--model-inference-batch-size" in result.reason


def test_init_only_does_not_persist_ppo_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "latest.json"

    main(
        (
            "--run-dir",
            str(tmp_path),
            "--init-only",
            "--d-model",
            "4",
            "--layers",
            "1",
            "--heads",
            "1",
            "--max-tokens",
            "512",
            "--ppo-profile",
            "detailed",
        )
    )

    capsys.readouterr()
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert metadata.train_config == TrainConfig()


def test_new_run_rejects_existing_run_without_force(
    tmp_path: Path,
) -> None:
    initialized = initialize_training_run(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        ),
        train_config=TrainConfig(seed=1),
    )

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--run-dir',\n"
                f"    {str(tmp_path)!r},\n"
                "    '--init-only',\n"
                "    '--d-model',\n"
                "    '8',\n"
                "    '--layers',\n"
                "    '1',\n"
                "    '--heads',\n"
                "    '1',\n"
                "    '--max-tokens',\n"
                "    '512',\n"
                "    '--seed',\n"
                "    '2',\n"
                "))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "training run already exists" in completed.stderr
    assert "--force-new-run" in completed.stderr
    metadata = read_torch_checkpoint_metadata(
        initialized.checkpoint_path
    )
    assert metadata.model_config == ModelConfig(
        d_model=4,
        layers=1,
        heads=1,
        max_tokens=64,
    )
    assert metadata.train_config == TrainConfig(seed=1)
    assert len(read_metrics(tmp_path)) == 1


def test_force_new_run_reinitializes_existing_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    initialize_training_run(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        ),
        train_config=TrainConfig(seed=1),
    )
    checkpoint_path = tmp_path / "checkpoints" / "latest.json"

    main(
        (
            "--run-dir",
            str(tmp_path),
            "--init-only",
            "--force-new-run",
            "--d-model",
            "8",
            "--layers",
            "1",
            "--heads",
            "1",
            "--max-tokens",
            "512",
            "--seed",
            "2",
        )
    )

    output = capsys.readouterr().out
    assert f"checkpoint: {checkpoint_path}" in output
    metadata = read_torch_checkpoint_metadata(checkpoint_path)
    assert metadata.model_config == ModelConfig(
        d_model=8,
        layers=1,
        heads=1,
        max_tokens=512,
    )
    assert metadata.train_config == TrainConfig(seed=2)
    metrics = read_metrics(tmp_path)
    assert len(metrics) == 1
    assert metrics[0].run_id == tmp_path.name


def test_resume_does_not_append_initial_metric_before_coordinator(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = initialize_training_run(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        ),
        train_config=TrainConfig(),
    )
    metrics_before = read_metrics(tmp_path)
    coordinator_max_samples: list[int] = []

    def fake_run_training_coordinator(
        *,
        run_dir: Path,
        run_id: str,
        model_config: ModelConfig,
        train_config: TrainConfig,
        execution_config: ExecutionConfig,
        max_samples: int,
        resume: Path | None,
    ) -> Ok[TrainingLoopResult] | Rejected:
        assert run_dir == tmp_path
        assert run_id == tmp_path.name
        assert model_config == ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        )
        assert train_config == TrainConfig()
        assert execution_config == ExecutionConfig()
        assert resume == initialized.checkpoint_path
        coordinator_max_samples.append(max_samples)
        return Ok(
            value=TrainingLoopResult(
                total_rounds=0,
                total_samples=0,
                total_updates=0,
                checkpoint_path=initialized.checkpoint_path,
            )
        )

    monkeypatch.setattr(
        train_module,
        "run_training_coordinator",
        fake_run_training_coordinator,
    )

    main(
        (
            "--run-dir",
            str(tmp_path),
            "--resume",
            str(initialized.checkpoint_path),
            "--max-samples",
            "0",
        )
    )

    capsys.readouterr()
    assert coordinator_max_samples == [0]
    assert read_metrics(tmp_path) == metrics_before


def test_resume_without_run_dir_uses_checkpoint_run_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "source-run"
    initialized = initialize_training_run(
        run_dir=run_dir,
        run_id=run_dir.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        ),
        train_config=TrainConfig(),
    )
    metrics_before = read_metrics(run_dir)

    def fake_run_training_coordinator(
        *,
        run_dir: Path,
        run_id: str,
        model_config: ModelConfig,
        train_config: TrainConfig,
        execution_config: ExecutionConfig,
        max_samples: int,
        resume: Path | None,
    ) -> Ok[TrainingLoopResult] | Rejected:
        assert run_dir == tmp_path / "source-run"
        assert run_id == "source-run"
        assert model_config == ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        )
        assert train_config == TrainConfig()
        assert execution_config == ExecutionConfig()
        assert max_samples == 0
        assert resume == initialized.checkpoint_path
        return Ok(
            value=TrainingLoopResult(
                total_rounds=0,
                total_samples=0,
                total_updates=0,
                checkpoint_path=initialized.checkpoint_path,
            )
        )

    monkeypatch.setattr(
        train_module,
        "run_training_coordinator",
        fake_run_training_coordinator,
    )

    main(
        (
            "--resume",
            str(initialized.checkpoint_path),
            "--max-samples",
            "0",
        )
    )

    output = capsys.readouterr().out
    assert f"dashboard: {run_dir / 'index.html'}" in output
    assert f"checkpoint: {initialized.checkpoint_path}" in output
    assert read_metrics(run_dir) == metrics_before


def test_resume_rejects_mismatched_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "source-run"
    initialized = initialize_training_run(
        run_dir=run_dir,
        run_id=run_dir.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        ),
        train_config=TrainConfig(),
    )
    other_run_dir = tmp_path / "other-run"

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--run-dir',\n"
                f"    {str(other_run_dir)!r},\n"
                "    '--resume',\n"
                f"    {str(initialized.checkpoint_path)!r},\n"
                "    '--max-samples',\n"
                "    '0',\n"
                "))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "--run-dir must match the run directory that owns --resume"
        in completed.stderr
    )


def test_resume_rejects_checkpoint_outside_run_dir(
    tmp_path: Path,
) -> None:
    invalid_resume = tmp_path / "latest.json"

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main(('--resume', "
                f"{str(invalid_resume)!r}, '--max-samples', '0'))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "--resume must point to "
        "<run-dir>/checkpoints/<checkpoint>.json" in completed.stderr
    )


def test_resume_seed_mismatch_reports_cli_error(tmp_path: Path) -> None:
    initialized = initialize_training_run(
        run_dir=tmp_path,
        run_id=tmp_path.name,
        model_config=ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=64,
        ),
        train_config=TrainConfig(seed=3),
    )

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--resume',\n"
                f"    {str(initialized.checkpoint_path)!r},\n"
                "    '--seed',\n"
                "    '4',\n"
                "    '--max-samples',\n"
                "    '0',\n"
                "))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "--seed must match the checkpoint seed when using --resume"
        in completed.stderr
    )
    assert "AssertionError" not in completed.stderr


def test_resume_corrupt_checkpoint_reports_cli_error(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "source-run"
    checkpoint_path = run_dir / "checkpoints" / "latest.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("{not checkpoint json", encoding="utf-8")

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--resume',\n"
                f"    {str(checkpoint_path)!r},\n"
                "    '--max-samples',\n"
                "    '0',\n"
                "))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "checkpoint corruption:" in completed.stderr
    assert "latest.json" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_resume_invalid_utf8_checkpoint_reports_cli_error(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "source-run"
    checkpoint_path = run_dir / "checkpoints" / "latest.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"\xff")

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--resume',\n"
                f"    {str(checkpoint_path)!r},\n"
                "    '--max-samples',\n"
                "    '0',\n"
                "))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "checkpoint corruption:" in completed.stderr
    assert "manifest is not valid UTF-8" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_resume_directory_checkpoint_reports_cli_error(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "source-run"
    checkpoint_path = run_dir / "checkpoints" / "latest.json"
    checkpoint_path.mkdir(parents=True)

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main((\n"
                "    '--resume',\n"
                f"    {str(checkpoint_path)!r},\n"
                "    '--max-samples',\n"
                "    '0',\n"
                "))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "checkpoint corruption:" in completed.stderr
    assert "manifest file is not readable" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cuda_device_unavailable_reports_cli_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    exit_code: object = None

    try:
        main(
            (
                "--run-dir",
                str(tmp_path),
                "--init-only",
                "--model-ranks",
                "cuda:0",
                "--d-model",
                "4",
                "--layers",
                "1",
                "--heads",
                "1",
                "--max-tokens",
                str(MIN_CLI_MAX_TOKENS),
            )
        )
    except SystemExit as error:
        exit_code = error.code

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--model-ranks cuda is unavailable" in captured.err
    assert "Traceback" not in captured.err
    assert read_metrics(tmp_path) == ()
    assert not (tmp_path / "checkpoints").exists()


def test_training_interrupt_reports_cli_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_training_coordinator(
        *,
        run_dir: Path,
        run_id: str,
        model_config: ModelConfig,
        train_config: TrainConfig,
        execution_config: ExecutionConfig,
        max_samples: int,
        resume: Path | None,
    ) -> Ok[TrainingLoopResult] | Rejected:
        assert run_dir == tmp_path
        assert run_id == tmp_path.name
        assert model_config == ModelConfig(
            d_model=4,
            layers=1,
            heads=1,
            max_tokens=MIN_CLI_MAX_TOKENS,
        )
        assert train_config == TrainConfig()
        assert execution_config == ExecutionConfig()
        assert max_samples == 1
        assert resume == tmp_path / "checkpoints" / "latest.json"
        raise KeyboardInterrupt

    monkeypatch.setattr(
        train_module,
        "run_training_coordinator",
        fake_run_training_coordinator,
    )
    exit_code: object = None

    try:
        main(
            (
                "--run-dir",
                str(tmp_path),
                "--d-model",
                "4",
                "--layers",
                "1",
                "--heads",
                "1",
                "--max-tokens",
                str(MIN_CLI_MAX_TOKENS),
                "--max-samples",
                "1",
            )
        )
    except SystemExit as error:
        exit_code = error.code

    captured = capsys.readouterr()
    assert exit_code == 130
    assert captured.err == "training interrupted\n"
    assert "Traceback" not in captured.err


def test_cli_rejects_too_small_max_tokens() -> None:
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main(('--max-tokens', "
                f"{str(MIN_CLI_MAX_TOKENS - 1)!r}))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert f"must be >= {MIN_CLI_MAX_TOKENS}" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_rejects_duplicate_cuda_model_rank() -> None:
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.train import main\n"
                "main(('--model-ranks', 'cuda:0,0'))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "duplicate CUDA model rank index: 0" in completed.stderr
    assert "Traceback" not in completed.stderr


def initialize_training_run(
    *,
    run_dir: Path,
    run_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> InitializedTrainingRun:
    result = _initialize_training_run(
        run_dir=run_dir,
        run_id=run_id,
        model_config=model_config,
        train_config=train_config,
        execution_config=ExecutionConfig(),
    )
    assert isinstance(result, Ok)
    return result.value


def read_torch_checkpoint_metadata(
    path: Path,
) -> TorchCheckpointMetadata:
    result = _read_torch_checkpoint_metadata(path)
    assert isinstance(result, Ok)
    return result.value
