"""Tests for execution-only training configuration."""

from __future__ import annotations

import subprocess
import sys

from server.foundation.result import Ok, Rejected
from server.training.runtime.config import (
    ExecutionConfig,
    ExecutionTimeouts,
    ModelRankPlacement,
    parse_cpu_set,
    parse_model_rank_placement,
)


def test_parse_cpu_set_ranges() -> None:
    parsed = parse_cpu_set("4-7")

    assert isinstance(parsed, Ok)
    assert parsed.value == (4, 5, 6, 7)


def test_parse_cpu_set_rejects_duplicate_cpu() -> None:
    parsed = parse_cpu_set("0-3,2,6")

    assert isinstance(parsed, Rejected)
    assert "duplicate CPU" in parsed.reason


def test_parse_cpu_set_rejects_empty_text() -> None:
    parsed = parse_cpu_set("")

    assert isinstance(parsed, Rejected)
    assert parsed.reason == "CPU set must not be empty"


def test_parse_cpu_set_rejects_descending_range() -> None:
    parsed = parse_cpu_set("3-1")

    assert isinstance(parsed, Rejected)
    assert "descending" in parsed.reason


def test_parse_cpu_set_rejects_non_numeric_part() -> None:
    parsed = parse_cpu_set("0,a")

    assert isinstance(parsed, Rejected)
    assert "invalid CPU number" in parsed.reason


def test_parse_model_rank_accepts_none() -> None:
    parsed = parse_model_rank_placement("none")

    assert isinstance(parsed, Ok)
    assert parsed.value == ModelRankPlacement(kind="none", devices=())


def test_parse_model_rank_rejects_inline() -> None:
    parsed = parse_model_rank_placement("inline")

    assert isinstance(parsed, Rejected)
    assert "--model-ranks" in parsed.reason


def test_parse_model_rank_accepts_mps() -> None:
    parsed = parse_model_rank_placement("mps")

    assert isinstance(parsed, Ok)
    assert parsed.value == ModelRankPlacement(
        kind="mps", devices=("mps",)
    )


def test_parse_model_rank_accepts_cuda_indices() -> None:
    parsed = parse_model_rank_placement("cuda:0,2")

    assert isinstance(parsed, Ok)
    assert parsed.value == ModelRankPlacement(
        kind="cuda", devices=("cuda:0", "cuda:2")
    )


def test_parse_model_rank_rejects_cuda_without_index() -> None:
    parsed = parse_model_rank_placement("cuda")

    assert isinstance(parsed, Rejected)
    assert "--model-ranks" in parsed.reason


def test_parse_model_rank_rejects_duplicate_cuda_index() -> None:
    parsed = parse_model_rank_placement("cuda:0,0")

    assert isinstance(parsed, Rejected)
    assert "duplicate CUDA model rank index" in parsed.reason


def test_execution_config_derives_default_single_worker() -> None:
    config = ExecutionConfig()

    assert config.worker_process_count() == 1
    assert config.worker_cpu_set(0) == ()
    assert config.model_rank_process_count() == 0
    assert not config.uses_model_rank_processes()
    assert config.timeouts == ExecutionTimeouts()


def test_default_timeouts_fit_cpu_rollouts() -> None:
    timeouts = ExecutionTimeouts()

    assert timeouts.round_seconds == 600.0
    assert timeouts.rollout_sample_seconds == 900.0
    assert timeouts.sampling_stop_seconds == 600.0


def test_execution_timeouts_reject_non_positive_values() -> None:
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from server.training.runtime.config import "
                "ExecutionTimeouts\n"
                "ExecutionTimeouts(update_seconds=0.0)\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "AssertionError" in completed.stderr


def test_execution_timeouts_keep_sampling_stages_separate() -> None:
    timeouts = ExecutionTimeouts(
        sampling_start_seconds=11.0,
        rollout_sample_seconds=22.0,
        sampling_stop_seconds=33.0,
    )

    assert timeouts.sampling_start_seconds == 11.0
    assert timeouts.rollout_sample_seconds == 22.0
    assert timeouts.sampling_stop_seconds == 33.0


def test_execution_config_derives_workers_from_cpu_list() -> None:
    config = ExecutionConfig(worker_cpus=(4, 5, 6, 7))

    assert config.worker_process_count() == 4
    assert config.worker_cpu_set(0) == (4,)
    assert config.worker_cpu_set(3) == (7,)


def test_execution_config_maps_workers_by_modulo() -> None:
    config = ExecutionConfig(
        worker_cpus=(4, 5, 6, 7, 8),
        model_ranks=ModelRankPlacement(
            kind="cuda", devices=("cuda:0", "cuda:1")
        ),
    )

    assert config.worker_process_count() == 5
    assert config.model_rank_process_count() == 2
    assert config.model_rank_index_for_worker(0) == 0
    assert config.model_rank_index_for_worker(1) == 1
    assert config.model_rank_index_for_worker(2) == 0
