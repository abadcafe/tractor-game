"""Tests for training metrics and dashboard output."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from server.result import Ok, Rejected
from server.training.dashboard import (
    render_dashboard_html,
    write_dashboard,
)
from server.training.json_types import JsonObject, JsonValue
from server.training.metrics import (
    TrainingMetric,
    append_metric,
    metrics_path,
    read_metrics,
)


def test_metrics_append_and_read_round_trip(tmp_path: Path) -> None:
    metric = _sample_metric()

    append_result = append_metric(tmp_path, metric)

    assert isinstance(append_result, Ok)
    assert read_metrics(tmp_path) == (metric,)


def test_read_metrics_skips_corrupt_record(tmp_path: Path) -> None:
    metric = _sample_metric()
    append_result = append_metric(tmp_path, metric)
    assert isinstance(append_result, Ok)
    path = metrics_path(tmp_path)
    with path.open("a", encoding="utf-8") as file:
        file.write("{partial")

    assert read_metrics(tmp_path) == (metric,)


def test_read_metrics_skips_invalid_utf8_record(tmp_path: Path) -> None:
    metric = _sample_metric()
    append_result = append_metric(tmp_path, metric)
    assert isinstance(append_result, Ok)
    path = metrics_path(tmp_path)
    with path.open("ab") as file:
        file.write(
            b'{"run_id":"bad'
            b"\xff"
            b'","total_games":11,'
            b'"total_updates":3,'
            b'"process_games_per_second":2.5,'
            b'"last_round_decisions_per_second":121.0,'
            b'"last_team0_reward":0.5,'
            b'"last_team1_reward":-0.5,'
            b'"last_generated_action_count":18,'
            b'"last_accepted_action_count":17,'
            b'"last_decision_count":16,'
            b'"last_average_action_choices":4.5,'
            b'"policy_loss":null,'
            b'"value_loss":null,'
            b'"entropy":null,'
            b'"approx_kl":null,'
            b'"clip_fraction":null,'
            b'"checkpoint_path":null}\n'
        )

    assert read_metrics(tmp_path) == (metric,)


def test_append_metric_rejects_non_finite_float(
    tmp_path: Path,
) -> None:
    result = append_metric(
        tmp_path, _sample_metric(policy_loss=math.nan)
    )

    assert isinstance(result, Rejected)
    assert "policy_loss" in result.reason
    assert not metrics_path(tmp_path).exists()


def test_append_metric_rejects_non_finite_ppo_profile(
    tmp_path: Path,
) -> None:
    result = append_metric(
        tmp_path, _sample_metric(ppo_argument_decode_seconds=math.inf)
    )

    assert isinstance(result, Rejected)
    assert "ppo_argument_decode_seconds" in result.reason
    assert not metrics_path(tmp_path).exists()


def test_read_metrics_skips_non_finite_record(tmp_path: Path) -> None:
    metric = _sample_metric()
    append_result = append_metric(tmp_path, metric)
    assert isinstance(append_result, Ok)
    path = metrics_path(tmp_path)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            '{"run_id":"bad",'
            '"total_games":11,'
            '"total_updates":3,'
            '"process_games_per_second":NaN,'
            '"last_round_decisions_per_second":121.0,'
            '"last_team0_reward":0.5,'
            '"last_team1_reward":-0.5,'
            '"last_generated_action_count":18,'
            '"last_accepted_action_count":17,'
            '"last_decision_count":16,'
            '"last_average_action_choices":4.5,'
            '"policy_loss":null,'
            '"value_loss":null,'
            '"entropy":null,'
            '"approx_kl":null,'
            '"clip_fraction":null,'
            '"checkpoint_path":null}\n'
        )

    assert read_metrics(tmp_path) == (metric,)


def test_read_metrics_accepts_missing_nullable_schema_fields(
    tmp_path: Path,
) -> None:
    metric = _sample_metric()
    append_result = append_metric(tmp_path, metric)
    assert isinstance(append_result, Ok)
    path = metrics_path(tmp_path)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            '{"run_id":"bad",'
            '"total_games":11,'
            '"total_updates":3,'
            '"process_games_per_second":2.5,'
            '"last_round_decisions_per_second":121.0,'
            '"last_team0_reward":0.5,'
            '"last_team1_reward":-0.5,'
            '"last_generated_action_count":18,'
            '"last_accepted_action_count":17,'
            '"last_decision_count":16,'
            '"last_average_action_choices":4.5,'
            '"policy_loss":null,'
            '"value_loss":null,'
            '"entropy":null,'
            '"approx_kl":null,'
            '"clip_fraction":null,'
            '"checkpoint_path":null}\n'
        )

    metrics = read_metrics(tmp_path)
    assert len(metrics) == 2
    assert metrics[0] == metric
    assert metrics[1].run_id == "bad"
    assert metrics[1].ppo_update_seconds is None
    assert metrics[1].checkpoint_path is None


def test_read_metrics_skips_invalid_nullable_float_record(
    tmp_path: Path,
) -> None:
    metric = _sample_metric()
    append_result = append_metric(tmp_path, metric)
    assert isinstance(append_result, Ok)

    _write_raw_metric_record(
        metrics_path(tmp_path), policy_loss=math.nan
    )

    assert read_metrics(tmp_path) == (metric,)


def test_read_metrics_skips_out_of_range_nullable_seconds_record(
    tmp_path: Path,
) -> None:
    metric = _sample_metric()
    append_result = append_metric(tmp_path, metric)
    assert isinstance(append_result, Ok)

    _write_raw_metric_record(
        metrics_path(tmp_path), ppo_update_seconds=-1.0
    )

    assert read_metrics(tmp_path) == (metric,)


def test_read_metrics_skips_out_of_range_nullable_fraction_record(
    tmp_path: Path,
) -> None:
    metric = _sample_metric()
    append_result = append_metric(tmp_path, metric)
    assert isinstance(append_result, Ok)

    _write_raw_metric_record(
        metrics_path(tmp_path), ppo_argument_decode_fraction=2.0
    )

    assert read_metrics(tmp_path) == (metric,)


def test_dashboard_mentions_metrics_file() -> None:
    html = render_dashboard_html(title="Tractor Training")
    assert "metrics.jsonl" in html
    assert "total_games" in html
    assert "innerHTML" not in html
    assert "createElement" in html
    assert "textContent" in html


def test_dashboard_escapes_title_html() -> None:
    html = render_dashboard_html(
        title='<script>alert("x")</script> & Training'
    )

    assert "<title>&lt;script&gt;" in html
    assert '<script>alert("x")</script>' not in html
    assert "&amp; Training" in html


def test_write_dashboard_creates_index(tmp_path: Path) -> None:
    result = write_dashboard(tmp_path, title="Tractor Training")
    assert isinstance(result, Ok)
    path = result.value
    assert path.exists()
    assert path.name == "index.html"


def test_write_dashboard_rejects_write_failure(
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

    result = write_dashboard(tmp_path, title="Tractor Training")

    assert isinstance(result, Rejected)
    assert "dashboard write failed" in result.reason


def _sample_metric(
    *,
    policy_loss: float | None = 0.1,
    ppo_argument_decode_seconds: float | None = 0.04,
) -> TrainingMetric:
    return TrainingMetric(
        run_id="run-1",
        total_games=10,
        total_updates=2,
        process_games_per_second=1.5,
        last_round_decisions_per_second=120.0,
        last_team0_reward=0.25,
        last_team1_reward=-0.25,
        last_generated_action_count=17,
        last_accepted_action_count=16,
        last_decision_count=15,
        last_average_action_choices=3.5,
        policy_loss=policy_loss,
        value_loss=0.2,
        entropy=0.3,
        approx_kl=0.01,
        clip_fraction=0.25,
        ppo_update_seconds=0.5,
        ppo_minibatch_loss_seconds=0.3,
        ppo_observation_batch_seconds=0.01,
        ppo_observation_encode_seconds=0.08,
        ppo_value_head_seconds=0.02,
        ppo_argument_select_seconds=0.01,
        ppo_argument_prefix_tensorize_seconds=0.01,
        ppo_argument_decode_seconds=ppo_argument_decode_seconds,
        ppo_argument_distribution_seconds=0.02,
        ppo_backward_seconds=0.12,
        ppo_optimizer_step_seconds=0.03,
        ppo_argument_decode_fraction=0.08,
        ppo_argument_prefix_batch_count=2,
        ppo_argument_prefix_row_count=30,
        ppo_argument_prefix_token_count=45,
        ppo_argument_prefix_valid_token_count=40,
        ppo_argument_prefix_padding_token_count=5,
        checkpoint_path="checkpoint.json",
    )


def _write_raw_metric_record(
    path: Path, **overrides: JsonValue
) -> None:
    record = _sample_metric_record()
    record.update(overrides)
    with path.open("a", encoding="utf-8") as file:
        json.dump(record, file, allow_nan=True)
        file.write("\n")


def _sample_metric_record() -> JsonObject:
    return {
        "run_id": "raw-run",
        "total_games": 10,
        "total_updates": 2,
        "process_games_per_second": 1.5,
        "last_round_decisions_per_second": 120.0,
        "last_team0_reward": 0.25,
        "last_team1_reward": -0.25,
        "last_generated_action_count": 17,
        "last_accepted_action_count": 16,
        "last_decision_count": 15,
        "last_average_action_choices": 3.5,
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 0.3,
        "approx_kl": 0.01,
        "clip_fraction": 0.25,
        "ppo_update_seconds": 0.5,
        "ppo_minibatch_loss_seconds": 0.3,
        "ppo_observation_batch_seconds": 0.01,
        "ppo_observation_encode_seconds": 0.08,
        "ppo_value_head_seconds": 0.02,
        "ppo_argument_select_seconds": 0.01,
        "ppo_argument_prefix_tensorize_seconds": 0.01,
        "ppo_argument_decode_seconds": 0.04,
        "ppo_argument_distribution_seconds": 0.02,
        "ppo_backward_seconds": 0.12,
        "ppo_optimizer_step_seconds": 0.03,
        "ppo_argument_decode_fraction": 0.08,
        "ppo_argument_prefix_batch_count": 2,
        "ppo_argument_prefix_row_count": 30,
        "ppo_argument_prefix_token_count": 45,
        "ppo_argument_prefix_valid_token_count": 40,
        "ppo_argument_prefix_padding_token_count": 5,
        "checkpoint_path": "checkpoint.json",
    }
