"""Black-box tests for SQLite training metrics."""

import math
from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training.metrics import (
    TrainingMetric,
    append_metric,
    read_metric_boundaries,
    read_metric_records,
    read_metrics,
    reconcile_metrics_with_checkpoint,
)
from server.training.persistence.schema import database_path


def test_append_metric_persists_typed_record(tmp_path: Path) -> None:
    metric = _metric(total_updates=1)

    appended = append_metric(tmp_path, metric)
    read = read_metric_records(tmp_path)

    assert isinstance(appended, Ok)
    assert appended.value.sequence == 1
    assert isinstance(read, Ok)
    assert len(read.value) == 1
    assert read.value[0].total_updates == 1
    assert database_path(tmp_path).exists()
    assert not (tmp_path / "metrics.jsonl").exists()


def test_read_metric_records_returns_latest_limit_in_order(
    tmp_path: Path,
) -> None:
    for update in range(1, 4):
        assert isinstance(append_metric(tmp_path, _metric(update)), Ok)

    result = read_metric_records(tmp_path, limit=2)

    assert isinstance(result, Ok)
    assert [record.total_updates for record in result.value] == [2, 3]


def test_read_metric_records_supports_incremental_cursor(
    tmp_path: Path,
) -> None:
    for update in range(1, 4):
        assert isinstance(append_metric(tmp_path, _metric(update)), Ok)

    result = read_metric_records(tmp_path, after_sequence=1)

    assert isinstance(result, Ok)
    assert [record.sequence for record in result.value] == [2, 3]


def test_read_metric_boundaries_returns_count_first_and_latest(
    tmp_path: Path,
) -> None:
    for update in range(3):
        assert isinstance(append_metric(tmp_path, _metric(update)), Ok)

    result = read_metric_boundaries(tmp_path)

    assert isinstance(result, Ok)
    assert result.value is not None
    assert result.value.count == 3
    assert result.value.first.total_updates == 0
    assert result.value.latest.total_updates == 2


def test_append_metric_rejects_duplicate_update(tmp_path: Path) -> None:
    assert isinstance(append_metric(tmp_path, _metric(1)), Ok)

    result = append_metric(tmp_path, _metric(1))

    assert isinstance(result, Rejected)


def test_append_metric_rejects_non_finite_value(tmp_path: Path) -> None:
    metric = _metric(1).model_copy(update={"policy_loss": math.nan})

    result = append_metric(tmp_path, metric)

    assert isinstance(result, Rejected)


def test_reconcile_metrics_removes_records_ahead_of_checkpoint(
    tmp_path: Path,
) -> None:
    for update in range(1, 4):
        assert isinstance(append_metric(tmp_path, _metric(update)), Ok)

    reconciled = reconcile_metrics_with_checkpoint(
        tmp_path,
        total_rounds=20,
        total_samples=200,
        total_updates=2,
    )
    read = read_metrics(tmp_path)

    assert isinstance(reconciled, Ok)
    assert isinstance(read, Ok)
    assert [metric.total_updates for metric in read.value] == [1, 2]


def _metric(total_updates: int = 1) -> TrainingMetric:
    return TrainingMetric(
        total_games=total_updates * 10,
        total_samples=total_updates * 100,
        total_updates=total_updates,
        process_games_per_second=2.0,
        process_samples_per_second=20.0,
        last_round_decisions_per_second=10.0,
        last_team0_reward=1.0,
        last_team1_reward=-1.0,
        last_generated_action_count=40,
        last_accepted_action_count=30,
        last_decision_count=20,
        last_average_action_choices=3.5,
        policy_loss=0.1,
        value_loss=0.2,
        entropy=0.3,
        approx_kl=0.01,
        clip_fraction=0.04,
        ppo_update_seconds=1.5,
        ppo_minibatch_loss_seconds=0.2,
        ppo_observation_batch_seconds=0.1,
        ppo_observation_encode_seconds=0.1,
        ppo_value_head_seconds=0.1,
        ppo_argument_select_seconds=0.1,
        ppo_argument_decode_seconds=0.1,
        ppo_argument_distribution_seconds=0.1,
        ppo_backward_seconds=0.2,
        ppo_optimizer_step_seconds=0.1,
        ppo_argument_decode_fraction=0.25,
        ppo_argument_trace_batch_count=2,
        ppo_argument_trace_row_count=10,
        ppo_argument_trace_token_count=100,
        ppo_argument_trace_valid_token_count=80,
        ppo_argument_trace_padding_token_count=20,
        checkpoint_path=None,
    )
