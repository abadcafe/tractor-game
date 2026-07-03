"""Tests for training metrics and dashboard output."""

from __future__ import annotations

from pathlib import Path

from server.training.dashboard import (
    render_dashboard_html,
    write_dashboard,
)
from server.training.metrics import (
    TrainingMetric,
    append_metric,
    read_metrics,
)


def test_metrics_append_and_read_round_trip(tmp_path: Path) -> None:
    metric = TrainingMetric(
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
        policy_loss=0.1,
        value_loss=0.2,
        entropy=0.3,
        approx_kl=0.01,
        clip_fraction=0.25,
        checkpoint_path="checkpoint.json",
    )

    append_metric(tmp_path, metric)

    assert read_metrics(tmp_path) == (metric,)


def test_dashboard_mentions_metrics_file() -> None:
    html = render_dashboard_html(title="Tractor Training")
    assert "metrics.jsonl" in html
    assert "total_games" in html


def test_write_dashboard_creates_index(tmp_path: Path) -> None:
    path = write_dashboard(tmp_path, title="Tractor Training")
    assert path.exists()
    assert path.name == "index.html"
