"""Black-box tests for distributed PPO minibatch scheduling."""

from __future__ import annotations

from server.training.ppo.batch_schedule import (
    plan_distributed_minibatches,
)


def test_plan_distributed_minibatches_enforces_global_batch_size() -> (
    None
):
    schedule = plan_distributed_minibatches(
        local_sample_counts=(6, 4),
        global_minibatch_size=4,
    )

    assert schedule.global_counts == (4, 4, 2)
    assert tuple(
        (span.start, span.end) for span in schedule.spans_for_rank(0)
    ) == ((0, 2), (2, 4), (4, 6))
    assert tuple(
        (span.start, span.end) for span in schedule.spans_for_rank(1)
    ) == ((0, 2), (2, 4), (4, 4))


def test_plan_distributed_minibatches_preserves_unbalanced() -> None:
    schedule = plan_distributed_minibatches(
        local_sample_counts=(1, 9),
        global_minibatch_size=4,
    )

    assert schedule.global_counts == (4, 4, 2)
    assert tuple(
        span.count() for span in schedule.spans_for_rank(0)
    ) == (1, 0, 0)
    assert tuple(
        span.count() for span in schedule.spans_for_rank(1)
    ) == (3, 4, 2)
    assert sum(schedule.global_counts) == 10


def test_plan_distributed_minibatches_balances_active_ranks() -> None:
    schedule = plan_distributed_minibatches(
        local_sample_counts=(4, 4, 4),
        global_minibatch_size=5,
    )

    assert schedule.global_counts == (5, 5, 2)
    assert tuple(
        tuple(span.count() for span in schedule.spans_for_rank(rank))
        for rank in range(3)
    ) == ((2, 1, 1), (2, 2, 0), (1, 2, 1))
