"""Deterministic global-minibatch scheduling across PPO ranks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MinibatchSpan:
    """One contiguous rank-local slice assigned to a global step."""

    start: int
    end: int

    def __post_init__(self) -> None:
        assert self.start >= 0
        assert self.end >= self.start

    def count(self) -> int:
        """Return the rank-local sample count."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class DistributedBatchSchedule:
    """All rank-local spans composing each global minibatch."""

    local_sample_counts: tuple[int, ...]
    rank_spans: tuple[tuple[MinibatchSpan, ...], ...]
    global_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        assert self.local_sample_counts
        assert all(count >= 0 for count in self.local_sample_counts)
        assert len(self.rank_spans) == len(self.local_sample_counts)
        assert all(
            len(spans) == len(self.global_counts)
            for spans in self.rank_spans
        )
        assert all(count > 0 for count in self.global_counts)
        for rank, spans in enumerate(self.rank_spans):
            assert (
                sum(span.count() for span in spans)
                == (self.local_sample_counts[rank])
            )
        for step_index, global_count in enumerate(self.global_counts):
            assert (
                sum(
                    spans[step_index].count()
                    for spans in self.rank_spans
                )
                == global_count
            )

    def spans_for_rank(self, rank: int) -> tuple[MinibatchSpan, ...]:
        """Return one rank's local contribution to every global step."""
        assert 0 <= rank < len(self.rank_spans)
        return self.rank_spans[rank]


def plan_distributed_minibatches(
    *,
    local_sample_counts: tuple[int, ...],
    global_minibatch_size: int,
) -> DistributedBatchSchedule:
    """Partition every local sample into bounded global minibatches."""
    assert local_sample_counts
    assert all(count >= 0 for count in local_sample_counts)
    assert sum(local_sample_counts) > 0
    assert global_minibatch_size > 0
    remaining = list(local_sample_counts)
    offsets = [0 for _count in local_sample_counts]
    rank_spans: list[list[MinibatchSpan]] = [
        [] for _count in local_sample_counts
    ]
    global_counts: list[int] = []
    step_index = 0
    while (remaining_count := sum(remaining)) > 0:
        global_count = min(global_minibatch_size, remaining_count)
        allocations = _proportional_step_allocations(
            remaining=tuple(remaining),
            target=global_count,
            first_rank=step_index % len(remaining),
        )
        for rank, allocation in enumerate(allocations):
            start = offsets[rank]
            end = start + allocation
            rank_spans[rank].append(MinibatchSpan(start=start, end=end))
            offsets[rank] = end
            remaining[rank] -= allocation
        global_counts.append(global_count)
        step_index += 1
    return DistributedBatchSchedule(
        local_sample_counts=local_sample_counts,
        rank_spans=tuple(tuple(spans) for spans in rank_spans),
        global_counts=tuple(global_counts),
    )


def _proportional_step_allocations(
    *, remaining: tuple[int, ...], target: int, first_rank: int
) -> tuple[int, ...]:
    assert remaining
    assert all(count >= 0 for count in remaining)
    assert 0 < target <= sum(remaining)
    assert 0 <= first_rank < len(remaining)
    total = sum(remaining)
    products = tuple(target * count for count in remaining)
    allocations = [product // total for product in products]
    unallocated = target - sum(allocations)
    remainder_order = sorted(
        (rank for rank, count in enumerate(remaining) if count > 0),
        key=lambda rank: (
            -(products[rank] % total),
            (rank - first_rank) % len(remaining),
        ),
    )
    assert unallocated <= len(remainder_order)
    for rank in remainder_order[:unallocated]:
        assert allocations[rank] < remaining[rank]
        allocations[rank] += 1
    return tuple(allocations)


__all__ = (
    "DistributedBatchSchedule",
    "MinibatchSpan",
    "plan_distributed_minibatches",
)
