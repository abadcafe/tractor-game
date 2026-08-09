"""Plan model-rank inference micro-batches by semantic trace shape."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InferenceShapePlan:
    """Execution buckets and padding telemetry."""

    buckets: tuple[tuple[int, ...], ...]
    original_padded_tokens: int
    planned_padded_tokens: int

    def __post_init__(self) -> None:
        assert self.buckets
        assert all(bucket for bucket in self.buckets)
        assert self.original_padded_tokens > 0
        assert self.planned_padded_tokens > 0
        assert self.planned_padded_tokens <= self.original_padded_tokens

    def bucket_count(self) -> int:
        """Return number of planned model invocations."""
        return len(self.buckets)

    def saved_padding_tokens(self) -> int:
        """Return semantic padding tokens avoided by this plan."""
        return self.original_padded_tokens - self.planned_padded_tokens


def plan_inference_shape_batches(
    generation_step_counts: tuple[int, ...],
    *,
    rows: tuple[int, ...],
) -> InferenceShapePlan:
    """Return micro-batches balancing padding and launches."""
    assert generation_step_counts
    assert rows
    counts = _row_generation_counts(generation_step_counts, rows=rows)
    original_tokens = _padded_token_count(counts)
    exact_buckets = _exact_shape_buckets(rows=rows, counts=counts)
    if not _should_split_by_shape(
        exact_buckets,
        generation_step_counts=generation_step_counts,
        original_tokens=original_tokens,
    ):
        return InferenceShapePlan(
            buckets=(rows,),
            original_padded_tokens=original_tokens,
            planned_padded_tokens=original_tokens,
        )
    planned_tokens = sum(
        _padded_token_count(
            _row_generation_counts(generation_step_counts, rows=bucket)
        )
        for bucket in exact_buckets
    )
    return InferenceShapePlan(
        buckets=exact_buckets,
        original_padded_tokens=original_tokens,
        planned_padded_tokens=planned_tokens,
    )


def _row_generation_counts(
    counts: tuple[int, ...], *, rows: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(counts[row] for row in rows)


def _padded_token_count(counts: tuple[int, ...]) -> int:
    assert counts
    return len(counts) * max(counts)


def _should_split_by_shape(
    buckets: tuple[tuple[int, ...], ...],
    *,
    generation_step_counts: tuple[int, ...],
    original_tokens: int,
) -> bool:
    assert buckets
    if sum(len(bucket) for bucket in buckets) < 32:
        return False
    if len(buckets) <= 1 or any(len(bucket) < 16 for bucket in buckets):
        return False
    planned_tokens = sum(
        _padded_token_count(
            _row_generation_counts(generation_step_counts, rows=bucket)
        )
        for bucket in buckets
    )
    return (original_tokens - planned_tokens) * 4 >= original_tokens


def _exact_shape_buckets(
    *, rows: tuple[int, ...], counts: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    buckets: dict[int, list[int]] = {}
    for row, step_count in zip(rows, counts, strict=True):
        if step_count not in buckets:
            buckets[step_count] = []
        buckets[step_count].append(row)
    return tuple(tuple(bucket) for bucket in buckets.values())
