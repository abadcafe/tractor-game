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
    if len(rows) <= 1 or not _should_split_by_shape(counts):
        return InferenceShapePlan(
            buckets=(rows,),
            original_padded_tokens=original_tokens,
            planned_padded_tokens=original_tokens,
        )
    exact_buckets = _exact_shape_buckets(rows=rows, counts=counts)
    buckets = _coalesce_small_shape_buckets(exact_buckets)
    planned_tokens = sum(
        _padded_token_count(
            _row_generation_counts(generation_step_counts, rows=bucket)
        )
        for bucket in buckets
    )
    return InferenceShapePlan(
        buckets=buckets,
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


def _should_split_by_shape(counts: tuple[int, ...]) -> bool:
    assert counts
    max_count = max(counts)
    useful_steps = sum(counts)
    padded_steps = len(counts) * max_count
    padding_waste = padded_steps - useful_steps
    return padding_waste >= max(len(counts), 8)


def _exact_shape_buckets(
    *, rows: tuple[int, ...], counts: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    buckets: dict[int, list[int]] = {}
    for row, step_count in zip(rows, counts, strict=True):
        if step_count not in buckets:
            buckets[step_count] = []
        buckets[step_count].append(row)
    return tuple(tuple(bucket) for bucket in buckets.values())


def _coalesce_small_shape_buckets(
    buckets: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    pending: list[int] = []
    result: list[tuple[int, ...]] = []
    for bucket in buckets:
        if len(bucket) >= 4:
            if pending:
                result.append(tuple(pending))
                pending.clear()
            result.append(bucket)
            continue
        pending.extend(bucket)
        if len(pending) >= 4:
            result.append(tuple(pending))
            pending.clear()
    if pending:
        if result:
            result[-1] = (*result[-1], *tuple(pending))
        else:
            result.append(tuple(pending))
    return tuple(result)
