"""Tests for model-rank inference shape planning."""

from __future__ import annotations

from server.training.runtime.model_rank.shape_planner import (
    plan_inference_shape_batches,
)


def test_plan_inference_shape_batches_keeps_uniform_rows() -> None:
    plan = plan_inference_shape_batches((3, 3, 3), rows=(0, 1, 2))

    assert plan.buckets == ((0, 1, 2),)
    assert plan.bucket_count() == 1
    assert plan.saved_padding_tokens() == 0


def test_plan_inference_shape_batches_splits_large_padding_waste() -> (
    None
):
    plan = plan_inference_shape_batches(
        (1, 1, 1, 1, 8, 8, 8, 8),
        rows=(0, 1, 2, 3, 4, 5, 6, 7),
    )

    assert plan.buckets == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert plan.bucket_count() == 2
    assert plan.saved_padding_tokens() == 28


def test_plan_inference_shape_batches_coalesces_small_buckets() -> None:
    plan = plan_inference_shape_batches(
        (1, 2, 8, 8),
        rows=(0, 1, 2, 3),
    )

    assert plan.buckets == ((0, 1, 2, 3),)
    assert plan.bucket_count() == 1
    assert plan.saved_padding_tokens() == 0
