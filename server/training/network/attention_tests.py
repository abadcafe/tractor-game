"""Black-box tests for structure-aware observation attention."""

from __future__ import annotations

import torch
from torch import Tensor

from server.training.network.attention import (
    StructuredObservationEncoder,
)


def test_inactive_structure_axis_is_coordinate_invariant() -> None:
    encoder, values, padding = _fixture()
    first = _encoded_rows(((2, 0, 0), (0, 0, 0)))
    shifted = _encoded_rows(((42, 0, 0), (0, 0, 0)))

    assert torch.allclose(
        _encode(encoder, values, padding, first),
        _encode(encoder, values, padding, shifted),
        atol=1e-6,
        rtol=1e-6,
    )


def test_active_structure_axis_is_invariant_to_shared_shift() -> None:
    encoder, values, padding = _fixture()
    first = _encoded_rows(((2, 0, 0), (5, 0, 0)))
    shifted = _encoded_rows(((32, 0, 0), (35, 0, 0)))

    assert torch.allclose(
        _encode(encoder, values, padding, first),
        _encode(encoder, values, padding, shifted),
        atol=1e-6,
        rtol=1e-6,
    )


def test_active_structure_axis_uses_relative_difference() -> None:
    encoder, values, padding = _fixture()
    first = _encoded_rows(((2, 0, 0), (5, 0, 0)))
    changed = _encoded_rows(((2, 0, 0), (6, 0, 0)))

    assert not torch.allclose(
        _encode(encoder, values, padding, first),
        _encode(encoder, values, padding, changed),
    )


def test_active_coordinate_zero_differs_from_absent_axis() -> None:
    encoder, values, padding = _fixture()
    active_zero = _encoded_rows(((1, 0, 0), (2, 0, 0)))
    absent = _encoded_rows(((0, 0, 0), (2, 0, 0)))

    assert not torch.allclose(
        _encode(encoder, values, padding, active_zero),
        _encode(encoder, values, padding, absent),
    )


def test_structure_axes_are_independent() -> None:
    encoder, values, padding = _fixture()
    round_event = _encoded_rows(((2, 0, 0), (5, 0, 0)))
    trick = _encoded_rows(((0, 2, 0), (0, 5, 0)))

    assert not torch.allclose(
        _encode(encoder, values, padding, round_event),
        _encode(encoder, values, padding, trick),
    )


def test_structure_attention_is_finite_and_fully_differentiable() -> (
    None
):
    encoder, values, padding = _fixture()
    encoded = _encoded_rows(
        (
            (2, 1, 1),
            (2, 1, 1),
        )
    )

    output = _encode(encoder, values, padding, encoded)
    torch.autograd.backward(output.square().sum())

    assert bool(torch.isfinite(output).all().item())
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in encoder.parameters()
    )


def _fixture() -> tuple[StructuredObservationEncoder, Tensor, Tensor]:
    encoder = StructuredObservationEncoder(d_model=8, layers=1, heads=1)
    values = torch.randn(1, 2, 8)
    padding = torch.zeros((1, 2), dtype=torch.bool)
    return encoder, values, padding


def _encoded_rows(rows: tuple[tuple[int, int, int], ...]) -> Tensor:
    return torch.tensor((rows,), dtype=torch.long)


def _encode(
    encoder: StructuredObservationEncoder,
    values: Tensor,
    padding_mask: Tensor,
    encoded_structure_coordinates: Tensor,
) -> Tensor:
    return encoder(
        values,
        padding_mask=padding_mask,
        encoded_structure_coordinates=encoded_structure_coordinates,
    )
