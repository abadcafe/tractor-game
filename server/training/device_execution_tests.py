"""Tests for device-specific model execution policy."""

from __future__ import annotations

import pytest
import torch

from server.training.device_execution import DeviceExecutionPolicy


def test_device_execution_policy_keeps_cpu_in_float32() -> None:
    policy = DeviceExecutionPolicy.for_device(torch.device("cpu"))

    assert not policy.uses_bfloat16


def test_device_execution_policy_enables_supported_cuda_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def supports_bfloat16() -> bool:
        return True

    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", supports_bfloat16
    )

    policy = DeviceExecutionPolicy.for_device(torch.device("cuda:0"))

    assert policy.uses_bfloat16


def test_device_execution_policy_keeps_unsupported_cuda_in_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def supports_bfloat16() -> bool:
        return False

    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", supports_bfloat16
    )

    policy = DeviceExecutionPolicy.for_device(torch.device("cuda:0"))

    assert not policy.uses_bfloat16
