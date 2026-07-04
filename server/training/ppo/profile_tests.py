"""Tests for PPO update profiling."""

from __future__ import annotations

import pytest
import torch

import server.training.ppo.profile as profile_module
from server.training.ppo.profile import PPOProfileAccumulator


def test_start_finish_basic_cuda_synchronizes_update_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    sync_devices: list[torch.device | None] = []
    sync_count_at_perf_counter: list[int] = []
    perf_counter_values = iter((10.0, 17.0))

    def fake_synchronize(device: torch.device | None = None) -> None:
        sync_devices.append(device)

    def fake_perf_counter() -> float:
        sync_count_at_perf_counter.append(len(sync_devices))
        return next(perf_counter_values)

    monkeypatch.setattr(torch.cuda, "synchronize", fake_synchronize)
    monkeypatch.setattr(
        profile_module.time,
        "perf_counter",
        fake_perf_counter,
    )

    accumulator = PPOProfileAccumulator.start(
        device=device,
        mode="basic",
    )
    profile = accumulator.finish()

    assert sync_devices == [device, device]
    assert sync_count_at_perf_counter == [1, 2]
    assert profile.update_seconds == 7.0
    assert profile.minibatch_loss_seconds == 0.0
