"""Tests for CPU affinity runtime control."""

from __future__ import annotations

import pytest

from server.foundation.result import Ok, Rejected
from server.training.runtime import affinity
from server.training.runtime.affinity import (
    apply_cpu_affinity,
    current_cpu_affinity,
    preflight_cpu_affinity,
)


def test_apply_cpu_affinity_empty_cpu_set_reads_current_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaffinity(pid: int) -> set[int]:
        assert pid == 0
        return {3, 1}

    monkeypatch.setattr(affinity.sys, "platform", "linux")
    monkeypatch.setattr(
        affinity.os, "sched_getaffinity", fake_getaffinity
    )

    applied = apply_cpu_affinity(label="coordinator", cpus=())

    assert isinstance(applied, Ok)
    assert applied.value.label == "coordinator"
    assert applied.value.requested_cpus == ()
    assert applied.value.active_cpus == (1, 3)


def test_apply_cpu_affinity_rejects_non_linux_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(affinity.sys, "platform", "darwin")

    applied = apply_cpu_affinity(label="worker-0", cpus=(4,))

    assert isinstance(applied, Rejected)
    assert "unavailable" in applied.reason
    assert "worker-0" in applied.reason


def test_apply_cpu_affinity_sets_requested_linux_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[tuple[int, set[int]]] = []

    def fake_setaffinity(pid: int, cpus: set[int]) -> None:
        requested.append((pid, cpus))

    def fake_getaffinity(pid: int) -> set[int]:
        assert pid == 0
        return {5, 4}

    monkeypatch.setattr(affinity.sys, "platform", "linux")
    monkeypatch.setattr(
        affinity.os, "sched_setaffinity", fake_setaffinity
    )
    monkeypatch.setattr(
        affinity.os, "sched_getaffinity", fake_getaffinity
    )

    applied = apply_cpu_affinity(label="worker-1", cpus=(4, 5))

    assert isinstance(applied, Ok)
    assert requested == [(0, {4, 5})]
    assert applied.value.active_cpus == (4, 5)


def test_apply_cpu_affinity_rejects_operating_system_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_setaffinity(pid: int, cpus: set[int]) -> None:
        assert pid == 0
        assert cpus == {7}
        raise OSError

    monkeypatch.setattr(affinity.sys, "platform", "linux")
    monkeypatch.setattr(
        affinity.os, "sched_setaffinity", fake_setaffinity
    )

    applied = apply_cpu_affinity(label="worker-2", cpus=(7,))

    assert isinstance(applied, Rejected)
    assert "affinity apply failed" in applied.reason


def test_preflight_cpu_affinity_restores_original_cpu_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[tuple[int, set[int]]] = []
    active_sets: list[set[int]] = [{0, 1}]

    def fake_setaffinity(pid: int, cpus: set[int]) -> None:
        assert pid == 0
        requested.append((pid, set(cpus)))
        active_sets[0] = set(cpus)

    def fake_getaffinity(pid: int) -> set[int]:
        assert pid == 0
        return set(active_sets[0])

    monkeypatch.setattr(affinity.sys, "platform", "linux")
    monkeypatch.setattr(
        affinity.os, "sched_setaffinity", fake_setaffinity
    )
    monkeypatch.setattr(
        affinity.os, "sched_getaffinity", fake_getaffinity
    )

    checked = preflight_cpu_affinity(label="worker-0", cpus=(3, 4))

    assert isinstance(checked, Ok)
    assert checked.value.active_cpus == (3, 4)
    assert requested == [(0, {3, 4}), (0, {0, 1})]
    assert active_sets == [{0, 1}]


def test_preflight_cpu_affinity_rejects_invalid_cpu_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[tuple[int, set[int]]] = []

    def fake_setaffinity(pid: int, cpus: set[int]) -> None:
        assert pid == 0
        requested.append((pid, set(cpus)))
        if cpus == {9}:
            raise OSError

    def fake_getaffinity(pid: int) -> set[int]:
        assert pid == 0
        return {0, 1}

    monkeypatch.setattr(affinity.sys, "platform", "linux")
    monkeypatch.setattr(
        affinity.os, "sched_setaffinity", fake_setaffinity
    )
    monkeypatch.setattr(
        affinity.os, "sched_getaffinity", fake_getaffinity
    )

    checked = preflight_cpu_affinity(label="worker-0", cpus=(9,))

    assert isinstance(checked, Rejected)
    assert "CPU affinity preflight failed" in checked.reason
    assert requested == [(0, {9}), (0, {0, 1})]


def test_current_cpu_affinity_returns_empty_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(affinity.sys, "platform", "freebsd")

    assert current_cpu_affinity() == ()


def test_current_cpu_affinity_returns_empty_on_os_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaffinity(pid: int) -> set[int]:
        assert pid == 0
        raise OSError

    monkeypatch.setattr(affinity.sys, "platform", "linux")
    monkeypatch.setattr(
        affinity.os, "sched_getaffinity", fake_getaffinity
    )

    assert current_cpu_affinity() == ()
