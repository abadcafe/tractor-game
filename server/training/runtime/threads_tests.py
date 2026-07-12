"""Tests for torch thread runtime control."""

from __future__ import annotations

import pytest

from server.foundation.result import Ok, Rejected
from server.training.runtime import threads
from server.training.runtime.threads import apply_torch_thread_config


def test_apply_torch_thread_config_inherits_current_thread_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_calls: list[int] = []
    interop_calls: list[int] = []

    def fake_num_threads() -> int:
        return 8

    def fake_num_interop_threads() -> int:
        return 2

    def fake_set_num_threads(value: int) -> None:
        set_calls.append(value)

    def fake_set_num_interop_threads(value: int) -> None:
        interop_calls.append(value)

    monkeypatch.setattr(
        threads.torch, "get_num_threads", fake_num_threads
    )
    monkeypatch.setattr(
        threads.torch,
        "get_num_interop_threads",
        fake_num_interop_threads,
    )
    monkeypatch.setattr(
        threads.torch, "set_num_threads", fake_set_num_threads
    )
    monkeypatch.setattr(
        threads.torch,
        "set_num_interop_threads",
        fake_set_num_interop_threads,
    )

    applied = apply_torch_thread_config(
        num_threads=None,
        num_interop_threads=None,
    )

    assert isinstance(applied, Ok)
    assert applied.value.requested_num_threads is None
    assert applied.value.requested_num_interop_threads is None
    assert applied.value.active_num_threads == 8
    assert applied.value.active_num_interop_threads == 2
    assert set_calls == []
    assert interop_calls == []


def test_apply_torch_thread_config_applies_changed_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_calls: list[int] = []
    interop_calls: list[int] = []

    def fake_num_threads() -> int:
        return 4

    def fake_num_interop_threads() -> int:
        return 1

    def fake_set_num_threads(value: int) -> None:
        set_calls.append(value)

    def fake_set_num_interop_threads(value: int) -> None:
        interop_calls.append(value)

    monkeypatch.setattr(
        threads.torch, "get_num_threads", fake_num_threads
    )
    monkeypatch.setattr(
        threads.torch,
        "get_num_interop_threads",
        fake_num_interop_threads,
    )
    monkeypatch.setattr(
        threads.torch, "set_num_threads", fake_set_num_threads
    )
    monkeypatch.setattr(
        threads.torch,
        "set_num_interop_threads",
        fake_set_num_interop_threads,
    )

    applied = apply_torch_thread_config(
        num_threads=2,
        num_interop_threads=3,
    )

    assert isinstance(applied, Ok)
    assert applied.value.requested_num_threads == 2
    assert applied.value.requested_num_interop_threads == 3
    assert set_calls == [2]
    assert interop_calls == [3]


def test_apply_torch_thread_config_skips_matching_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_calls: list[int] = []
    interop_calls: list[int] = []

    def fake_num_threads() -> int:
        return 4

    def fake_num_interop_threads() -> int:
        return 2

    def fake_set_num_threads(value: int) -> None:
        set_calls.append(value)

    def fake_set_num_interop_threads(value: int) -> None:
        interop_calls.append(value)

    monkeypatch.setattr(
        threads.torch, "get_num_threads", fake_num_threads
    )
    monkeypatch.setattr(
        threads.torch,
        "get_num_interop_threads",
        fake_num_interop_threads,
    )
    monkeypatch.setattr(
        threads.torch, "set_num_threads", fake_set_num_threads
    )
    monkeypatch.setattr(
        threads.torch,
        "set_num_interop_threads",
        fake_set_num_interop_threads,
    )

    applied = apply_torch_thread_config(
        num_threads=4,
        num_interop_threads=2,
    )

    assert isinstance(applied, Ok)
    assert set_calls == []
    assert interop_calls == []


def test_apply_torch_thread_config_rejects_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_num_threads() -> int:
        return 8

    def fake_num_interop_threads() -> int:
        return 2

    def fake_set_num_threads(value: int) -> None:
        assert value == 1
        raise RuntimeError

    monkeypatch.setattr(
        threads.torch, "get_num_threads", fake_num_threads
    )
    monkeypatch.setattr(
        threads.torch,
        "get_num_interop_threads",
        fake_num_interop_threads,
    )
    monkeypatch.setattr(
        threads.torch, "set_num_threads", fake_set_num_threads
    )

    applied = apply_torch_thread_config(
        num_threads=1,
        num_interop_threads=None,
    )

    assert isinstance(applied, Rejected)
    assert "must be applied before use" in applied.reason
