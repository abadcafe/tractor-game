"""CPU affinity application for training processes."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from server.foundation import result as _result
from server.training.runtime.config import CpuSet


@dataclass(frozen=True, slots=True)
class CpuAffinityStatus:
    """Observed CPU affinity after an affinity operation."""

    label: str
    requested_cpus: CpuSet
    active_cpus: CpuSet


def apply_cpu_affinity(
    *,
    label: str,
    cpus: CpuSet,
) -> _result.Ok[CpuAffinityStatus] | _result.Rejected:
    """Bind the current process to a CPU set."""
    assert label
    if not cpus:
        return _result.Ok(
            value=CpuAffinityStatus(
                label=label,
                requested_cpus=(),
                active_cpus=current_cpu_affinity(),
            )
        )
    if not sys.platform.startswith("linux"):
        return _result.Rejected(
            reason=(
                f"CPU affinity is unavailable for {label} on "
                f"{sys.platform}"
            )
        )
    try:
        os.sched_setaffinity(0, set(cpus))
        active = tuple(sorted(os.sched_getaffinity(0)))
    except OSError:
        return _result.Rejected(
            reason=f"CPU affinity apply failed for {label}: {cpus}"
        )
    return _result.Ok(
        value=CpuAffinityStatus(
            label=label,
            requested_cpus=cpus,
            active_cpus=active,
        )
    )


def preflight_cpu_affinity(
    *,
    label: str,
    cpus: CpuSet,
) -> _result.Ok[CpuAffinityStatus] | _result.Rejected:
    """Validate a CPU affinity request and restore the caller mask."""
    assert label
    if not cpus:
        return _result.Ok(
            value=CpuAffinityStatus(
                label=label,
                requested_cpus=(),
                active_cpus=current_cpu_affinity(),
            )
        )
    if not sys.platform.startswith("linux"):
        return _result.Rejected(
            reason=(
                f"CPU affinity is unavailable for {label} on "
                f"{sys.platform}"
            )
        )
    try:
        original = set(os.sched_getaffinity(0))
    except OSError:
        return _result.Rejected(
            reason=f"CPU affinity preflight failed for {label}: {cpus}"
        )
    restore_failed = False
    try:
        os.sched_setaffinity(0, set(cpus))
        active = tuple(sorted(os.sched_getaffinity(0)))
    except OSError:
        return _result.Rejected(
            reason=f"CPU affinity preflight failed for {label}: {cpus}"
        )
    finally:
        try:
            os.sched_setaffinity(0, original)
        except OSError:
            restore_failed = True
    if restore_failed:
        return _result.Rejected(
            reason=(
                "CPU affinity restore failed after preflight for "
                f"{label}"
            )
        )
    requested = set(cpus)
    if not requested.issubset(set(active)):
        return _result.Rejected(
            reason=f"CPU affinity preflight failed for {label}: {cpus}"
        )
    return _result.Ok(
        value=CpuAffinityStatus(
            label=label,
            requested_cpus=cpus,
            active_cpus=active,
        )
    )


def current_cpu_affinity() -> CpuSet:
    """Return current Linux CPU affinity, or empty if unavailable."""
    if not sys.platform.startswith("linux"):
        return ()
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except OSError:
        return ()
