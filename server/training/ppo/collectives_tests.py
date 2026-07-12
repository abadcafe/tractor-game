"""Black-box tests for functional PPO tensor collectives."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.distributed as dist

from server.training.ppo.collectives import (
    all_reduce_max,
    all_reduce_sum,
)


def test_all_reduce_sum_preserves_input_without_warning(
    tmp_path: Path,
) -> None:
    source = torch.tensor([2.0, 3.0])

    with _SingleRankProcessGroup(tmp_path / "sum"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = all_reduce_sum(source)

    assert result is not source
    assert torch.equal(result, source)
    assert not caught


def test_all_reduce_max_preserves_input_without_warning(
    tmp_path: Path,
) -> None:
    source = torch.tensor(7, dtype=torch.long)

    with _SingleRankProcessGroup(tmp_path / "max"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = all_reduce_max(source)

    assert result is not source
    assert torch.equal(result, source)
    assert not caught


class _SingleRankProcessGroup:
    def __init__(self, rendezvous_path: Path) -> None:
        self._rendezvous_path = rendezvous_path

    def __enter__(self) -> None:
        assert not dist.is_initialized()
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{self._rendezvous_path.as_posix()}",
            rank=0,
            world_size=1,
        )

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exception_type, exception, traceback
        dist.destroy_process_group()
