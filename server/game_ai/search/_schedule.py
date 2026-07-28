"""Deterministic paired-sample allocation for sequential halving."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HalvingRound:
    """One round's survivor width and new samples per candidate."""

    survivor_count: int
    samples_per_candidate: int

    def __post_init__(self) -> None:
        assert self.survivor_count > 1
        assert self.samples_per_candidate > 0


@dataclass(frozen=True, slots=True)
class HalvingSchedule:
    """Complete immutable compute schedule for one root decision."""

    rounds: tuple[HalvingRound, ...]
    sample_count: int

    def __post_init__(self) -> None:
        assert self.rounds
        assert self.sample_count == sum(
            item.survivor_count * item.samples_per_candidate
            for item in self.rounds
        )

    @classmethod
    def build(
        cls,
        *,
        candidate_count: int,
        sample_budget: int,
    ) -> HalvingSchedule:
        """Allocate equal paired blocks without exceeding the budget."""
        assert candidate_count > 1
        survivor_counts: list[int] = []
        survivors = candidate_count
        while survivors > 1:
            survivor_counts.append(survivors)
            survivors = (survivors + 1) // 2
        minimum = sum(survivor_counts)
        assert sample_budget >= minimum
        samples = [1 for _count in survivor_counts]
        remaining = sample_budget - minimum
        while True:
            affordable = [
                index
                for index, count in enumerate(survivor_counts)
                if count <= remaining
            ]
            if not affordable:
                break
            selected = min(
                affordable,
                key=lambda index: (
                    survivor_counts[index] * samples[index],
                    -index,
                ),
            )
            samples[selected] += 1
            remaining -= survivor_counts[selected]
        rounds = tuple(
            HalvingRound(
                survivor_count=count,
                samples_per_candidate=samples[index],
            )
            for index, count in enumerate(survivor_counts)
        )
        return cls(
            rounds=rounds,
            sample_count=sum(
                item.survivor_count * item.samples_per_candidate
                for item in rounds
            ),
        )


__all__ = ("HalvingRound", "HalvingSchedule")
