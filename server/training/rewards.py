"""Torch-free reward semantics shared by training and inference."""

from __future__ import annotations

from server.game import Partnership, partnership_of
from server.game.rules.progression import TerminalProgress
from server.game.snapshots import PlayerSnapshot
from server.training.progress import (
    PartnershipProgress,
    RoundScore,
    zero_sum_rewards,
)


def round_rewards(
    *,
    before: PlayerSnapshot,
    after: PlayerSnapshot,
) -> tuple[float, float]:
    """Return zero-sum partnership rewards for a completed round."""
    scoring = after.scoring
    assert scoring is not None
    winning_partnership = scoring.winning_partnership
    declarer = after.declarer
    assert declarer is not None
    declarer_partnership = partnership_of(declarer)
    if before.declarer is not None:
        assert partnership_of(before.declarer) == declarer_partnership
    first_before = PartnershipProgress(
        level=before.partnership_levels.at(Partnership.FIRST),
        is_declarer=declarer_partnership == Partnership.FIRST,
    )
    second_before = PartnershipProgress(
        level=before.partnership_levels.at(Partnership.SECOND),
        is_declarer=declarer_partnership == Partnership.SECOND,
    )
    first_after = PartnershipProgress(
        level=(
            TerminalProgress.WIN
            if after.winning_partnership == Partnership.FIRST
            else after.partnership_levels.at(Partnership.FIRST)
        ),
        is_declarer=winning_partnership == Partnership.FIRST,
    )
    second_after = PartnershipProgress(
        level=(
            TerminalProgress.WIN
            if after.winning_partnership == Partnership.SECOND
            else after.partnership_levels.at(Partnership.SECOND)
        ),
        is_declarer=winning_partnership == Partnership.SECOND,
    )
    reward = zero_sum_rewards(
        first_before=first_before,
        second_before=second_before,
        first_after=first_after,
        second_after=second_after,
        score=RoundScore(
            declarer_partnership=declarer_partnership,
            total_defender_points=scoring.total_defender_points,
            mandatory_levels=before.mandatory_levels,
        ),
    )
    return (
        reward.first_partnership,
        reward.second_partnership,
    )


__all__ = ("round_rewards",)
