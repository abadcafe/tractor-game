"""Project internal contracts into explicit player-visible states."""

from server.game.rules.cards import Rank
from server.game.snapshots.contract import (
    NoTrump,
    SuitedTrump,
    TrumpSnapshot,
)

from . import phases
from .round_values import Contract


def contract_snapshot(contract: Contract) -> TrumpSnapshot:
    if contract.trump_suit is None:
        return NoTrump(rank=contract.trump_rank)
    return SuitedTrump(
        rank=contract.trump_rank,
        suit=contract.trump_suit,
    )


def deal_trump_rank(phase: phases.DealBid) -> Rank:
    declarer = phase.fixed_declarer
    team = 0 if declarer is None else int(declarer) % 2
    return phase.levels[team]
