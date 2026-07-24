"""Derive round contracts from declarations and team levels."""

from server.game.config import Seat
from server.game.rules import bidding
from server.game.rules.cards import Rank, Suit

from .round_values import Declaration, TeamLevels
from .topology import team


def rule_declaration(
    declaration: Declaration | None,
) -> bidding.Declaration | None:
    if declaration is None:
        return None
    return bidding.Declaration(cards=declaration.cards)


def declaration_suit(
    declaration: Declaration | None,
) -> Suit | None:
    rule_value = rule_declaration(declaration)
    if rule_value is None:
        return None
    return rule_value.suit


def trump_rank(
    levels: TeamLevels,
    declarer: Seat | None,
) -> Rank:
    if declarer is None:
        return levels[0]
    return levels[team(declarer)]
