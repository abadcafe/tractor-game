"""Derive round contracts from declarations and team levels."""

import server.game.rules.bidding as bidding
from server.game.rules.cards import Rank, Suit
from server.game.seating import Partnership, Seat, partnership_of

from .round_values import Declaration, TeamLevels


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
        return levels.at(Partnership.FIRST)
    return levels.at(partnership_of(declarer))
