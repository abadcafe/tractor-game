"""Rule-driven automatic decision policy."""

from __future__ import annotations

import random
from typing import final

from server.foundation.result import Ok, Rejected
from server.game import commands
from server.game.rules import bidding, play
from server.game.rules.cards import CardId
from server.game_runtime.player import PlayerView

from ._policy import DecisionCommand, DecisionRequest


@final
class AutoPolicy:
    """Choose one rule-legal command without external services."""

    def __init__(self, random_source: random.Random) -> None:
        self._random = random_source

    def observe(self, view: PlayerView) -> Ok[None] | Rejected:
        """Accept every view without retaining policy state."""
        del view
        return Ok(None)

    async def decide(
        self,
        request: DecisionRequest,
    ) -> Ok[DecisionCommand] | Rejected:
        """Choose one legal strategic command."""
        return Ok(_automatic_command(request, self._random))


def _automatic_command(
    request: DecisionRequest,
    random_source: random.Random,
) -> DecisionCommand:
    snapshot = request.view.snapshot
    action = request.action
    if action == "bid":
        current = (
            None
            if snapshot.bid_winner is None
            else bidding.Declaration(cards=snapshot.bid_winner.cards)
        )
        reveals = bidding.legal_reveals(
            snapshot.hand,
            snapshot.trump_rank,
            current,
        )
        if reveals and random_source.random() < 0.15:
            reveal = random_source.choice(reveals)
            return commands.RevealBid(
                card_ids=tuple(CardId(card.id) for card in reveal.cards)
            )
        return commands.PassBid()
    if action == "discard":
        selected = random_source.sample(
            list(snapshot.hand),
            len(snapshot.bottom_cards),
        )
        return commands.Bury(
            card_ids=tuple(CardId(card.id) for card in selected)
        )
    if action == "stir":
        current = (
            None
            if snapshot.bid_winner is None
            else bidding.Declaration(cards=snapshot.bid_winner.cards)
        )
        reveals = tuple(
            reveal
            for reveal in bidding.legal_reveals(
                snapshot.hand,
                snapshot.trump_rank,
                current,
            )
            if len(reveal.cards) == 2
        )
        if reveals and random_source.random() < 0.15:
            reveal = random_source.choice(reveals)
            return commands.Stir(
                card_ids=tuple(CardId(card.id) for card in reveal.cards)
            )
        return commands.PassStir()
    assert action == "play"
    lead = None
    if snapshot.trick is not None and snapshot.trick.slots:
        lead = snapshot.trick.slots[0].cards
    selected = play.choose_legal_play(
        snapshot.hand,
        lead,
        snapshot.trump_suit,
        snapshot.trump_rank,
    )
    return commands.Play(
        card_ids=tuple(CardId(card.id) for card in selected)
    )


__all__ = ("AutoPolicy",)
