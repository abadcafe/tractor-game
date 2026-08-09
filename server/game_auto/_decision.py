"""Rule-driven automatic decisions over a complete player snapshot."""

from __future__ import annotations

import random

from server.game import Seat, commands
from server.game.rules import bidding, play
from server.game.rules.cards import CardId
from server.game.snapshots import PlayerSnapshot

type AutoCommand = (
    commands.RevealBid
    | commands.PassBid
    | commands.Stir
    | commands.PassStir
    | commands.Bury
    | commands.Play
)


def choose_auto_command(
    *,
    actor: Seat,
    snapshot: PlayerSnapshot,
    random_source: random.Random,
) -> AutoCommand:
    """Choose one rule-legal strategic command without runtime state."""
    action = snapshot.awaiting_action
    if action == "bid":
        if (
            snapshot.bid_winner is not None
            and snapshot.bid_winner.actor == actor
        ):
            return commands.PassBid()
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
