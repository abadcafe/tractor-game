"""Condition and sample complete round deals for particle proposals."""

from __future__ import annotations

import random
from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game import (
    GameConfig,
    GameSeed,
    PartnershipMap,
    RoundDeal,
    Seat,
    next_seat,
    seats,
)
from server.game.rules.cards import Card, Rank, create_decks
from server.game.snapshots.events import BidEventSnapshot

from ._history import ObservedFrame


@dataclass(frozen=True, slots=True)
class DealConstraints:
    """Player-known constraints on one complete shuffled deal."""

    config: GameConfig
    round_number: int
    partnership_levels: PartnershipMap[Rank]
    fixed_declarer: Seat | None
    start_actor: Seat
    fixed_positions: tuple[tuple[int, Card], ...]
    bid_events: tuple[BidEventSnapshot, ...]


def deal_constraints(
    *,
    viewer: Seat,
    frames: tuple[ObservedFrame, ...],
) -> Ok[DealConstraints] | Rejected:
    """Extract exact deal constraints from a complete round history."""
    first = frames[0].snapshot
    if first.phase != "DEAL_BID":
        return Rejected(
            reason="AI round history does not start at deal"
        )
    first_actors = [
        seat
        for seat, count in first.remaining_cards.items()
        if count == 1
    ]
    if len(first_actors) != 1:
        return Rejected(reason="AI cannot determine opening deal actor")
    start_actor = first_actors[0]
    fixed: dict[int, Card] = {}
    previous_ids: set[str] = set()
    for frame in frames:
        snapshot = frame.snapshot
        if snapshot.phase != "DEAL_BID":
            break
        current_ids = {card.id for card in snapshot.hand}
        added = current_ids - previous_ids
        total = sum(snapshot.remaining_cards.values())
        if added:
            if len(added) != 1:
                return Rejected(
                    reason="AI observed multiple cards in one deal step"
                )
            position = total - 1
            if _deal_actor(start_actor, position) != viewer:
                return Rejected(
                    reason="AI observed a card dealt to another seat"
                )
            card = next(
                item for item in snapshot.hand if item.id in added
            )
            fixed[8 + position] = card
        previous_ids = current_ids
    initial_bottom = _known_initial_bottom(frames)
    if initial_bottom is not None:
        for index, card in enumerate(initial_bottom):
            fixed[index] = card
    latest = frames[-1].snapshot
    return Ok(
        DealConstraints(
            config=GameConfig(mandatory_levels=latest.mandatory_levels),
            round_number=latest.round_number,
            partnership_levels=latest.partnership_levels,
            fixed_declarer=first.declarer,
            start_actor=start_actor,
            fixed_positions=tuple(sorted(fixed.items())),
            bid_events=tuple(latest.bid_events),
        )
    )


def sample_round_deal(
    *,
    constraints: DealConstraints,
    random_source: random.Random,
) -> Ok[RoundDeal] | Rejected:
    """Sample one complete deal satisfying all direct constraints."""
    canonical = tuple(create_decks())
    by_id = {card.id: card for card in canonical}
    positions: list[Card | None] = [None for _ in canonical]
    used: set[str] = set()
    for position, card in constraints.fixed_positions:
        expected = by_id.get(card.id)
        if (
            expected != card
            or position < 0
            or position >= len(positions)
        ):
            return Rejected(reason="AI deal constraint is invalid")
        if positions[position] is not None or card.id in used:
            return Rejected(reason="AI deal constraint is duplicated")
        positions[position] = card
        used.add(card.id)
    for event in constraints.bid_events:
        for card in event.cards:
            if card.id in used:
                assigned = next(
                    index
                    for index, value in enumerate(positions)
                    if value is not None and value.id == card.id
                )
                if (
                    assigned < 8
                    or assigned >= 8 + event.deal_ordinal
                    or _deal_actor(
                        constraints.start_actor,
                        assigned - 8,
                    )
                    != event.actor
                ):
                    return Rejected(
                        reason="AI reveal conflicts with known deal"
                    )
                continue
            eligible = [
                8 + deal_index
                for deal_index in range(event.deal_ordinal)
                if positions[8 + deal_index] is None
                and _deal_actor(
                    constraints.start_actor,
                    deal_index,
                )
                == event.actor
            ]
            if not eligible:
                return Rejected(
                    reason="AI reveal has no possible deal position"
                )
            position = random_source.choice(eligible)
            positions[position] = card
            used.add(card.id)
    remaining = [card for card in canonical if card.id not in used]
    random_source.shuffle(remaining)
    remaining_iter = iter(remaining)
    completed = tuple(
        next(remaining_iter) if card is None else card
        for card in positions
    )
    return Ok(
        RoundDeal(
            config=constraints.config,
            continuation_seed=GameSeed(random_source.getrandbits(64)),
            round_number=constraints.round_number,
            partnership_levels=constraints.partnership_levels,
            fixed_declarer=constraints.fixed_declarer,
            start_actor=constraints.start_actor,
            shuffled_cards=completed,
        )
    )


def _known_initial_bottom(
    frames: tuple[ObservedFrame, ...],
) -> tuple[Card, ...] | None:
    for frame in frames:
        exchange = frame.snapshot.own_initial_bottom_exchange
        if exchange is not None:
            return exchange.picked_up_bottom_cards
        if (
            frame.snapshot.awaiting_action == "discard"
            and frame.snapshot.bottom_cards
            and not frame.snapshot.stir_events
        ):
            return frame.snapshot.bottom_cards
    return None


def _deal_actor(start: Seat, deal_index: int) -> Seat:
    assert deal_index >= 0
    actor = start
    for _ in range(deal_index % len(seats())):
        actor = next_seat(actor)
    return actor


__all__ = ("DealConstraints", "deal_constraints", "sample_round_deal")
