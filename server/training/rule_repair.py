"""Rule-backed emergency actions for keeping self-play moving."""

from __future__ import annotations

from itertools import combinations

from server.protocol import StateSnapshot
from server.result import Ok, Rejected
from server.rules.cards import Card
from server.rules.follow import is_legal_follow
from server.training.selection_actions import (
    ActionQuery,
    GeneratedAction,
    SelectionChoice,
    SelectionTrace,
    build_action_query,
    decode_selection_action,
)

MAX_REPAIR_COMBINATIONS_SCANNED: int = 20000


def repair_action(
    *,
    player_index: int,
    snapshot: StateSnapshot,
) -> Ok[GeneratedAction] | Rejected:
    """Return a conservative legal-ish action without using hints."""
    query = build_action_query(
        player_index=player_index,
        snapshot=snapshot,
    )
    if snapshot.awaiting_action == "bid":
        return _pass_action(query)
    if snapshot.awaiting_action == "stir":
        return _pass_action(query)
    if snapshot.awaiting_action == "discard":
        return _cards_action(
            query=query,
            snapshot=snapshot,
            cards=_discard_cards(query, snapshot),
        )
    if snapshot.awaiting_action == "play":
        return _cards_action(
            query=query,
            snapshot=snapshot,
            cards=_play_cards(snapshot),
        )
    return decode_selection_action(query, SelectionTrace(choices=()))


def _pass_action(query: ActionQuery) -> Ok[GeneratedAction] | Rejected:
    return decode_selection_action(
        query,
        SelectionTrace(choices=(SelectionChoice("pass"),)),
    )


def _cards_action(
    *,
    query: ActionQuery,
    snapshot: StateSnapshot,
    cards: tuple[Card, ...],
) -> Ok[GeneratedAction] | Rejected:
    hand_ids = [card.id for card in snapshot.player_hand]
    choices = [
        SelectionChoice("select_card", hand_ids.index(card_item.id))
        for card_item in cards
    ]
    if query.kind in ("bid", "stir", "lead_play"):
        choices.append(SelectionChoice("stop"))
    return decode_selection_action(
        query,
        SelectionTrace(choices=tuple(choices)),
    )


def _discard_cards(
    query: ActionQuery,
    snapshot: StateSnapshot,
) -> tuple[Card, ...]:
    sorted_hand = sorted(
        snapshot.player_hand,
        key=lambda card: (
            card.points,
            card.suit.value,
            card.rank.value,
        ),
    )
    discard_count = (
        8 if query.discard_count is None else query.discard_count
    )
    return tuple(sorted_hand[:discard_count])


def _play_cards(snapshot: StateSnapshot) -> tuple[Card, ...]:
    hand = list(snapshot.player_hand)
    if not hand:
        return ()
    lead_cards = _lead_cards(snapshot)
    if not lead_cards:
        return (hand[0],)
    lead_count = len(lead_cards)
    if lead_count <= 0:
        return (hand[0],)
    scanned = 0
    for combo in combinations(hand, lead_count):
        scanned += 1
        candidate = list(combo)
        if is_legal_follow(
            hand,
            candidate,
            lead_cards,
            snapshot.trump_suit,
            snapshot.trump_rank,
        ):
            return tuple(candidate)
        if scanned >= MAX_REPAIR_COMBINATIONS_SCANNED:
            break
    return tuple(hand[:lead_count])


def _lead_cards(snapshot: StateSnapshot) -> list[Card]:
    trick = snapshot.trick
    if trick is None:
        return []
    for slot in trick.slots:
        if slot.player == trick.lead_player:
            return list(slot.cards)
    return []
