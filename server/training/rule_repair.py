"""Rule-backed emergency actions for keeping self-play moving."""

from __future__ import annotations

from itertools import combinations

from server.protocol import StateSnapshot
from server.result import Ok, Rejected
from server.rules.cards import Card
from server.rules.follow import is_legal_follow
from server.training.action_tokens import (
    ACTION_BID_TOKEN_ID,
    ACTION_DISCARD_TOKEN_ID,
    ACTION_PASS_TOKEN_ID,
    ACTION_PLAY_TOKEN_ID,
    ACTION_STIR_TOKEN_ID,
    BEGIN_TOKEN_ID,
    FIRST_CARD_TOKEN_ID,
    STOP_TOKEN_ID,
    GeneratedAction,
    ModelActionKind,
    build_action_query,
    decode_action_tokens,
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
        return _decode(
            player_index=player_index,
            snapshot=snapshot,
            token_ids=(
                BEGIN_TOKEN_ID,
                ACTION_PASS_TOKEN_ID,
                STOP_TOKEN_ID,
            ),
        )
    if snapshot.awaiting_action == "stir":
        return _decode(
            player_index=player_index,
            snapshot=snapshot,
            token_ids=(
                BEGIN_TOKEN_ID,
                ACTION_PASS_TOKEN_ID,
                STOP_TOKEN_ID,
            ),
        )
    if snapshot.awaiting_action == "discard":
        return _cards_action(
            player_index=player_index,
            snapshot=snapshot,
            action_kind="discard",
            cards=_discard_cards(snapshot),
        )
    if snapshot.awaiting_action == "play":
        return _cards_action(
            player_index=player_index,
            snapshot=snapshot,
            action_kind="play",
            cards=_play_cards(snapshot),
        )
    return decode_action_tokens(query, ())


def _cards_action(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    action_kind: ModelActionKind,
    cards: tuple[Card, ...],
) -> Ok[GeneratedAction] | Rejected:
    token_id = _action_token_id(action_kind)
    slot_tokens: list[int] = []
    hand_ids = [card.id for card in snapshot.player_hand]
    for card_item in cards:
        slot = hand_ids.index(card_item.id)
        slot_tokens.append(FIRST_CARD_TOKEN_ID + slot)
    return _decode(
        player_index=player_index,
        snapshot=snapshot,
        token_ids=(
            BEGIN_TOKEN_ID,
            token_id,
            *tuple(slot_tokens),
            STOP_TOKEN_ID,
        ),
    )


def _discard_cards(snapshot: StateSnapshot) -> tuple[Card, ...]:
    query = build_action_query(player_index=0, snapshot=snapshot)
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


def _decode(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    token_ids: tuple[int, ...],
) -> Ok[GeneratedAction] | Rejected:
    return decode_action_tokens(
        build_action_query(
            player_index=player_index,
            snapshot=snapshot,
        ),
        token_ids,
    )


def _action_token_id(action_kind: ModelActionKind) -> int:
    if action_kind == "bid":
        return ACTION_BID_TOKEN_ID
    if action_kind == "stir":
        return ACTION_STIR_TOKEN_ID
    if action_kind == "discard":
        return ACTION_DISCARD_TOKEN_ID
    return ACTION_PLAY_TOKEN_ID
