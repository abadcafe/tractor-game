"""Player message parsing for the game aggregate."""

from __future__ import annotations

from typing import TypeGuard, assert_never

from server.actions import (
    BidAction,
    CardActionKind,
    DiscardAction,
    GameActionKind,
    NextRoundAction,
    PlayAction,
    SkipBidAction,
    SkipStirAction,
    StirAction,
)
from server.protocol import PlayerMessage
from server.result import Ok, Rejected
from server.rules.cards import Card
from server.rules.rejections.bid import EmptyBidRejected
from server.rules.rejections.card import CardNotInHandRejected
from server.sm import round_sm
from server.sm.rejections.protocol import (
    GameNotStartedRejected,
    InvalidCardFormatRejected,
    MissingActionTypeRejected,
    MissingCardIdRejected,
    UnknownActionTypeRejected,
)
from server.sm.types import BidEvent

type GameAction = (
    BidAction
    | SkipBidAction
    | PlayAction
    | StirAction
    | SkipStirAction
    | DiscardAction
    | NextRoundAction
)


def action_kind(action: GameAction) -> GameActionKind:
    if isinstance(action, BidAction):
        return "bid"
    if isinstance(action, SkipBidAction):
        return "skip_bid"
    if isinstance(action, PlayAction):
        return "play"
    if isinstance(action, StirAction):
        return "stir"
    if isinstance(action, SkipStirAction):
        return "skip_stir"
    if isinstance(action, DiscardAction):
        return "discard"
    return "next_round"


def parse_player_message(
    *,
    round_state: round_sm.RoundState | None,
    player_index: int,
    message: PlayerMessage,
) -> Ok[GameAction] | Rejected:
    raw = message.raw
    action_type_raw = raw.get("type")
    action_type: str | None = (
        action_type_raw if isinstance(action_type_raw, str) else None
    )
    if action_type is None:
        return MissingActionTypeRejected()

    pass_val_raw = raw.get("pass", False)
    is_pass = isinstance(pass_val_raw, bool) and pass_val_raw

    if action_type == "bid":
        if is_pass:
            return Ok(value=SkipBidAction())
        return _parse_card_action(
            round_state=round_state,
            player_index=player_index,
            cards_raw=raw.get("cards"),
            action_type=action_type,
        )

    if action_type == "stir":
        if is_pass:
            return Ok(value=SkipStirAction())
        return _parse_card_action(
            round_state=round_state,
            player_index=player_index,
            cards_raw=raw.get("cards"),
            action_type=action_type,
        )

    if action_type == "discard":
        return _parse_card_action(
            round_state=round_state,
            player_index=player_index,
            cards_raw=raw.get("cards"),
            action_type="discard",
        )

    if action_type == "play":
        return _parse_card_action(
            round_state=round_state,
            player_index=player_index,
            cards_raw=raw.get("cards"),
            action_type="play",
        )

    if action_type == "next_round":
        return Ok(value=NextRoundAction())

    return UnknownActionTypeRejected(action_type)


def resolve_cards(
    *,
    round_state: round_sm.RoundState | None,
    player_index: int,
    card_ids: list[str],
) -> Ok[list[Card]] | Rejected:
    """
    Resolve card id strings to Card objects from the player's current
    hand.
    """
    if round_state is None:
        return GameNotStartedRejected()
    hand = round_state.players_hand[player_index]
    if (
        round_state.phase == "STIRRING"
        and round_state.stirring_state is not None
        and round_state.stirring_state.phase == "EXCHANGING"
        and round_state.stirring_state.exchanging_player == player_index
        and round_state.stirring_state.exchange_state is not None
    ):
        hand = (
            round_state.stirring_state.exchange_state.hand_after_pickup
        )
    card_map = {card.id: card for card in hand}

    result: list[Card] = []
    for card_id in card_ids:
        if card_id not in card_map:
            return CardNotInHandRejected(
                card_id, player_index=player_index, current=True
            )
        result.append(card_map[card_id])
    return Ok(value=result)


def bid_event_from_action(
    *, player_index: int, action: BidAction
) -> Ok[BidEvent] | Rejected:
    cards = action.cards
    if not cards:
        return EmptyBidRejected()
    if cards[0].is_joker:
        kind = "joker"
        joker_type = "big" if cards[0].is_big_joker else "small"
        suit = None
    else:
        kind = "trump_rank"
        suit = cards[0].suit
        joker_type = None

    return Ok(
        value=BidEvent(
            player=player_index,
            cards=cards,
            kind=kind,
            suit=suit,
            joker_type=joker_type,
            count=action.count,
        )
    )


def _parse_card_action(
    *,
    round_state: round_sm.RoundState | None,
    player_index: int,
    cards_raw: object,
    action_type: CardActionKind,
) -> Ok[GameAction] | Rejected:
    card_ids_result = _extract_card_ids(cards_raw)
    if isinstance(card_ids_result, Rejected):
        return card_ids_result
    resolved_result = resolve_cards(
        round_state=round_state,
        player_index=player_index,
        card_ids=card_ids_result.value,
    )
    if isinstance(resolved_result, Rejected):
        return resolved_result

    cards = resolved_result.value
    if action_type == "bid":
        return Ok(value=BidAction(cards=cards, count=len(cards)))
    if action_type == "stir":
        return Ok(value=StirAction(cards=cards))
    if action_type == "discard":
        return Ok(value=DiscardAction(cards=cards))
    if action_type == "play":
        return Ok(value=PlayAction(cards=cards))
    assert_never(action_type)


def _is_str_dict(val: object) -> TypeGuard[dict[str, object]]:
    return isinstance(val, dict)


def _is_obj_list(val: object) -> TypeGuard[list[object]]:
    return isinstance(val, list)


def _extract_card_ids(cards_val: object) -> Ok[list[str]] | Rejected:
    if not _is_obj_list(cards_val):
        return Ok(value=[])
    ids: list[str] = []
    for item in cards_val:
        if isinstance(item, str):
            ids.append(item)
        elif _is_str_dict(item):
            id_val = item.get("id")
            if isinstance(id_val, str):
                ids.append(id_val)
            else:
                return MissingCardIdRejected(item)
        else:
            return InvalidCardFormatRejected(item)
    return Ok(value=ids)
