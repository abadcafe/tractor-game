"""Physical card ownership operations for immutable hands."""

from server.foundation.result import Ok, Rejected
from server.game.rules.cards import Card, CardId
from server.game.seating import Seat

from .round_values import Hands


def resolve_cards(
    hand: tuple[Card, ...],
    card_ids: tuple[CardId, ...],
) -> Ok[tuple[Card, ...]] | Rejected:
    by_id = {card.id: card for card in hand}
    result: list[Card] = []
    used: set[str] = set()
    for card_id in card_ids:
        value = str(card_id)
        if value in used:
            return Rejected(f"不能重复选择牌 {value}")
        card = by_id.get(value)
        if card is None:
            return Rejected(f"当前手牌中没有牌 {value}")
        result.append(card)
        used.add(value)
    return Ok(tuple(result))


def replace_hand(
    hands: Hands,
    seat: Seat,
    hand: tuple[Card, ...],
) -> Hands:
    return hands.replace(seat, hand)
