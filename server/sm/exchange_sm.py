"""Exchange (换底牌) state machine for 升级 (Shengji/Tractor).

The declarer picks up the 8 bottom cards, then discards 8 cards from
their
combined hand. The discarded cards become the new bottom cards used for
scoring.
"""

from pydantic import BaseModel, ConfigDict

from server.result import Ok, Rejected
from server.rules.cards import Card
from server.rules.rejections.card import (
    CardNotInHandRejected,
    DuplicateCardRejected,
)

from .rejections.stirring import InvalidExchangeCountRejected
from .types import ExchangePhase

# ---- Models ----


class ExchangeInput(BaseModel):
    """Input to create an exchange state."""

    model_config = ConfigDict(frozen=True)

    declarer_player: int
    bottom_cards: list[Card]
    declarer_hand: list[Card]


class ExchangeResult(BaseModel):
    """Result after the declarer completes the discard."""

    model_config = ConfigDict(frozen=True)

    new_hand: list[Card]
    new_bottom_cards: list[Card]


class ExchangeState(BaseModel):
    """State of the exchange process."""

    model_config = ConfigDict(frozen=True)

    phase: ExchangePhase
    hand_after_pickup: list[Card]
    count: int
    declarer_player: int
    result: ExchangeResult | None


# ---- State Machine ----


def create_exchange(input: ExchangeInput) -> ExchangeState:
    """Combine declarer hand + bottom cards. Set phase to PICKED_UP."""
    hand_after_pickup = list(input.declarer_hand) + list(
        input.bottom_cards
    )
    return ExchangeState(
        phase="PICKED_UP",
        hand_after_pickup=hand_after_pickup,
        count=len(input.bottom_cards),
        declarer_player=input.declarer_player,
        result=None,
    )


def discard(
    state: ExchangeState, cards: list[Card]
) -> Ok[ExchangeState] | Rejected:
    """Validate and discard cards from hand_after_pickup.

    Returns Ok(new_state) on success, Rejected(reason) on invalid input.
    """
    if len(cards) != state.count:
        return InvalidExchangeCountRejected(
            required_count=state.count, actual_count=len(cards)
        )

    hand_ids = {c.id for c in state.hand_after_pickup}
    discard_ids = [c.id for c in cards]

    for cid in discard_ids:
        if cid not in hand_ids:
            return CardNotInHandRejected(cid)

    # Check for duplicate card IDs in input
    seen: set[str] = set()
    for cid in discard_ids:
        if cid in seen:
            return DuplicateCardRejected(cid)
        seen.add(cid)

    discard_set = set(discard_ids)
    new_hand = [
        c for c in state.hand_after_pickup if c.id not in discard_set
    ]

    result = ExchangeResult(
        new_hand=new_hand,
        new_bottom_cards=list(cards),
    )

    return Ok(
        ExchangeState(
            phase="COMPLETE",
            hand_after_pickup=state.hand_after_pickup,
            count=state.count,
            declarer_player=state.declarer_player,
            result=result,
        )
    )
