"""Private lead and follow legality algorithms."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected

from ._follow_analysis import analyze_follow
from ._ordering import effective_suit
from ._patterns import analyze_patterns
from ._rejections.card import CardNotInHandRejected
from ._rejections.play import (
    EmptyFollowRejected,
    EmptyLeadRejected,
    IllegalFollowShapeRejected,
    MustExhaustLeadSuitRejected,
    MustExhaustTrumpRejected,
    MustFollowHigherPatternRejected,
    MustFollowLeadSuitRejected,
    MustFollowPairsRejected,
    MustFollowTrumpRejected,
    WrongFollowCountRejected,
)
from ._rejections.text import PlayShapeInfo
from .cards import Card, Rank, Suit
from .cards.faces import CardFace


def is_legal_lead(
    hand: list[Card],
    played_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
    _other_players_hands: list[list[Card]],
) -> bool:
    """Verify that a leading play attempt can be submitted.

    All leading plays are throw attempts. A failed throw is still
    accepted by
    the protocol and resolved to the forced small sub-play by
    resolve_lead_throw.
    This predicate only checks the input constraints that can still
    reject the
    message: non-empty, in hand, and one effective suit.
    """
    if not played_cards:
        return False

    # Step 1: played_cards must be a subset of hand
    hand_ids = {c.id for c in hand}
    for c in played_cards:
        if c.id not in hand_ids:
            return False

    # Step 2: all played cards must have the same effective suit
    eff_suits = {
        effective_suit(c, trump_suit, trump_rank) for c in played_cards
    }
    if len(eff_suits) != 1:
        return False

    return True


def is_legal_follow(
    hand: list[Card],
    played_cards: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    """Verify that a following play is legal per spec section 6.2.

    Validates:
    1. played count == lead count
    2. played_cards are all in hand
    3. Suit-following rules
    4. Sub-play priority rules (spec 6.2 steps 7a/7b/7c)
    """
    analysis_result = analyze_follow(
        hand=hand,
        lead_cards=lead_cards,
        trump_suit=trump_suit,
        trump_rank=trump_rank,
    )
    if isinstance(analysis_result, Rejected):
        return False
    return analysis_result.value.validate_cards(played_cards)


def illegal_follow_rejection(
    hand: list[Card],
    played_cards: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> Rejected:
    """Return a structured rejection for an illegal following play."""
    if is_legal_follow(
        hand, played_cards, lead_cards, trump_suit, trump_rank
    ):
        return IllegalFollowShapeRejected(
            _play_shape_info(lead_cards, trump_suit, trump_rank)
        )

    if not lead_cards:
        return EmptyLeadRejected()

    lead_count = len(lead_cards)
    if len(played_cards) != lead_count:
        return WrongFollowCountRejected(lead_count)

    if not played_cards:
        return EmptyFollowRejected()

    hand_ids = {card.id for card in hand}
    for card in played_cards:
        if card.id not in hand_ids:
            return CardNotInHandRejected(card.id, current=True)

    lead_eff = effective_suit(lead_cards[0], trump_suit, trump_rank)
    suit_in_hand = [
        card
        for card in hand
        if effective_suit(card, trump_suit, trump_rank) == lead_eff
    ]
    suit_in_played = [
        card
        for card in played_cards
        if effective_suit(card, trump_suit, trump_rank) == lead_eff
    ]

    if (
        len(suit_in_hand) >= lead_count
        and len(suit_in_played) != lead_count
    ):
        if lead_eff == "trump":
            return MustFollowTrumpRejected()
        return MustFollowLeadSuitRejected(lead_eff)

    if len(suit_in_hand) < lead_count and len(suit_in_played) < len(
        suit_in_hand
    ):
        if lead_eff == "trump":
            return MustExhaustTrumpRejected(len(suit_in_hand))
        return MustExhaustLeadSuitRejected(lead_eff, len(suit_in_hand))

    pair_reason = _explain_follow_pair_priority(
        suit_in_hand,
        suit_in_played,
        lead_cards,
        trump_suit,
        trump_rank,
    )
    if pair_reason is not None:
        return pair_reason

    return IllegalFollowShapeRejected(
        _play_shape_info(lead_cards, trump_suit, trump_rank)
    )


def _explain_follow_pair_priority(
    hand_suit_cards: list[Card],
    played_suit_cards: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> Rejected | None:
    analysis_result = analyze_follow(
        hand=hand_suit_cards,
        lead_cards=lead_cards,
        trump_suit=trump_suit,
        trump_rank=trump_rank,
    )
    assert isinstance(analysis_result, Ok)
    analysis = analysis_result.value
    played_pair_faces: frozenset[CardFace] = (
        frozenset()
        if not played_suit_cards
        else frozenset(
            analyze_patterns(
                played_suit_cards, trump_suit, trump_rank
            ).pair_faces
        )
    )
    hand_pair_count = len(analysis.pair_planner.pair_faces)
    if len(played_pair_faces) < analysis.pair_floor:
        return MustFollowPairsRejected(
            lead_pair_count=analysis.lead_pair_count,
            lead_suit=analysis.lead_effective_suit,
            hand_pair_count=hand_pair_count,
            pair_floor=analysis.pair_floor,
        )

    if not analysis.pair_planner.pair_selection_is_valid(
        played_pair_faces
    ):
        return MustFollowHigherPatternRejected(
            _play_shape_info(lead_cards, trump_suit, trump_rank)
        )
    return None


def _play_shape_info(
    cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> PlayShapeInfo:
    if not cards:
        return PlayShapeInfo(kind="empty", suit=None, card_count=0)
    suit = effective_suit(cards[0], trump_suit, trump_rank)
    patterns = analyze_patterns(cards, trump_suit, trump_rank).patterns
    if len(cards) == 1:
        return PlayShapeInfo(kind="single", suit=suit, card_count=1)
    if len(patterns) == 1:
        pattern = patterns[0]
        if pattern.pair_count >= 2:
            return PlayShapeInfo(
                kind="tractor",
                suit=suit,
                card_count=len(cards),
                pair_count=pattern.pair_count,
            )
        if pattern.pair_count == 1:
            return PlayShapeInfo(
                kind="pair",
                suit=suit,
                card_count=len(cards),
                pair_count=1,
            )
    return PlayShapeInfo(kind="cards", suit=suit, card_count=len(cards))
