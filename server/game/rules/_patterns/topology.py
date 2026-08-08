"""Fixed card-face tiers used exclusively for tractor structure."""

from __future__ import annotations

from server.game.rules._ordering import EffectiveSuit
from server.game.rules.cards import Rank, Suit, suited_ranks
from server.game.rules.cards.faces import CardFace

from .model import PairTier


def pair_tiers(
    effective_suit: EffectiveSuit,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> tuple[PairTier, ...]:
    """Return fixed weak-to-strong structural tiers."""
    if effective_suit != "trump":
        assert effective_suit != Suit.JOKER
        return tuple(
            (CardFace(suit=effective_suit, rank=rank),)
            for rank in suited_ranks()
            if rank != trump_rank
        )

    joker_tiers: tuple[PairTier, PairTier] = (
        (CardFace(suit=Suit.JOKER, rank=Rank.SMALL_JOKER),),
        (CardFace(suit=Suit.JOKER, rank=Rank.BIG_JOKER),),
    )
    if trump_suit is None:
        level_tier = tuple(
            CardFace(suit=suit, rank=trump_rank)
            for suit in Suit
            if suit != Suit.JOKER
        )
        return (level_tier, *joker_tiers)

    ordinary_tiers = tuple(
        (CardFace(suit=trump_suit, rank=rank),)
        for rank in suited_ranks()
        if rank != trump_rank
    )
    vice_level_tier = tuple(
        CardFace(suit=suit, rank=trump_rank)
        for suit in Suit
        if suit not in (Suit.JOKER, trump_suit)
    )
    main_level_tier = (CardFace(suit=trump_suit, rank=trump_rank),)
    return (
        *ordinary_tiers,
        vice_level_tier,
        main_level_tier,
        *joker_tiers,
    )
