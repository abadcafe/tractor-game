"""Tests for rules.compare public interface."""

from typing import Literal

from server.rules.cards import POINTS_MAP, Card, Rank, Suit
from server.rules.compare import (
    can_win,
    compare_plays,
    compare_plays_against_lead,
)


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit,
        rank=rank,
        points=POINTS_MAP[rank],
    )


class TestCanWin:
    def test_can_win_all_lead_suit(self) -> None:
        """All cards are lead suit -> can win."""
        cards = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.HEARTS, Rank.KING),
        ]
        assert (
            can_win(cards, Suit.HEARTS, Suit.SPADES, Rank.TWO) is True
        )

    def test_can_win_all_trump(self) -> None:
        """
        All cards are trump -> can win (trump = lead_eff when lead is
        trump).
        """
        cards = [_card(Suit.JOKER, Rank.BIG_JOKER)]
        assert can_win(cards, "trump", Suit.SPADES, Rank.TWO) is True

    def test_can_win_lead_is_trump_play_trump(self) -> None:
        """Lead is trump, play is trump -> can win."""
        cards = [
            _card(Suit.SPADES, Rank.ACE)
        ]  # trump when trump_suit=spade
        assert can_win(cards, "trump", Suit.SPADES, Rank.TWO) is True

    def test_can_win_off_suit_non_trump(self) -> None:
        """Card is neither lead suit nor trump -> cannot win."""
        cards = [_card(Suit.DIAMONDS, Rank.ACE)]
        assert (
            can_win(cards, Suit.HEARTS, Suit.SPADES, Rank.TWO) is False
        )

    def test_can_win_off_suit_but_trump(self) -> None:
        """Card is not lead suit but is trump -> can win."""
        # sp2 is trump when trump_suit=spade, trump_rank=2
        cards = [_card(Suit.SPADES, Rank.TWO)]
        assert (
            can_win(cards, Suit.HEARTS, Suit.SPADES, Rank.TWO) is True
        )

    def test_can_win_mixed_lead_and_off_suit(self) -> None:
        """
        One card is lead suit, one is off-suit non-trump -> cannot win.
        """
        cards = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.DIAMONDS, Rank.KING),
        ]
        assert (
            can_win(cards, Suit.HEARTS, Suit.SPADES, Rank.TWO) is False
        )

    def test_can_win_mixed_lead_and_trump(self) -> None:
        """One card is lead suit, one is trump -> CAN win.

        Per spec 8.2: any card that is (not lead_suit AND not trump) ->
        cannot win.
        Trump cards are always OK. So hA + sp2(trump) -> both valid ->
        can win.
        """
        # hA (lead suit) + sp2 (trump, trump_suit=spade, trump_rank=2)
        cards = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.SPADES, Rank.TWO),
        ]
        assert (
            can_win(cards, Suit.HEARTS, Suit.SPADES, Rank.TWO) is True
        )


class TestComparePlays:
    # --- can_win gating ---
    def test_compare_plays_a_wins_by_eligibility(self) -> None:
        """A can win, B cannot -> A wins."""
        a = [_card(Suit.HEARTS, Rank.THREE)]
        b = [_card(Suit.DIAMONDS, Rank.ACE)]  # off-suit, not trump
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0

    def test_compare_plays_b_wins_by_eligibility(self) -> None:
        """B can win, A cannot -> B wins."""
        a = [_card(Suit.DIAMONDS, Rank.ACE)]
        b = [_card(Suit.HEARTS, Rank.THREE)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result < 0

    def test_compare_plays_neither_can_win(self) -> None:
        """Neither can win -> tie (0)."""
        a = [_card(Suit.DIAMONDS, Rank.ACE)]
        b = [_card(Suit.CLUBS, Rank.KING)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result == 0

    # --- Trump vs non-trump ---
    def test_compare_plays_trump_beats_non_trump(self) -> None:
        """All-trump play beats all-lead-suit play."""
        # spA is trump (trump_suit=spade)
        a = [_card(Suit.SPADES, Rank.ACE)]
        b = [_card(Suit.HEARTS, Rank.ACE)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0

    # --- Sub-level comparison ---
    def test_compare_plays_pair_beats_single(self) -> None:
        """
        Pair (level 2) beats single (level 1), even if single has higher
        rank.
        """
        # hA pair vs hK single -- pair wins by level
        a = [
            _card(Suit.HEARTS, Rank.ACE, 1),
            _card(Suit.HEARTS, Rank.ACE, 2),
        ]
        b = [_card(Suit.HEARTS, Rank.KING)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0

    def test_compare_plays_tractor_beats_pair(self) -> None:
        """Tractor (level 3) beats pair (level 2)."""
        a = [
            _card(Suit.HEARTS, Rank.THREE, 1),
            _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1),
            _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        b = [
            _card(Suit.HEARTS, Rank.ACE, 1),
            _card(Suit.HEARTS, Rank.ACE, 2),
        ]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0

    def test_compare_plays_same_level_higher_rank_wins(self) -> None:
        """Same sub-level: higher max rank wins."""
        a = [_card(Suit.HEARTS, Rank.ACE)]
        b = [_card(Suit.HEARTS, Rank.KING)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0

    def test_compare_plays_same_level_same_rank_tie(self) -> None:
        """Same sub-level, same max rank -> tie."""
        a = [_card(Suit.HEARTS, Rank.ACE, 1)]
        b = [_card(Suit.HEARTS, Rank.ACE, 2)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result == 0

    def test_compare_plays_both_trump_higher_wins(self) -> None:
        """Both trump: higher trump_rank_order wins."""
        a = [_card(Suit.JOKER, Rank.BIG_JOKER)]
        b = [_card(Suit.JOKER, Rank.SMALL_JOKER)]
        result = compare_plays(a, b, "trump", Suit.SPADES, Rank.TWO)
        assert result > 0

    # --- Trump sub-type comparison (spec 2.3 / 8.4) ---
    def test_compare_plays_trump_suit_level_beats_other_suit_level(
        self,
    ) -> None:
        """
        Trump-suit level card beats other-suit level card at same rank.

        trump_suit=♥, trump_rank=5:
          ♥5 = 主花色级牌 (spec value=80)
          ♠5 = 其他花色级牌 (spec value=70)
        ♥5 should win.
        """
        a = [_card(Suit.HEARTS, Rank.FIVE)]
        b = [_card(Suit.SPADES, Rank.FIVE)]
        # lead_eff=♠ (spades is trump, so both are trump)
        result = compare_plays(
            a, b, Suit.SPADES, Suit.HEARTS, Rank.FIVE
        )
        assert result > 0

    def test_compare_plays_trump_suit_level_beats_diamond_level(
        self,
    ) -> None:
        """
        Trump-suit level card beats diamond-level card (lowest
        other-suit level).

        trump_suit=♥, trump_rank=5:
          ♥5 = 80
          ♦5 = 70
        ♥5 should win.
        """
        a = [_card(Suit.HEARTS, Rank.FIVE)]
        b = [_card(Suit.DIAMONDS, Rank.FIVE)]
        result = compare_plays(
            a, b, Suit.DIAMONDS, Suit.HEARTS, Rank.FIVE
        )
        assert result > 0

    def test_compare_plays_other_suit_level_cards_tie(self) -> None:
        """
        Other-suit level cards are equal; earlier play order wins the
        trick.

        trump_rank=5, trump_suit=♥:
          ♣5 = 70
          ♠5 = 70
        """
        a = [_card(Suit.SPADES, Rank.FIVE)]
        b = [_card(Suit.CLUBS, Rank.FIVE)]
        result = compare_plays(
            a, b, Suit.HEARTS, Suit.HEARTS, Rank.FIVE
        )
        assert result == 0

    def test_compare_plays_no_trump_level_cards_tie(self) -> None:
        """In no-trump rounds, all trump-rank suits are equal."""
        a = [_card(Suit.SPADES, Rank.TWO)]
        b = [_card(Suit.HEARTS, Rank.TWO)]
        result = compare_plays(a, b, "trump", None, Rank.TWO)
        assert result == 0

    def test_compare_plays_trump_pair_sub_type_diff(self) -> None:
        """Both trump pairs at same rank, different sub-types.

        trump_suit=♥, trump_rank=5:
          ♥5♥5 = 主花色级牌对子 (max rank = 80)
          ♠5♠5 = 其他花色级牌对子 (max rank = 70)
        ♥5♥5 should win.
        """
        a = [
            _card(Suit.HEARTS, Rank.FIVE, 1),
            _card(Suit.HEARTS, Rank.FIVE, 2),
        ]
        b = [
            _card(Suit.SPADES, Rank.FIVE, 1),
            _card(Suit.SPADES, Rank.FIVE, 2),
        ]
        result = compare_plays(
            a, b, Suit.SPADES, Suit.HEARTS, Rank.FIVE
        )
        assert result > 0

    def test_compare_plays_trump_suit_non_level_vs_other_suit_level(
        self,
    ) -> None:
        """
        Trump-suit non-level card vs other-suit level card at same rank.

        trump_suit=♥, trump_rank=K:
          ♥A = 主花色非级牌 (spec value=45+14=59)
          ♠K = 其他花色级牌 (spec value=70)
        ♠K should win (70 > 59).
        """
        a = [_card(Suit.HEARTS, Rank.ACE)]
        b = [_card(Suit.SPADES, Rank.KING)]
        result = compare_plays(
            a, b, Suit.SPADES, Suit.HEARTS, Rank.KING
        )
        assert result < 0


class TestComparePlaysAgainstLead:
    def test_structurally_invalid_trump_kill_cannot_win(self) -> None:
        lead = [
            _card(Suit.HEARTS, Rank.ACE, 1),
            _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING, 1),
            _card(Suit.HEARTS, Rank.QUEEN, 1),
        ]
        valid_kill = [
            _card(Suit.SPADES, Rank.FOUR, 1),
            _card(Suit.SPADES, Rank.FOUR, 2),
            _card(Suit.SPADES, Rank.FIVE, 1),
            _card(Suit.SPADES, Rank.SIX, 1),
        ]
        invalid_big_cards = [
            _card(Suit.JOKER, Rank.BIG_JOKER, 1),
            _card(Suit.JOKER, Rank.SMALL_JOKER, 1),
            _card(Suit.SPADES, Rank.ACE, 1),
            _card(Suit.SPADES, Rank.KING, 1),
        ]

        result = compare_plays_against_lead(
            invalid_big_cards,
            valid_kill,
            lead,
            Suit.SPADES,
            Rank.TWO,
        )

        assert result < 0

    def test_matching_trump_kills_compare_by_main_pattern(
        self,
    ) -> None:
        lead = [
            _card(Suit.HEARTS, Rank.ACE, 1),
            _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.KING, 1),
            _card(Suit.HEARTS, Rank.QUEEN, 1),
        ]
        high_pair_kill = [
            _card(Suit.SPADES, Rank.KING, 1),
            _card(Suit.SPADES, Rank.KING, 2),
            _card(Suit.SPADES, Rank.THREE, 1),
            _card(Suit.SPADES, Rank.FOUR, 1),
        ]
        low_pair_with_big_jokers = [
            _card(Suit.SPADES, Rank.QUEEN, 1),
            _card(Suit.SPADES, Rank.QUEEN, 2),
            _card(Suit.JOKER, Rank.BIG_JOKER, 1),
            _card(Suit.JOKER, Rank.SMALL_JOKER, 1),
        ]

        result = compare_plays_against_lead(
            high_pair_kill,
            low_pair_with_big_jokers,
            lead,
            Suit.SPADES,
            Rank.TWO,
        )

        assert result > 0

    def test_all_single_throw_kills_compare_by_highest_trump(
        self,
    ) -> None:
        lead = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.HEARTS, Rank.QUEEN),
        ]
        big_joker_kill = [
            _card(Suit.JOKER, Rank.BIG_JOKER),
            _card(Suit.SPADES, Rank.THREE),
            _card(Suit.SPADES, Rank.FOUR),
        ]
        small_joker_kill = [
            _card(Suit.JOKER, Rank.SMALL_JOKER),
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.SPADES, Rank.KING),
        ]

        result = compare_plays_against_lead(
            big_joker_kill,
            small_joker_kill,
            lead,
            Suit.SPADES,
            Rank.TWO,
        )

        assert result > 0

    def test_same_highest_trump_returns_tie_for_play_order(
        self,
    ) -> None:
        lead = [
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.HEARTS, Rank.KING),
        ]
        first_kill = [
            _card(Suit.JOKER, Rank.BIG_JOKER, 1),
            _card(Suit.SPADES, Rank.THREE, 1),
        ]
        later_kill = [
            _card(Suit.JOKER, Rank.BIG_JOKER, 2),
            _card(Suit.SPADES, Rank.FOUR, 1),
        ]

        result = compare_plays_against_lead(
            first_kill,
            later_kill,
            lead,
            Suit.SPADES,
            Rank.TWO,
        )

        assert result == 0


class TestSubLevelComparison:
    def test_compare_plays_lower_rank_pair_beats_higher_rank_single(
        self,
    ) -> None:
        """
        Pair (level 2) beats single (level 1) even when pair has lower
        rank.

        h3 pair (rank 3) vs hA single (rank A): pair wins because level
        2 > level 1.
        The existing test_compare_plays_pair_beats_single uses hA pair
        vs hK single,
        which doesn't actually test the edge case (pair has higher rank
        too).
        """
        a = [
            _card(Suit.HEARTS, Rank.THREE, 1),
            _card(Suit.HEARTS, Rank.THREE, 2),
        ]
        b = [_card(Suit.HEARTS, Rank.ACE)]
        result = compare_plays(a, b, Suit.HEARTS, Suit.SPADES, Rank.TWO)
        assert result > 0  # pair wins despite lower rank
