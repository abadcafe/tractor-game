"""Tests for sm.scoring module."""
import pytest
from server.sm.card_model import Card, Suit, Rank
from server.sm.types import PlayType, CompletedTrick, CompletedTrickSlot
from server.sm.scoring import calculate_score, RoundResult


def _card(suit: Suit, rank: Rank, deck: int = 1) -> Card:
    """Create a card with correct point values per spec: 5=5, 10=10, K=10, else 0."""
    pts_map: dict[Rank, int] = {
        Rank.FIVE: 5, Rank.TEN: 10, Rank.KING: 10,
    }
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=pts_map.get(rank, 0), deck=deck,
    )


def _completed_trick(
    lead_type: PlayType, card_count: int, winner: int,
    *,
    throw_sub_pattern: str = "singles",
) -> CompletedTrick:
    """Create a minimal CompletedTrick for scoring tests.

    calculate_score uses last_trick.lead_type to determine multiplier
    and last_trick.winner to determine which team gets the ambush bonus.
    card_count is encoded in the slot's cards length.

    For THROW type, throw_sub_pattern controls the card composition so
    calculate_score can distinguish sub-patterns:
      "singles" - all different ranks (no pair sub-pattern), multiplier x2
      "pair"    - at least one pair (same rank), multiplier x4
      "tractor" - at least a 4-card tractor, multiplier 2^N
    """
    if lead_type == PlayType.THROW:
        if throw_sub_pattern == "singles":
            # Each card has a different rank -> no pair possible
            ranks = [Rank.THREE, Rank.FOUR, Rank.SIX, Rank.SEVEN,
                     Rank.EIGHT, Rank.NINE, Rank.JACK, Rank.QUEEN]
            cards = [_card(Suit.SPADES, ranks[i]) for i in range(card_count)]
        elif throw_sub_pattern == "pair":
            # Two cards of same rank -> pair sub-pattern
            cards = [_card(Suit.SPADES, Rank.THREE, 1),
                     _card(Suit.SPADES, Rank.THREE, 2)]
        elif throw_sub_pattern == "tractor":
            # 4+ cards forming consecutive pairs -> tractor sub-pattern
            cards = [_card(Suit.SPADES, Rank.THREE, 1),
                     _card(Suit.SPADES, Rank.THREE, 2),
                     _card(Suit.SPADES, Rank.FOUR, 1),
                     _card(Suit.SPADES, Rank.FOUR, 2)]
            # Extend if card_count > 4
            extra_ranks = [Rank.SIX, Rank.SEVEN]
            i = 0
            while len(cards) < card_count:
                r = extra_ranks[i % len(extra_ranks)]
                cards.append(_card(Suit.SPADES, r, 1))
                cards.append(_card(Suit.SPADES, r, 2))
                i += 1
            cards = cards[:card_count]
        else:
            cards = [_card(Suit.SPADES, Rank.THREE)] * card_count
    else:
        cards = [_card(Suit.SPADES, Rank.THREE)] * card_count

    # Place cards in the lead player's slot so _find_lead_cards finds them
    # directly, not via the fallback path.
    lead_player = 0
    lead_slot = CompletedTrickSlot(player=lead_player, cards=cards)
    slots: list[CompletedTrickSlot] = [lead_slot]
    if winner != lead_player:
        winner_slot = CompletedTrickSlot(player=winner, cards=[])
        slots.append(winner_slot)
    return CompletedTrick(
        lead_player=lead_player, lead_type=lead_type, slots=slots,
        winner=winner, points=0,
    )


class TestCalculateScore:
    def test_calculate_score_big_light(self) -> None:
        """Defender 0 points -> declarer +3, no switch."""
        result = calculate_score(
            defender_points=0,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=0),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.declarer_level_change == 3
        assert result.switch_declarer is False
        assert result.next_declarer_team == 0
        assert result.next_declarer_player == 3  # partner of player 0
        assert result.team0_new_level == Rank.FIVE  # TWO + 3 = FIVE
        assert result.team1_new_level == Rank.TWO   # unchanged

    def test_calculate_score_small_light(self) -> None:
        """Defender 1-39 points -> declarer +2."""
        result = calculate_score(
            defender_points=35,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=0),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.declarer_level_change == 2
        assert result.team0_new_level == Rank.FOUR  # TWO + 2 = FOUR
        assert result.team1_new_level == Rank.TWO   # unchanged

    def test_calculate_score_plus1(self) -> None:
        """Defender 40-79 -> declarer +1."""
        result = calculate_score(
            defender_points=50,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=0),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.declarer_level_change == 1
        assert result.team0_new_level == Rank.THREE  # TWO + 1 = THREE
        assert result.team1_new_level == Rank.TWO    # unchanged

    def test_calculate_score_switch(self) -> None:
        """Defender 80-119 -> switch declarer."""
        result = calculate_score(
            defender_points=100,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.declarer_level_change == 0
        assert result.switch_declarer is True
        assert result.next_declarer_team == 1
        assert result.team0_new_level == Rank.TWO  # TWO + 0 = TWO (no change for declarer)
        assert result.team1_new_level == Rank.TWO  # defender gets abs(0) = 0 advance

    def test_calculate_score_defender_plus1(self) -> None:
        """Defender 120-159 -> defender +1, switch."""
        result = calculate_score(
            defender_points=130,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.FIVE,
            team1_level=Rank.THREE,
        )
        assert result.declarer_level_change == -1
        assert result.switch_declarer is True
        assert result.team0_new_level == Rank.FOUR  # FIVE - 1 = FOUR
        assert result.team1_new_level == Rank.FOUR  # THREE + abs(-1) = FOUR

    def test_calculate_score_defender_plus2(self) -> None:
        """Defender 160-199 -> defender +2."""
        result = calculate_score(
            defender_points=180,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.FIVE,
            team1_level=Rank.THREE,
        )
        assert result.declarer_level_change == -2
        assert result.team0_new_level == Rank.THREE  # FIVE - 2 = THREE
        assert result.team1_new_level == Rank.FIVE   # THREE + abs(-2) = FIVE

    def test_calculate_score_defender_plus3(self) -> None:
        """Defender 200 -> defender +3."""
        result = calculate_score(
            defender_points=200,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.FIVE,
            team1_level=Rank.THREE,
        )
        assert result.declarer_level_change == -3
        assert result.team0_new_level == Rank.TWO    # FIVE - 3 = TWO (clamped at 2)
        assert result.team1_new_level == Rank.SIX     # THREE + abs(-3) = SIX


class TestAmbushMultiplier:
    def test_ambush_single_x2(self) -> None:
        """Single play ambush = x2."""
        bottom = [_card(Suit.SPADES, Rank.FIVE), _card(Suit.SPADES, Rank.TEN)]
        result = calculate_score(
            defender_points=10,
            bottom_cards=bottom,
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        # bottom_base = 5+10 = 15; bonus = 15*2 = 30; total = 10+30 = 40
        assert result.bottom_card_bonus == 30
        assert result.total_defender_points == 40

    def test_ambush_pair_x4(self) -> None:
        """Pair play ambush = x4."""
        bottom = [_card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2)]
        result = calculate_score(
            defender_points=10,
            bottom_cards=bottom,
            last_trick=_completed_trick(PlayType.PAIR, 2, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        # bottom_base = 5+5 = 10; bonus = 10*4 = 40; total = 10+40 = 50
        assert result.bottom_card_bonus == 40
        assert result.total_defender_points == 50

    def test_ambush_tractor_4card_x16(self) -> None:
        """4-card tractor ambush = x16 (2^4)."""
        bottom = [
            _card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2),
            _card(Suit.SPADES, Rank.TEN, 1), _card(Suit.SPADES, Rank.TEN, 2),
        ]
        result = calculate_score(
            defender_points=10,
            bottom_cards=bottom,
            last_trick=_completed_trick(PlayType.TRACTOR, 4, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        # bottom_base = 5+5+10+10 = 30; bonus = 30*16 = 480; total = 10+480 = 490
        assert result.bottom_card_bonus == 480
        assert result.total_defender_points == 490

    def test_ambush_tractor_6card_x64(self) -> None:
        """6-card tractor ambush = x64 (2^6)."""
        bottom = [
            _card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2),
            _card(Suit.SPADES, Rank.TEN, 1), _card(Suit.SPADES, Rank.TEN, 2),
            _card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2),
        ]
        result = calculate_score(
            defender_points=10,
            bottom_cards=bottom,
            last_trick=_completed_trick(PlayType.TRACTOR, 6, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        # bottom_base = 5+5+10+10+10+10 = 50; bonus = 50*64 = 3200; total = 10+3200 = 3210
        assert result.bottom_card_bonus == 3200
        assert result.total_defender_points == 3210

    def test_ambush_tractor_n_card(self) -> None:
        """N-card tractor ambush = 2^N."""
        # 8-card tractor (4 pairs)
        bottom = [
            _card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2),
            _card(Suit.SPADES, Rank.SIX, 1), _card(Suit.SPADES, Rank.SIX, 2),
            _card(Suit.SPADES, Rank.NINE, 1), _card(Suit.SPADES, Rank.NINE, 2),
            _card(Suit.SPADES, Rank.TEN, 1), _card(Suit.SPADES, Rank.TEN, 2),
        ]
        result = calculate_score(
            defender_points=0,
            bottom_cards=bottom,
            last_trick=_completed_trick(PlayType.TRACTOR, 8, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        # bottom_base = 5+5+0+0+0+0+10+10 = 30; multiplier = 2^8 = 256; bonus = 7680
        assert result.bottom_card_bonus == 30 * 256

    def test_ambush_throw_with_tractor(self) -> None:
        """THROW containing a tractor uses tractor multiplier (2^N)."""
        bottom = [
            _card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2),
            _card(Suit.SPADES, Rank.TEN, 1), _card(Suit.SPADES, Rank.TEN, 2),
        ]
        # THROW with 4 cards that contain a 4-card tractor sub-pattern
        result = calculate_score(
            defender_points=0,
            bottom_cards=bottom,
            last_trick=_completed_trick(PlayType.THROW, 4, winner=1, throw_sub_pattern="tractor"),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        # THROW with tractor sub-pattern -> 2^4 = 16; bottom_base = 30; bonus = 30*16 = 480
        assert result.bottom_card_bonus == 30 * 16

    def test_ambush_throw_with_pair(self) -> None:
        """THROW containing only pairs (no tractor) uses x4."""
        bottom = [
            _card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2),
        ]
        # THROW with 2 cards that form a pair sub-pattern (no tractor)
        result = calculate_score(
            defender_points=0,
            bottom_cards=bottom,
            last_trick=_completed_trick(PlayType.THROW, 2, winner=1, throw_sub_pattern="pair"),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        # THROW with pair sub-pattern -> x4; bottom_base = 5+5 = 10; bonus = 10*4 = 40
        assert result.bottom_card_bonus == 40

    def test_ambush_throw_all_singles_x2(self) -> None:
        """THROW of all singles uses x2."""
        bottom = [_card(Suit.SPADES, Rank.FIVE), _card(Suit.SPADES, Rank.TEN)]
        # THROW with 2 cards that are all singles (no pair sub-pattern)
        result = calculate_score(
            defender_points=0,
            bottom_cards=bottom,
            last_trick=_completed_trick(PlayType.THROW, 2, winner=1, throw_sub_pattern="singles"),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        # THROW all singles -> x2; bottom_base = 5+10 = 15; bonus = 15*2 = 30
        assert result.bottom_card_bonus == (5 + 10) * 2

    def test_no_ambush_declarer_wins_last(self) -> None:
        """No ambush when declarer wins last trick."""
        bottom = [_card(Suit.SPADES, Rank.FIVE), _card(Suit.SPADES, Rank.TEN)]
        result = calculate_score(
            defender_points=10,
            bottom_cards=bottom,
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=0),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.bottom_card_bonus == 0
        assert result.total_defender_points == 10


class TestDeclarerRotation:
    def test_calculate_score_next_declarer_stays(self) -> None:
        """When declarer stays, next_declarer_player = partner."""
        result = calculate_score(
            defender_points=0,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=0),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.next_declarer_team == 0
        assert result.next_declarer_player == 3  # partner of 0
        assert result.team0_new_level == Rank.FIVE  # TWO + 3 = FIVE
        assert result.team1_new_level == Rank.TWO   # unchanged

    def test_calculate_score_next_declarer_switches(self) -> None:
        """When declarer switches, next_declarer_player = counterclockwise_next(declarer)."""
        result = calculate_score(
            defender_points=100,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.next_declarer_team == 1
        assert result.next_declarer_player == 1  # CCW next of 0
        assert result.team0_new_level == Rank.TWO  # TWO + 0 = TWO
        assert result.team1_new_level == Rank.TWO  # defender gets abs(0) = 0 advance


class TestBoundaryValues:
    @pytest.mark.parametrize(
        "points,expected_change,expected_switch",
        [
            (0, 3, False),
            (1, 2, False),
            (39, 2, False),
            (40, 1, False),
            (79, 1, False),
            (80, 0, True),
            (119, 0, True),
            (120, -1, True),
            (159, -1, True),
            (160, -2, True),
            (199, -2, True),
            (200, -3, True),
        ],
    )
    def test_calculate_score_boundary_values(
        self, points: int, expected_change: int, expected_switch: bool,
    ) -> None:
        result = calculate_score(
            defender_points=points,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=0),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.declarer_level_change == expected_change
        assert result.switch_declarer == expected_switch


class TestDeclarerTeam1:
    def test_declarer_team1_big_light(self) -> None:
        """Declarer team=1, defender 0 points -> declarer +3."""
        result = calculate_score(
            defender_points=0,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=1),
            declarer_team=1,
            declarer_player=1,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.declarer_level_change == 3
        assert result.switch_declarer is False
        assert result.next_declarer_team == 1
        assert result.next_declarer_player == 2  # partner of player 1
        assert result.team1_new_level == Rank.FIVE  # TWO + 3 = FIVE
        assert result.team0_new_level == Rank.TWO   # unchanged

    def test_declarer_team1_switch(self) -> None:
        """Declarer team=1, defender 100 points -> switch to team 0."""
        result = calculate_score(
            defender_points=100,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=0),
            declarer_team=1,
            declarer_player=1,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.declarer_level_change == 0
        assert result.switch_declarer is True
        assert result.next_declarer_team == 0
        assert result.next_declarer_player == 3  # CCW next of 1 is 3
        assert result.team1_new_level == Rank.TWO  # declarer team unchanged
        assert result.team0_new_level == Rank.TWO  # defender gets 0 advance


class TestOver200:
    def test_defender_over_200_from_bonus(self) -> None:
        """When ambush bonus pushes total over 200, still gets -3 and switch."""
        # 6-card tractor ambush: 2^6=64, bottom_base=50 -> bonus=3200
        # total = 50 + 3200 = 3250, well over 200
        bottom = [
            _card(Suit.SPADES, Rank.FIVE, 1), _card(Suit.SPADES, Rank.FIVE, 2),
            _card(Suit.SPADES, Rank.TEN, 1), _card(Suit.SPADES, Rank.TEN, 2),
            _card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2),
        ]
        result = calculate_score(
            defender_points=50,
            bottom_cards=bottom,
            last_trick=_completed_trick(PlayType.TRACTOR, 6, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        # total = 50 + 3200 = 3250 -> fallback: -3, switch
        assert result.declarer_level_change == -3
        assert result.switch_declarer is True
        assert result.total_defender_points == 3250
        assert result.team0_new_level == Rank.TWO  # TWO - 3 clamped at TWO
        assert result.team1_new_level == Rank.FIVE  # TWO + 3 = FIVE

    def test_defender_points_exactly_200(self) -> None:
        """Defender points exactly 200 -> -3, switch."""
        result = calculate_score(
            defender_points=200,
            bottom_cards=[],
            last_trick=_completed_trick(PlayType.SINGLE, 1, winner=1),
            declarer_team=0,
            declarer_player=0,
            team0_level=Rank.TWO,
            team1_level=Rank.TWO,
        )
        assert result.declarer_level_change == -3
        assert result.switch_declarer is True
        assert result.total_defender_points == 200
