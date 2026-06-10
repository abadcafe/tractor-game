"""Tests for sm.trick module."""
from typing import Literal

from server.sm.card_model import Card, Suit, Rank
from server.sm.trick import (
    TrickState, TrickInput, TrickResult,
    create_trick, play,
)
from server.sm.result import Ok, Rejected


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
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


class TestCreateTrick:
    def test_create_trick_initial_state(self) -> None:
        """Initial state: LEADING, no cards played."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.SPADES, Rank.ACE)],
        ]
        state = create_trick(TrickInput(
            lead_player=0,
            hands=hands,
            trump_suit=Suit.SPADES,
            trump_rank=Rank.TWO,
            defender_points=0,
            declarer_team=0,
        ))
        assert state.phase == "LEADING"
        assert state.lead_player == 0
        assert state.cur == 0
        assert state.played == 0

    def test_create_trick_lead_player(self) -> None:
        """Current player starts as lead_player."""
        hands = [[_card(Suit.HEARTS, Rank.ACE)]] * 4
        state = create_trick(TrickInput(
            lead_player=2,
            hands=hands,
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
            defender_points=0,
            declarer_team=0,
        ))
        assert state.cur == 2


class TestPlayLead:
    def test_play_lead_single(self) -> None:
        """Leading a single card transitions to FOLLOWING."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        assert state.phase == "FOLLOWING"
        assert state.played == 1

    def test_play_lead_pair(self) -> None:
        """Leading a pair transitions to FOLLOWING."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)],
            [_card(Suit.HEARTS, Rank.KING, 1), _card(Suit.HEARTS, Rank.KING, 2)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        pair = hands[0][:2]
        result = play(state, player=0, cards=pair)
        assert isinstance(result, Ok)
        state = result.value
        assert state.phase == "FOLLOWING"

    def test_play_lead_sets_following(self) -> None:
        """Lead play transitions to FOLLOWING phase."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        assert state.phase == "FOLLOWING"


class TestPlayFollow:
    def test_play_follow_single(self) -> None:
        """Following with a single card advances cur."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        assert state.cur == 1  # CCW next of 0
        result = play(state, player=1, cards=[hands[1][0]])
        assert isinstance(result, Ok)
        state = result.value
        assert state.played == 2

    def test_play_follow_after_lead(self) -> None:
        """After lead, following players play in CCW order."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])  # lead
        assert isinstance(result, Ok)
        state = result.value
        assert state.cur == 1  # next in CCW


class TestPlayResolve:
    def test_play_four_plays_resolve(self) -> None:
        """After 4 plays, trick resolves to RESOLVED."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=1, cards=[hands[1][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=3, cards=[hands[3][0]])  # CCW: 1->3
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=2, cards=[hands[2][0]])  # CCW: 3->2
        assert isinstance(result, Ok)
        state = result.value
        assert state.phase == "RESOLVED"

    def test_play_resolve_determines_winner(self) -> None:
        """Winner is determined by compare_plays: highest same-suit card wins."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],   # player 0: ♥A (highest)
            [_card(Suit.HEARTS, Rank.KING)],   # player 1: ♥K
            [_card(Suit.HEARTS, Rank.QUEEN)],  # player 2: ♥Q
            [_card(Suit.HEARTS, Rank.JACK)],   # player 3: ♥J
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=1, cards=[hands[1][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=3, cards=[hands[3][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=2, cards=[hands[2][0]])
        assert isinstance(result, Ok)
        state = result.value
        assert state.phase == "RESOLVED"
        # ♥A should win (highest rank in lead suit)
        result = _get_result(state)
        assert result.winner == 0

    def test_play_resolve_counts_points(self) -> None:
        """Points are summed from all played cards using card.points field."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],      # 0 pts
            [_card(Suit.HEARTS, Rank.KING)],      # 10 pts
            [_card(Suit.HEARTS, Rank.FIVE)],      # 5 pts
            [_card(Suit.HEARTS, Rank.JACK)],      # 0 pts
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=1, cards=[hands[1][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=3, cards=[hands[3][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=2, cards=[hands[2][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = _get_result(state)
        assert result.points == 15  # K(10) + 5(5) = 15

    def test_play_resolve_defender_points_update(self) -> None:
        """If defender wins, updated_defender_points includes trick points."""
        hands = [
            [_card(Suit.HEARTS, Rank.FIVE)],     # 5 pts (team 0)
            [_card(Suit.HEARTS, Rank.ACE)],      # team 1
            [_card(Suit.HEARTS, Rank.KING)],     # team 1, 10 pts
            [_card(Suit.HEARTS, Rank.THREE)],    # team 0
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=10, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=1, cards=[hands[1][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=3, cards=[hands[3][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=2, cards=[hands[2][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = _get_result(state)
        # Player 1 (team 1, defender) has ♥A which wins
        # Defender gets 5 + 10 = 15 trick points, total = 10 + 15 = 25
        assert result.winner == 1
        assert result.updated_defender_points == 25

    def test_play_resolve_completed_trick(self) -> None:
        """Resolved trick produces a CompletedTrick with all slots."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=1, cards=[hands[1][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=3, cards=[hands[3][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=2, cards=[hands[2][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = _get_result(state)
        assert result.completed_trick is not None
        assert len(result.completed_trick.slots) == 4
        assert result.completed_trick.lead_player == 0

    def test_play_lead_trump_beats_non_trump(self) -> None:
        """Trump card beats non-trump card."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],       # 0: non-trump ♥A
            [_card(Suit.SPADES, Rank.ACE)],        # 1: trump ♠A (spades=trump)
            [_card(Suit.HEARTS, Rank.QUEEN)],      # 2: non-trump
            [_card(Suit.HEARTS, Rank.JACK)],       # 3: non-trump
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=1, cards=[hands[1][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=3, cards=[hands[3][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=2, cards=[hands[2][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = _get_result(state)
        # Player 1's ♠A (trump) should beat player 0's ♥A (non-trump)
        assert result.winner == 1


class TestPlayValidation:
    def test_play_wrong_player_rejected(self) -> None:
        """Playing from wrong player is rejected."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=2, cards=[hands[2][0]])
        assert isinstance(result, Rejected)
        # not player 0's turn

    def test_play_not_in_hand_rejected(self) -> None:
        """Playing cards not in hand is rejected."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        fake = _card(Suit.DIAMONDS, Rank.ACE)
        result = play(state, player=0, cards=[fake])
        assert isinstance(result, Rejected)

    def test_play_follow_must_follow_suit(self) -> None:
        """Following player must play cards of the led effective suit if they have any."""
        # Lead: ♥A (single). Player 1 has ♥K and ♠Q.
        # Player 1 MUST follow hearts, not spades.
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],       # 0: leads ♥A
            [_card(Suit.HEARTS, Rank.KING), _card(Suit.SPADES, Rank.QUEEN)],  # 1: has ♥ and ♠
            [_card(Suit.HEARTS, Rank.QUEEN)],      # 2: ♥Q
            [_card(Suit.HEARTS, Rank.JACK)],       # 3: ♥J
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        # Player 0 leads ♥A
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        # Player 1 tries to play ♠Q (off-suit) -- should be rejected
        result = play(state, player=1, cards=[hands[1][1]])
        assert isinstance(result, Rejected)
        # ♠Q -- illegal
        # Player 1 plays ♥K (correct follow-suit) -- should succeed
        result = play(state, player=1, cards=[hands[1][0]])  # ♥K -- legal
        assert isinstance(result, Ok)
        state = result.value
        assert state.played == 2

    def test_play_follow_no_suit_can_play_anything(self) -> None:
        """Following player with no cards of the led suit can play any card."""
        # Lead: ♥A (single). Player 1 has only ♠ cards.
        # Player 1 can play any ♠ card.
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],       # 0: leads ♥A
            [_card(Suit.SPADES, Rank.QUEEN)],      # 1: only ♠ (no hearts)
            [_card(Suit.HEARTS, Rank.KING)],       # 2: ♥K
            [_card(Suit.HEARTS, Rank.JACK)],       # 3: ♥J
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        # Player 1 has no hearts, can play ♠Q
        result = play(state, player=1, cards=[hands[1][0]])
        assert isinstance(result, Ok)
        state = result.value
        assert state.played == 2

    def test_play_all_four_complete(self) -> None:
        """Playing all 4 cards resolves the trick."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=1, cards=[hands[1][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=3, cards=[hands[3][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=2, cards=[hands[2][0]])
        assert isinstance(result, Ok)
        state = result.value
        assert state.phase == "RESOLVED"
        assert state.played == 4


class TestPlayResolved:
    def test_play_on_resolved_trick_rejected(self) -> None:
        """Calling play() after the trick is resolved raises ValueError."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[hands[0][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=1, cards=[hands[1][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=3, cards=[hands[3][0]])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=2, cards=[hands[2][0]])
        assert isinstance(result, Ok)
        state = result.value
        assert state.phase == "RESOLVED"
        # No cards left in hands, but we can still verify the guard
        result = play(state, player=0, cards=[])
        assert isinstance(result, Rejected)


class TestPlayEmptyCards:
    def test_play_empty_cards_list_rejected(self) -> None:
        """Playing an empty cards list is rejected (not in hand)."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE)],
            [_card(Suit.HEARTS, Rank.KING)],
            [_card(Suit.HEARTS, Rank.QUEEN)],
            [_card(Suit.HEARTS, Rank.JACK)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=[])
        assert isinstance(result, Rejected)


class TestPlayFollowPairSuit:
    def test_play_follow_pair_must_follow_suit(self) -> None:
        """Following a pair lead: must play pair of the led suit if possible."""
        # Lead: pair of ♥A (2 cards). Player 1 has ♥K pair and ♠Q pair.
        # Player 1 MUST follow hearts pair.
        hands = [
            [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)],
            [_card(Suit.HEARTS, Rank.KING, 1), _card(Suit.HEARTS, Rank.KING, 2),
             _card(Suit.SPADES, Rank.QUEEN, 1), _card(Suit.SPADES, Rank.QUEEN, 2)],
            [_card(Suit.HEARTS, Rank.QUEEN, 1), _card(Suit.HEARTS, Rank.QUEEN, 2)],
            [_card(Suit.HEARTS, Rank.JACK, 1), _card(Suit.HEARTS, Rank.JACK, 2)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        # Player 0 leads ♥A pair
        result = play(state, player=0, cards=hands[0][:2])
        assert isinstance(result, Ok)
        state = result.value
        # Player 1 tries to play ♠Q pair (off-suit) -- should be rejected
        result = play(state, player=1, cards=hands[1][2:4])
        assert isinstance(result, Rejected)
        # Player 1 plays ♥K pair (correct follow-suit) -- should succeed
        result = play(state, player=1, cards=hands[1][:2])
        assert isinstance(result, Ok)
        state = result.value
        assert state.played == 2

    def test_play_follow_no_pair_can_play_anything(self) -> None:
        """Following a pair lead: with no pair of led suit, can play any pair."""
        # Lead: pair of ♥A. Player 1 has only ♠ cards (no ♥ pair).
        hands = [
            [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)],
            [_card(Suit.SPADES, Rank.QUEEN, 1), _card(Suit.SPADES, Rank.QUEEN, 2)],
            [_card(Suit.HEARTS, Rank.KING, 1), _card(Suit.HEARTS, Rank.KING, 2)],
            [_card(Suit.HEARTS, Rank.JACK, 1), _card(Suit.HEARTS, Rank.JACK, 2)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=hands[0][:2])
        assert isinstance(result, Ok)
        state = result.value
        # Player 1 has no hearts pair, plays ♠Q pair
        result = play(state, player=1, cards=hands[1][:2])
        assert isinstance(result, Ok)
        state = result.value
        assert state.played == 2


def _get_result(state: TrickState) -> TrickResult:
    """Extract result from a RESOLVED TrickState."""
    assert state.phase == "RESOLVED"
    assert state.result is not None
    return state.result


class TestPlayFollowTractorSuit:
    def test_play_follow_tractor_must_follow_suit(self) -> None:
        """Following a tractor lead: must play matching-length tractor of same suit if possible."""
        # Lead: tractor of ♥A pair + ♥K pair (4 cards).
        # Player 1 has both ♥Q pair + ♥J pair (matching tractor) and ♠Q pair + ♠J pair.
        # Player 1 MUST follow with hearts tractor.
        hands = [
            [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
             _card(Suit.HEARTS, Rank.KING, 1), _card(Suit.HEARTS, Rank.KING, 2)],
            [_card(Suit.HEARTS, Rank.QUEEN, 1), _card(Suit.HEARTS, Rank.QUEEN, 2),
             _card(Suit.HEARTS, Rank.JACK, 1), _card(Suit.HEARTS, Rank.JACK, 2),
             _card(Suit.SPADES, Rank.QUEEN, 1), _card(Suit.SPADES, Rank.QUEEN, 2),
             _card(Suit.SPADES, Rank.JACK, 1), _card(Suit.SPADES, Rank.JACK, 2)],
            [_card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2)],
            [_card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2)],
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        # Player 0 leads ♥A+♥K tractor (4 cards)
        lead_cards = hands[0][:4]
        result = play(state, player=0, cards=lead_cards)
        assert isinstance(result, Ok)
        state = result.value
        # Player 1 tries to play ♠Q+♠J pair (off-suit tractor) -- should be rejected
        off_suit = hands[1][4:8]
        result = play(state, player=1, cards=off_suit)
        assert isinstance(result, Rejected)
        # Player 1 plays ♥Q+♥J pair (correct follow-suit tractor) -- should succeed
        on_suit = hands[1][:4]
        result = play(state, player=1, cards=on_suit)
        assert isinstance(result, Ok)
        state = result.value
        assert state.played == 2


class TestPlayResolveNewComparison:
    def test_resolve_pair_beats_single_same_suit(self) -> None:
        """When all 4 play pairs, the highest pair wins."""
        hands = [
            [_card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2)],  # 0: h3 pair
            [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)],      # 1: hA pair
            [_card(Suit.HEARTS, Rank.KING, 1), _card(Suit.HEARTS, Rank.KING, 2)],     # 2: hK pair
            [_card(Suit.HEARTS, Rank.QUEEN, 1), _card(Suit.HEARTS, Rank.QUEEN, 2)],   # 3: hQ pair
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=hands[0])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=1, cards=hands[1])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=3, cards=hands[3])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=2, cards=hands[2])
        assert isinstance(result, Ok)
        state = result.value
        result = _get_result(state)
        assert result.winner == 1  # hA pair wins

    def test_resolve_trump_pair_beats_non_trump_pair(self) -> None:
        """Trump pair beats non-trump pair."""
        hands = [
            [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)],      # 0: hA pair (non-trump)
            [_card(Suit.SPADES, Rank.THREE, 1), _card(Suit.SPADES, Rank.THREE, 2)],   # 1: sp3 pair (trump)
            [_card(Suit.HEARTS, Rank.KING, 1), _card(Suit.HEARTS, Rank.KING, 2)],     # 2: hK pair
            [_card(Suit.HEARTS, Rank.QUEEN, 1), _card(Suit.HEARTS, Rank.QUEEN, 2)],   # 3: hQ pair
        ]
        state = create_trick(TrickInput(
            lead_player=0, hands=hands,
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO,
            defender_points=0, declarer_team=0,
        ))
        result = play(state, player=0, cards=hands[0])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=1, cards=hands[1])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=3, cards=hands[3])
        assert isinstance(result, Ok)
        state = result.value
        result = play(state, player=2, cards=hands[2])
        assert isinstance(result, Ok)
        state = result.value
        result = _get_result(state)
        assert result.winner == 1  # trump pair wins
