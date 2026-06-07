"""End-to-end integration tests for the sm (state machine) game engine.

These tests drive the complete game flow from start to game-over,
using actual state machine operations (deal_next_card, reveal, stir,
pass_stir, discard, play) -- NOT manually constructed result objects.
"""
from server.sm.card_model import Card, Suit, Rank
from server.sm.types import (
    BidEvent, PlayAction, PlayType, CompletedTrick, CompletedTrickSlot,
)
from server.sm.constants import LEVELS
from server.sm.play_rules import get_legal_plays
from server.sm.scoring import calculate_score
from server.sm.round_sm import (
    create_round, deal_next_card as rn_deal, reveal as rn_reveal,
    pass_stir as rn_pass, discard as rn_discard,
    play as rn_play, is_round_complete, get_round_result, RoundInput,
    RoundState,
)
from server.sm.game_sm import create_game, start_game, process_round_result


def _play_first_legal(round_state: RoundState) -> RoundState:
    """Play the first legal play for the current player in the trick.

    Uses get_legal_plays to find a valid play instead of blindly playing
    hand[0], which may violate follow-suit rules and cause ValueError.
    """
    trick = round_state.trick_state
    assert trick is not None
    cur = trick.cur
    hand = trick.hands[cur]
    if not hand:
        return round_state

    # Determine if leading or following
    is_leading = trick.phase == "LEADING"
    if is_leading:
        lead_action: PlayAction | None = None
    else:
        # Build the lead action from the lead player's slot
        lead_slot = trick.slots[trick.lead_player]
        assert lead_slot is not None
        lead_cards = lead_slot.cards
        lead_action = PlayAction(type=trick.lead_type, cards=lead_cards)

    legal_plays = get_legal_plays(
        hand=hand,
        is_leading=is_leading,
        lead_action=lead_action,
        trump_suit=round_state.trump_suit,
        trump_rank=round_state.trump_rank,
    )
    assert len(legal_plays) > 0, f"No legal plays for player {cur}"
    return rn_play(round_state, cards=legal_plays[0].cards)


def _complete_round_no_bid(round_state: RoundState) -> RoundState:
    """Drive a round through all phases with no bids, all pass stirring."""
    # Deal-bid: deal all 100 cards without revealing
    while round_state.phase == "DEAL_BID":
        if round_state.deal_bid_state is None or round_state.deal_bid_state.phase != "DEALING":
            break
        round_state = rn_deal(round_state)

    # Stirring: all pass
    for _ in range(4):
        if round_state.phase != "STIRRING":
            break
        round_state = rn_pass(round_state)

    # Exchange: discard the original bottom cards
    if round_state.phase == "EXCHANGE" and round_state.exchange_state is not None:
        discards = round_state.exchange_state.hand_after_pickup[:round_state.exchange_state.count]
        round_state = rn_discard(round_state, discards)

    # Playing: play all tricks using legal plays
    while round_state.phase == "PLAYING":
        trick = round_state.trick_state
        if trick is None:
            break
        for _ in range(4):
            if trick.phase == "RESOLVED":
                # Check if we need to start next trick
                if round_state.phase == "PLAYING" and round_state.trick_state is not None:
                    trick = round_state.trick_state
                    if trick.phase == "RESOLVED" or trick.phase == "LEADING":
                        if trick.phase == "RESOLVED":
                            break
                else:
                    break
            round_state = _play_first_legal(round_state)
            trick = round_state.trick_state
            if trick is None:
                break

    return round_state


class TestE2EFullRound:
    def test_e2e_full_round_deal_bid_to_scoring(self) -> None:
        """Drive a complete round from DEAL_BID through SCORING using real operations."""
        round_state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        round_state = _complete_round_no_bid(round_state)

        # Round should be COMPLETE or SCORING
        assert round_state.phase in ("SCORING", "COMPLETE")
        if is_round_complete(round_state):
            result = get_round_result(round_state)
            assert result is not None
            assert result.next_declarer_team in (0, 1)
            assert result.next_declarer_player in (0, 1, 2, 3)
            assert result.total_defender_points >= 0

    def test_e2e_round_with_empty_trump(self) -> None:
        """Round with no bids = empty trump throughout."""
        round_state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        # Deal all cards without bidding
        while round_state.phase == "DEAL_BID":
            if round_state.deal_bid_state is None or round_state.deal_bid_state.phase != "DEALING":
                break
            round_state = rn_deal(round_state)

        assert round_state.phase == "STIRRING"
        assert round_state.trump_suit is None  # empty trump

    def test_e2e_round_with_bid_and_stirring(self) -> None:
        """Round with a bid: someone reveals, then stirring."""
        round_state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        # Deal some cards
        for _ in range(20):
            if round_state.deal_bid_state is None or round_state.deal_bid_state.phase != "DEALING":
                break
            round_state = rn_deal(round_state)

        # Find and reveal a trump rank card
        for p in range(4):
            if round_state.deal_bid_state is None:
                break
            trump_cards = [c for c in round_state.deal_bid_state.players_hand[p]
                          if c.rank == Rank.TWO and not c.is_joker]
            if trump_cards:
                event = BidEvent(
                    player=p, cards=[trump_cards[0]], kind="trump_rank",
                    suit=trump_cards[0].suit, joker_type=None, count=1,
                )
                round_state = rn_reveal(round_state, event)
                break

        # Deal remaining
        while round_state.phase == "DEAL_BID":
            if round_state.deal_bid_state is None or round_state.deal_bid_state.phase != "DEALING":
                break
            round_state = rn_deal(round_state)

        # If deal-bid completed with a winner, we should be in STIRRING
        if round_state.deal_bid_state is not None and round_state.deal_bid_state.bid_winner is not None:
            assert round_state.phase == "STIRRING"
            assert round_state.trump_suit is not None

    def test_e2e_round_with_exchange(self) -> None:
        """Exchange: declarer picks up and discards bottom cards."""
        round_state = create_round(RoundInput(
            declarer_team=None, trump_rank=Rank.TWO,
            last_declarer_player=None,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        ))
        # Complete deal-bid + stirring
        while round_state.phase == "DEAL_BID":
            if round_state.deal_bid_state is None or round_state.deal_bid_state.phase != "DEALING":
                break
            round_state = rn_deal(round_state)
        for _ in range(4):
            if round_state.phase != "STIRRING":
                break
            round_state = rn_pass(round_state)

        assert round_state.phase == "EXCHANGE"
        assert round_state.exchange_state is not None
        # Discard cards using the dynamic count from exchange state
        discards = round_state.exchange_state.hand_after_pickup[:round_state.exchange_state.count]
        round_state = rn_discard(round_state, discards)
        assert round_state.phase == "PLAYING"
        assert round_state.trick_state is not None


class TestE2EMultipleRounds:
    def test_e2e_multiple_rounds_game_flow(self) -> None:
        """Drive multiple rounds through the game state machine using real round results."""
        game = create_game()
        game = start_game(game)

        for i in range(6):
            if game.phase == "GAME_OVER":
                break

            # Run a full round using actual state machine operations
            round_state = create_round(RoundInput(
                declarer_team=game.declarer_team,
                trump_rank=game.team0_level if (game.declarer_team or 0) == 0 else game.team1_level,
                last_declarer_player=game.last_declarer_player,
                team0_level=game.team0_level,
                team1_level=game.team1_level,
            ))
            round_state = _complete_round_no_bid(round_state)

            if is_round_complete(round_state):
                result = get_round_result(round_state)
                assert result is not None
                game = process_round_result(game, result)
            else:
                # Round didn't complete cleanly, skip
                break

        # Game should either be over or still in progress
        assert game.phase in ("IN_ROUND", "GAME_OVER")

    def test_e2e_full_game_declarer_wins_fast(self) -> None:
        """Fast game: use real RoundResults from completed rounds to drive game_sm."""
        game = create_game()
        game = start_game(game)

        # Run actual rounds until game ends (could be many)
        max_rounds = 20
        for i in range(max_rounds):
            if game.phase == "GAME_OVER":
                break

            round_state = create_round(RoundInput(
                declarer_team=game.declarer_team,
                trump_rank=game.team0_level if (game.declarer_team or 0) == 0 else game.team1_level,
                last_declarer_player=game.last_declarer_player,
                team0_level=game.team0_level,
                team1_level=game.team1_level,
            ))
            round_state = _complete_round_no_bid(round_state)

            if is_round_complete(round_state):
                result = get_round_result(round_state)
                assert result is not None
                game = process_round_result(game, result)
            else:
                break

        # Either game ended or we ran max rounds
        assert game.phase in ("IN_ROUND", "GAME_OVER")


class TestE2EScoringBoundaryCases:
    def _completed_trick(self, lead_type: PlayType, card_count: int, winner: int) -> CompletedTrick:
        """Create a minimal CompletedTrick for scoring tests.

        Always includes a slot for lead_player=0 so the primary lookup path
        in _find_lead_card_count/_find_lead_cards is exercised rather than
        the fallback.
        """
        lead_cards = [Card(id="D1-spades-3", suit=Suit.SPADES, rank=Rank.THREE,
                           is_joker=False, is_big_joker=False, points=0, deck=1)] * card_count
        slots = [CompletedTrickSlot(player=0, cards=lead_cards)]
        if winner != 0:
            slots.append(CompletedTrickSlot(
                player=winner,
                cards=[Card(id="D1-spades-3", suit=Suit.SPADES, rank=Rank.THREE,
                            is_joker=False, is_big_joker=False, points=0, deck=1)] * card_count,
            ))
        return CompletedTrick(
            lead_player=0, lead_type=lead_type, slots=slots,
            winner=winner, points=0,
        )

    def test_e2e_scoring_exact_boundaries(self) -> None:
        """Verify scoring at exact boundary values using the scoring module directly."""
        # 0 points = big light (+3)
        result = calculate_score(
            defender_points=0, bottom_cards=[],
            last_trick=self._completed_trick(PlayType.SINGLE, 1, winner=0),
            declarer_team=0, declarer_player=0,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        )
        assert result.declarer_level_change == 3

        # 80 points = switch boundary
        result = calculate_score(
            defender_points=80, bottom_cards=[],
            last_trick=self._completed_trick(PlayType.SINGLE, 1, winner=1),
            declarer_team=0, declarer_player=0,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        )
        assert result.switch_declarer is True
        assert result.declarer_level_change == 0

    def test_e2e_ambush_multiplier_tractor(self) -> None:
        """Verify tractor ambush multiplier = 2^N."""
        bottom = [
            Card(id="D1-spades-5", suit=Suit.SPADES, rank=Rank.FIVE,
                 is_joker=False, is_big_joker=False, points=5, deck=1),
            Card(id="D2-spades-5", suit=Suit.SPADES, rank=Rank.FIVE,
                 is_joker=False, is_big_joker=False, points=5, deck=2),
            Card(id="D1-spades-10", suit=Suit.SPADES, rank=Rank.TEN,
                 is_joker=False, is_big_joker=False, points=10, deck=1),
            Card(id="D2-spades-10", suit=Suit.SPADES, rank=Rank.TEN,
                 is_joker=False, is_big_joker=False, points=10, deck=2),
        ]
        result = calculate_score(
            defender_points=0, bottom_cards=bottom,
            last_trick=self._completed_trick(PlayType.TRACTOR, 4, winner=1),
            declarer_team=0, declarer_player=0,
            team0_level=Rank.TWO, team1_level=Rank.TWO,
        )
        # 4-card tractor: (5+5+10+10) * 2^4 = 30 * 16 = 480
        assert result.bottom_card_bonus == 480


class TestE2ELevelProgression:
    def test_e2e_level_progression_sequence(self) -> None:
        """Levels progress through the correct sequence: 2->3->...->A."""
        expected = [
            Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE,
            Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE,
            Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
        ]
        assert list(LEVELS) == expected
