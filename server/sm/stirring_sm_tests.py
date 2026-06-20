"""Tests for sm.stirring_sm module."""
from typing import Literal

from .card_model import Card, Suit, Rank
from server.result import Ok, Rejected
from .stirring_sm import (
    StirringState,
    StirInput,
    create_stirring, pass_stir, stir, stir_discard, get_stir_result,
)


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit, rank=rank,
        is_joker=(suit == Suit.JOKER),
        is_big_joker=(rank == Rank.BIG_JOKER),
        points=0, deck=deck,
    )


def _make_input(
    *,
    trump_suit: Suit | None = Suit.HEARTS,
    trump_rank: Rank = Rank.TWO,
    declarer_player: int = 0,
    initial_bid_cards: list[Card] | None = None,
    bottom_cards: list[Card] | None = None,
    players_hand: list[list[Card]] | None = None,
) -> StirInput:
    """Create StirInput with defaults for bottom_cards and players_hand."""
    if bottom_cards is None:
        bottom_cards = [
            _card(Suit.DIAMONDS, Rank.THREE, 1),
            _card(Suit.DIAMONDS, Rank.FOUR, 1),
            _card(Suit.DIAMONDS, Rank.FIVE, 1),
            _card(Suit.DIAMONDS, Rank.SIX, 1),
            _card(Suit.CLUBS, Rank.THREE, 1),
            _card(Suit.CLUBS, Rank.FOUR, 1),
            _card(Suit.CLUBS, Rank.FIVE, 1),
            _card(Suit.CLUBS, Rank.SIX, 1),
        ]
    if players_hand is None:
        # 4 players, each with 25 cards (enough for exchanges)
        players_hand = [[] for _ in range(4)]
        for i in range(4):
            cards: list[Card] = []
            for rank in [Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
                         Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
                         Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
                cards.append(_card(Suit.HEARTS if i == 0 else Suit.SPADES, rank, 1))
                cards.append(_card(Suit.HEARTS if i == 0 else Suit.SPADES, rank, 2))
            # Add one trump-rank card per suit for stirring tests
            cards.append(_card(Suit.SPADES, Rank.TWO, 1))
            cards.append(_card(Suit.HEARTS, Rank.TWO, 1))
            players_hand[i] = cards
    return StirInput(
        trump_suit=trump_suit,
        trump_rank=trump_rank,
        initial_bid_cards=[] if initial_bid_cards is None else initial_bid_cards,
        declarer_player=declarer_player,
        bottom_cards=bottom_cards,
        players_hand=players_hand,
    )


def _complete_initial_exchange(state: StirringState) -> StirringState:
    """Complete the initial EXCHANGING sub-phase by discarding bottom cards."""
    assert state.phase == "EXCHANGING"
    assert state.exchange_state is not None
    assert state.exchanging_player is not None
    discards = state.exchange_state.hand_after_pickup[:state.exchange_state.count]
    result = stir_discard(state, player=state.exchanging_player, cards=discards)
    assert isinstance(result, Ok), f"stir_discard rejected: {result.reason}"
    return result.value

class TestCreateStirring:
    def test_create_stirring_starts_in_exchanging(self) -> None:
        """Initial state: EXCHANGING phase, current_player = declarer."""
        state = create_stirring(_make_input(
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
            declarer_player=0,
        ))
        assert state.phase == "EXCHANGING"
        assert state.current_player == 0  # declarer must exchange first
        assert len(state.pass_set) == 0
        assert len(state.actions) == 0
        assert state.exchanging_player == 0

    def test_create_stirring_has_exchange_state(self) -> None:
        """Initial state has exchange_state set up for declarer."""
        state = create_stirring(_make_input(
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
            declarer_player=0,
        ))
        assert state.exchange_state is not None
        assert state.exchange_state.phase == "PICKED_UP"
        assert state.exchange_state.count == 8  # BOTTOM_CARD_COUNT

    def test_create_stirring_with_empty_trump(self) -> None:
        """空主 (trump_suit=None) also starts in EXCHANGING."""
        state = create_stirring(_make_input(
            trump_suit=None,
            trump_rank=Rank.TWO,
            declarer_player=0,
        ))
        assert state.phase == "EXCHANGING"
        assert state.current_player == 0

    def test_create_stirring_single_initial_bid_uses_single_priority(self) -> None:
        """A single initial bid can be over-stirred by any trump-rank pair."""
        initial_bid = [_card(Suit.SPADES, Rank.FIVE, 1)]
        players_hand = [[], [], [], [
            _card(Suit.DIAMONDS, Rank.FIVE, 1),
            _card(Suit.DIAMONDS, Rank.FIVE, 2),
        ]]
        state = create_stirring(_make_input(
            trump_suit=Suit.SPADES,
            trump_rank=Rank.FIVE,
            declarer_player=0,
            initial_bid_cards=initial_bid,
            players_hand=players_hand,
        ))
        state = _complete_initial_exchange(state)

        result = stir(state, player=state.current_player, cards=players_hand[3])

        assert isinstance(result, Ok)
        assert result.value.trump_suit == Suit.DIAMONDS

    def test_create_stirring_joker_initial_bid_uses_joker_priority(self) -> None:
        """A big-joker initial bid is already max priority."""
        initial_bid = [
            _card(Suit.JOKER, Rank.BIG_JOKER, 1),
            _card(Suit.JOKER, Rank.BIG_JOKER, 2),
        ]
        players_hand = [[], [], [], [
            _card(Suit.SPADES, Rank.FIVE, 1),
            _card(Suit.SPADES, Rank.FIVE, 2),
        ]]
        state = create_stirring(_make_input(
            trump_suit=None,
            trump_rank=Rank.FIVE,
            declarer_player=0,
            initial_bid_cards=initial_bid,
            players_hand=players_hand,
        ))

        assert state.current_priority == 205
        state = _complete_initial_exchange(state)
        assert state.phase == "COMPLETE"


class TestStirDiscard:
    def test_stir_discard_completes_initial_exchange(self) -> None:
        """After stir_discard, phase transitions to WAITING and skips exchanger."""
        state = create_stirring(_make_input(
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
            declarer_player=0,
        ))
        assert state.phase == "EXCHANGING"
        # Discard bottom cards (first 8 from combined hand)
        assert state.exchange_state is not None
        discards = state.exchange_state.hand_after_pickup[:8]
        result = stir_discard(state, player=0, cards=discards)
        assert isinstance(result, Ok)
        new_state = result.value
        assert new_state.phase == "WAITING"
        assert new_state.current_player == 1  # CCW next of 0
        assert new_state.pass_set == frozenset({0})
        assert new_state.exchange_state is None
        assert new_state.exchanging_player is None

    def test_stir_discard_wrong_player_rejected(self) -> None:
        """Only the exchanging player can discard."""
        state = create_stirring(_make_input(
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
            declarer_player=0,
        ))
        discards = state.exchange_state.hand_after_pickup[:8] if state.exchange_state else []
        result = stir_discard(state, player=1, cards=discards)
        assert isinstance(result, Rejected)
        assert "炒主者" in result.reason

    def test_stir_discard_wrong_card_count_rejected(self) -> None:
        """Discarding wrong number of cards is rejected."""
        state = create_stirring(_make_input(
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
            declarer_player=0,
        ))
        assert state.exchange_state is not None
        # Discard only 7 cards instead of 8
        discards = state.exchange_state.hand_after_pickup[:7]
        result = stir_discard(state, player=0, cards=discards)
        assert isinstance(result, Rejected)

    def test_stir_discard_not_in_exchanging_rejected(self) -> None:
        """stir_discard is rejected when not in EXCHANGING phase."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
            declarer_player=0,
        )))
        assert state.phase == "WAITING"
        result = stir_discard(state, player=0, cards=[])
        assert isinstance(result, Rejected)
        assert "换底牌阶段" in result.reason

    def test_stir_discard_updates_hands_and_bottom(self) -> None:
        """After stir_discard, players_hand and bottom_cards are updated."""
        bottom = [_card(Suit.DIAMONDS, Rank.THREE, 1)]
        hands = [
            [_card(Suit.HEARTS, Rank.ACE, 1)],
            [], [], [],
        ]
        state = create_stirring(_make_input(
            trump_suit=Suit.HEARTS,
            declarer_player=0,
            bottom_cards=bottom,
            players_hand=hands,
        ))
        assert state.exchange_state is not None
        # Discard the one bottom card
        discards = state.exchange_state.hand_after_pickup[:1]
        result = stir_discard(state, player=0, cards=discards)
        assert isinstance(result, Ok)
        new_state = result.value
        # Player 0's hand should be updated (original + bottom - discards)
        assert len(new_state.players_hand[0]) == 1  # 1 original + 1 bottom - 1 discard
        # Bottom cards should be the discarded cards
        assert len(new_state.bottom_cards) == 1
        assert new_state.bottom_cards[0].id == discards[0].id


class TestPassStir:
    def test_pass_stir_rejected_in_exchanging(self) -> None:
        """Pass is rejected during EXCHANGING sub-phase."""
        state = create_stirring(_make_input(
            trump_suit=Suit.HEARTS,
            trump_rank=Rank.TWO,
            declarer_player=0,
        ))
        assert state.phase == "EXCHANGING"
        result = pass_stir(state, player=0)
        assert isinstance(result, Rejected)
        assert "换底牌" in result.reason

    def test_pass_stir_adds_to_pass_set(self) -> None:
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        assert state.pass_set == frozenset({0})
        result = pass_stir(state, player=state.current_player)
        assert isinstance(result, Ok)
        assert state.current_player in result.value.pass_set

    def test_pass_stir_advances_player(self) -> None:
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cur = state.current_player
        result = pass_stir(state, player=cur)
        assert isinstance(result, Ok)
        from .constants import next_player_ccw
        assert result.value.current_player == next_player_ccw(cur)

    def test_pass_stir_all_pass_complete(self) -> None:
        """When the three non-exchanging players pass, phase becomes COMPLETE."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        assert state.pass_set == frozenset({0})
        for _ in range(3):
            result = pass_stir(state, player=state.current_player)
            assert isinstance(result, Ok)
            state = result.value
        assert state.phase == "COMPLETE"
        assert state.pass_set == frozenset({0, 1, 2, 3})

    def test_pass_stir_wrong_player_rejected(self) -> None:
        """Pass from a player who is not current_player is rejected."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        # current_player is 1; pass from player 0 should be rejected
        result = pass_stir(state, player=0)
        assert isinstance(result, Rejected)


class TestStir:
    def test_stir_rejected_in_exchanging(self) -> None:
        """Stir is rejected during EXCHANGING sub-phase."""
        state = create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        ))
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Rejected)
        assert "反主" in result.reason

    def test_stir_pair_accepted(self) -> None:
        """Pair of trump rank in higher-priority suit is accepted."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit == Suit.SPADES
        assert result.value.phase == "EXCHANGING"  # transitions to EXCHANGING

    def test_stir_pair_changes_trump(self) -> None:
        """Stir changes trump suit to the pair's suit."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.CLUBS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit == Suit.HEARTS

    def test_stir_pair_resets_pass_set(self) -> None:
        """After a stir, pass_set is reset (others can counter-stir)."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        assert state.pass_set == frozenset({0})
        # Player 1 passes first
        result = pass_stir(state, player=state.current_player)
        assert isinstance(result, Ok)
        state = result.value
        assert len(state.pass_set) == 2
        # Player 3 stirs
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert len(result.value.pass_set) == 0

    def test_stir_single_rejected(self) -> None:
        """Single card stir is rejected (must be pair)."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.SPADES, Rank.TWO, 1)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Rejected)
        assert "对子" in result.reason

    def test_stir_same_suit_rejected(self) -> None:
        """Stirring with the same suit as current trump is rejected."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.HEARTS, Rank.TWO, 1), _card(Suit.HEARTS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Rejected)
        assert "优先级" in result.reason

    def test_stir_lower_priority_rejected(self) -> None:
        """Pair with lower priority than current trump suit is rejected."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO, declarer_player=0,
        )))
        # ♣ pair has lower priority than current ♠ trump
        cards = [_card(Suit.CLUBS, Rank.TWO, 1), _card(Suit.CLUBS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Rejected)
        assert "优先级" in result.reason

    def test_stir_joker_pair_accepted(self) -> None:
        """Joker pair stir is always accepted (highest priority)."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.SPADES, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit is None  # 无主

    def test_stir_joker_pair_sets_no_trump(self) -> None:
        """Joker pair stir sets trump_suit=None."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.JOKER, Rank.SMALL_JOKER, 1), _card(Suit.JOKER, Rank.SMALL_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit is None

    def test_stir_empty_trump_diamond_pair_accepted(self) -> None:
        """空主: even ♦ pair can stir (lowest priority still beats no trump)."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.DIAMONDS, Rank.TWO, 1), _card(Suit.DIAMONDS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit == Suit.DIAMONDS

    def test_stir_empty_trump_joker_pair_accepted(self) -> None:
        """空主: joker pair is accepted, sets trump to None (per SI-006)."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.trump_suit is None  # stays None

    def test_stir_empty_trump_small_joker_then_big_joker(self) -> None:
        """空主: small joker pair → big joker pair accepted (higher priority)."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        )))
        # Player stirs with small joker pair
        small_jokers = [_card(Suit.JOKER, Rank.SMALL_JOKER, 1), _card(Suit.JOKER, Rank.SMALL_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=small_jokers)
        assert isinstance(result, Ok)
        assert result.value.trump_suit is None
        assert result.value.current_priority == 204
        # Must complete exchange before continuing
        state = _complete_initial_exchange(result.value)
        # Next player stirs with big joker pair
        big_jokers = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=big_jokers)
        assert isinstance(result, Ok)
        assert result.value.trump_suit is None
        assert result.value.current_priority == 205

    def test_stir_empty_trump_big_joker_then_small_joker_rejected(self) -> None:
        """空主: after big joker pair, phase is COMPLETE (no higher stir possible)."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        )))
        # Player stirs with big joker pair
        big_jokers = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=big_jokers)
        assert isinstance(result, Ok)
        assert result.value.current_priority == 205
        # After exchange, goes directly to COMPLETE (max priority)
        state = _complete_initial_exchange(result.value)
        assert state.phase == "COMPLETE"

    def test_stir_empty_trump_joker_no_infinite_loop(self) -> None:
        """空主: two players with joker pairs cannot alternate infinitely."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=None, trump_rank=Rank.TWO, declarer_player=0,
        )))
        # Player stirs with big joker pair
        big_jokers = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=big_jokers)
        assert isinstance(result, Ok)
        state = _complete_initial_exchange(result.value)
        assert state.current_priority == 205
        # Big joker pair is max priority → COMPLETE directly, no pass needed
        assert state.phase == "COMPLETE"

    def test_stir_big_joker_pair_skips_waiting(self) -> None:
        """Big joker pair (max priority) skips WAITING, goes directly to COMPLETE after exchange."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        big_jokers = [_card(Suit.JOKER, Rank.BIG_JOKER, 1), _card(Suit.JOKER, Rank.BIG_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=big_jokers)
        assert isinstance(result, Ok)
        assert result.value.phase == "EXCHANGING"
        assert result.value.current_priority == 205
        # Complete the exchange — should go to COMPLETE, not WAITING
        state = _complete_initial_exchange(result.value)
        assert state.phase == "COMPLETE"

    def test_stir_small_joker_pair_goes_to_waiting(self) -> None:
        """Small joker pair (not max) goes to WAITING — others could still stir."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        small_jokers = [_card(Suit.JOKER, Rank.SMALL_JOKER, 1), _card(Suit.JOKER, Rank.SMALL_JOKER, 2)]
        result = stir(state, player=state.current_player, cards=small_jokers)
        assert isinstance(result, Ok)
        assert result.value.phase == "EXCHANGING"
        assert result.value.current_priority == 204
        # Complete the exchange — should go to WAITING (big joker could still beat it)
        stirrer = result.value.current_player
        state = _complete_initial_exchange(result.value)
        assert state.phase == "WAITING"
        assert state.pass_set == frozenset({stirrer})

    def test_stir_wrong_player_rejected(self) -> None:
        """Stir from a player who is not current_player is rejected."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        wrong_player = 0  # current is 1
        result = stir(state, player=wrong_player, cards=cards)
        assert isinstance(result, Rejected)
        assert "回合" in result.reason

    def test_last_stir_player_is_skipped_after_exchange(self) -> None:
        """Player who just stirred is not asked again in the same stir cycle."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        # Player 1 stirs
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        stirrer = result.value.current_player
        state = _complete_initial_exchange(result.value)
        assert state.pass_set == frozenset({stirrer})

        # Other three players pass; the turn never returns to the stirrer.
        for _ in range(3):
            result = pass_stir(state, player=state.current_player)
            assert isinstance(result, Ok)
            state = result.value
        assert state.phase == "COMPLETE"
        assert state.current_player != stirrer

    def test_stir_updates_current_priority(self) -> None:
        """Successful stir updates current_priority to the new bid_value."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        # Initial: ♦ pair priority = 200
        assert state.current_priority == 200

        # Stir with ♠ pair (priority 203)
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.current_priority == 203

    def test_stir_transitions_to_exchanging(self) -> None:
        """Successful stir transitions phase to EXCHANGING."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.phase == "EXCHANGING"
        assert result.value.exchanging_player == state.current_player
        assert result.value.exchange_state is not None


class TestStirFullFlow:
    def test_stir_full_flow_multiple_stirs(self) -> None:
        """Multiple stirs: each higher-priority pair overrides."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        # Player 1 stirs with ♣ pair (beats ♦)
        cards_club = [_card(Suit.CLUBS, Rank.TWO, 1), _card(Suit.CLUBS, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards_club)
        assert isinstance(result, Ok)
        state = _complete_initial_exchange(result.value)
        assert state.trump_suit == Suit.CLUBS
        # Player stirs with ♠ pair (beats ♣)
        cards_spade = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards_spade)
        assert isinstance(result, Ok)
        state = _complete_initial_exchange(result.value)
        assert state.trump_suit == Suit.SPADES
        # Remaining non-exchanging players pass
        for _ in range(3):
            result = pass_stir(state, player=state.current_player)
            assert isinstance(result, Ok)
            state = result.value
        assert state.phase == "COMPLETE"

    def test_stir_complete_result(self) -> None:
        """COMPLETE state produces correct StirResult."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.HEARTS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        # Non-exchanging players pass
        for _ in range(3):
            result = pass_stir(state, player=state.current_player)
            assert isinstance(result, Ok)
            state = result.value
        assert state.phase == "COMPLETE"
        # Use get_stir_result to extract StirResult
        stir_result = get_stir_result(state)
        assert stir_result.final_trump_suit == Suit.HEARTS
        assert stir_result.stir_count == 0

    def test_stir_complete_result_with_stirs(self) -> None:
        """COMPLETE state after stirs has correct stir_count and result."""
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS, trump_rank=Rank.TWO, declarer_player=0,
        )))
        # One stir
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        state = _complete_initial_exchange(result.value)
        # Non-exchanging players pass
        for _ in range(3):
            result = pass_stir(state, player=state.current_player)
            assert isinstance(result, Ok)
            state = result.value
        assert state.phase == "COMPLETE"
        stir_result = get_stir_result(state)
        assert stir_result.final_trump_suit == Suit.SPADES
        assert stir_result.stir_count == 1

    def test_stir_discard_after_stir_updates_hands(self) -> None:
        """After a stir, stir_discard updates the stirring player's hand."""
        bottom = [_card(Suit.DIAMONDS, Rank.THREE, 1)]
        hands = [
            [_card(Suit.HEARTS, Rank.ACE, 1)],
            [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)],
            [], [],
        ]
        state = _complete_initial_exchange(create_stirring(_make_input(
            trump_suit=Suit.DIAMONDS,
            declarer_player=0,
            bottom_cards=bottom,
            players_hand=hands,
        )))
        # Player 1 stirs with ♠ pair
        cards = [_card(Suit.SPADES, Rank.TWO, 1), _card(Suit.SPADES, Rank.TWO, 2)]
        result = stir(state, player=state.current_player, cards=cards)
        assert isinstance(result, Ok)
        assert result.value.phase == "EXCHANGING"
        assert result.value.exchanging_player == state.current_player
        # Player 1 discards
        assert result.value.exchange_state is not None
        discards = result.value.exchange_state.hand_after_pickup[:1]
        assert result.value.exchanging_player is not None
        result2 = stir_discard(result.value, player=result.value.exchanging_player, cards=discards)
        assert isinstance(result2, Ok)
        assert result2.value.phase == "WAITING"
        # Player 1's hand should be updated
        assert len(result2.value.players_hand[state.current_player]) == 2  # 2 original + 1 bottom - 1 discard
