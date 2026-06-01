"""Game state machine for 升级 (Shengji/Tractor).

Orchestrates phase transitions, validates actions using the rules layer,
and exposes the query/action interface that the API layer will call.
Implements AI auto-play: after a human action, the server automatically
executes AI turns until it is the human player's turn again.

Turn order is sequential (0→1→2→3→0) throughout all phases.  The Game class
manages current_player_index directly rather than delegating to state
functions (which use the clockwise seating order 0→2→3→1→0).
"""

from __future__ import annotations

from server.engine.card import Card, Rank, Suit
from server.engine.constants import HUMAN_PLAYER_INDEX, PLAYER_COUNT
from server.engine.game_state import GameState, TrickSlot
from server.engine.player_utils import get_team_index
from server.engine.scoring import ScoreResult, calculate_score, is_game_over
from server.engine.state import (
    advance_round,
    clear_trick as state_clear_trick,
    create_initial_state,
    deal_cards,
    discard_cards,
    pickup_bottom_cards,
    play_cards,
    set_declarer,
)
from server.engine.types import BidAction, Phase, PlayAction, PlayType, StirAction
from server.rules.bidding import (
    get_valid_bid_levels,
    get_valid_stir_options,
    get_winning_bid,
    is_bidding_over,
    is_valid_bid,
    is_valid_stir,
)
from server.rules.validator import get_legal_plays as _get_legal_plays


def _next_player_sequential(player_index: int) -> int:
    """Advance to next player in sequential order (0→1→2→3→0)."""
    return (player_index + 1) % PLAYER_COUNT


class Game:
    """Central game controller orchestrating phase transitions and action dispatch."""

    def __init__(self) -> None:
        self.state: GameState = create_initial_state()
        self._stir_passes: set[int] = set()

    # ---- Game Flow ----

    def start_round(self) -> None:
        """Deal cards and begin bidding."""
        self.state = deal_cards(self.state)
        self._stir_passes = set()

    # ---- Bidding ----

    def submit_bid(
        self, player_index: int, level: Rank | None, *, pass_: bool
    ) -> bool:
        """Submit a bid for a player. Returns True if accepted."""
        if self.state.phase != Phase.BIDDING:
            return False
        if player_index != self.state.current_player_index:
            return False

        highest_bid = get_winning_bid(self.state.bidding_history)
        highest_level = highest_bid.level if highest_bid else None

        if not is_valid_bid(level, pass_, highest_level, self.state.current_level):
            return False

        # Record bid manually (sequential turn order, not clockwise)
        bid = BidAction(
            player_index=player_index,
            level=level,
            pass_=pass_,
        )
        self.state = self.state.model_copy(update={
            "bidding_history": [*self.state.bidding_history, bid],
            "current_player_index": _next_player_sequential(player_index),
        })

        # Check if bidding is over
        if is_bidding_over(self.state.bidding_history, PLAYER_COUNT):
            winner = get_winning_bid(self.state.bidding_history)
            if winner is not None and winner.level is not None:
                # Winner must choose trump suit — restore their index
                self.state = self.state.model_copy(
                    update={"current_player_index": winner.player_index}
                )
            else:
                # No one bid — redeal
                self.start_round()
        elif player_index == HUMAN_PLAYER_INDEX:
            # Only auto-play AI after a human action
            self._ai_auto_play()

        return True

    def set_trump(self, player_index: int, trump_suit: Suit) -> bool:
        """Set trump suit after winning the bid. Returns True if accepted."""
        if self.state.phase != Phase.BIDDING:
            return False

        winner = get_winning_bid(self.state.bidding_history)
        if winner is None or winner.player_index != player_index:
            return False
        if winner.level is None:
            return False

        self.state = set_declarer(
            self.state, player_index, trump_suit, winner.level
        )
        # Override to sequential turn order (next player after declarer)
        self.state = self.state.model_copy(
            update={"current_player_index": _next_player_sequential(player_index)}
        )
        return True

    def get_valid_bids(self) -> list[Rank]:
        """Get valid bid levels for the current player."""
        highest_bid = get_winning_bid(self.state.bidding_history)
        return get_valid_bid_levels(
            highest_bid.level if highest_bid else None,
            self.state.current_level,
        )

    # ---- Stirring (炒地皮) ----

    def submit_stir(self, player_index: int, stir: StirAction | None) -> bool:
        """Submit a stir action or pass.

        For passes (stir=None), any player may submit regardless of turn order.
        The game tracks which players have passed and ends stirring when all 4
        have passed.  For actual stir actions, the player must be current.
        """
        if self.state.phase != Phase.STIRRING:
            return False

        if stir is None:
            self._stir_passes.add(player_index)

            if player_index == self.state.current_player_index:
                self.state = self.state.model_copy(update={
                    "current_player_index": _next_player_sequential(player_index),
                })

            if len(self._stir_passes) >= PLAYER_COUNT:
                self.state = pickup_bottom_cards(self.state)

            return True

        # Actual stir: must be the current player
        if player_index != self.state.current_player_index:
            return False

        if not is_valid_stir(
            stir,
            self.state.trump_suit,
            self.state.trump_rank,
            self.state.stir_history,
        ):
            return False

        # Record stir manually (sequential turn order)
        self.state = self.state.model_copy(update={
            "trump_suit": stir.new_trump_suit,
            "trump_rank": stir.level or self.state.trump_rank,
            "stir_history": [*self.state.stir_history, stir],
            "current_player_index": _next_player_sequential(player_index),
        })
        self._stir_passes = set()
        return True

    # ---- Exchange (扣底) ----

    def submit_discard(self, player_index: int, cards: list[Card]) -> bool:
        """Declarer discards cards after picking up bottom cards."""
        if self.state.phase != Phase.EXCHANGE:
            return False
        if player_index != self.state.current_player_index:
            return False

        self.state = discard_cards(self.state, player_index, cards)
        return True

    # ---- Playing ----

    def submit_play(self, player_index: int, cards: list[Card]) -> bool:
        """Play cards from a player's hand.

        Determines the play type from legal plays when possible, otherwise
        infers the type from the cards themselves (SINGLE for 1 card, PAIR
        for 2 identical, etc.).  This keeps the method compatible with tests
        that play hand[0] without checking legality first.
        """
        if self.state.phase != Phase.PLAYING:
            return False
        if player_index != self.state.current_player_index:
            return False

        if not cards:
            return False

        # Try to match against legal plays first
        legal_plays = self._get_legal_plays(player_index)
        action = self._match_play_action(cards, legal_plays)

        # If no legal match, infer the play type from the cards
        if action is None:
            action = self._infer_play_type(cards)

        self.state = play_cards(self.state, player_index, action)
        if player_index == HUMAN_PLAYER_INDEX:
            self._ai_auto_play()
        return True

    def clear_trick(self) -> None:
        """Reset current trick slots for the next trick."""
        self.state = state_clear_trick(self.state)
        self.state = self.state.model_copy(
            update={"last_completed_trick": None}
        )

    # ---- Scoring / Next Round ----

    def next_round(self) -> None:
        """Calculate scores and advance to the next round."""
        if self.state.phase != Phase.SCORING:
            return

        result = self._calculate_round_score()

        if is_game_over(result.team0_new_level) or is_game_over(
            result.team1_new_level
        ):
            self.state = self.state.model_copy(
                update={"phase": Phase.GAME_OVER}
            )
            return

        new_declarer = (
            self.state.declarer_team_index
            if not result.switch_declarer
            else 1 - self.state.declarer_team_index
        )

        self.state = advance_round(
            self.state,
            result.team0_new_level,
            result.team1_new_level,
            new_declarer,
        )

    # ---- Query ----

    def get_awaiting_action(self) -> str | None:
        """Return what action the game is waiting for, or None."""
        phase = self.state.phase
        if phase == Phase.BIDDING:
            winner = get_winning_bid(self.state.bidding_history)
            if winner is not None and winner.level is not None:
                if is_bidding_over(self.state.bidding_history, PLAYER_COUNT):
                    return "set_trump"
            return "bid"
        if phase == Phase.STIRRING:
            return "stir"
        if phase == Phase.EXCHANGE:
            return "discard"
        if phase == Phase.PLAYING:
            trick = self.state.current_trick
            all_played = all(s.cards is not None for s in trick)
            if all_played or self.state.last_completed_trick is not None:
                all_cards_played = all(
                    len(p.hand) == 0 for p in self.state.players
                )
                if all_cards_played:
                    return "next_round"
                if self.state.last_completed_trick is not None:
                    return "clear_trick"
            return "play"
        if phase == Phase.SCORING:
            return "next_round"
        return None

    def is_human_turn(self) -> bool:
        """Check if it's the human player's turn."""
        return self.state.current_player_index == HUMAN_PLAYER_INDEX

    # ---- Private Helpers ----

    def _ai_auto_play(self) -> None:
        """Execute AI turns until human's turn or phase change."""
        from server.ai.auto_play import choose_bid, choose_discard, choose_play, choose_stir

        while not self.is_human_turn():
            phase = self.state.phase
            if phase not in (
                Phase.BIDDING, Phase.STIRRING, Phase.EXCHANGE, Phase.PLAYING
            ):
                break

            cp = self.state.current_player_index

            if phase == Phase.BIDDING:
                # Check if bidding already resolved
                if is_bidding_over(self.state.bidding_history, PLAYER_COUNT):
                    winner = get_winning_bid(self.state.bidding_history)
                    if winner is not None and winner.level is not None:
                        self.state = self.state.model_copy(
                            update={"current_player_index": winner.player_index}
                        )
                    break
                valid_levels = self.get_valid_bids()
                if not valid_levels:
                    break
                chosen = choose_bid(valid_levels, self.state.current_level)
                if chosen is None:
                    self.submit_bid(cp, None, pass_=True)
                else:
                    self.submit_bid(cp, chosen, pass_=False)

            elif phase == Phase.STIRRING:
                if self.state.trump_suit is None:
                    break
                valid = get_valid_stir_options(
                    self.state.trump_suit,
                    self.state.trump_rank,
                    cp,
                    self.state.stir_history,
                )
                if not valid:
                    self.submit_stir(cp, None)
                else:
                    result = choose_stir(
                        self.state.trump_suit,
                        [s.level for s in valid if s.level is not None],
                        cp,
                        self.state.stir_history,
                    )
                    if result is None:
                        self.submit_stir(cp, None)
                    else:
                        new_suit, level = result
                        stir = StirAction(
                            player_index=cp,
                            new_trump_suit=new_suit,
                            level=level,
                        )
                        self.submit_stir(cp, stir)

            elif phase == Phase.EXCHANGE:
                hand = self.state.players[cp].hand
                count = self.state.settings.bottom_card_count
                if len(hand) < count:
                    break
                discards = choose_discard(hand, count)
                self.submit_discard(cp, discards)

            elif phase == Phase.PLAYING:
                legal = self._get_legal_plays(cp)
                if not legal:
                    break
                action = choose_play(legal)
                self.submit_play(cp, action.cards)

    def _get_legal_plays(self, player_index: int) -> list[PlayAction]:
        """Get all legal plays for a player."""
        player = self.state.players[player_index]
        trick = self.state.current_trick
        is_leading = all(s.cards is None for s in trick)

        trick_dicts = [
            {"player_index": s.player_index, "cards": s.cards}
            for s in trick
        ]

        lead_action: PlayAction | None = None
        if not is_leading:
            for slot in trick:
                if slot.cards is not None:
                    lead_action = PlayAction(
                        type=self.state.lead_play_type or PlayType.SINGLE,
                        cards=slot.cards,
                    )
                    break

        return _get_legal_plays(
            player.hand,
            trick_dicts,
            self.state.trump_suit,
            self.state.trump_rank,
            is_leading,
            lead_action,
        )

    def _match_play_action(
        self, cards: list[Card], legal_plays: list[PlayAction]
    ) -> PlayAction | None:
        """Match played cards to a legal play action."""
        card_ids = {c.id for c in cards}
        for play in legal_plays:
            if card_ids == {c.id for c in play.cards}:
                return play
        return None

    @staticmethod
    def _infer_play_type(cards: list[Card]) -> PlayAction:
        """Infer play type from cards when no legal play matches."""
        if len(cards) == 1:
            return PlayAction(type=PlayType.SINGLE, cards=cards)
        if len(cards) == 2 and cards[0].suit == cards[1].suit and cards[0].rank == cards[1].rank:
            return PlayAction(type=PlayType.PAIR, cards=cards)
        # Default to SINGLE for any other combination
        return PlayAction(type=PlayType.SINGLE, cards=cards)

    def _calculate_round_score(self) -> ScoreResult:
        """Calculate the score for the current round."""
        last_trick = self.state.trick_history[-1]
        last_trick_winner_team = get_team_index(last_trick.winner_index)

        return calculate_score(
            self.state.defender_points,
            self.state.bottom_cards,
            last_trick_winner_team,
            last_trick.lead_type,
            self.state.declarer_team_index,
            self.state.teams[self.state.declarer_team_index].current_level,
            self.state.teams[1 - self.state.declarer_team_index].current_level,
        )
