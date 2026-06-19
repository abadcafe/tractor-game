"""Automatic player implementation."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from itertools import combinations

from server.actions import (
    BidAction,
    DiscardAction,
    NextRoundAction,
    PlayAction,
    SkipBidAction,
    SkipStirAction,
    StirAction,
)
from server.messages import PlayerMessage, StateMessage
from server.player.base import GameView, Player
from server.sm.card_model import Card, Rank, Suit
from server.snapshot import StateSnapshot

logger = logging.getLogger(__name__)

type PlayerAction = BidAction | SkipBidAction | StirAction | SkipStirAction | DiscardAction | PlayAction | NextRoundAction


@dataclass(frozen=True)
class TrickSlotDecisionKey:
    """Card ids visible in one trick slot for AutoPlayer retry suppression."""

    player: int
    card_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrickDecisionKey:
    """Trick fields that can change whether a failed card action is worth retrying."""

    lead_player: int
    current_player: int
    slots: tuple[TrickSlotDecisionKey, ...]


@dataclass(frozen=True)
class AutoDecisionKey:
    """Player-facing state identity for suppressing repeated failed card actions."""

    phase: str
    awaiting_action: str | None
    hand_card_ids: tuple[str, ...]
    trick: TrickDecisionKey | None
    bid_winner_card_ids: tuple[str, ...]
    action_hint_card_ids: tuple[tuple[str, ...], ...]


class AutoPlayer(Player):
    """Built-in client-like player driven only by player-facing snapshots.

    Uses create_task for actions to avoid blocking the on_state call chain,
    which prevents race conditions when _push_state_to_all iterates over players.
    """

    def __init__(self, index: int) -> None:
        super().__init__(index)
        self._action_count = 0
        self._failed_card_actions: dict[AutoDecisionKey, set[tuple[str, ...]]] = {}
        self._last_attempt_key: AutoDecisionKey | None = None
        self._last_attempt_cards: tuple[str, ...] | None = None

    async def run(self, game: GameView) -> None:
        """Start this player by requesting current state with seq=0.

        This is the player-driven entry point: instead of Game pushing initial
        state, each player independently asks for the current state. The
        returned StateMessage drives all further actions.
        """
        await game.receive(self.index, PlayerMessage(seq=0, raw={}))

    async def on_state(self, game: GameView, message: StateMessage) -> None:
        snapshot = message.state
        seq = message.seq
        error = message.error
        key = self._decision_key(snapshot)
        if (
            error is not None
            and self._is_card_action_rejection(error)
            and self._last_attempt_key == key
            and self._last_attempt_cards is not None
        ):
            self._failed_card_actions.setdefault(key, set()).add(self._last_attempt_cards)
        elif error is None and self._last_attempt_key != key:
            self._last_attempt_key = None
            self._last_attempt_cards = None

        if snapshot.phase == "DEAL_BID" and snapshot.awaiting_action == "bid":
            await self._handle_deal_bid(snapshot, game, seq=seq)
        elif snapshot.phase == "STIRRING" and snapshot.awaiting_action == "stir":
            await self._handle_stir(snapshot, game, seq=seq)
        elif snapshot.phase == "STIRRING" and snapshot.awaiting_action == "discard":
            await self._handle_discard(snapshot, game, seq=seq)
        elif snapshot.phase == "PLAYING" and snapshot.awaiting_action == "play":
            await self._handle_play(snapshot, game, seq=seq)
        elif snapshot.phase == "WAITING" and snapshot.awaiting_action == "next_round":
            await self._handle_next_round(snapshot, game, seq=seq)

    async def _handle_deal_bid(self, snapshot: StateSnapshot, game: GameView, *, seq: int) -> None:
        """Bid only from player-facing action_hints; otherwise pass."""
        candidates = self._hint_candidates(snapshot)
        if not candidates:
            self._submit_action(game, seq, SkipBidAction(), snapshot)
            return
        chosen = self._prefer_longer_candidate(candidates)
        self._submit_action(game, seq, BidAction(cards=chosen, count=len(chosen)), snapshot)

    async def _handle_stir(self, snapshot: StateSnapshot, game: GameView, *, seq: int) -> None:
        """Optionally stir using only player-facing action_hints."""
        candidates = self._hint_candidates(snapshot)
        if candidates and random.random() < 0.5:
            chosen = random.choice(candidates)
            self._submit_action(game, seq, StirAction(cards=chosen), snapshot)
            return
        self._submit_action(game, seq, SkipStirAction(), snapshot)

    async def _handle_discard(self, snapshot: StateSnapshot, game: GameView, *, seq: int) -> None:
        """Randomly discard cards during STIRRING EXCHANGING sub-phase."""
        hand = snapshot.player_hand
        stir = snapshot.stirring_state
        count = stir.exchange_count if stir is not None and stir.exchange_count is not None else 8
        if len(hand) >= count and count > 0:
            cards = random.sample(hand, count)
        elif len(hand) > 0:
            cards = list(hand)
        else:
            return
        action = DiscardAction(cards=cards)
        self._submit_action(game, seq, action, snapshot)

    async def _handle_play(self, snapshot: StateSnapshot, game: GameView, *, seq: int) -> None:
        """Pick a play using only the same snapshot information a human sees."""
        candidates = self._hint_candidates(snapshot) or self._play_candidates_from_snapshot(snapshot)
        if not candidates:
            logger.warning("AutoPlayer %d: no fallback play found", self.index)
            return
        chosen = random.choice(candidates)
        action = PlayAction(cards=chosen)
        self._action_count += 1
        if self._action_count % 50 == 0:
            logger.info("AutoPlayer %d: action #%d in PLAYING", self.index, self._action_count)
        self._submit_action(game, seq, action, snapshot)

    def _hint_candidates(self, snapshot: StateSnapshot) -> list[list[Card]]:
        return self._filter_failed_candidates(snapshot, [list(cards) for cards in snapshot.action_hints])

    def _play_candidates_from_snapshot(self, snapshot: StateSnapshot) -> list[list[Card]]:
        hand = list(snapshot.player_hand)
        if not hand:
            return []
        if snapshot.trick is None or not snapshot.trick.slots:
            return self._filter_failed_candidates(snapshot, [[card] for card in hand])

        lead_slot = snapshot.trick.slots[snapshot.trick.lead_player]
        lead_cards = list(lead_slot.cards)
        if not lead_cards:
            return self._filter_failed_candidates(snapshot, [[card] for card in hand])

        lead_count = len(lead_cards)
        if len(hand) < lead_count:
            return []

        lead_eff = self._effective_suit(lead_cards[0], snapshot.trump_suit, snapshot.trump_rank)
        same_eff_cards = [
            card for card in hand
            if self._effective_suit(card, snapshot.trump_suit, snapshot.trump_rank) == lead_eff
        ]
        other_cards = [
            card for card in hand
            if self._effective_suit(card, snapshot.trump_suit, snapshot.trump_rank) != lead_eff
        ]

        candidates: list[list[Card]] = []
        if len(same_eff_cards) >= lead_count:
            candidates.extend(self._candidate_combinations(same_eff_cards, lead_count))
        else:
            needed = lead_count - len(same_eff_cards)
            candidates.extend(
                same_eff_cards + fill
                for fill in self._candidate_combinations(other_cards, needed)
            )

        return self._filter_failed_candidates(snapshot, candidates)

    def _filter_failed_candidates(self, snapshot: StateSnapshot, candidates: list[list[Card]]) -> list[list[Card]]:
        failed = self._failed_card_actions.get(self._decision_key(snapshot), set())
        result: list[list[Card]] = []
        seen: set[tuple[str, ...]] = set()
        for candidate in candidates:
            key = self._cards_key(candidate)
            if key in failed or key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        return result

    @staticmethod
    def _prefer_longer_candidate(candidates: list[list[Card]]) -> list[Card]:
        return max(candidates, key=len)

    @staticmethod
    def _candidate_combinations(cards: list[Card], count: int, limit: int = 40) -> list[list[Card]]:
        if count <= 0:
            return [[]]
        if len(cards) < count:
            return []
        combos = [list(combo) for combo in combinations(cards, count)]
        combos.sort(key=AutoPlayer._combo_sort_key)
        return combos[:limit]

    @staticmethod
    def _combo_sort_key(cards: list[Card]) -> tuple[int, int]:
        rank_counts: dict[Rank, int] = {}
        for card in cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        pair_like = sum(count // 2 for count in rank_counts.values())
        point_cards = sum(1 for card in cards if card.rank in (Rank.FIVE, Rank.TEN, Rank.KING))
        return (-pair_like, point_cards)

    @staticmethod
    def _effective_suit(card: Card, trump_suit: Suit | None, trump_rank: Rank) -> str:
        if card.is_joker or card.rank == trump_rank or (trump_suit is not None and card.suit == trump_suit):
            return "trump"
        return card.suit.value

    def _submit_action(
        self,
        game: GameView,
        seq: int,
        action: PlayerAction,
        snapshot: StateSnapshot,
    ) -> None:
        if isinstance(action, BidAction | StirAction | DiscardAction | PlayAction) and action.cards:
            self._last_attempt_key = self._decision_key(snapshot)
            self._last_attempt_cards = self._cards_key(action.cards)
        else:
            self._last_attempt_key = None
            self._last_attempt_cards = None
        asyncio.create_task(game.receive(self.index, self._message_from_action(seq, action)))

    @staticmethod
    def _message_from_action(seq: int, action: PlayerAction) -> PlayerMessage:
        if isinstance(action, BidAction):
            return PlayerMessage(
                seq=seq,
                raw={"type": "bid", "cards": [card.id for card in action.cards]},
            )
        if isinstance(action, SkipBidAction):
            return PlayerMessage(seq=seq, raw={"type": "bid", "pass": True})
        if isinstance(action, StirAction):
            return PlayerMessage(
                seq=seq,
                raw={"type": "stir", "cards": [card.id for card in action.cards]},
            )
        if isinstance(action, SkipStirAction):
            return PlayerMessage(seq=seq, raw={"type": "stir", "pass": True})
        if isinstance(action, DiscardAction):
            return PlayerMessage(
                seq=seq,
                raw={"type": "discard", "cards": [card.id for card in action.cards]},
            )
        if isinstance(action, PlayAction):
            return PlayerMessage(
                seq=seq,
                raw={"type": "play", "cards": [card.id for card in action.cards]},
            )
        return PlayerMessage(seq=seq, raw={"type": "next_round"})

    def _decision_key(self, snapshot: StateSnapshot) -> AutoDecisionKey:
        trick_key: TrickDecisionKey | None = None
        if snapshot.trick is not None:
            trick_key = TrickDecisionKey(
                lead_player=snapshot.trick.lead_player,
                current_player=snapshot.trick.current_player,
                slots=tuple(
                    TrickSlotDecisionKey(
                        player=slot.player,
                        card_ids=tuple(card.id for card in slot.cards),
                    )
                    for slot in snapshot.trick.slots
                ),
            )
        bid_winner_cards: tuple[str, ...] = ()
        if snapshot.bid_winner is not None:
            bid_winner_cards = tuple(card.id for card in snapshot.bid_winner.cards)
        return AutoDecisionKey(
            phase=snapshot.phase,
            awaiting_action=snapshot.awaiting_action,
            hand_card_ids=tuple(card.id for card in snapshot.player_hand),
            trick=trick_key,
            bid_winner_card_ids=bid_winner_cards,
            action_hint_card_ids=tuple(tuple(card.id for card in hint) for hint in snapshot.action_hints),
        )

    @staticmethod
    def _cards_key(cards: list[Card]) -> tuple[str, ...]:
        return tuple(sorted(card.id for card in cards))

    @staticmethod
    def _is_card_action_rejection(error: str) -> bool:
        """Return whether an error proves the last card choice was invalid."""
        return error != ""

    async def _handle_next_round(self, snapshot: StateSnapshot, game: GameView, *, seq: int) -> None:
        """Submit NextRoundAction."""
        action = NextRoundAction()
        self._submit_action(game, seq, action, snapshot)
