"""Automatic player implementation."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from server.player.base import GameView, Player
from server.protocol import (
    AwaitingAction,
    PlayerMessage,
    RoundPhase,
    StateMessage,
    StateSnapshot,
)
from server.result import Ok
from server.rules.cards import Card, Rank, Suit
from server.rules.follow import is_legal_follow
from server.rules.hints import (
    get_legal_play_hints,
    sort_play_action_hints,
)

logger = logging.getLogger(__name__)

type CardActionType = Literal["bid", "stir", "discard", "play"]
MAX_AUTO_PLAY_CANDIDATES: int = 40
MAX_AUTO_COMBINATIONS_SCANNED: int = 5000


@dataclass(frozen=True)
class TrickSlotDecisionKey:
    """
    Card ids visible in one trick slot for AutoPlayer retry suppression.
    """

    player: int
    card_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrickDecisionKey:
    """
    Trick fields that can change whether a failed card action is worth
    retrying.
    """

    lead_player: int
    current_player: int
    slots: tuple[TrickSlotDecisionKey, ...]


@dataclass(frozen=True)
class AutoDecisionKey:
    """
    Player-facing state identity for suppressing repeated failed card
    actions.
    """

    phase: RoundPhase
    awaiting_action: AwaitingAction | None
    hand_card_ids: tuple[str, ...]
    trick: TrickDecisionKey | None
    bid_winner_card_ids: tuple[str, ...]
    action_hint_card_ids: tuple[tuple[str, ...], ...]


class AutoPlayer(Player):
    """
    Built-in client-like player driven only by player-facing snapshots.

    Uses create_task for actions to avoid blocking the on_state call
    chain,
    which prevents race conditions when _push_state_to_all iterates over
    players.
    """

    def __init__(self, index: int) -> None:
        super().__init__(index)
        self._action_count = 0
        self._failed_card_actions: dict[
            AutoDecisionKey, set[tuple[str, ...]]
        ] = {}
        self._last_attempt_key: AutoDecisionKey | None = None
        self._last_attempt_cards: tuple[str, ...] | None = None

    async def run(self, game: GameView) -> None:
        """Start this player by requesting current state with seq=0.

        This is the player-driven entry point: instead of Game pushing
        initial
        state, each player independently asks for the current state. The
        returned StateMessage drives all further actions.
        """
        await game.receive(self.index, PlayerMessage(seq=0, raw={}))

    async def on_state(
        self, game: GameView, message: StateMessage
    ) -> None:
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
            self._failed_card_actions.setdefault(key, set()).add(
                self._last_attempt_cards
            )
        elif error is None and self._last_attempt_key != key:
            self._last_attempt_key = None
            self._last_attempt_cards = None

        if (
            snapshot.phase == "DEAL_BID"
            and snapshot.awaiting_action == "bid"
        ):
            await self._handle_deal_bid(snapshot, game, seq=seq)
        elif (
            snapshot.phase == "STIRRING"
            and snapshot.awaiting_action == "stir"
        ):
            await self._handle_stir(snapshot, game, seq=seq)
        elif (
            snapshot.phase == "STIRRING"
            and snapshot.awaiting_action == "discard"
        ):
            await self._handle_discard(snapshot, game, seq=seq)
        elif (
            snapshot.phase == "PLAYING"
            and snapshot.awaiting_action == "play"
        ):
            await self._handle_play(snapshot, game, seq=seq)
        elif (
            snapshot.phase == "WAITING"
            and snapshot.awaiting_action == "next_round"
        ):
            await self._handle_next_round(snapshot, game, seq=seq)

    async def _handle_deal_bid(
        self, snapshot: StateSnapshot, game: GameView, *, seq: int
    ) -> None:
        """Bid only from player-facing action_hints; otherwise pass."""
        candidates = self._hint_candidates(snapshot)
        if not candidates:
            self._submit_message(
                game, seq, {"type": "bid", "pass": True}, snapshot, None
            )
            return
        chosen = candidates[0]
        self._submit_card_action(game, seq, "bid", chosen, snapshot)

    async def _handle_stir(
        self, snapshot: StateSnapshot, game: GameView, *, seq: int
    ) -> None:
        """Optionally stir using only player-facing action_hints."""
        candidates = self._hint_candidates(snapshot)
        if candidates and random.random() < 0.5:
            chosen = random.choice(candidates)
            self._submit_card_action(
                game, seq, "stir", chosen, snapshot
            )
            return
        self._submit_message(
            game, seq, {"type": "stir", "pass": True}, snapshot, None
        )

    async def _handle_discard(
        self, snapshot: StateSnapshot, game: GameView, *, seq: int
    ) -> None:
        """
        Randomly discard cards during STIRRING EXCHANGING sub-phase.
        """
        hand = snapshot.player_hand
        stir = snapshot.stirring_state
        count = (
            stir.exchange_count
            if stir is not None and stir.exchange_count is not None
            else 8
        )
        if len(hand) >= count and count > 0:
            cards = random.sample(hand, count)
        elif len(hand) > 0:
            cards = list(hand)
        else:
            return
        self._submit_card_action(game, seq, "discard", cards, snapshot)

    async def _handle_play(
        self, snapshot: StateSnapshot, game: GameView, *, seq: int
    ) -> None:
        """
        Pick a play using only the same snapshot information a human
        sees.
        """
        candidates = self._hint_candidates(
            snapshot
        ) or self._play_candidates_from_snapshot(snapshot)
        if not candidates:
            logger.warning(
                "AutoPlayer %d: no fallback play found", self.index
            )
            return
        chosen = random.choice(candidates)
        self._action_count += 1
        if self._action_count % 50 == 0:
            logger.info(
                "AutoPlayer %d: action #%d in PLAYING",
                self.index,
                self._action_count,
            )
        self._submit_card_action(game, seq, "play", chosen, snapshot)

    def _hint_candidates(
        self, snapshot: StateSnapshot
    ) -> list[list[Card]]:
        return self._filter_failed_candidates(
            snapshot, [list(cards) for cards in snapshot.action_hints]
        )

    def _play_candidates_from_snapshot(
        self, snapshot: StateSnapshot
    ) -> list[list[Card]]:
        hand = list(snapshot.player_hand)
        if not hand:
            return []
        lead_cards = _lead_cards(snapshot)
        if not lead_cards:
            return self._filter_failed_candidates(
                snapshot, [[card] for card in hand]
            )

        hints_result = get_legal_play_hints(
            hand,
            lead_cards,
            snapshot.trump_suit,
            snapshot.trump_rank,
            max_hints=MAX_AUTO_PLAY_CANDIDATES,
        )
        if isinstance(hints_result, Ok):
            return self._filter_failed_candidates(
                snapshot, hints_result.value
            )

        fallback_candidates = _bounded_legal_follow_candidates(
            hand,
            lead_cards,
            snapshot.trump_suit,
            snapshot.trump_rank,
        )
        return self._filter_failed_candidates(
            snapshot, fallback_candidates
        )

    def _filter_failed_candidates(
        self,
        snapshot: StateSnapshot,
        candidates: list[list[Card]],
    ) -> list[list[Card]]:
        failed = self._failed_card_actions.get(
            self._decision_key(snapshot), set()
        )
        result: list[list[Card]] = []
        seen: set[tuple[str, ...]] = set()
        for candidate in candidates:
            key = self._cards_key(candidate)
            if key in failed or key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        return result

    def _submit_card_action(
        self,
        game: GameView,
        seq: int,
        action_type: CardActionType,
        cards: list[Card],
        snapshot: StateSnapshot,
    ) -> None:
        self._submit_message(
            game,
            seq,
            {"type": action_type, "cards": [card.id for card in cards]},
            snapshot,
            self._cards_key(cards) if cards else None,
        )

    def _submit_message(
        self,
        game: GameView,
        seq: int,
        raw: dict[str, object],
        snapshot: StateSnapshot,
        attempted_cards: tuple[str, ...] | None,
    ) -> None:
        if attempted_cards is not None:
            self._last_attempt_key = self._decision_key(snapshot)
            self._last_attempt_cards = attempted_cards
        else:
            self._last_attempt_key = None
            self._last_attempt_cards = None
        asyncio.create_task(
            game.receive(self.index, PlayerMessage(seq=seq, raw=raw))
        )

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
            bid_winner_cards = tuple(
                card.id for card in snapshot.bid_winner.cards
            )
        return AutoDecisionKey(
            phase=snapshot.phase,
            awaiting_action=snapshot.awaiting_action,
            hand_card_ids=tuple(
                card.id for card in snapshot.player_hand
            ),
            trick=trick_key,
            bid_winner_card_ids=bid_winner_cards,
            action_hint_card_ids=tuple(
                tuple(card.id for card in hint)
                for hint in snapshot.action_hints
            ),
        )

    @staticmethod
    def _cards_key(cards: list[Card]) -> tuple[str, ...]:
        return tuple(sorted(card.id for card in cards))

    @staticmethod
    def _is_card_action_rejection(error: str) -> bool:
        """
        Return whether an error proves the last card choice was invalid.
        """
        return error != ""

    async def _handle_next_round(
        self, snapshot: StateSnapshot, game: GameView, *, seq: int
    ) -> None:
        """Submit NextRoundAction."""
        self._submit_message(
            game, seq, {"type": "next_round"}, snapshot, None
        )


def _lead_cards(snapshot: StateSnapshot) -> list[Card]:
    trick = snapshot.trick
    if trick is None:
        return []
    for slot in trick.slots:
        if slot.player == trick.lead_player:
            return list(slot.cards)
    return []


def _bounded_legal_follow_candidates(
    hand: list[Card],
    lead_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> list[list[Card]]:
    lead_count = len(lead_cards)
    if lead_count == 0 or len(hand) < lead_count:
        return []

    result: list[list[Card]] = []
    scanned = 0
    for combo in combinations(hand, lead_count):
        scanned += 1
        candidate = list(combo)
        if is_legal_follow(
            hand, candidate, lead_cards, trump_suit, trump_rank
        ):
            result.append(candidate)
            if len(result) >= MAX_AUTO_PLAY_CANDIDATES:
                break
        if scanned >= MAX_AUTO_COMBINATIONS_SCANNED:
            break
    return sort_play_action_hints(result, trump_suit, trump_rank)
