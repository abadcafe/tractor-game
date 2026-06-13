"""Player abstraction for the Tractor game.

Defines the Player ABC, AutoPlayer (random AI), HumanPlayer (WebSocket-driven),
and the GameView Protocol that describes the Game interface players rely on.

Game.act() dispatches player action types (from server.actions) to the
appropriate sm state machine operations.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Protocol

from fastapi import WebSocket, WebSocketDisconnect

from server.actions import (
    BidAction,
    DiscardAction,
    NextRoundAction,
    PlayAction,
    SkipBidAction,
    SkipStirAction,
    StirAction,
)
from server.sm.card_model import Card, Suit
from server.sm.comparator import bid_value
from server.snapshot import StateSnapshot

logger = logging.getLogger(__name__)


# ---- GameView Protocol (DIP: defined at the consumer, not the implementer) ----


class GameView(Protocol):
    """Protocol describing the Game interface that Player subclasses rely on.

    Players call game.snapshot(index) to read state and
    game.act(index, action) to submit actions. No other
    Game methods are used from on_state() or its helpers.

    Game structurally satisfies this Protocol; no explicit
    inheritance declaration is needed.
    """

    def snapshot(self, for_player: int) -> StateSnapshot: ...
    async def act(
        self,
        player_index: int,
        action: BidAction | SkipBidAction | StirAction | SkipStirAction | DiscardAction | PlayAction | NextRoundAction,
    ) -> None: ...


# ---- Player ABC ----


class Player(ABC):
    """Abstract base class for game players.

    The game engine pushes state to each player via on_state(game).
    Subclasses must NOT call game.act() directly from on_state; all
    actions must be submitted via asyncio.create_task(game.act(...))
    so that state transitions remain serialized.
    """

    def __init__(self, index: int) -> None:
        self.index = index

    @abstractmethod
    async def on_state(self, game: GameView, *, seq: int = 0, error: str | None = None) -> None:
        """Called by Game when it pushes state to this player.

        Args:
            game: A GameView providing snapshot() and act().
                  Call game.snapshot(self.index) to get the player's
                  view of the state, and game.act(self.index, action)
                  to submit an action.
            seq: State sequence number for this push.
            error: Error message to include (only for the acting player).
        """

    async def send_error(self, message: str) -> None:
        """Send an error message to this player.

        AutoPlayer ignores errors; HumanPlayer forwards them via WebSocket.
        """
        pass


# ---- AutoPlayer ----


class AutoPlayer(Player):
    """AI player that makes random legal decisions.

    Uses create_task for actions to avoid blocking the on_state call chain,
    which prevents race conditions when _push_state_to_all iterates over players.
    A small delay is added before each action to prevent rapid cascading.
    """

    def __init__(self, index: int) -> None:
        super().__init__(index)
        self._action_count = 0

    async def on_state(self, game: GameView, *, seq: int = 0, error: str | None = None) -> None:
        snapshot = game.snapshot(self.index)

        if snapshot.phase == "DEAL_BID":
            await self._handle_deal_bid(snapshot, game)
        elif snapshot.phase == "STIRRING" and snapshot.awaiting_action == "stir":
            await self._handle_stir(snapshot, game)
        elif snapshot.phase == "EXCHANGE" and snapshot.awaiting_action == "discard":
            await self._handle_discard(snapshot, game)
        elif snapshot.phase == "PLAYING" and snapshot.awaiting_action == "play":
            await self._handle_play(snapshot, game)
        elif snapshot.phase == "COMPLETE" and snapshot.awaiting_action == "next_round":
            await self._handle_next_round(snapshot, game)

    async def _handle_deal_bid(self, snapshot: StateSnapshot, game: GameView) -> None:
        """Bid during DEAL_BID phase if we have a competitive bid.

        Only bids if: (a) no bid_winner yet, or (b) we can beat the
        current winner's priority.  Prefers pairs over singles.
        A 50% random factor prevents always-bidding determinism.
        """
        hand = snapshot.player_hand
        trump_rank = snapshot.trump_rank

        # 50% chance to even consider bidding (prevents deterministic always-bid)
        if random.random() >= 0.5:
            return

        # Group trump-rank cards by suit
        suit_groups: dict[Suit, list[Card]] = {}
        jokers: list[Card] = []
        for c in hand:
            if getattr(c, "rank", None) != trump_rank and not getattr(c, "is_joker", False):
                continue
            if getattr(c, "is_joker", False):
                jokers.append(c)
            else:
                suit_groups.setdefault(c.suit, []).append(c)

        # Best possible bid: prefer joker pair > trump-rank pair > trump-rank single
        best_cards: list[Card] = []

        # Check joker pairs (requires 2 of same rank)
        sj = [c for c in jokers if getattr(c, "rank", None) == "SJ"]
        bj = [c for c in jokers if getattr(c, "rank", None) == "BJ"]
        if len(bj) >= 2:
            best_cards = bj[:2]
        elif len(sj) >= 2:
            best_cards = sj[:2]

        # Check trump-rank pairs (any suit)
        if not best_cards:
            for cards in suit_groups.values():
                if len(cards) >= 2:
                    best_cards = cards[:2]
                    break

        # Fall back to single trump-rank card
        if not best_cards:
            for cards in suit_groups.values():
                if cards:
                    best_cards = [cards[0]]
                    break

        if not best_cards:
            return

        # Check if existing bid_winner has higher or equal priority
        from server.sm.card_model import Rank
        trump_rank_enum = Rank(trump_rank)
        best_priority = bid_value(best_cards, trump_rank_enum)
        bid_winner = snapshot.bid_winner
        if bid_winner is not None:
            winner_priority = bid_value(bid_winner.cards, trump_rank_enum)
            if best_priority <= winner_priority:
                return  # can't beat current winner

        action = BidAction(cards=best_cards, count=len(best_cards))
        asyncio.create_task(game.act(self.index, action))

    async def _handle_stir(self, snapshot: StateSnapshot, game: GameView) -> None:
        """Act during STIRRING phase."""
        if snapshot.current_player != self.index:
            return
        hand = snapshot.player_hand
        trump_rank = snapshot.trump_rank
        current_trump_suit = snapshot.trump_suit
        # Find pairs of trump rank cards that can beat current trump
        trump_cards = [c for c in hand if getattr(c, "rank", None) == trump_rank]
        if len(trump_cards) >= 2 and random.random() < 0.5:
            suit_groups: dict[Suit, list[Card]] = {}
            for c in trump_cards:
                suit_groups.setdefault(c.suit, []).append(c)
            valid_pair = None
            for suit, suit_cards in suit_groups.items():
                if len(suit_cards) >= 2:
                    # Skip same-suit as current trump (can't stir to same suit)
                    if current_trump_suit is not None and suit == current_trump_suit:
                        continue
                    valid_pair = suit_cards[:2]
                    break
            if valid_pair:
                action = StirAction(cards=valid_pair)
            else:
                action = SkipStirAction()
        else:
            action = SkipStirAction()
        asyncio.create_task(game.act(self.index, action))

    async def _handle_discard(self, snapshot: StateSnapshot, game: GameView) -> None:
        """Randomly discard cards during EXCHANGE phase."""
        if snapshot.current_player != self.index:
            return
        hand = snapshot.player_hand
        exc = snapshot.exchange_state
        count = exc.count if exc is not None else 8
        if len(hand) >= count and count > 0:
            cards = random.sample(hand, count)
        elif len(hand) > 0:
            cards = list(hand)
        else:
            return
        action = DiscardAction(cards=cards)
        asyncio.create_task(game.act(self.index, action))

    async def _handle_play(self, snapshot: StateSnapshot, game: GameView) -> None:
        """Pick a random legal play."""
        if snapshot.current_player != self.index:
            return
        legal = snapshot.legal_actions
        if not legal:
            logger.warning("AutoPlayer %d: no legal actions in PLAYING phase!", self.index)
            return
        chosen = random.choice(legal)
        # chosen is a list[Card] (plain card list from get_legal_plays)
        action = PlayAction(cards=chosen)
        self._action_count += 1
        if self._action_count % 50 == 0:
            logger.info("AutoPlayer %d: action #%d in PLAYING", self.index, self._action_count)
        asyncio.create_task(game.act(self.index, action))

    async def _handle_next_round(self, snapshot: StateSnapshot, game: GameView) -> None:
        """Submit NextRoundAction."""
        action = NextRoundAction()
        asyncio.create_task(game.act(self.index, action))


# ---- HumanPlayer ----


class HumanPlayer(Player):
    """Human player that communicates via WebSocket."""

    def __init__(self, index: int, ws: WebSocket | None = None) -> None:
        super().__init__(index)
        self._ws = ws

    async def on_state(self, game: GameView, *, seq: int = 0, error: str | None = None) -> None:
        """Push state to the human player via WebSocket.

        Catches any exception from send_json (e.g. WebSocket disconnected,
        websockets library AssertionError) to prevent a broken connection
        from crashing the entire _push_state_to_all() chain or the
        dealing loop. The human player will receive fresh state when
        they reconnect.
        """
        if self._ws is None:
            return
        snapshot = game.snapshot(self.index)
        try:
            msg: dict[str, object] = {
                "type": "state",
                "seq": seq,
                "awaiting": snapshot.awaiting_action,
                "state": snapshot.to_dict(),
            }
            if error is not None:
                msg["error"] = error
            await self._ws.send_json(msg)
        except (WebSocketDisconnect, OSError):
            logger.debug("Failed to push state to human player %d (WS likely disconnected)", self.index, exc_info=True)

    def set_ws(self, ws: WebSocket | None) -> None:
        """Replace the WebSocket reference."""
        self._ws = ws

    def clear_ws_if_current(self, ws: WebSocket) -> None:
        """Clear the WebSocket reference only if it still points to the given instance.

        Used in finally blocks to avoid clearing a connection that has already
        been replaced by a new one (connection takeover).
        """
        if self._ws is ws:
            self._ws = None

    def is_connected(self) -> bool:
        """Return True if this player has an active WebSocket connection."""
        return self._ws is not None

    async def send_error(self, message: str) -> None:
        """No-op: errors are now merged into state messages.

        Kept for GameView Protocol compatibility.
        """
        pass

    async def close_ws(self) -> None:
        """Close the WebSocket connection if active, then clear the reference."""
        if self._ws is not None:
            ws = self._ws
            self._ws = None
            try:
                await ws.close()
            except (WebSocketDisconnect, OSError):
                logger.debug("Failed to close WS for player %d (already disconnected)", self.index, exc_info=True)
