"""Player abstraction for the Tractor game.

Defines the Player ABC, AutoPlayer (random AI), HumanPlayer (WebSocket-driven),
and player-facing action dataclasses (BidAction, StirAction, PlayAction, etc.).

These PlayerAction types are distinct from the sm internal types
(BidEvent, StirAction, PlayAction in server/sm/types.py). Game.act()
converts between them.
"""

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _log_task_exception(task: asyncio.Task) -> None:
    """Log exceptions from fire-and-forget create_task calls."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.exception("Unhandled exception in player action task: %s", exc)


# ---- PlayerAction types ----


@dataclass
class BidAction:
    """Cards the player reveals during bidding."""

    cards: list
    count: int


@dataclass
class StirAction:
    """Pair of cards to stir with (change trump suit)."""

    cards: list


@dataclass
class SkipStirAction:
    """Pass during stirring."""

    pass


@dataclass
class DiscardAction:
    """Cards to discard for the bottom pile."""

    cards: list


@dataclass
class PlayAction:
    """Cards to play in the current trick."""

    cards: list


@dataclass
class NextRoundAction:
    """Signal to proceed to the next round."""

    pass


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
    async def on_state(self, game: Any) -> None:
        """Called by Game when it pushes state to this player.

        Args:
            game: The Game instance. Call game.snapshot(self.index) to
                  get the player's view of the state, and
                  game.act(self.index, action) to submit an action.
        """


# ---- AutoPlayer ----


class AutoPlayer(Player):
    """AI player that makes random legal decisions."""

    async def on_state(self, game: Any) -> None:
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

    async def _handle_deal_bid(self, snapshot: Any, game: Any) -> None:
        """Randomly decide whether to bid during DEAL_BID phase.

        During DEAL_BID, all players may bid regardless of current_player.
        """
        hand = snapshot.player_hand
        trump_rank = snapshot.trump_rank
        trump_cards = [c for c in hand if getattr(c, "rank", None) == trump_rank]
        if trump_cards and random.random() < 0.5:
            card = random.choice(trump_cards)
            action = BidAction(cards=[card], count=1)
            task = asyncio.create_task(game.act(self.index, action))
            task.add_done_callback(_log_task_exception)

    async def _handle_stir(self, snapshot: Any, game: Any) -> None:
        """Act during STIRRING phase."""
        if snapshot.current_player != self.index:
            return
        hand = snapshot.player_hand
        trump_rank = snapshot.trump_rank
        # Find pairs of trump rank cards (simplified: just pass if no cards)
        trump_cards = [c for c in hand if getattr(c, "rank", None) == trump_rank]
        if len(trump_cards) >= 2 and random.random() < 0.5:
            cards = trump_cards[:2]
            action = StirAction(cards=cards)
        else:
            action = SkipStirAction()
        task = asyncio.create_task(game.act(self.index, action))
        task.add_done_callback(_log_task_exception)

    async def _handle_discard(self, snapshot: Any, game: Any) -> None:
        """Randomly discard cards during EXCHANGE phase."""
        if snapshot.current_player != self.index:
            return
        hand = snapshot.player_hand
        # Default: discard up to 8 cards or the entire hand, whichever is smaller
        discard_count = min(len(hand), 8)
        if discard_count > 0:
            cards = random.sample(hand, discard_count)
        else:
            cards = []
        action = DiscardAction(cards=cards)
        task = asyncio.create_task(game.act(self.index, action))
        task.add_done_callback(_log_task_exception)

    async def _handle_play(self, snapshot: Any, game: Any) -> None:
        """Pick a random legal play."""
        if snapshot.current_player != self.index:
            return
        legal = snapshot.legal_actions
        if not legal:
            return
        chosen = random.choice(legal)
        # chosen is a sm.PlayAction Pydantic model with .cards attribute
        action = PlayAction(cards=chosen.cards)
        task = asyncio.create_task(game.act(self.index, action))
        task.add_done_callback(_log_task_exception)

    async def _handle_next_round(self, snapshot: Any, game: Any) -> None:
        """Submit NextRoundAction."""
        if snapshot.current_player != self.index:
            return
        action = NextRoundAction()
        task = asyncio.create_task(game.act(self.index, action))
        task.add_done_callback(_log_task_exception)


# ---- HumanPlayer ----


class HumanPlayer(Player):
    """Human player that communicates via WebSocket."""

    def __init__(self, index: int, ws: Any = None) -> None:
        super().__init__(index)
        self._ws = ws

    async def on_state(self, game: Any) -> None:
        """Push state to the human player via WebSocket."""
        if self._ws is None:
            return
        snapshot = game.snapshot(self.index)
        await self._ws.send_json({
            "type": "state",
            "awaiting": snapshot.awaiting_action,
            "state": snapshot.to_dict(),
        })

    def set_ws(self, ws: Any) -> None:
        """Replace the WebSocket reference."""
        self._ws = ws

    def is_connected(self) -> bool:
        """Return True if this player has an active WebSocket connection."""
        return self._ws is not None

    async def close_ws(self) -> None:
        """Close the WebSocket connection if active, then clear the reference."""
        if self._ws is not None:
            ws = self._ws
            self._ws = None
            try:
                await ws.close()
            except Exception:
                pass
