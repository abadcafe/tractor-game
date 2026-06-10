"""Player abstraction for the Tractor game.

Defines the Player ABC, AutoPlayer (random AI), HumanPlayer (WebSocket-driven),
and player-facing action dataclasses (BidAction, StirAction, PlayAction, etc.).

Game.act() dispatches these player action types to the appropriate sm
state machine operations.
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
        """Bid during DEAL_BID phase if we have a competitive bid.

        Only bids if: (a) no bid_winner yet, or (b) we can beat the
        current winner.  Prefers pairs over singles to avoid the noise
        of guaranteed-losing single-card bids after a pair has won.
        A 50% random factor prevents always-bidding determinism.
        """
        hand = snapshot.player_hand
        trump_rank = snapshot.trump_rank

        # 50% chance to even consider bidding (prevents deterministic always-bid)
        if random.random() >= 0.5:
            return

        # Group trump-rank cards by suit
        suit_groups: dict[Any, list] = {}
        jokers: list = []
        for c in hand:
            if getattr(c, "rank", None) != trump_rank and not getattr(c, "is_joker", False):
                continue
            if getattr(c, "is_joker", False):
                jokers.append(c)
            else:
                suit_groups.setdefault(c.suit, []).append(c)

        # Best possible bid: prefer joker pair > trump-rank pair > trump-rank single
        best_cards: list = []
        best_count = 0
        best_suit = None
        best_kind = ""  # "joker" or "trump_rank"

        # Check joker pairs (requires 2 of same rank)
        sj = [c for c in jokers if getattr(c, "rank", None) == "SJ"]
        bj = [c for c in jokers if getattr(c, "rank", None) == "BJ"]
        if len(bj) >= 2:
            best_cards = bj[:2]
            best_count = 2
            best_kind = "joker"
        elif len(sj) >= 2:
            best_cards = sj[:2]
            best_count = 2
            best_kind = "joker"

        # Check trump-rank pairs (any suit)
        if not best_cards:
            for suit, cards in suit_groups.items():
                if len(cards) >= 2:
                    best_cards = cards[:2]
                    best_count = 2
                    best_suit = suit
                    best_kind = "trump_rank"
                    break

        # Fall back to single trump-rank card
        if not best_cards:
            for suit, cards in suit_groups.items():
                if cards:
                    best_cards = [cards[0]]
                    best_count = 1
                    best_suit = suit
                    best_kind = "trump_rank"
                    break

        if not best_cards:
            return

        # Check if existing bid_winner would block us
        bid_winner = snapshot.bid_winner
        if bid_winner is not None:
            winner_count = getattr(bid_winner, "count", 0)
            winner_kind = getattr(bid_winner, "kind", "")
            # Joker pair always beats trump-rank pair; same-kind needs higher count
            if winner_kind == "joker" and best_kind != "joker":
                return  # can't beat joker pair with trump-rank
            if winner_kind == "joker" and best_kind == "joker":
                # Both joker pairs: must be bigger joker
                winner_joker = getattr(bid_winner, "joker_type", "")
                if winner_joker == "big":
                    return  # big joker pair is unbeatable
                # winner has small joker, we need big joker
                if len(bj) < 2:
                    return
            if winner_count >= best_count and winner_kind == best_kind:
                return  # same kind, count not higher

        action = BidAction(cards=best_cards, count=best_count)
        task = asyncio.create_task(game.act(self.index, action))
        task.add_done_callback(_log_task_exception)

    async def _handle_stir(self, snapshot: Any, game: Any) -> None:
        """Act during STIRRING phase."""
        if snapshot.current_player != self.index:
            return
        hand = snapshot.player_hand
        trump_rank = snapshot.trump_rank
        current_trump_suit = snapshot.trump_suit
        # Find pairs of trump rank cards that can beat current trump
        trump_cards = [c for c in hand if getattr(c, "rank", None) == trump_rank]
        if len(trump_cards) >= 2 and random.random() < 0.5:
            suit_groups: dict[Any, list] = {}
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
        task = asyncio.create_task(game.act(self.index, action))
        task.add_done_callback(_log_task_exception)

    async def _handle_discard(self, snapshot: Any, game: Any) -> None:
        """Randomly discard cards during EXCHANGE phase."""
        if snapshot.current_player != self.index:
            return
        hand = snapshot.player_hand
        # exchange_state is a dict in the snapshot; extract count
        exc = snapshot.exchange_state
        if exc is not None:
            count = exc.get("count", 8) if isinstance(exc, dict) else getattr(exc, "count", 8)
        else:
            count = 8
        if len(hand) >= count and count > 0:
            cards = random.sample(hand, count)
        elif len(hand) > 0:
            cards = list(hand)
        else:
            return
        action = DiscardAction(cards=cards)
        task = asyncio.create_task(game.act(self.index, action))
        task.add_done_callback(_log_task_exception)

    async def _handle_play(self, snapshot: Any, game: Any) -> None:
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

    async def send_error(self, message: str) -> None:
        """Send an error message to the human player via WebSocket."""
        if self._ws is None:
            return
        await self._ws.send_json({
            "type": "error",
            "message": message,
        })

    async def close_ws(self) -> None:
        """Close the WebSocket connection if active, then clear the reference."""
        if self._ws is not None:
            ws = self._ws
            self._ws = None
            try:
                await ws.close()
            except Exception:
                pass
