"""Player abstraction for the Tractor game.

Defines the Player ABC, AutoPlayer (random AI), HumanPlayer (WebSocket-driven),
and the GameView Protocol that describes the Game interface players rely on.

Both Player subclasses are self-contained:
- AutoPlayer: receives state push → internal decision → game.act(self.index, action)
- HumanPlayer: receives WS message → internal validation + parsing → game.act(self.index, action)

Game.act() dispatches player action types (from server.actions) to the
appropriate sm state machine operations.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Protocol, TypeGuard

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
from server.sm.result import Ok, Rejected, StateResult
from server.snapshot import StateSnapshot

logger = logging.getLogger(__name__)

# Type alias for the action union used in GameView.act()
PlayerAction = BidAction | SkipBidAction | StirAction | SkipStirAction | DiscardAction | PlayAction | NextRoundAction


# ---- GameView Protocol (DIP: defined at the consumer, not the implementer) ----


class GameView(Protocol):
    """Protocol describing the Game interface that Player subclasses rely on.

    Players call game.snapshot(index) to read state and
    game.act(index, action) to submit actions.

    HumanPlayer additionally uses resolve_cards(), current_seq, and is_over()
    for its WS protocol handling (seq validation, action parsing, game-over check).

    Game structurally satisfies this Protocol; no explicit
    inheritance declaration is needed.
    """

    def snapshot(self, for_player: int) -> StateSnapshot: ...
    async def act(
        self,
        player_index: int,
        seq: int,
        action: PlayerAction,
    ) -> None: ...
    def resolve_cards(self, player_index: int, card_ids: list[str]) -> Ok[list[Card]] | Rejected: ...
    @property
    def current_seq(self) -> int: ...
    def is_over(self) -> bool: ...


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
            error: Error message (unicast to acting player only).
        """


# ---- AutoPlayer ----


class AutoPlayer(Player):
    """AI player that makes random legal decisions.

    Uses create_task for actions to avoid blocking the on_state call chain,
    which prevents race conditions when _push_state_to_all iterates over players.
    """

    def __init__(self, index: int) -> None:
        super().__init__(index)
        self._action_count = 0

    async def run(self, game: GameView) -> None:
        """Start this player by sending an initial next_round confirmation.

        This is the player-driven entry point: instead of Game pushing initial
        state, each player independently starts by
        requesting the current state. The first next_round confirmation in
        WAITING phase gets the player into the game loop; subsequent on_state
        pushes drive all further actions.
        """
        await game.act(self.index, 0, NextRoundAction())

    async def on_state(self, game: GameView, *, seq: int = 0, error: str | None = None) -> None:
        # Error pushes indicate a stale/rejected action. Re-read current
        # state and retry with the correct seq — the game state hasn't
        # changed, so our decision logic should produce the same action,
        # but now with seq matching the current _seq.
        snapshot = game.snapshot(self.index)

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
        """Bid during DEAL_BID phase if we have a competitive bid.

        Only bids if: (a) no bid_winner yet, or (b) we can beat the
        current winner's priority.  Prefers pairs over singles.
        A 50% random factor prevents always-bidding determinism.
        In sync round-robin mode, must send SkipBidAction when choosing
        not to bid so the turn advances.

        In subsequent rounds (declarer_team is set), only players on
        the declarer team may bid — others must skip.
        """
        from server.sm.card_model import Rank

        # In subsequent rounds, non-declarer-team players cannot bid
        if snapshot.declarer_team is not None:
            from server.sm.constants import get_team_index
            if get_team_index(self.index) != snapshot.declarer_team:
                asyncio.create_task(game.act(self.index, seq, SkipBidAction()))
                return
        hand = snapshot.player_hand
        trump_rank = snapshot.trump_rank

        # 50% chance to even consider bidding (prevents deterministic always-bid)
        if random.random() >= 0.5:
            asyncio.create_task(game.act(self.index, seq, SkipBidAction()))
            return

        # Group trump-rank cards by suit
        suit_groups: dict[Suit, list[Card]] = {}
        jokers: list[Card] = []
        for c in hand:
            if c.rank != trump_rank and not c.is_joker:
                continue
            if c.is_joker:
                jokers.append(c)
            else:
                suit_groups.setdefault(c.suit, []).append(c)

        # Best possible bid: prefer joker pair > trump-rank pair > trump-rank single
        best_cards: list[Card] = []

        # Check joker pairs (requires 2 of same rank)
        sj = [c for c in jokers if c.rank == Rank.SMALL_JOKER]
        bj = [c for c in jokers if c.rank == Rank.BIG_JOKER]
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
            asyncio.create_task(game.act(self.index, seq, SkipBidAction()))
            return

        # Check if existing bid_winner has higher or equal priority
        best_priority = bid_value(best_cards, trump_rank)
        bid_winner = snapshot.bid_winner
        if bid_winner is not None:
            winner_priority = bid_value(bid_winner.cards, trump_rank)
            if best_priority <= winner_priority:
                asyncio.create_task(game.act(self.index, seq, SkipBidAction()))
                return  # can't beat current winner

        action = BidAction(cards=best_cards, count=len(best_cards))
        asyncio.create_task(game.act(self.index, seq, action))

    async def _handle_stir(self, snapshot: StateSnapshot, game: GameView, *, seq: int) -> None:
        """Act during STIRRING phase."""
        hand = snapshot.player_hand
        trump_rank = snapshot.trump_rank
        current_trump_suit = snapshot.trump_suit
        # Find pairs of trump rank cards that can beat current trump
        trump_cards = [c for c in hand if c.rank == trump_rank]
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
        asyncio.create_task(game.act(self.index, seq, action))

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
        asyncio.create_task(game.act(self.index, seq, action))

    async def _handle_play(self, snapshot: StateSnapshot, game: GameView, *, seq: int) -> None:
        """Pick a random legal play."""
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
        asyncio.create_task(game.act(self.index, seq, action))

    async def _handle_next_round(self, snapshot: StateSnapshot, game: GameView, *, seq: int) -> None:
        """Submit NextRoundAction."""
        action = NextRoundAction()
        asyncio.create_task(game.act(self.index, seq, action))


# ---- HumanPlayer ----


class HumanPlayer(Player):
    """Human player that manages its own WebSocket lifecycle.

    Self-contained: receives WS messages, validates seq, parses actions,
    and calls game.act(self.index, action). Server's WS endpoint just
    delegates to handle_connection(websocket, game).
    """

    def __init__(self, index: int) -> None:
        super().__init__(index)
        self._ws: WebSocket | None = None

    async def on_state(self, game: GameView, *, seq: int = 0, error: str | None = None) -> None:
        """Push state to the human player via WebSocket.

        Catches any exception from send_json (e.g. WebSocket disconnected)
        to prevent a broken connection from crashing the entire
        _push_state_to_all() chain. The human player will receive fresh
        state when they reconnect.
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

    async def handle_connection(self, websocket: WebSocket, game: GameView) -> None:
        """Take over the full WS connection lifecycle.

        Handles: connection takeover, game-over fast path, accept,
        receive loop (seq validation + action parsing + game.act),
        and cleanup.

        Args:
            websocket: The incoming WebSocket connection.
            game: The GameView for the game this player belongs to.
        """
        # Connection takeover: close old connection if present
        if self._ws is not None:
            old_ws = self._ws
            self._ws = None
            try:
                await old_ws.close()
            except (WebSocketDisconnect, OSError):
                logger.debug("Failed to close old WS for player %d during takeover", self.index, exc_info=True)

        # Game-over fast path: accept, send final state, close
        if game.is_over():
            await websocket.accept()
            snapshot = game.snapshot(self.index)
            try:
                await websocket.send_json({
                    "type": "state",
                    "seq": game.current_seq,
                    "awaiting": snapshot.awaiting_action,
                    "state": snapshot.to_dict(),
                })
            except (WebSocketDisconnect, OSError):
                logger.debug("Failed to send final state in game-over branch", exc_info=True)
                return
            try:
                await websocket.close()
            except (WebSocketDisconnect, OSError):
                logger.debug("Failed to close WS in game-over branch", exc_info=True)
            return

        # Normal path: accept, bind, receive loop
        await websocket.accept()
        self._ws = websocket

        try:
            while True:
                try:
                    raw = await websocket.receive_json()
                except (WebSocketDisconnect, OSError):
                    logger.debug("WS receive loop ended (client disconnected)")
                    break

                # receive_json() returns Any. Use _is_str_dict TypeGuard to narrow
                # to dict[str, object] instead of dict[Unknown, Unknown].
                if not _is_str_dict(raw):
                    continue
                t = raw.get("type")
                action_type: str | None = t if isinstance(t, str) else None
                s = raw.get("seq", 0)
                client_seq: int = s if isinstance(s, int) else 0
                pass_val_raw = raw.get("pass", False)
                is_pass: bool = isinstance(pass_val_raw, bool) and pass_val_raw
                cards_raw = raw.get("cards")
                card_ids_result = _extract_card_ids(cards_raw)

                # Action parsing
                parse_result = self._parse_action(game, self.index, action_type, is_pass, card_ids_result)
                if isinstance(parse_result, Rejected):
                    snapshot = game.snapshot(self.index)
                    try:
                        await websocket.send_json({
                            "type": "state",
                            "seq": game.current_seq,
                            "awaiting": snapshot.awaiting_action,
                            "state": snapshot.to_dict(),
                            "error": parse_result.reason,
                        })
                    except (WebSocketDisconnect, OSError):
                        break
                    continue

                # Submit action to game
                await game.act(self.index, client_seq, parse_result.value)

                # After game.act, check if game is over
                if game.is_over():
                    # The GAME_OVER state push comes from on_state().
                    # Just exit the loop; on_state() already sent it.
                    break

        finally:
            self._clear_ws_if_current(websocket)

    @staticmethod
    def _parse_action(
        game: GameView,
        player_index: int,
        action_type: str | None,
        is_pass: bool,
        card_ids_result: StateResult[list[str]],
    ) -> StateResult[PlayerAction]:
        """Parse a WS action into a PlayerAction.

        Fields are pre-extracted from the raw WS message by handle_connection.
        Uses game.resolve_cards() to convert card IDs to Card objects.
        Returns Rejected for unknown action types or card resolution failures.
        """
        if action_type is None:
            return Rejected(reason="missing action type")

        if action_type == "bid":
            if is_pass:
                return Ok(value=SkipBidAction())
            if isinstance(card_ids_result, Rejected):
                return card_ids_result
            resolved_result = game.resolve_cards(player_index, card_ids_result.value)
            if isinstance(resolved_result, Rejected):
                return resolved_result
            return Ok(value=BidAction(cards=resolved_result.value, count=len(resolved_result.value)))

        elif action_type == "stir":
            if is_pass:
                return Ok(value=SkipStirAction())
            if isinstance(card_ids_result, Rejected):
                return card_ids_result
            resolved_result = game.resolve_cards(player_index, card_ids_result.value)
            if isinstance(resolved_result, Rejected):
                return resolved_result
            return Ok(value=StirAction(cards=resolved_result.value))

        elif action_type == "discard":
            if isinstance(card_ids_result, Rejected):
                return card_ids_result
            resolved_result = game.resolve_cards(player_index, card_ids_result.value)
            if isinstance(resolved_result, Rejected):
                return resolved_result
            return Ok(value=DiscardAction(cards=resolved_result.value))

        elif action_type == "play":
            if isinstance(card_ids_result, Rejected):
                return card_ids_result
            resolved_result = game.resolve_cards(player_index, card_ids_result.value)
            if isinstance(resolved_result, Rejected):
                return resolved_result
            return Ok(value=PlayAction(cards=resolved_result.value))

        elif action_type == "next_round":
            return Ok(value=NextRoundAction())

        else:
            return Rejected(reason=f"unknown action type: {action_type}")

    def _clear_ws_if_current(self, ws: WebSocket) -> None:
        """Clear the WebSocket reference only if it still points to the given instance.

        Used in finally blocks to avoid clearing a connection that has already
        been replaced by a new one (connection takeover).
        """
        if self._ws is ws:
            self._ws = None

    def is_connected(self) -> bool:
        """Return True if this player has an active WebSocket connection."""
        return self._ws is not None

    async def close_ws(self) -> None:
        """Close the WebSocket connection if active, then clear the reference.

        Sends a final state push before closing so the client receives
        the last known state (e.g., when the game is being deleted).
        """
        if self._ws is not None:
            ws = self._ws
            self._ws = None
            try:
                await ws.close()
            except (WebSocketDisconnect, OSError):
                logger.debug("Failed to close WS for player %d (already disconnected)", self.index, exc_info=True)


# ---- WS message parsing helpers ----


def _is_str_dict(val: object) -> TypeGuard[dict[str, object]]:
    """Narrow object to dict[str, object] — string keys, unknown values.

    isinstance(val, dict) narrows to dict[Unknown, Unknown] which triggers
    reportUnknownVariableType in strict mode. TypeGuard narrows to
    dict[str, object] instead, which is properly typed.
    """
    return isinstance(val, dict)


def _is_obj_list(val: object) -> TypeGuard[list[object]]:
    """Narrow object to list[object] — properly typed element type.

    isinstance(val, list) narrows to list[Unknown] which triggers
    reportUnknownVariableType in strict mode. TypeGuard narrows to
    list[object] instead, which is properly typed.
    """
    return isinstance(val, list)


def _extract_card_ids(cards_val: object) -> StateResult[list[str]]:
    """Extract card ID strings from the 'cards' field of a WS message.

    cards_val comes from raw.get("cards") which may be a list of strings
    or dicts with an "id" key. Returns Rejected if any card format is
    invalid. Returns Ok([]) if cards_val is not a list.
    """
    if not _is_obj_list(cards_val):
        return Ok(value=[])
    ids: list[str] = []
    for item in cards_val:
        if isinstance(item, str):
            ids.append(item)
        elif _is_str_dict(item):
            id_val = item.get("id")
            if isinstance(id_val, str):
                ids.append(id_val)
            else:
                return Rejected(reason=f"Invalid card format: missing 'id' in {item}")
        else:
            return Rejected(reason=f"Invalid card format: {item}")
    return Ok(value=ids)
