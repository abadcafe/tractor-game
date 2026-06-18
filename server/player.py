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
from dataclasses import dataclass
from itertools import combinations
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
from server.sm.card_model import Card, Rank, Suit
from server.sm.result import Ok, Rejected, StateResult
from server.snapshot import StateSnapshot

logger = logging.getLogger(__name__)

# Type alias for the action union used in GameView.act()
PlayerAction = BidAction | SkipBidAction | StirAction | SkipStirAction | DiscardAction | PlayAction | NextRoundAction


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
        """Start this player by sending an initial next_round confirmation.

        This is the player-driven entry point: instead of Game pushing initial
        state, each player independently starts by
        requesting the current state. The first next_round confirmation in
        WAITING phase gets the player into the game loop; subsequent on_state
        pushes drive all further actions.
        """
        await game.act(self.index, 0, NextRoundAction())

    async def on_state(self, game: GameView, *, seq: int = 0, error: str | None = None) -> None:
        snapshot = game.snapshot(self.index)
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
        asyncio.create_task(game.act(self.index, seq, action))

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
        return not error.startswith("stale action:")

    async def _handle_next_round(self, snapshot: StateSnapshot, game: GameView, *, seq: int) -> None:
        """Submit NextRoundAction."""
        action = NextRoundAction()
        self._submit_action(game, seq, action, snapshot)


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
