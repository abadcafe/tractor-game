#!/usr/bin/env python3
"""
Play a full GAME of Tractor via CDP-controlled real Chromium (headless=False).
Records ALL bugs and playability issues to aaa.md.

A full game = multiple rounds until GAME_OVER (one team reaches ACE).
Human player is index 3 (South, Team 0 with partner North=0).
Opponents: West=1, East=2 (Team 1).
"""

import json
import os
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("DISPLAY", ":0")

from playwright.sync_api import sync_playwright

# ---- Configuration ----
SERVER_URL = "http://127.0.0.1:8000"
SCREENSHOT_DIR = Path("/tmp/tractor-playthrough")
LOG_FILE = SCREENSHOT_DIR / "playthrough.log"
BUG_FILE = Path("/home/lfw/works/tractor-game/aaa.md")
MAX_GAME_DURATION_SECONDS = 3600
POLL_INTERVAL_MS = 300

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Logger ----
log_entries: list[dict[str, Any]] = []
bugs_found: list[dict[str, Any]] = []


def log_event(category: str, message: str, data: dict[str, Any] | None = None) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "message": message,
        "data": data,
    }
    log_entries.append(entry)
    print(f"[{category}] {message}", flush=True)


def record_bug(
    category: str,
    description: str,
    severity: str = "medium",
    phase: str | None = None,
    state_data: dict[str, Any] | None = None,
    screenshot_name: str | None = None,
) -> None:
    """Record a bug with all context."""
    bug = {
        "category": category,
        "description": description,
        "severity": severity,
        "phase": phase,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state_data": state_data,
        "screenshot": screenshot_name,
    }
    # Deduplicate by category+description
    existing = [b for b in bugs_found if b["category"] == category and b["description"] == description]
    if not existing:
        bugs_found.append(bug)
        log_event("BUG", f"[{severity}] {category}: {description}")


# ---- Injected page script: DOM observation + persistent WS state tracking ----
INJECTED_SCRIPT = r"""
(function() {
  if (window.__TRACTOR_INSTRUMENTED) return;
  window.__TRACTOR_INSTRUMENTED = true;
  window.__GAME_LOG = [];
  window.__LAST_STATE = null;
  window.__SCORING_SEEN = false;
  window.__ALL_STATES = [];
  window.__ERROR_MESSAGES = [];

  function pushLog(type, data) {
    const entry = { time: Date.now(), type, data };
    window.__GAME_LOG.push(entry);
    console.log('[TRACTOR_CDP]', type, JSON.stringify(data));
  }

  function startObserver() {
    if (!document.body) { setTimeout(startObserver, 50); return; }
    new MutationObserver(function(mutations) {
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.nodeType === 1) {
            if (node.classList && node.classList.contains('error-toast')) {
              const text = node.textContent || '';
              pushLog('error_toast', { text });
              window.__ERROR_MESSAGES.push({ time: Date.now(), text });
            }
            if (node.classList && node.classList.contains('game-over-overlay')) {
              pushLog('game_over', { text: node.textContent });
            }
            if (node.classList && node.classList.contains('scoring-overlay')) {
              if (!window.__SCORING_SEEN) {
                window.__SCORING_SEEN = true;
                pushLog('scoring', { text: node.textContent });
              }
            }
          }
        }
      }
    }).observe(document.body, { childList: true, subtree: true });
  }
  startObserver();

  const OrigWS = window.WebSocket;
  window.WebSocket = function(...args) {
    const ws = new OrigWS(...args);
    window.__TRACTOR_WS = ws;
    ws.addEventListener('message', (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'state') {
          window.__LAST_STATE = msg;
          // Keep last 500 states for debugging
          window.__ALL_STATES.push({ time: Date.now(), state: msg });
          if (window.__ALL_STATES.length > 500) window.__ALL_STATES.shift();
          if (msg.state && msg.state.phase !== 'COMPLETE') {
            window.__SCORING_SEEN = false;
          }
        } else if (msg.type === 'error') {
          pushLog('server_error', { message: msg.message });
          window.__ERROR_MESSAGES.push({ time: Date.now(), text: msg.message });
        }
      } catch {}
    });
    return ws;
  };

  console.log('[TRACTOR_CDP] Instrumentation loaded');
})();
"""


def take_screenshot(page: Any, name: str) -> None:
    path = SCREENSHOT_DIR / f"{name}.png"
    try:
        page.screenshot(path=path, full_page=False)
        log_event("screenshot", f"Saved {path}")
    except Exception as e:
        log_event("screenshot", f"Failed to save {path}: {e}")


# ---- Card utilities ----
RANK_ORDER: dict[str, int] = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
    "9": 7, "10": 8, "J": 9, "Q": 10, "K": 11, "A": 12,
    "SJ": 13, "BJ": 14,
}

POINT_RANKS: set[str] = {"5", "10", "K"}


def card_value(c: dict[str, Any]) -> int:
    return RANK_ORDER.get(c.get("rank", ""), -1)


def card_points(c: dict[str, Any]) -> int:
    rank = c.get("rank", "")
    if rank == "5": return 5
    if rank == "10": return 10
    if rank == "K": return 10
    return 0


def is_trump_card(c: dict[str, Any], trump_suit: str | None, trump_rank: str) -> bool:
    return c.get("suit") == "joker" or c.get("rank") == trump_rank or c.get("suit") == trump_suit


def card_sort_key(c: dict[str, Any], trump_suit: str | None, trump_rank: str) -> tuple[int, int, int]:
    """Sort key: (is_trump desc, rank_value desc, suit_order)."""
    suit_order = {"spades": 0, "hearts": 1, "clubs": 2, "diamonds": 3, "joker": 4}
    is_t = 1 if is_trump_card(c, trump_suit, trump_rank) else 0
    return (-is_t, -card_value(c), suit_order.get(c.get("suit", ""), 5))


def _extract_error_texts(page: Any, log_type: str, data_key: str) -> list[str]:
    """Extract text strings from page.__GAME_LOG entries of a given type.

    Performs all extraction in the browser to avoid Any/Unknown type issues
    from page.evaluate in pyright strict mode. Returns only string values.
    """
    try:
        raw = page.evaluate(
            """(args) => {
                const log = window.__GAME_LOG || [];
                const filtered = log
                    .filter(e => e.type === args.logType)
                    .map(e => (e.data && e.data[args.dataKey]) || '')
                    .filter(t => typeof t === 'string' && t.trim());
                return JSON.stringify(filtered);
            }""",
            {"logType": log_type, "dataKey": data_key},
        )
        if not isinstance(raw, str):
            return []
        # json.loads returns Any; since we JSON-stringified a string[],
        # we can safely assert the type
        parsed: list[str] = json.loads(raw)
        return parsed
    except Exception:
        return []


def _extract_error_messages(page: Any) -> list[str]:
    """Extract error message strings from the page's __ERROR_MESSAGES array."""
    try:
        raw = page.evaluate("""
            () => {
                const errs = window.__ERROR_MESSAGES || [];
                const filtered = errs.map(e => e.text || '').filter(t => typeof t === 'string' && t.trim());
                return JSON.stringify(filtered);
            }
        """)
        if not isinstance(raw, str):
            return []
        # json.loads returns Any; since we JSON-stringified a string[],
        # we can safely assert the type
        parsed: list[str] = json.loads(raw)
        return parsed
    except Exception:
        return []


# ---- Enhanced Strategy ----

def evaluate_trick_state(state: dict[str, Any]) -> dict[str, Any]:
    trick = state.get("trick")
    if trick is None:
        return {"is_leading": True, "lead_suit": None, "cards_played": 0, "lead_cards": []}
    slots = trick.get("slots", [])
    played = [s for s in slots if s.get("cards") and len(s["cards"]) > 0]
    lead_player = trick.get("lead_player")
    lead_suit = None
    lead_cards: list[dict[str, Any]] = []
    for s in played:
        if s.get("player") == lead_player and s.get("cards"):
            lead_suit = s["cards"][0].get("suit")
            lead_cards = s["cards"]
            break
    return {
        "is_leading": len(played) == 0,
        "lead_suit": lead_suit,
        "cards_played": len(played),
        "slots": slots,
        "lead_cards": lead_cards,
    }


def get_team(player: int) -> int:
    """Team 0: {0,3}, Team 1: {1,2}."""
    return 0 if player in (0, 3) else 1


def choose_play_action(
    hand: Sequence[dict[str, Any]],
    legal_actions: Sequence[Sequence[dict[str, Any]]],
    trick_info: dict[str, Any],
    trump_suit: str | None,
    trump_rank: str,
    state: dict[str, Any],
) -> Sequence[dict[str, Any]] | None:
    """Choose best play action with competitive strategy.

    As Team 0 player (index 3):
    - If we're declarer_team: try to win tricks and prevent defenders from getting points
    - If we're defender: try to win point-rich tricks
    """
    if not legal_actions:
        return None

    human_team = 0  # player 3 is always team 0
    declarer_team = state.get("declarer_team")
    is_defender = declarer_team is not None and declarer_team != human_team
    defender_points = state.get("defender_points", 0)

    def action_strength(action: Sequence[dict[str, Any]]) -> int:
        return sum(card_value(c) for c in action)

    def action_trump_count(action: Sequence[dict[str, Any]]) -> int:
        return sum(1 for c in action if is_trump_card(c, trump_suit, trump_rank))

    def action_point_value(action: Sequence[dict[str, Any]]) -> int:
        return sum(card_points(c) for c in action)

    if trick_info["is_leading"]:
        # ---- LEADING STRATEGY ----
        pairs = [a for a in legal_actions if len(a) == 2 and a[0].get("rank") == a[1].get("rank")]
        tractors = [a for a in legal_actions if len(a) >= 4]
        singles = [a for a in legal_actions if len(a) == 1]
        # throws = multi-card plays that aren't tractors (for future use)

        if is_defender:
            # As defender, lead with strong trumps to control tricks
            # Or lead non-trump aces to draw out trumps
            trump_singles = [a for a in singles if is_trump_card(a[0], trump_suit, trump_rank)]
            non_trump_singles = [a for a in singles if not is_trump_card(a[0], trump_suit, trump_rank)]

            # Lead point cards in suits where opponents likely have to follow
            point_singles = [a for a in non_trump_singles if card_points(a[0]) > 0]
            if point_singles:
                return max(point_singles, key=lambda a: card_points(a[0]))

            # Lead tractors/pairs to maintain control
            if tractors:
                non_trump_tractors = [a for a in tractors if action_trump_count(a) == 0]
                if non_trump_tractors:
                    return max(non_trump_tractors, key=action_strength)
                return max(tractors, key=action_strength)

            if pairs:
                non_trump_pairs = [a for a in pairs if action_trump_count(a) == 0]
                if non_trump_pairs:
                    return max(non_trump_pairs, key=action_strength)
                return max(pairs, key=action_strength)

            # Lead high non-trump cards to force opponents to use trumps
            if non_trump_singles:
                return max(non_trump_singles, key=lambda a: card_value(a[0]))

            # Last resort: play lowest trump
            if trump_singles:
                return min(trump_singles, key=lambda a: card_value(a[0]))

            return max(legal_actions, key=action_strength)
        else:
            # As declarer, be more strategic about preserving trumps
            # Lead non-trump to draw out opponent trumps early
            non_trump_singles = [a for a in singles if not is_trump_card(a[0], trump_suit, trump_rank)]

            if tractors:
                non_trump_tractors = [a for a in tractors if action_trump_count(a) == 0]
                if non_trump_tractors:
                    return max(non_trump_tractors, key=action_strength)
                return max(tractors, key=action_strength)

            if pairs:
                non_trump_pairs = [a for a in pairs if action_trump_count(a) == 0]
                if non_trump_pairs:
                    return max(non_trump_pairs, key=action_strength)

            if non_trump_singles:
                # Lead from longest suit for better control
                return max(non_trump_singles, key=lambda a: card_value(a[0]))

            if singles:
                return max(singles, key=lambda a: card_value(a[0]))

            return max(legal_actions, key=action_strength)
    else:
        # ---- FOLLOWING STRATEGY ----
        lead_suit = trick_info.get("lead_suit")
        # lead_cards used for determining trick structure
        cards_played = trick_info.get("cards_played", 0)
        slots = trick_info.get("slots", [])

        # Determine current trick winner and points
        trick_winner_team = -1
        trick_points = 0
        current_winning_strength = -1

        for s in slots:
            cards_in_slot = s.get("cards", [])
            if cards_in_slot:
                trick_points += sum(card_points(c) for c in cards_in_slot)
                strength = sum(card_value(c) for c in cards_in_slot)
                all_trump = all(is_trump_card(c, trump_suit, trump_rank) for c in cards_in_slot)
                if all_trump:
                    strength += 1000  # Trump beats non-trump
                if strength > current_winning_strength:
                    current_winning_strength = strength
                    trick_winner_team = get_team(s.get("player", -1))

        # Is our partner currently winning?
        partner_winning = trick_winner_team == human_team
        # Are there point cards in the trick?
        trick_has_points = trick_points > 0

        suit_matching: list[Sequence[dict[str, Any]]] = []
        trump_plays: list[Sequence[dict[str, Any]]] = []
        off_plays: list[Sequence[dict[str, Any]]] = []

        for a in legal_actions:
            if lead_suit and all(c.get("suit") == lead_suit for c in a):
                suit_matching.append(a)
            elif all(is_trump_card(c, trump_suit, trump_rank) for c in a):
                trump_plays.append(a)
            else:
                off_plays.append(a)

        if is_defender:
            # As defender: try to win tricks with points
            if suit_matching:
                if partner_winning and cards_played >= 2:
                    # Partner is winning and we're last or nearly last - dump low cards
                    return min(suit_matching, key=action_strength)
                # Try to win the trick
                # Play just enough to beat current best
                winning_plays = [a for a in suit_matching if action_strength(a) > current_winning_strength - 1000]
                if winning_plays:
                    return min(winning_plays, key=action_strength)  # Win by smallest margin
                return max(suit_matching, key=action_strength)  # Can't beat, dump lowest

            if trump_plays:
                if partner_winning:
                    # Don't waste trumps if partner is winning
                    # But if we're last to play and partner is winning, don't overtrump
                    if cards_played >= 3:
                        # We're last, partner won - play something else
                        if off_plays:
                            return min(off_plays, key=action_strength)
                        return min(trump_plays, key=action_strength)
                    # Not last yet, still risky - play low
                    return min(trump_plays, key=action_strength)
                # Need to win - use minimum trump to win
                if trick_has_points:
                    return min(trump_plays, key=action_strength)
                # No points - maybe don't waste trumps
                if defender_points < 40:  # Need points badly
                    return min(trump_plays, key=action_strength)
                # Have enough points, save trumps
                if off_plays:
                    return min(off_plays, key=action_strength)
                return min(trump_plays, key=action_strength)

            # Can't follow suit, no trumps
            if off_plays:
                # Dump cards with no points
                no_point_dumps = [a for a in off_plays if action_point_value(a) == 0]
                if no_point_dumps:
                    return max(no_point_dumps, key=action_strength)
                return max(off_plays, key=action_strength)

        else:
            # As declarer: control the game, prevent defenders from getting points
            if suit_matching:
                if partner_winning and not trick_has_points:
                    # Partner winning, no points at stake - save strong cards
                    return min(suit_matching, key=action_strength)
                # Play strong to win
                return max(suit_matching, key=action_strength)

            if trump_plays:
                if partner_winning and cards_played >= 2:
                    # Partner likely winning, don't overtrump
                    if off_plays:
                        return min(off_plays, key=action_strength)
                return min(trump_plays, key=action_strength)

            if off_plays:
                no_point_dumps = [a for a in off_plays if action_point_value(a) == 0]
                if no_point_dumps:
                    return max(no_point_dumps, key=action_strength)
                return max(off_plays, key=action_strength)

        return max(legal_actions, key=action_strength)


def run_playthrough() -> None:
    round_history: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            executable_path="/usr/bin/chromium",
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--window-size=1400,950",
            ],
        )
        context = browser.new_context(viewport={"width": 1400, "height": 950})
        context.add_init_script(script=INJECTED_SCRIPT)
        page = context.new_page()

        error_flags: dict[str, bool] = {
            "bid_failed": False,
            "bid_done": False,
            "play_failed": False,
            "stir_failed": False,
            "stir_done": False,
        }
        action_history: list[dict[str, Any]] = []
        action_count = 0
        max_actions = 5000
        next_round_sent = False
        previous_state_msg: dict[str, Any] | None = None
        bid_cooldown = 0  # Prevent bid spam; countdown each poll

        page.on("console", lambda msg: log_event("console", f"[{msg.type}] {msg.text}", {"type": msg.type}))
        def on_page_error(err: Any) -> None:
            log_event("pageerror", str(err), {"error": str(err)})
            record_bug("pageerror", str(err), severity="high")

        def on_request_failed(req: Any) -> None:
            log_event("requestfailed", f"{req.method} {req.url} failed", {"url": req.url})
            record_bug("request_failed", f"{req.method} {req.url}", severity="low")

        page.on("pageerror", on_page_error)
        page.on("requestfailed", on_request_failed)

        log_event("navigation", f"Opening {SERVER_URL}")
        page.goto(SERVER_URL, wait_until="networkidle")
        time.sleep(2)
        take_screenshot(page, "00_initial_load")

        log_event("wait", "Waiting for WebSocket connection and initial state...")
        for _ in range(30):
            try:
                state = page.evaluate("() => window.__LAST_STATE")
                if state:
                    log_event("state", "Initial state received", {"phase": state.get("state", {}).get("phase")})
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            log_event("error", "Timed out waiting for initial state")
            take_screenshot(page, "error_no_initial_state")
            record_bug("init_timeout", "Timed out waiting for initial game state", severity="critical")
            browser.close()
            return

        take_screenshot(page, "01_game_started")

        def send_action(action: dict[str, Any]) -> bool:
            try:
                result = page.evaluate("""
                    (action) => {
                        if (!window.__TRACTOR_WS || window.__TRACTOR_WS.readyState !== 1) return false;
                        window.__TRACTOR_WS.send(JSON.stringify(action));
                        return true;
                    }
                """, action)
                if result:
                    action_history.append({"time": time.time(), "action": action})
                    log_event("ws_action", f"Sent {action['type']}", action)
                    return True
                else:
                    log_event("ws_action", f"WS not ready for {action['type']}")
                    record_bug("ws_not_ready", f"WebSocket not ready for action {action['type']}", severity="medium")
                    return False
            except Exception as e:
                log_event("ws_action", f"Failed to send {action['type']}: {e}")
                record_bug("ws_send_failed", f"Failed to send {action['type']}: {e}", severity="high")
                return False

        def get_human_action(state_msg: dict[str, Any]) -> dict[str, Any] | None:
            state = state_msg.get("state", {})
            phase = state.get("phase")
            awaiting = state_msg.get("awaiting")
            current_player = state.get("current_player")
            human_index = 3

            # COMPLETE phase: ALL players must send next_round, regardless of current_player
            if phase == "COMPLETE" and awaiting == "next_round":
                nonlocal next_round_sent
                if not next_round_sent:
                    next_round_sent = True
                    return {"type": "next_round"}
                return None

            # During DEAL_BID, we can bid even if it's not our "turn" for dealing
            # Bidding happens during the dealing process - we should try to bid
            # each time we see new cards in our hand that qualify
            if phase == "DEAL_BID":
                nonlocal bid_cooldown
                if bid_cooldown > 0:
                    bid_cooldown -= 1
                    return None

                # If our last bid was rejected, don't keep trying this round
                if error_flags["bid_failed"]:
                    return None

                hand = state.get("player_hand", [])
                trump_rank = state.get("trump_rank")
                bid_winner = state.get("bid_winner")
                declarer_team = state.get("declarer_team")
                human_team = 0  # player 3 is team 0

                # If we're not the declarer team and there's already a bid winner
                # from the declarer team, we can't bid (非庄家方不能叫牌)
                if declarer_team is not None and declarer_team != human_team and bid_winner is not None:
                    return None

                # If we already bid successfully (bid_done) and there's a winner
                # that's us, no need to bid again
                if error_flags["bid_done"] and bid_winner is not None:
                    return None

                jokers = [c for c in hand if c.get("suit") == "joker"]
                bj = [c for c in jokers if c.get("rank") == "BJ"]
                sj = [c for c in jokers if c.get("rank") == "SJ"]

                candidates: list[list[dict[str, Any]]] = []
                # Priority: big joker pair > small joker pair > trump-rank pairs > trump-rank singles
                if len(bj) >= 2:
                    candidates.append([bj[0], bj[1]])
                if len(sj) >= 2:
                    candidates.append([sj[0], sj[1]])
                trump_cards = [c for c in hand if c.get("rank") == trump_rank and c.get("suit") != "joker"]
                suits: dict[str, list[dict[str, Any]]] = {}
                for c in trump_cards:
                    suits.setdefault(c["suit"], []).append(c)
                for suit_cards in suits.values():
                    if len(suit_cards) >= 2:
                        candidates.append(suit_cards[:2])
                    elif len(suit_cards) >= 1:
                        candidates.append([suit_cards[0]])

                if not candidates:
                    return None

                def cand_priority(cand: list[dict[str, Any]]) -> tuple[int, int]:
                    is_pair = len(cand) >= 2
                    is_joker = cand[0].get("suit") == "joker"
                    return (1 if is_pair else 0, 1 if is_joker else 0)

                candidates.sort(key=cand_priority, reverse=True)
                best = candidates[0]

                if bid_winner is not None:
                    winner_cards = bid_winner.get("cards", [])
                    # Can't beat joker pair
                    if winner_cards and winner_cards[0].get("suit") == "joker" and len(winner_cards) >= 2:
                        return None
                    # If winner has pair and we only have single, don't bid
                    if len(winner_cards) >= 2 and len(best) < 2:
                        return None
                    # Same pair count - check priority
                    if len(winner_cards) >= 2 and len(best) >= 2:
                        # Joker beats trump-rank
                        if winner_cards[0].get("suit") == "joker":
                            return None

                # Send bid and set cooldown to prevent spam
                bid_cooldown = 5  # Wait 5 poll cycles before trying again
                error_flags["bid_failed"] = False  # Reset for next attempt
                return {"type": "bid", "cards": [c["id"] for c in best]}

            # Not our turn for other phases
            if current_player != human_index:
                return None

            if phase == "STIRRING" and awaiting == "stir":
                if error_flags.get("stir_failed") or error_flags.get("stir_done"):
                    return {"type": "stir", "pass": True}
                hand = state.get("player_hand", [])
                trump_rank = state.get("trump_rank")
                current_trump_suit = state.get("trump_suit")
                jokers = [c for c in hand if c.get("suit") == "joker"]
                bj = [c for c in jokers if c.get("rank") == "BJ"]
                sj = [c for c in jokers if c.get("rank") == "SJ"]
                # Stir with joker pairs (highest priority)
                if len(bj) >= 2:
                    error_flags["stir_done"] = True
                    return {"type": "stir", "cards": [bj[0]["id"], bj[1]["id"]]}
                if len(sj) >= 2:
                    error_flags["stir_done"] = True
                    return {"type": "stir", "cards": [sj[0]["id"], sj[1]["id"]]}
                trump_cards = [c for c in hand if c.get("rank") == trump_rank and c.get("suit") != "joker"]
                suits: dict[str, list[dict[str, Any]]] = {}
                for c in trump_cards:
                    suits.setdefault(c["suit"], []).append(c)
                for suit, suit_cards in suits.items():
                    if len(suit_cards) >= 2:
                        if current_trump_suit is None or current_trump_suit == "joker" or suit != current_trump_suit:
                            error_flags["stir_done"] = True
                            return {"type": "stir", "cards": [suit_cards[0]["id"], suit_cards[1]["id"]]}
                return {"type": "stir", "pass": True}

            if phase == "EXCHANGE" and awaiting == "discard":
                hand = state.get("player_hand", [])
                count = state.get("exchange_state", {}).get("count", 8)
                trump_suit = state.get("trump_suit")
                trump_rank = state.get("trump_rank")

                # Strategic discard: keep trumps and point cards, dump low non-trump
                def discard_priority(c: dict[str, Any]) -> tuple[int, int, int]:
                    """Lower = more likely to discard."""
                    is_t = 1 if is_trump_card(c, trump_suit, trump_rank) else 0
                    pts = card_points(c)
                    val = card_value(c)
                    return (is_t, pts, val)  # Discard: non-trump first, then no-points, then low-value

                sorted_hand = sorted(hand, key=discard_priority)
                to_discard = sorted_hand[:count]
                return {"type": "discard", "cards": [c["id"] for c in to_discard]}

            if phase == "PLAYING" and awaiting == "play":
                legal = state.get("legal_actions", [])
                if not legal:
                    log_event("warning", "No legal actions in PLAYING phase!")
                    record_bug("no_legal_actions", "No legal actions available in PLAYING phase", severity="high", phase="PLAYING")
                    return None
                trick_info = evaluate_trick_state(state)
                trump_suit = state.get("trump_suit")
                trump_rank = state.get("trump_rank")
                best = choose_play_action(
                    state.get("player_hand", []),
                    legal,
                    trick_info,
                    trump_suit,
                    trump_rank,
                    state,
                )
                if best is None:
                    return None
                return {"type": "play", "cards": [c["id"] for c in best]}

            return None

        start_time = time.time()
        last_phase = None
        last_phase_change_time = start_time
        last_screenshot_time = start_time
        last_action_time = 0.0
        action_cooldown = 1.0
        consecutive_errors = 0
        phase_stuck_threshold = 180
        round_count = 0
        last_team0_level = "2"
        last_team1_level = "2"
        game_over = False
        trick_count = 0
        last_trick_history_len = 0
        last_error_check_time = 0.0
        state_msg: dict[str, Any] | None = None

        while True:
            elapsed = time.time() - start_time
            if elapsed > MAX_GAME_DURATION_SECONDS:
                log_event("timeout", f"Game exceeded {MAX_GAME_DURATION_SECONDS}s, stopping")
                take_screenshot(page, "timeout")
                record_bug("game_timeout", f"Game exceeded {MAX_GAME_DURATION_SECONDS}s without completing", severity="critical")
                break

            if action_count > max_actions:
                log_event("limit", f"Exceeded max actions {max_actions}, stopping")
                record_bug("action_limit", f"Exceeded max actions {max_actions}", severity="high")
                break

            try:
                state_msg = page.evaluate("() => window.__LAST_STATE")
            except Exception:
                state_msg = None
                consecutive_errors += 1
                if consecutive_errors > 30:
                    log_event("error", "State polling failed repeatedly, aborting")
                    record_bug("ws_disconnect", "WebSocket disconnected and state polling failed", severity="critical")
                    break
                time.sleep(POLL_INTERVAL_MS / 1000)
                continue

            consecutive_errors = 0

            if state_msg:
                state = state_msg.get("state", {})
                phase = state.get("phase")
                current_player = state.get("current_player")
                is_human_turn = current_player == 3
                team0_level = state.get("team0_level")
                team1_level = state.get("team1_level")
                defender_points = state.get("defender_points", 0)
                trick_history = state.get("trick_history", [])

                # Track trick progress
                if len(trick_history) != last_trick_history_len:
                    new_tricks = trick_history[last_trick_history_len:]
                    for t in new_tricks:
                        trick_count += 1
                        winner = t.get("winner", -1)
                        points = t.get("points", 0)
                        log_event("trick", f"Trick #{trick_count}: winner=player{winner}(team{get_team(winner)}), points={points}")
                    last_trick_history_len = len(trick_history)

                # Detect round change via level changes
                if team0_level != last_team0_level or team1_level != last_team1_level:
                    round_count += 1
                    level_change = f"team0={last_team0_level}->{team0_level}, team1={last_team1_level}->{team1_level}"
                    log_event("round_change", f"Round {round_count} complete: {level_change}")
                    round_history.append({
                        "round": round_count,
                        "team0": f"{last_team0_level}->{team0_level}",
                        "team1": f"{last_team1_level}->{team1_level}",
                        "defender_points": defender_points,
                    })
                    last_team0_level = team0_level
                    last_team1_level = team1_level
                    next_round_sent = False
                    trick_count = 0
                    last_trick_history_len = 0

                # Check for state regression (phase going backwards unexpectedly)
                phase_order = {"IDLE": 0, "DEAL_BID": 1, "STIRRING": 2, "EXCHANGE": 3, "PLAYING": 4, "COMPLETE": 5, "GAME_OVER": 6}
                if previous_state_msg and last_phase:
                    prev_state = previous_state_msg.get("state", {})
                    prev_phase = prev_state.get("phase")
                    if (prev_phase in phase_order and phase in phase_order
                            and phase_order.get(phase, 0) < phase_order.get(prev_phase, 0)
                            and prev_phase != "COMPLETE" and phase != "DEAL_BID"):
                        # Don't flag COMPLETE->DEAL_BID (that's normal round transition)
                        record_bug("phase_regression", f"Phase regressed from {prev_phase} to {phase}", severity="high", phase=phase)

                previous_state_msg = state_msg

                # Check WebSocket health
                try:
                    ws_state = page.evaluate("() => window.__TRACTOR_WS ? window.__TRACTOR_WS.readyState : -1")
                    if ws_state != 1 and last_phase is not None:
                        log_event("ws_status", f"WebSocket readyState={ws_state}")
                        record_bug("ws_disconnect", f"WebSocket disconnected (readyState={ws_state})", severity="high", phase=phase)
                except Exception:
                    pass

                # Check for error messages from server
                if time.time() - last_error_check_time > 2:
                    last_error_check_time = time.time()
                    try:
                        # Get errors since last check by clearing them after reading
                        recent_errors = page.evaluate("""
                            () => {
                                const errs = window.__ERROR_MESSAGES.splice(0);
                                return errs.map(e => e.text || '');
                            }
                        """)
                        for text in recent_errors:
                            if not isinstance(text, str):
                                continue
                            if "叫牌" in text or "不能叫牌" in text:
                                error_flags["bid_failed"] = True
                                bid_cooldown = 10  # Wait longer after rejection
                                log_event("info", f"Bid rejected (expected): {text}")
                                # Don't record as bug - bid rejection is normal game flow
                            elif "反主" in text or "不能反主" in text:
                                error_flags["stir_failed"] = True
                                log_event("error", f"Stir rejected: {text}")
                                record_bug("stir_rejected", text, severity="medium", phase="STIRRING")
                            elif "出牌" in text or "无效的出牌" in text:
                                log_event("error", f"Play rejected: {text}")
                                record_bug("action_rejected", text, severity="medium", phase=phase)
                            elif "弃牌" in text:
                                log_event("error", f"Discard rejected: {text}")
                                record_bug("discard_rejected", text, severity="medium", phase="EXCHANGE")
                            elif text.strip():
                                log_event("error", f"Other error: {text}")
                                record_bug("other_error", text, severity="low", phase=phase)
                    except Exception:
                        pass

                # Validate state consistency
                if phase == "PLAYING":
                    hand = state.get("player_hand", [])
                    legal = state.get("legal_actions", [])
                    if is_human_turn and not legal:
                        record_bug("no_legal_actions_playing", "Human's turn in PLAYING but no legal actions", severity="high", phase="PLAYING",
                                   state_data={"current_player": current_player, "hand_size": len(hand)})

                    # Check hand size consistency
                    hand_counts = state.get("player_hand_counts", [])
                    if len(hand_counts) == 4:
                        total = sum(hand_counts)
                        if phase == "PLAYING" and total > 100:
                            record_bug("card_count_overflow", f"Total cards in play: {total} (expected <=100)", severity="medium", phase="PLAYING")

                # DEAL_BID progress is monitored via phase_stuck check

                if phase != last_phase:
                    if last_phase is not None and (time.time() - last_phase_change_time) > phase_stuck_threshold:
                        record_bug("phase_stuck", f"Phase {last_phase} stuck for {int(time.time() - last_phase_change_time)}s before transitioning to {phase}",
                                   severity="critical", phase=last_phase)

                    log_event("phase_change", f"Phase changed to {phase}", {
                        "phase": phase,
                        "current_player": current_player,
                        "is_human_turn": is_human_turn,
                        "team0_level": team0_level,
                        "team1_level": team1_level,
                        "defender_points": defender_points,
                    })
                    take_screenshot(page, f"phase_{phase.lower()}_{int(elapsed)}")
                    last_phase = phase
                    last_phase_change_time = time.time()
                    error_flags["bid_failed"] = False
                    error_flags["bid_done"] = False
                    error_flags["play_failed"] = False
                    error_flags["stir_failed"] = False
                    error_flags["stir_done"] = False
                    if phase != "COMPLETE":
                        next_round_sent = False

                # Check for phase stuck
                if last_phase is not None and (time.time() - last_phase_change_time) > phase_stuck_threshold:
                    record_bug("phase_stuck", f"Phase {phase} stuck for {int(time.time() - last_phase_change_time)}s",
                               severity="critical", phase=phase)
                    take_screenshot(page, f"stuck_{phase.lower()}_{int(elapsed)}")
                    break

                # Periodic screenshot
                if time.time() - last_screenshot_time > 30:
                    take_screenshot(page, f"tick_{int(elapsed)}")
                    last_screenshot_time = time.time()

                if phase == "GAME_OVER":
                    game_over = True
                    log_event("game_over", "GAME_OVER phase detected")
                    take_screenshot(page, "99_game_over")
                    winning_team = state.get("winning_team")
                    human_won = winning_team == 0
                    log_event("result", f"Game over! Winning team: {winning_team} ({'WE WON!' if human_won else 'we lost'}), team0_level={team0_level}, team1_level={team1_level}")
                    break

                # Human action
                if (time.time() - last_action_time) > action_cooldown:
                    action = get_human_action(state_msg)
                    if action:
                        if send_action(action):
                            action_count += 1
                            last_action_time = time.time()
                            time.sleep(0.5)
                            continue

            time.sleep(POLL_INTERVAL_MS / 1000)

        # ---- Post-game analysis ----
        # Extract error texts via browser-side filtering to avoid pyright Any issues
        error_toast_texts = _extract_error_texts(page, "error_toast", "text")
        if error_toast_texts:
            log_event("summary", f"Error toasts observed: {len(error_toast_texts)}")
            for text in error_toast_texts:
                record_bug("error_toast", text, severity="medium")

        server_error_texts = _extract_error_texts(page, "server_error", "message")
        if server_error_texts:
            log_event("summary", f"Server errors observed: {len(server_error_texts)}")
            for msg in server_error_texts:
                record_bug("server_error", msg, severity="medium")

        # Get raw page log for JSON report (untyped, for logging only)
        page_log_raw: list[dict[str, Any]] = []
        try:
            raw_json = page.evaluate("() => JSON.stringify(window.__GAME_LOG || [])")
            if isinstance(raw_json, str):
                # json.loads returns Any; we know the structure from our JS code
                page_log_raw = json.loads(raw_json)
        except Exception:
            pass

        all_error_msgs = _extract_error_messages(page)

        log_event("summary", f"Actions={action_count}, events={len(page_log_raw)}, bugs={len(bugs_found)}, rounds={round_count}")

        # ---- Save full JSON log ----
        final_state_data = state_msg if state_msg is not None else None
        report = {
            "meta": {
                "start_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.time() - start_time,
                "server_url": SERVER_URL,
                "browser": "chromium_headless_false",
                "game_completed": game_over,
            },
            "entries": log_entries,
            "page_events": page_log_raw,
            "action_history": action_history,
            "bugs_found": bugs_found,
            "round_history": round_history,
            "final_state": final_state_data,
            "all_errors": all_error_msgs,
        }
        LOG_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

        browser.close()
        log_event("done", f"Playthrough complete. Logs saved to {LOG_FILE}")

    # ---- Write bug report to aaa.md ----
    final_report_state = state_msg if state_msg is not None else None
    write_bug_report(bugs_found, round_history, game_over, final_report_state, time.time() - start_time)


def write_bug_report(
    bugs: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    game_completed: bool,
    final_state: dict[str, Any] | None,
    duration: float,
) -> None:
    """Write comprehensive bug report to aaa.md."""
    lines: list[str] = []
    lines.append("# 🎮 拖拉机游戏完整一局测试报告\n")
    lines.append(f"**测试时间**: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**游戏时长**: {duration:.1f}秒 ({duration/60:.1f}分钟)")
    lines.append(f"**游戏完成**: {'✅ 是' if game_completed else '❌ 否'}")

    if final_state:
        state = final_state.get("state", {})
        winning_team = state.get("winning_team")
        team0_level = state.get("team0_level", "?")
        team1_level = state.get("team1_level", "?")
        human_team = 0
        lines.append(f"**获胜方**: Team {winning_team} ({'🏆 我们赢了!' if winning_team == human_team else '😞 我们输了'})")
        lines.append(f"**最终等级**: Team0={team0_level}, Team1={team1_level}")

    lines.append(f"**总回合数**: {len(rounds)}")
    lines.append(f"**发现问题数**: {len(bugs)}")
    lines.append("")

    # Round history
    if rounds:
        lines.append("## 📊 回合历史\n")
        lines.append("| 回合 | Team0等级变化 | Team1等级变化 | 闲家得分 |")
        lines.append("|------|-------------|-------------|---------|")
        for r in rounds:
            lines.append(f"| {r['round']} | {r['team0']} | {r['team1']} | {r.get('defender_points', '?')} |")
        lines.append("")

    # Bug summary
    if bugs:
        lines.append("## 🐛 发现的问题\n")

        # Group by severity
        critical = [b for b in bugs if b["severity"] == "critical"]
        high = [b for b in bugs if b["severity"] == "high"]
        medium = [b for b in bugs if b["severity"] == "medium"]
        low = [b for b in bugs if b["severity"] == "low"]

        lines.append(f"**严重程度分布**: 🔴 严重={len(critical)}, 🟠 高={len(high)}, 🟡 中={len(medium)}, 🟢 低={len(low)}")
        lines.append("")

        for severity_name, severity_bugs in [("🔴 严重 (Critical)", critical), ("🟠 高 (High)", high), ("🟡 中 (Medium)", medium), ("🟢 低 (Low)", low)]:
            if not severity_bugs:
                continue
            lines.append(f"### {severity_name}\n")
            for i, bug in enumerate(severity_bugs, 1):
                lines.append(f"#### {i}. [{bug['category']}] {bug['description']}\n")
                if bug.get("phase"):
                    lines.append(f"- **阶段**: {bug['phase']}")
                if bug.get("state_data"):
                    lines.append(f"- **状态数据**: `{json.dumps(bug['state_data'], ensure_ascii=False)}`")
                if bug.get("screenshot"):
                    lines.append(f"- **截图**: {bug['screenshot']}")
                lines.append(f"- **时间**: {bug.get('timestamp', 'unknown')}")
                lines.append("")
    else:
        lines.append("## ✅ 未发现问题\n")
        lines.append("本局游戏运行完美，未发现任何 bug！\n")

    # Detailed event log reference
    lines.append("## 📋 详细日志\n")
    lines.append(f"完整 JSON 日志: `{LOG_FILE}`")
    lines.append(f"截图目录: `{SCREENSHOT_DIR}/`")
    lines.append("")

    BUG_FILE.write_text("\n".join(lines))
    print(f"\n{'='*60}")
    print(f"Bug report written to {BUG_FILE}")
    print(f"Found {len(bugs)} bugs")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        run_playthrough()
    except Exception as e:
        log_event("fatal", f"Playthrough crashed: {e}", {"error": str(e)})
        record_bug("fatal_crash", f"Playthrough script crashed: {e}", severity="critical")
        try:
            write_bug_report(bugs_found, [], False, None, 0)
        except Exception:
            pass
        raise
