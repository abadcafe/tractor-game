#!/usr/bin/env python3
"""
Visible Playwright playthrough pytest test for the Tractor browser UI.

Default behavior:
- build the frontend into static/
- start uvicorn with websockets-sansio
- open a tab in an existing debug-enabled Chromium when available,
otherwise
  launch a visible Chromium window
- drive a full game through the rendered UI
- write screenshots and reports under test-results/playthrough/
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Literal, Protocol, TextIO, TypeGuard, cast
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type Severity = Literal["critical", "high", "medium", "low"]
type UiActionKind = Literal[
    "bid", "next_round", "stir_pass", "discard", "play"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test-results" / "playthrough"
HUMAN_PLAYER = 3
THIS_FILE = Path(__file__).name
CLEAR_SELECTION_BUTTON = "清牌"
BID_ACTION_BUTTON_SELECTOR = ".hand-actions .action-panel--bid button"
NEXT_ROUND_BUTTON_SELECTOR = ".scoring-overlay__next-round"
DEFAULT_MAX_SECONDS = 12 * 60 * 60

RANK_ORDER: dict[str, int] = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "10": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12,
    "SJ": 13,
    "BJ": 14,
}


class LocatorLike(Protocol):
    @property
    def first(self) -> LocatorLike: ...

    def click(self, *, timeout: float | None = None) -> None: ...

    def count(self) -> int: ...

    def is_enabled(self, *, timeout: float | None = None) -> bool: ...

    def scroll_into_view_if_needed(
        self, *, timeout: float | None = None
    ) -> None: ...


class PageLike(Protocol):
    def get_by_role(
        self,
        role: str,
        *,
        name: str,
        exact: bool | None = None,
    ) -> LocatorLike: ...

    def locator(self, selector: str) -> LocatorLike: ...

    def evaluate(self, expression: str) -> object: ...

    def screenshot(self, *, path: str, full_page: bool) -> None: ...

    def goto(
        self, url: str, *, wait_until: str | None = None
    ) -> object: ...

    def on(
        self, event: str, callback: Callable[[object], None]
    ) -> None: ...

    def close(self) -> None: ...


class BrowserContextLike(Protocol):
    def add_init_script(self, *, script: str) -> None: ...

    def new_page(self) -> PageLike: ...

    def close(self) -> None: ...


class BrowserLike(Protocol):
    @property
    def contexts(self) -> list[BrowserContextLike]: ...

    def new_context(
        self, *, viewport: dict[str, int]
    ) -> BrowserContextLike: ...

    def close(self) -> None: ...


class BrowserTypeLike(Protocol):
    def launch(
        self,
        *,
        headless: bool,
        executable_path: str | None,
        args: list[str],
    ) -> BrowserLike: ...

    def connect_over_cdp(
        self,
        endpoint_url: str,
    ) -> BrowserLike: ...


class PlaywrightLike(Protocol):
    @property
    def chromium(self) -> BrowserTypeLike: ...

    def stop(self) -> None: ...


class PlaywrightStarterLike(Protocol):
    def start(self) -> PlaywrightLike: ...


class PlaywrightModuleLike(Protocol):
    def sync_playwright(self) -> PlaywrightStarterLike: ...


class ConsoleMessageLike(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def text(self) -> str: ...


class RequestLike(Protocol):
    @property
    def method(self) -> str: ...

    @property
    def url(self) -> str: ...


INJECTED_SCRIPT = r"""
(() => {
  if (window.__TRACTOR_PLAYWRIGHT_INSTALLED) return;
  window.__TRACTOR_PLAYWRIGHT_INSTALLED = true;
  window.__TRACTOR_LAST_STATE_JSON = "";
  window.__TRACTOR_EVENTS = [];
  window.__TRACTOR_ERRORS = [];

  function saveEvents() {
    window.__TRACTOR_EVENTS_JSON = JSON.stringify(
      window.__TRACTOR_EVENTS
    );
    window.__TRACTOR_ERRORS_JSON = JSON.stringify(
      window.__TRACTOR_ERRORS
    );
  }

  function record(type, data) {
    const entry = { time: Date.now(), type, data };
    window.__TRACTOR_EVENTS.push(entry);
    if (window.__TRACTOR_EVENTS.length > 1000) {
      window.__TRACTOR_EVENTS.shift();
    }
    saveEvents();
    console.log("[TRACTOR_PLAYWRIGHT]", type, JSON.stringify(data));
  }

  function observeDom() {
    if (!document.body) {
      window.setTimeout(observeDom, 50);
      return;
    }
    new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (!(node instanceof HTMLElement)) continue;
          if (node.classList.contains("error-toast")) {
            const text = node.textContent || "";
            window.__TRACTOR_ERRORS.push({ time: Date.now(), text });
            record("error_toast", { text });
          }
          if (node.classList.contains("game-over-overlay")) {
            record(
              "game_over_overlay",
              { text: node.textContent || "" }
            );
          }
          if (node.classList.contains("scoring-overlay")) {
            record("scoring_overlay", { text: node.textContent || "" });
          }
        }
      }
      saveEvents();
    }).observe(document.body, { childList: true, subtree: true });
  }
  observeDom();

  const OriginalWebSocket = window.WebSocket;
  const WrappedWebSocket = function(...args) {
    const ws = new OriginalWebSocket(...args);
    window.__TRACTOR_WS = ws;
    record("ws_opening", { url: String(args[0] || "") });
    ws.addEventListener("open", () => record("ws_open", {}));
    ws.addEventListener("close", (event) => {
      record("ws_close", { code: event.code, reason: event.reason });
    });
    ws.addEventListener("error", () => record("ws_error", {}));
    ws.addEventListener("message", (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message && message.type === "state") {
          window.__TRACTOR_LAST_STATE_JSON = JSON.stringify(message);
          const state = message.state || {};
          record("state", {
            seq: message.seq,
            awaiting: state.awaiting_action,
            phase: state.phase,
            current_player: state.trick
              ? state.trick.current_player
              : null
          });
        }
      } catch (error) {
        record("ws_message_parse_error", { message: String(error) });
      }
    });
    return ws;
  };
  WrappedWebSocket.prototype = OriginalWebSocket.prototype;
  WrappedWebSocket.CONNECTING = OriginalWebSocket.CONNECTING;
  WrappedWebSocket.OPEN = OriginalWebSocket.OPEN;
  WrappedWebSocket.CLOSING = OriginalWebSocket.CLOSING;
  WrappedWebSocket.CLOSED = OriginalWebSocket.CLOSED;
  window.WebSocket = WrappedWebSocket;
  saveEvents();
})();
"""


@dataclass(frozen=True)
class Config:
    project_root: Path
    output_dir: Path
    server_url: str
    port: int
    max_seconds: int
    browser_executable: str | None
    cdp_url: str | None
    prefer_existing_browser: bool
    start_server: bool
    build_frontend: bool
    keep_open: bool


@dataclass(frozen=True)
class EventEntry:
    timestamp: str
    category: str
    message: str
    data: JsonObject | None = None

    def to_json(self) -> JsonObject:
        return {
            "timestamp": self.timestamp,
            "category": self.category,
            "message": self.message,
            "data": self.data,
        }


@dataclass(frozen=True)
class BugEntry:
    category: str
    description: str
    severity: Severity
    phase: str | None
    timestamp: str
    screenshot: str | None
    data: JsonObject | None

    def to_json(self) -> JsonObject:
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "screenshot": self.screenshot,
            "data": self.data,
        }


@dataclass(frozen=True)
class RoundEntry:
    index: int
    team0: str
    team1: str
    defender_points: int

    def to_json(self) -> JsonObject:
        return {
            "index": self.index,
            "team0": self.team0,
            "team1": self.team1,
            "defender_points": self.defender_points,
        }


@dataclass(frozen=True)
class UiAction:
    kind: UiActionKind
    card_ids: tuple[str, ...] = ()


@dataclass
class Recorder:
    output_dir: Path
    events: list[EventEntry] = field(default_factory=list[EventEntry])
    bugs: list[BugEntry] = field(default_factory=list[BugEntry])

    def event(
        self,
        category: str,
        message: str,
        data: JsonObject | None = None,
    ) -> None:
        entry = EventEntry(
            timestamp=utc_now(),
            category=category,
            message=message,
            data=data,
        )
        self.events.append(entry)
        print(f"[{category}] {message}", flush=True)

    def bug(
        self,
        category: str,
        description: str,
        severity: Severity,
        phase: str | None = None,
        screenshot: str | None = None,
        data: JsonObject | None = None,
    ) -> None:
        for existing in self.bugs:
            if (
                existing.category == category
                and existing.description == description
            ):
                return
        self.bugs.append(
            BugEntry(
                category=category,
                description=description,
                severity=severity,
                phase=phase,
                timestamp=utc_now(),
                screenshot=screenshot,
                data=data,
            )
        )
        self.event("bug", f"{severity}: {category}: {description}")


@dataclass
class ManagedServer:
    process: subprocess.Popen[str]
    log_file: TextIO

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log_file.close()


@dataclass(frozen=True)
class BrowserSession:
    playwright: PlaywrightLike
    browser: BrowserLike
    context: BrowserContextLike
    page: PageLike
    attached_existing_browser: bool


@dataclass(frozen=True)
class BrowserEndpoint:
    browser_id: str
    label: str
    port: int
    endpoint_url: str
    source: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_playwright_module() -> PlaywrightModuleLike:
    return cast(
        PlaywrightModuleLike,
        importlib.import_module("playwright.sync_api"),
    )


def playwright_exception_types() -> tuple[type[Exception], ...]:
    try:
        module = importlib.import_module("playwright.sync_api")
    except ModuleNotFoundError:
        return (Exception,)
    candidate: object = getattr(module, "Error", None)
    if isinstance(candidate, type) and issubclass(candidate, Exception):
        return (candidate,)
    return (Exception,)


PLAYWRIGHT_EXCEPTIONS = playwright_exception_types()


def is_console_message(value: object) -> TypeGuard[ConsoleMessageLike]:
    return hasattr(value, "type") and hasattr(value, "text")


def is_request(value: object) -> TypeGuard[RequestLike]:
    return hasattr(value, "method") and hasattr(value, "url")


def parse_json_object(raw: str) -> JsonObject | None:
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return cast(JsonObject, parsed)
    return None


def parse_json_list(raw: str) -> list[JsonValue]:
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return cast(list[JsonValue], parsed)
    return []


def json_string_list(values: Sequence[str]) -> list[JsonValue]:
    return [value for value in values]


def object_field(data: JsonObject, key: str) -> JsonObject | None:
    value = data.get(key)
    return value if isinstance(value, dict) else None


def list_field(data: JsonObject, key: str) -> list[JsonValue]:
    value = data.get(key)
    return value if isinstance(value, list) else []


def string_field(data: JsonObject, key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


def int_field(data: JsonObject, key: str) -> int | None:
    value = data.get(key)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def card_id(card: JsonObject) -> str:
    value = card.get("id")
    return value if isinstance(value, str) else ""


def card_rank(card: JsonObject) -> str:
    value = card.get("rank")
    return value if isinstance(value, str) else ""


def card_suit(card: JsonObject) -> str:
    value = card.get("suit")
    return value if isinstance(value, str) else ""


def card_list(value: JsonValue | None) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    result: list[JsonObject] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
    return result


def card_matrix(value: JsonValue | None) -> list[list[JsonObject]]:
    if not isinstance(value, list):
        return []
    result: list[list[JsonObject]] = []
    for row in value:
        cards = card_list(row)
        if cards:
            result.append(cards)
    return result


def rank_value(card: JsonObject) -> int:
    return RANK_ORDER.get(card_rank(card), -1)


def card_points(card: JsonObject) -> int:
    rank = card_rank(card)
    if rank == "5":
        return 5
    if rank in ("10", "K"):
        return 10
    return 0


def is_trump(
    card: JsonObject, trump_suit: str | None, trump_rank: str
) -> bool:
    return (
        card_suit(card) == "joker"
        or card_rank(card) == trump_rank
        or (trump_suit is not None and card_suit(card) == trump_suit)
    )


def choose_discard_cards(state: JsonObject) -> tuple[str, ...]:
    hand = card_list(state.get("player_hand"))
    stirring_state = object_field(state, "stirring_state")
    count = (
        int_field(stirring_state, "exchange_count")
        if stirring_state is not None
        else None
    )
    if count is None or count <= 0:
        count = 8
    trump_suit = string_field(state, "trump_suit")
    trump_rank = string_field(state, "trump_rank") or "2"

    def discard_key(card: JsonObject) -> tuple[int, int, int]:
        trump_weight = (
            1 if is_trump(card, trump_suit, trump_rank) else 0
        )
        return (trump_weight, card_points(card), rank_value(card))

    chosen = sorted(hand, key=discard_key)[:count]
    return tuple(card_id(card) for card in chosen if card_id(card))


def action_hint_candidates(state: JsonObject) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for candidate in card_matrix(state.get("action_hints")):
        card_ids = tuple(
            card_id(card) for card in candidate if card_id(card)
        )
        if card_ids:
            result.append(card_ids)
    return result


def choose_bid_cards(state: JsonObject) -> tuple[str, ...]:
    candidates = action_hint_candidates(state)
    if not candidates:
        return ()
    return candidates[0]


def choose_play_candidates(state: JsonObject) -> list[tuple[str, ...]]:
    hints = action_hint_candidates(state)
    if hints:
        return hints
    return pick_free_play_candidates(state)


def pick_free_play_candidates(
    state: JsonObject, limit: int = 80
) -> list[tuple[str, ...]]:
    hand = card_list(state.get("player_hand"))
    if not hand:
        return []

    lead_count = 1
    lead_cards: list[JsonObject] = []
    trick = object_field(state, "trick")
    if trick is not None:
        lead_player = int_field(trick, "lead_player")
        slots = list_field(trick, "slots")
        if lead_player is not None and 0 <= lead_player < len(slots):
            lead_slot_raw = slots[lead_player]
            if isinstance(lead_slot_raw, dict):
                lead_cards = card_list(lead_slot_raw.get("cards"))
                if lead_cards:
                    lead_count = len(lead_cards)

    if not lead_cards:
        return [
            (card_id(card),) for card in hand[:limit] if card_id(card)
        ]

    trump_suit = string_field(state, "trump_suit")
    trump_rank = string_field(state, "trump_rank") or "2"
    lead_effective_suit = effective_suit(
        lead_cards[0], trump_suit, trump_rank
    )
    same_suit_cards = [
        card
        for card in hand
        if effective_suit(card, trump_suit, trump_rank)
        == lead_effective_suit
    ]
    other_cards = [card for card in hand if card not in same_suit_cards]

    candidates: list[list[JsonObject]] = []
    if len(same_suit_cards) >= lead_count:
        candidates.extend(
            card_combinations_prefer_pairs(
                same_suit_cards, lead_count, limit
            )
        )
    elif same_suit_cards:
        needed = lead_count - len(same_suit_cards)
        for fill in card_combinations_prefer_pairs(
            other_cards, needed, limit
        ):
            candidates.append(same_suit_cards + fill)
            if len(candidates) >= limit:
                break
    else:
        candidates.extend(
            card_combinations_prefer_pairs(hand, lead_count, limit)
        )

    return dedupe_card_candidates(candidates, limit)


def card_combinations_prefer_pairs(
    cards: list[JsonObject],
    count: int,
    limit: int,
) -> list[list[JsonObject]]:
    if count <= 0:
        return [[]]
    if len(cards) < count:
        return []
    combos = [list(combo) for combo in combinations(cards, count)]
    if count == 2:
        combos.sort(
            key=lambda combo: (
                0 if same_rank_pair(combo[0], combo[1]) else 1
            )
        )
    return combos[:limit]


def same_rank_pair(first: JsonObject, second: JsonObject) -> bool:
    return card_suit(first) == card_suit(second) and card_rank(
        first
    ) == card_rank(second)


def dedupe_card_candidates(
    candidates: Sequence[Sequence[JsonObject]], limit: int
) -> list[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []
    for candidate in candidates:
        key = tuple(
            sorted(card_id(card) for card in candidate if card_id(card))
        )
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
        if len(result) >= limit:
            break
    return result


def effective_suit(
    card: JsonObject, trump_suit: str | None, trump_rank: str
) -> str:
    suit = card_suit(card)
    rank = card_rank(card)
    if (
        suit == "joker"
        or rank == trump_rank
        or (trump_suit is not None and suit == trump_suit)
    ):
        return "trump"
    return suit


def plan_action(
    state_message: JsonObject,
    rejected_play_candidates: set[tuple[str, ...]],
) -> UiAction | None:
    state = object_field(state_message, "state")
    if state is None:
        return None

    awaiting = string_field(state, "awaiting_action")

    if awaiting == "bid":
        cards = choose_bid_cards(state)
        return UiAction(kind="bid", card_ids=cards) if cards else None
    if awaiting == "next_round":
        return UiAction(kind="next_round")
    if awaiting == "stir":
        return UiAction(kind="stir_pass")
    if awaiting == "discard":
        cards = choose_discard_cards(state)
        return (
            UiAction(kind="discard", card_ids=cards) if cards else None
        )
    if awaiting == "play":
        for cards in choose_play_candidates(state):
            if cards not in rejected_play_candidates:
                return UiAction(kind="play", card_ids=cards)
        return None
    return None


def css_attr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def click_button(
    page: PageLike, name: str, timeout_ms: int = 5000
) -> None:
    page.get_by_role("button", name=name, exact=True).first.click(
        timeout=timeout_ms
    )


def click_clear_selection_if_available(page: PageLike) -> None:
    locator = page.get_by_role(
        "button", name=CLEAR_SELECTION_BUTTON, exact=True
    )
    try:
        if locator.count() > 0 and locator.first.is_enabled(
            timeout=500
        ):
            locator.first.click(timeout=1000)
    except PLAYWRIGHT_EXCEPTIONS:
        return


def click_cards(page: PageLike, card_ids: Sequence[str]) -> None:
    click_clear_selection_if_available(page)
    for card in card_ids:
        selector = (
            f'.hand-view .card[data-card-id="{css_attr_value(card)}"]'
        )
        locator = page.locator(selector).first
        locator.scroll_into_view_if_needed(timeout=3000)
        locator.click(timeout=3000)


def click_first_bid_option(page: PageLike) -> None:
    locator = page.locator(BID_ACTION_BUTTON_SELECTOR).first
    locator.scroll_into_view_if_needed(timeout=3000)
    locator.click(timeout=3000)


def click_next_round(page: PageLike) -> None:
    overlay_button = page.locator(NEXT_ROUND_BUTTON_SELECTOR)
    if overlay_button.count() > 0:
        overlay_button.first.scroll_into_view_if_needed(timeout=3000)
        overlay_button.first.click(timeout=3000)
        return
    click_button(page, "下一轮")


def perform_action(
    page: PageLike, action: UiAction, recorder: Recorder
) -> bool:
    try:
        if action.kind == "bid":
            click_first_bid_option(page)
        elif action.kind == "next_round":
            click_next_round(page)
        elif action.kind == "stir_pass":
            click_button(page, "不反")
        elif action.kind == "discard":
            click_cards(page, action.card_ids)
            click_button(page, "换底牌")
        elif action.kind == "play":
            click_cards(page, action.card_ids)
            click_button(page, "出牌")
    except PLAYWRIGHT_EXCEPTIONS as error:
        recorder.bug(
            category="ui_action_failed",
            description=f"{action.kind}: {error}",
            severity="high",
        )
        return False
    recorder.event(
        "ui_action",
        f"{action.kind} {','.join(action.card_ids)}".strip(),
        {
            "kind": action.kind,
            "card_ids": json_string_list(action.card_ids),
        },
    )
    return True


def page_string(page: PageLike, expression: str) -> str | None:
    value: object = page.evaluate(expression)
    return value if isinstance(value, str) else None


def latest_state(page: PageLike) -> JsonObject | None:
    raw = page_string(
        page, "() => window.__TRACTOR_LAST_STATE_JSON || ''"
    )
    if raw is None or raw == "":
        return None
    return parse_json_object(raw)


def drain_browser_errors(page: PageLike) -> list[str]:
    raw = page_string(
        page,
        """() => {
            const raw = window.__TRACTOR_ERRORS_JSON || "[]";
            window.__TRACTOR_ERRORS = [];
            window.__TRACTOR_ERRORS_JSON = "[]";
            return raw;
        }""",
    )
    if raw is None:
        return []
    values = parse_json_list(raw)
    result: list[str] = []
    for value in values:
        if isinstance(value, dict):
            text = string_field(value, "text")
            if text is not None and text.strip():
                result.append(text)
    return result


def browser_events(page: PageLike) -> list[JsonValue]:
    try:
        raw = page_string(
            page, "() => window.__TRACTOR_EVENTS_JSON || '[]'"
        )
    except PLAYWRIGHT_EXCEPTIONS:
        return []
    return parse_json_list(raw) if raw is not None else []


def wait_for_initial_state(
    page: PageLike, timeout_seconds: int, recorder: Recorder
) -> JsonObject | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            state = latest_state(page)
        except PLAYWRIGHT_EXCEPTIONS as error:
            recorder.event("state_wait", f"state read failed: {error}")
            state = None
        if state is not None:
            return state
        time.sleep(0.25)
    return None


def take_screenshot(
    page: PageLike, output_dir: Path, name: str, recorder: Recorder
) -> str | None:
    path = output_dir / "screenshots" / f"{slug(name)}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        page.screenshot(path=str(path), full_page=False)
    except PLAYWRIGHT_EXCEPTIONS as error:
        recorder.event("screenshot", f"failed: {error}")
        return None
    recorder.event("screenshot", str(path))
    return str(path)


def slug(value: str) -> str:
    chars: list[str] = []
    for char in value:
        if char.isalnum() or char in ("-", "_"):
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "screenshot"


def run_build(config: Config, recorder: Recorder) -> None:
    recorder.event("build", "deno task build")
    subprocess.run(
        ["deno", "task", "build"],
        cwd=config.project_root,
        check=True,
    )


def server_ready(server_url: str) -> bool:
    try:
        with urlopen(
            Request(f"{server_url}/docs"), timeout=1
        ) as response:
            return response.status == 200
    except OSError, URLError, TimeoutError:
        return False


def start_server(config: Config, recorder: Recorder) -> ManagedServer:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = (config.output_dir / "uvicorn.log").open(
        "w", encoding="utf-8"
    )
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.web.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--ws",
        "websockets-sansio",
    ]
    recorder.event(
        "server",
        "starting uvicorn",
        {"command": json_string_list(command)},
    )
    process: subprocess.Popen[str] = subprocess.Popen(
        command,
        cwd=config.project_root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    managed = ManagedServer(process=process, log_file=log_file)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            managed.stop()
            raise SystemExit(
                f"uvicorn exited early with code {process.returncode}"
            )
        if server_ready(config.server_url):
            recorder.event("server", f"ready at {config.server_url}")
            return managed
        time.sleep(0.5)
    managed.stop()
    raise SystemExit(
        f"server did not become ready at {config.server_url}"
    )


def find_chromium(explicit_path: str | None) -> str | None:
    if explicit_path is not None:
        return explicit_path
    for candidate in ("chromium", "chromium-browser", "google-chrome"):
        path = shutil.which(candidate)
        if path is not None:
            return path
    return None


def cdp_url_ready(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme in ("ws", "wss"):
        port = parsed.port
        host = parsed.hostname or "127.0.0.1"
        return port is not None and tcp_port_ready(host, port)

    if parsed.scheme in ("http", "https"):
        port = parsed.port
        host = parsed.hostname or "127.0.0.1"
        if port is not None and tcp_port_ready(host, port):
            return True

    version_url = f"{url.rstrip('/')}/json/version"
    try:
        with urlopen(Request(version_url), timeout=0.5) as response:
            return response.status == 200
    except OSError, URLError, TimeoutError:
        return False


def tcp_port_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def known_devtools_paths() -> list[tuple[str, str, Path]]:
    home = Path.home()
    if sys.platform == "darwin":
        return [
            (
                "chrome",
                "Chrome",
                home
                / "Library/Application Support"
                / "Google"
                / "Chrome"
                / "DevToolsActivePort",
            ),
            (
                "chrome-canary",
                "Chrome Canary",
                home
                / "Library/Application Support"
                / "Google"
                / "Chrome Canary"
                / "DevToolsActivePort",
            ),
            (
                "chromium",
                "Chromium",
                home
                / "Library/Application Support"
                / "Chromium"
                / "DevToolsActivePort",
            ),
            (
                "edge",
                "Microsoft Edge",
                home
                / "Library/Application Support"
                / "Microsoft Edge"
                / "DevToolsActivePort",
            ),
        ]
    if sys.platform.startswith("linux"):
        return [
            (
                "chrome",
                "Chrome",
                home / ".config/google-chrome/DevToolsActivePort",
            ),
            (
                "chromium",
                "Chromium",
                home / ".config/chromium/DevToolsActivePort",
            ),
            (
                "edge",
                "Microsoft Edge",
                home / ".config/microsoft-edge/DevToolsActivePort",
            ),
        ]
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data is None:
            return []
        root = Path(local_app_data)
        return [
            (
                "chrome",
                "Chrome",
                root / "Google/Chrome/User Data/DevToolsActivePort",
            ),
            (
                "chromium",
                "Chromium",
                root / "Chromium/User Data/DevToolsActivePort",
            ),
            (
                "edge",
                "Microsoft Edge",
                root / "Microsoft/Edge/User Data/DevToolsActivePort",
            ),
        ]
    return []


def discover_devtools_endpoints() -> list[BrowserEndpoint]:
    result: list[BrowserEndpoint] = []
    for browser_id, label, path in known_devtools_paths():
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]
        if not lines:
            continue
        try:
            port = int(lines[0])
        except ValueError:
            continue
        if port <= 0 or port >= 65536:
            continue
        if not tcp_port_ready("127.0.0.1", port):
            continue
        ws_path = lines[1] if len(lines) > 1 else ""
        endpoint_url = (
            f"ws://127.0.0.1:{port}{ws_path}"
            if ws_path
            else f"http://127.0.0.1:{port}"
        )
        result.append(
            BrowserEndpoint(
                browser_id=browser_id,
                label=label,
                port=port,
                endpoint_url=endpoint_url,
                source=str(path),
            )
        )
    return result


def running_chromium_commands() -> list[str]:
    commands: list[str] = []
    for process_name in (
        "chromium",
        "chromium-browser",
        "google-chrome",
    ):
        result = subprocess.run(
            ["pgrep", "-a", process_name],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            continue
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line or " --type=" in line:
                continue
            if line not in commands:
                commands.append(line)
    return commands


def choose_existing_cdp_url(
    config: Config, recorder: Recorder
) -> str | None:
    if not config.prefer_existing_browser:
        return None
    if config.cdp_url is not None:
        if cdp_url_ready(config.cdp_url):
            return config.cdp_url
        raise SystemExit(
            f"CDP endpoint is not reachable: {config.cdp_url}. "
            "Start Chromium with --remote-debugging-port=9222 or omit"
            "--cdp-url."
        )
    endpoints = discover_devtools_endpoints()
    if endpoints:
        endpoint = endpoints[0]
        recorder.event(
            "browser",
            f"using existing {endpoint.label} at port {endpoint.port}",
            {
                "browser_id": endpoint.browser_id,
                "endpoint_url": endpoint.endpoint_url,
                "source": endpoint.source,
            },
        )
        return endpoint.endpoint_url
    for port in (9222, 9223, 9224):
        candidate = f"http://127.0.0.1:{port}"
        if cdp_url_ready(candidate):
            recorder.event(
                "browser", f"using existing Chromium at {candidate}"
            )
            return candidate
    running_commands = running_chromium_commands()
    if running_commands:
        command_preview = running_commands[0]
        raise SystemExit(
            "Chromium is already running, but it does not expose a"
            "reachable"
            "remote debugging endpoint on 9222/9223/9224.\n"
            f"Detected Chromium: {command_preview}\n"
            "Playwright cannot attach to a normal Chromium process"
            "after it"
            "has started. To reuse your browser window, fully quit"
            "Chromium"
            "and start it with:\n"
            "  chromium --remote-debugging-port=9222\n"
            "Then run this script again. To intentionally launch a"
            "separate"
            "visible browser, pass --new-browser."
        )
    recorder.event(
        "browser",
        "no running Chromium found; launching a new visible Chromium",
    )
    return None


def setup_browser(config: Config, recorder: Recorder) -> BrowserSession:
    if sys.platform.startswith("linux"):
        os.environ.setdefault("DISPLAY", ":0")

    playwright = load_playwright_module().sync_playwright().start()
    cdp_url = choose_existing_cdp_url(config, recorder)
    if cdp_url is not None:
        browser = playwright.chromium.connect_over_cdp(cdp_url)
        if browser.contexts:
            context = browser.contexts[0]
        else:
            context = browser.new_context(
                viewport={"width": 1400, "height": 950}
            )
        context.add_init_script(
            script="localStorage.removeItem('tractor-game-id');"
        )
        context.add_init_script(script=INJECTED_SCRIPT)
        page = context.new_page()
        return BrowserSession(
            playwright=playwright,
            browser=browser,
            context=context,
            page=page,
            attached_existing_browser=True,
        )

    executable = find_chromium(config.browser_executable)
    browser = playwright.chromium.launch(
        headless=False,
        executable_path=executable,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--window-size=1400,950",
        ],
    )
    context = browser.new_context(
        viewport={"width": 1400, "height": 950}
    )
    context.add_init_script(
        script="localStorage.removeItem('tractor-game-id');"
    )
    context.add_init_script(script=INJECTED_SCRIPT)
    page = context.new_page()
    return BrowserSession(
        playwright=playwright,
        browser=browser,
        context=context,
        page=page,
        attached_existing_browser=False,
    )


def attach_page_logging(page: PageLike, recorder: Recorder) -> None:
    def on_console(message: object) -> None:
        if not is_console_message(message):
            recorder.event("console", str(message))
            return
        recorder.event(
            "console",
            f"[{message.type}] {message.text}",
            {"type": message.type, "text": message.text},
        )

    def on_page_error(error: object) -> None:
        recorder.bug("page_error", str(error), "high")

    def on_request_failed(request: object) -> None:
        if not is_request(request):
            recorder.bug("request_failed", str(request), "medium")
            return
        recorder.bug(
            "request_failed",
            f"{request.method} {request.url}",
            "medium",
            data={"url": request.url, "method": request.method},
        )

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    page.on("requestfailed", on_request_failed)


def int_from_state(data: JsonObject, key: str, default: int) -> int:
    value = int_field(data, key)
    return value if value is not None else default


def run_playthrough(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(output_dir=config.output_dir)
    managed_server: ManagedServer | None = None
    browser_session: BrowserSession | None = None
    page: PageLike | None = None
    rounds: list[RoundEntry] = []
    final_state: JsonObject | None = None
    game_completed = False
    start_time = time.monotonic()

    try:
        if config.build_frontend:
            run_build(config, recorder)
        if config.start_server:
            managed_server = start_server(config, recorder)

        browser_session = setup_browser(config, recorder)
        page = browser_session.page
        attach_page_logging(page, recorder)
        recorder.event("navigation", config.server_url)
        page.goto(config.server_url, wait_until="domcontentloaded")

        initial_state = wait_for_initial_state(
            page, timeout_seconds=30, recorder=recorder
        )
        if initial_state is None:
            screenshot = take_screenshot(
                page, config.output_dir, "no_initial_state", recorder
            )
            recorder.bug(
                "initial_state_timeout",
                "browser did not receive a state message within 30"
                "seconds",
                "critical",
                screenshot=screenshot,
            )
            return

        take_screenshot(
            page, config.output_dir, "initial_state", recorder
        )

        last_phase: str | None = None
        last_phase_time = time.monotonic()
        last_screenshot_time = time.monotonic()
        last_action_seq: int | None = None
        last_action: UiAction | None = None
        rejected_play_candidates: set[tuple[str, ...]] = set()
        last_team0_level = "2"
        last_team1_level = "2"
        phase_stuck_seconds = 180
        action_count = 0

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > config.max_seconds:
                screenshot = take_screenshot(
                    page, config.output_dir, "timeout", recorder
                )
                recorder.bug(
                    "timeout",
                    f"game did not finish within {config.max_seconds}"
                    f"seconds",
                    "critical",
                    phase=last_phase,
                    screenshot=screenshot,
                )
                break

            state_message = latest_state(page)
            if state_message is None:
                time.sleep(0.25)
                continue

            final_state = state_message
            state = object_field(state_message, "state")
            if state is None:
                recorder.bug(
                    "bad_state_message",
                    "state message missing state object",
                    "critical",
                )
                break

            seq = int_field(state_message, "seq")
            phase = string_field(state, "phase") or "UNKNOWN"
            team0_level = string_field(state, "team0_level") or "?"
            team1_level = string_field(state, "team1_level") or "?"
            defender_points = int_from_state(
                state, "defender_points", 0
            )

            for text in drain_browser_errors(page):
                if last_action is not None:
                    last_action_seq = None
                    if last_action.kind == "play":
                        rejected_play_candidates.add(
                            last_action.card_ids
                        )
                        if "出牌" in text or "play" in text.lower():
                            recorder.event(
                                "candidate_rejected",
                                text,
                                {
                                    "kind": last_action.kind,
                                    "card_ids": json_string_list(
                                        last_action.card_ids
                                    ),
                                },
                            )
                            continue
                    if last_action.kind == "bid":
                        recorder.event(
                            "bid_error_toast",
                            text,
                            {
                                "kind": last_action.kind,
                                "card_ids": json_string_list(
                                    last_action.card_ids
                                ),
                            },
                        )
                recorder.bug(
                    "browser_error_toast", text, "medium", phase=phase
                )

            if (
                team0_level != last_team0_level
                or team1_level != last_team1_level
            ):
                rounds.append(
                    RoundEntry(
                        index=len(rounds) + 1,
                        team0=f"{last_team0_level}->{team0_level}",
                        team1=f"{last_team1_level}->{team1_level}",
                        defender_points=defender_points,
                    )
                )
                recorder.event(
                    "round",
                    f"{last_team0_level}->{team0_level},"
                    f"{last_team1_level}->{team1_level}",
                )
                last_team0_level = team0_level
                last_team1_level = team1_level

            if phase != last_phase:
                recorder.event(
                    "phase",
                    phase,
                    {
                        "seq": seq,
                        "awaiting": string_field(
                            state, "awaiting_action"
                        ),
                        "team0_level": team0_level,
                        "team1_level": team1_level,
                        "defender_points": defender_points,
                    },
                )
                take_screenshot(
                    page,
                    config.output_dir,
                    f"phase_{phase}_{int(elapsed)}",
                    recorder,
                )
                last_phase = phase
                last_phase_time = time.monotonic()
                rejected_play_candidates.clear()

            if time.monotonic() - last_phase_time > phase_stuck_seconds:
                screenshot = take_screenshot(
                    page, config.output_dir, f"stuck_{phase}", recorder
                )
                recorder.bug(
                    "phase_stuck",
                    f"phase {phase} did not change for"
                    f"{phase_stuck_seconds} seconds",
                    "critical",
                    phase=phase,
                    screenshot=screenshot,
                )
                break

            if time.monotonic() - last_screenshot_time > 30:
                take_screenshot(
                    page,
                    config.output_dir,
                    f"tick_{int(elapsed)}",
                    recorder,
                )
                last_screenshot_time = time.monotonic()

            if state.get("winning_team") is not None:
                game_completed = True
                take_screenshot(
                    page, config.output_dir, "game_over", recorder
                )
                recorder.event(
                    "result",
                    "game over",
                    {
                        "winning_team": state.get("winning_team"),
                        "team0_level": team0_level,
                        "team1_level": team1_level,
                    },
                )
                break

            if seq is not None and seq != last_action_seq:
                action = plan_action(
                    state_message, rejected_play_candidates
                )
                if action is not None and perform_action(
                    page, action, recorder
                ):
                    last_action_seq = seq
                    last_action = action
                    action_count += 1

            time.sleep(0.25)

        recorder.event(
            "summary",
            f"actions={action_count}, rounds={len(rounds)},"
            f"bugs={len(recorder.bugs)}",
        )
    finally:
        browser_event_log: list[JsonValue] = []
        if page is not None:
            browser_event_log = browser_events(page)
            write_reports(
                config=config,
                recorder=recorder,
                rounds=rounds,
                final_state=final_state,
                game_completed=game_completed,
                duration_seconds=time.monotonic() - start_time,
                browser_event_log=browser_event_log,
            )
        if config.keep_open and page is not None:
            recorder.event(
                "keep_open",
                "press Ctrl+C in this terminal to close the browser",
            )
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        if browser_session is not None:
            if browser_session.attached_existing_browser:
                if not config.keep_open:
                    try:
                        browser_session.page.close()
                    except PLAYWRIGHT_EXCEPTIONS:
                        pass
            else:
                browser_session.context.close()
                browser_session.browser.close()
            browser_session.playwright.stop()
        if managed_server is not None:
            managed_server.stop()


def write_reports(
    config: Config,
    recorder: Recorder,
    rounds: Sequence[RoundEntry],
    final_state: JsonObject | None,
    game_completed: bool,
    duration_seconds: float,
    browser_event_log: Sequence[JsonValue],
) -> None:
    report_path = config.output_dir / "playthrough.json"
    markdown_path = config.output_dir / "playthrough.md"
    report = {
        "meta": {
            "time": utc_now(),
            "server_url": config.server_url,
            "duration_seconds": duration_seconds,
            "browser": "chromium",
            "headless": False,
            "game_completed": game_completed,
        },
        "events": [entry.to_json() for entry in recorder.events],
        "browser_events": list(browser_event_log),
        "bugs": [bug.to_json() for bug in recorder.bugs],
        "rounds": [entry.to_json() for entry in rounds],
        "final_state": final_state,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_lines: list[str] = [
        "# Tractor Playwright Playthrough",
        "",
        f"- Time: {utc_now()}",
        f"- Server: {config.server_url}",
        f"- Duration: {duration_seconds:.1f}s",
        "- Headless: false",
        f"- Game completed: {str(game_completed).lower()}",
        f"- Rounds: {len(rounds)}",
        f"- Bugs: {len(recorder.bugs)}",
        f"- JSON report: `{report_path}`",
        f"- Screenshots: `{config.output_dir / 'screenshots'}`",
        "",
    ]
    if rounds:
        markdown_lines.extend(
            [
                "## Rounds",
                "",
                "| Round | Team 0 | Team 1 | Defender points |",
                "| --- | --- | --- | --- |",
            ]
        )
        for entry in rounds:
            markdown_lines.append(
                f"| {entry.index} | {entry.team0} | {entry.team1} |"
                f"{entry.defender_points} |"
            )
        markdown_lines.append("")

    markdown_lines.extend(["## Bugs", ""])
    if recorder.bugs:
        for index, bug in enumerate(recorder.bugs, start=1):
            markdown_lines.append(
                f"{index}. [{bug.severity}] {bug.category}:"
                f"{bug.description}"
            )
            if bug.phase is not None:
                markdown_lines.append(f"   Phase: {bug.phase}")
            if bug.screenshot is not None:
                markdown_lines.append(
                    f"   Screenshot: `{bug.screenshot}`"
                )
    else:
        markdown_lines.append(
            "No bugs recorded by the playthrough runner."
        )
    markdown_lines.append("")
    markdown_path.write_text(
        "\n".join(markdown_lines), encoding="utf-8"
    )
    print(f"Report written to {markdown_path}", flush=True)


def _env_bool(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return Path(value)


def _explicitly_selected_by_pytest() -> bool:
    for arg in sys.argv[1:]:
        if Path(arg).name == THIS_FILE:
            return True
    return False


def _should_run_playthrough() -> bool:
    return _explicitly_selected_by_pytest() or _env_bool(
        "TRACTOR_PLAYTHROUGH"
    )


def _load_report(path: Path) -> JsonObject:
    parsed = parse_json_object(path.read_text(encoding="utf-8"))
    assert parsed is not None, (
        f"playthrough report is not valid JSON: {path}"
    )
    return parsed


def _card(card_id_value: str, suit: str, rank: str) -> JsonObject:
    return {"id": card_id_value, "suit": suit, "rank": rank}


def _state_message(awaiting: str, state: JsonObject) -> JsonObject:
    return {"seq": 1, "state": {**state, "awaiting_action": awaiting}}


def test_plan_action_bids_first_hint_candidate() -> None:
    state: JsonObject = {
        "action_hints": [
            [_card("D1-spades-2", "spades", "2")],
            [
                _card("D1-hearts-2", "hearts", "2"),
                _card("D2-hearts-2", "hearts", "2"),
            ],
        ],
        "player_hand": [
            _card("D1-spades-2", "spades", "2"),
            _card("D1-hearts-2", "hearts", "2"),
            _card("D2-hearts-2", "hearts", "2"),
        ],
    }

    action = plan_action(_state_message("bid", state), set())

    assert action == UiAction(kind="bid", card_ids=("D1-spades-2",))


def test_plan_action_waits_for_auto_pass_without_bid_hints() -> None:
    state: JsonObject = {
        "action_hints": [],
        "player_hand": [_card("D1-spades-7", "spades", "7")],
    }

    action = plan_action(_state_message("bid", state), set())

    assert action is None


def test_plan_action_uses_first_play_hint_candidate() -> None:
    state: JsonObject = {
        "action_hints": [
            [_card("D1-diamonds-5", "diamonds", "5")],
            [_card("D1-diamonds-K", "diamonds", "K")],
        ],
        "player_hand": [
            _card("D1-diamonds-5", "diamonds", "5"),
            _card("D1-diamonds-K", "diamonds", "K"),
        ],
    }

    action = plan_action(_state_message("play", state), set())

    assert action == UiAction(kind="play", card_ids=("D1-diamonds-5",))


def test_plan_action_skips_rejected_play_hint_candidate() -> None:
    state: JsonObject = {
        "action_hints": [
            [_card("D1-diamonds-5", "diamonds", "5")],
            [_card("D1-diamonds-K", "diamonds", "K")],
        ],
        "player_hand": [
            _card("D1-diamonds-5", "diamonds", "5"),
            _card("D1-diamonds-K", "diamonds", "K"),
        ],
    }

    action = plan_action(
        _state_message("play", state), {("D1-diamonds-5",)}
    )

    assert action == UiAction(kind="play", card_ids=("D1-diamonds-K",))


def test_plan_action_free_leads_when_play_has_no_hints() -> None:
    state: JsonObject = {
        "action_hints": [],
        "player_hand": [
            _card("D1-clubs-3", "clubs", "3"),
            _card("D1-spades-A", "spades", "A"),
        ],
        "trick": {
            "lead_player": 3,
            "slots": [{}, {}, {}, {}],
        },
    }

    action = plan_action(_state_message("play", state), set())

    assert action == UiAction(kind="play", card_ids=("D1-clubs-3",))


@pytest.mark.playthrough
def test_visible_playwright_full_game_playthrough() -> None:
    if not _should_run_playthrough():
        pytest.skip(
            "visible browser playthrough is skipped by default; run "
            "`python -m pytest tests/test_playthrough_playwright.py"
            "-s`"
            "or set TRACTOR_PLAYTHROUGH=1"
        )

    port = _env_int("TRACTOR_PLAYTHROUGH_PORT", 8787)
    output_dir = _env_path(
        "TRACTOR_PLAYTHROUGH_OUTPUT_DIR", DEFAULT_OUTPUT_DIR
    )
    config = Config(
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        server_url=f"http://127.0.0.1:{port}",
        port=port,
        max_seconds=_env_int(
            "TRACTOR_PLAYTHROUGH_MAX_SECONDS", DEFAULT_MAX_SECONDS
        ),
        browser_executable=os.environ.get(
            "TRACTOR_PLAYTHROUGH_BROWSER"
        ),
        cdp_url=os.environ.get("TRACTOR_PLAYTHROUGH_CDP_URL"),
        prefer_existing_browser=not _env_bool(
            "TRACTOR_PLAYTHROUGH_NEW_BROWSER"
        ),
        start_server=True,
        build_frontend=not _env_bool("TRACTOR_PLAYTHROUGH_NO_BUILD"),
        keep_open=_env_bool("TRACTOR_PLAYTHROUGH_KEEP_OPEN"),
    )

    run_playthrough(config)

    report = _load_report(output_dir / "playthrough.json")
    meta = object_field(report, "meta")
    assert meta is not None
    bugs = list_field(report, "bugs")
    assert meta.get("game_completed") is True
    assert bugs == []
