import { assertEquals, assertNotEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderGameOverOverlay } from "../ui/components/game-over-overlay.ts";
import type { StateSnapshot } from "../core/types.ts";

const doc = new DOMParser().parseFromString(
  `<html><body><div id="app"></div></body></html>`,
  "text/html",
);
// @ts-ignore test setup
globalThis.document = doc;

function makeSnapshot(overrides: Partial<StateSnapshot> = {}): StateSnapshot {
  return {
    phase: "GAME_OVER",
    player_hand: [],
    bottom_cards: [],
    trump_rank: "2",
    trump_suit: null,
    declarer_team: null,
    declarer_player: null,
    current_player: 0,
    defender_points: 0,
    legal_actions: [],
    trick: null,
    trick_history: [],
    bid_events: [],
    bid_winner: null,
    awaiting_action: null,
    stirring_state: null,
    exchange_state: null,
    scoring: null,
    winning_team: 0,
    team0_level: "A",
    team1_level: "5",
    ...overrides,
  };
}

Deno.test("test_renderGameOverOverlay_shows_winner", () => {
  const snap = makeSnapshot({ winning_team: 0 });
  const el = renderGameOverOverlay(snap);
  const winnerEl = el.querySelector(".winner-text");
  assertNotEquals(winnerEl, null);
  const text = winnerEl?.textContent ?? "";
  assertEquals(text.includes("队伍0获胜!"), true);
});

Deno.test("test_renderGameOverOverlay_team1_wins", () => {
  const snap = makeSnapshot({ winning_team: 1 });
  const el = renderGameOverOverlay(snap);
  const winnerEl = el.querySelector(".winner-text");
  assertNotEquals(winnerEl, null);
  const text = winnerEl?.textContent ?? "";
  assertEquals(text.includes("队伍1获胜!"), true);
});

Deno.test("test_renderGameOverOverlay_null_winning_team", () => {
  const snap = makeSnapshot({ winning_team: null });
  const el = renderGameOverOverlay(snap);
  const winnerEl = el.querySelector(".winner-text");
  assertNotEquals(winnerEl, null);
  const text = winnerEl?.textContent ?? "";
  assertEquals(text.includes("游戏结束"), true);
});

Deno.test("test_renderGameOverOverlay_new_game_button", () => {
  const snap = makeSnapshot();
  let newGameCalled = false;
  const onNewGame = () => { newGameCalled = true; };
  const el = renderGameOverOverlay(snap, onNewGame);
  const buttons = el.querySelectorAll("button");
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("新游戏"), true);
});

Deno.test("test_renderGameOverOverlay_no_button_without_callback", () => {
  const snap = makeSnapshot();
  const el = renderGameOverOverlay(snap);
  const buttons = el.querySelectorAll("button");
  assertEquals(buttons.length, 0);
});

Deno.test("test_renderGameOverOverlay_new_game_callback", () => {
  const snap = makeSnapshot();
  let newGameCalled = false;
  const onNewGame = () => { newGameCalled = true; };
  const el = renderGameOverOverlay(snap, onNewGame);
  const buttons = el.querySelectorAll("button");
  const newGameButton = Array.from(buttons).find((b) => b.textContent === "新游戏");
  assertNotEquals(newGameButton, undefined);
  newGameButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(newGameCalled, true);
});
