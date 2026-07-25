import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderGameOverOverlay } from "../ui/components/game-over-overlay.ts";
import type { StateSnapshot } from "../core/types.ts";

const doc = new DOMParser().parseFromString(
  `<html><body><div id="app"></div></body></html>`,
  "text/html",
);
// @ts-ignore test setup
globalThis.document = doc;

function makeSnapshot(
  overrides: Partial<StateSnapshot> = {},
): StateSnapshot {
  return {
    phase: "WAITING",
    round_number: 1,
    hand: [],
    bottom_cards: [],
    trump: { kind: "no_trump", rank: "2" },
    declarer: null,
    defender_points: 0,
    action_hints: [],
    trick: null,
    last_completed_trick: null,
    defender_point_cards: [],
    bid_events: [],
    bid_winner: null,
    stir_events: [],
    own_initial_bottom_exchange: null,
    awaiting_action: null,
    stirring_state: null,
    scoring: null,
    winning_partnership: "first",
    partnership_levels: { first: "A", second: "5" },
    remaining_cards: { a: 13, b: 13, c: 13, d: 13 },
    mandatory_levels: ["A"],
    next_round_confirmed: [],
    ...overrides,
  };
}

Deno.test("test_renderGameOverOverlay_shows_winner", () => {
  const snap = makeSnapshot({ winning_partnership: "first" });
  const el = renderGameOverOverlay(snap, "c");
  const winnerEl = el.querySelector(".winner-text");
  assertNotEquals(winnerEl, null);
  const text = winnerEl?.textContent ?? "";
  assertEquals(text.includes("我们赢了！"), true);
});

Deno.test("test_renderGameOverOverlay_other_partnership_wins", () => {
  const snap = makeSnapshot({ winning_partnership: "second" });
  const el = renderGameOverOverlay(snap, "c");
  const winnerEl = el.querySelector(".winner-text");
  assertNotEquals(winnerEl, null);
  const text = winnerEl?.textContent ?? "";
  assertEquals(text.includes("对方获胜"), true);
});

Deno.test("test_renderGameOverOverlay_null_winning_partnership", () => {
  const snap = makeSnapshot({ winning_partnership: null });
  const el = renderGameOverOverlay(snap, "c");
  const winnerEl = el.querySelector(".winner-text");
  assertNotEquals(winnerEl, null);
  const text = winnerEl?.textContent ?? "";
  assertEquals(text.includes("游戏结束"), true);
});

Deno.test("test_renderGameOverOverlay_new_game_button", () => {
  const snap = makeSnapshot();
  const el = renderGameOverOverlay(snap, "c", () => {});
  const buttons = el.querySelectorAll("button");
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("新游戏"), true);
});

Deno.test("test_renderGameOverOverlay_no_button_without_callback", () => {
  const snap = makeSnapshot();
  const el = renderGameOverOverlay(snap, "c");
  const buttons = el.querySelectorAll("button");
  assertEquals(buttons.length, 0);
});

Deno.test("test_renderGameOverOverlay_new_game_callback", () => {
  const snap = makeSnapshot();
  let newGameCalled = false;
  const onNewGame = () => {
    newGameCalled = true;
  };
  const el = renderGameOverOverlay(snap, "c", onNewGame);
  const buttons = el.querySelectorAll("button");
  const newGameButton = Array.from(buttons).find((b) =>
    b.textContent === "新游戏"
  );
  assertNotEquals(newGameButton, undefined);
  newGameButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(newGameCalled, true);
});

Deno.test("test_renderGameOverOverlay_uses_viewer_team", () => {
  const snap = makeSnapshot({ winning_partnership: "second" });
  const el = renderGameOverOverlay(snap, "b");
  const text = el.querySelector(".winner-text")?.textContent ?? "";
  assertEquals(text.includes("我们赢了！"), true);
});
