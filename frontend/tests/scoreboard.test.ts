import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderScoreboard } from "../ui/components/scoreboard.ts";
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
    phase: "PLAYING",
    player_hand: [],
    bottom_cards: [],
    trump_rank: "2",
    trump_suit: null,
    declarer_team: null,
    declarer_player: null,
    defender_points: 15,
    action_hints: [],
    trick: null,
    last_completed_trick: null,
    defender_point_cards: [],
    failed_throw: null,
    bid_events: [],
    bid_winner: null,
    awaiting_action: null,
    stirring_state: null,
    scoring: null,
    winning_team: null,
    team0_level: "3",
    team1_level: "5",
    player_hand_counts: [13, 13, 13, 13],
    next_round_confirmed: [],
    ...overrides,
  };
}

Deno.test("test_renderScoreboard_shows_levels", () => {
  const snap = makeSnapshot();
  const el = renderScoreboard(snap);
  const text = el.textContent ?? "";
  assertEquals(text.includes("3"), true);
  assertEquals(text.includes("5"), true);
});

Deno.test("test_renderScoreboard_has_chat_box_placeholder", () => {
  const snap = makeSnapshot();
  const el = renderScoreboard(snap);
  const input = el.querySelector(".scoreboard__chat-input");
  assertEquals(input !== null, true);
  assertEquals(input?.getAttribute("disabled"), "true");
});

Deno.test("test_renderScoreboard_has_no_operation_tabs_or_duplicate_table_info", () => {
  const snap = makeSnapshot({
    defender_points: 25,
    bid_events: [{
      player: 1,
      cards: [{ id: "D1-hearts-2", suit: "hearts", rank: "2" }],
      kind: "trump_rank",
      suit: "hearts",
      joker_type: null,
      count: 1,
    }],
  });
  const el = renderScoreboard(snap);
  const buttons = el.querySelectorAll("button");
  const text = el.textContent ?? "";
  assertEquals(buttons.length, 0);
  assertEquals(text.includes("防守方得分"), false);
  assertEquals(text.includes("主牌"), false);
  assertEquals(text.includes("上一墩"), false);
  assertEquals(text.includes("底牌"), false);
  assertEquals(text.includes("抢主记录"), false);
  assertEquals(text.includes("♥2"), false);
  assertEquals(text.includes("25"), false);
});

Deno.test("test_renderScoreboard_stirring_status_uses_stirring_phase", () => {
  const exchangingSnap = makeSnapshot({
    phase: "STIRRING",
    awaiting_action: null,
    stirring_state: {
      phase: "EXCHANGING",
      trump_suit: "spades",
      current_player: 1,
      declarer_player: 0,
      exchanging_player: 1,
      exchange_count: 8,
    },
  });
  const exchangingEl = renderScoreboard(exchangingSnap);
  const exchangingText = exchangingEl.textContent ?? "";
  assertEquals(exchangingText.includes("换底牌"), true);
  assertEquals(exchangingText.includes("待反主"), false);

  const waitingSnap = makeSnapshot({
    phase: "STIRRING",
    awaiting_action: null,
    stirring_state: {
      phase: "WAITING",
      trump_suit: "spades",
      current_player: 2,
      declarer_player: 0,
      exchanging_player: null,
      exchange_count: null,
    },
  });
  const waitingEl = renderScoreboard(waitingSnap);
  const waitingText = waitingEl.textContent ?? "";
  assertEquals(waitingText.includes("待反主"), true);
  assertEquals(waitingText.includes("换底牌"), false);
});
