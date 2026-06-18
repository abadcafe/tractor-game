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

function makeSnapshot(overrides: Partial<StateSnapshot> = {}): StateSnapshot {
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
    trick_history: [],
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

Deno.test("test_renderScoreboard_shows_defender_points", () => {
  const snap = makeSnapshot({ defender_points: 25 });
  const el = renderScoreboard(snap);
  const text = el.textContent ?? "";
  assertEquals(text.includes("25"), true);
});
