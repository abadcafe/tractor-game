import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderGameTable } from "../ui/components/game-table.ts";
import type { StateSnapshot } from "../core/types.ts";

// Set up global document
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
    trump_suit: "hearts",
    declarer_team: 0,
    declarer_player: 3,
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

Deno.test("test_renderGameTable_shows_four_players", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap);
  const players = el.querySelectorAll(".player-area");
  assertEquals(players.length, 4);
});

Deno.test("test_renderGameTable_declarer_marker", () => {
  const snap = makeSnapshot({ declarer_player: 3 });
  const el = renderGameTable(snap);
  const markers = el.querySelectorAll(".declarer-marker");
  assertEquals(markers.length, 1);
  assertEquals(markers[0].textContent, "庄");
});

Deno.test("test_renderGameTable_current_player_highlight", () => {
  const snap = makeSnapshot({
    awaiting_action: "play",
    trick: { lead_player: 0, slots: [], current_player: 1 },
  });
  const el = renderGameTable(snap);
  const current = el.querySelectorAll(".player-area.current");
  assertEquals(current.length, 1);
});

Deno.test("test_renderGameTable_player_labels", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap);
  const labels = el.querySelectorAll(".player-label");
  const labelTexts = Array.from(labels).map((l) => l.textContent);
  assertEquals(labelTexts.includes("你"), true);
  assertEquals(labelTexts.includes("同伴"), true);
  assertEquals(labelTexts.includes("左家"), true);
  assertEquals(labelTexts.includes("右家"), true);
});
