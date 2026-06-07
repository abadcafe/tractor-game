import { assertEquals, assertNotEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderTrickView } from "../ui/components/trick-view.ts";
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
    current_player: 3,
    defender_points: 0,
    legal_actions: [],
    trick: {
      lead_player: 0,
      lead_type: "single",
      slots: [{ player: 0, cards: [{ id: "D1-clubs-7", suit: "clubs", rank: "7" }] }],
      current_player: 3,
    },
    trick_history: [],
    bid_events: [],
    bid_winner: null,
    awaiting_action: null,
    stirring_state: null,
    exchange_state: null,
    scoring: null,
    winning_team: null,
    team0_level: "2",
    team1_level: "2",
    ...overrides,
  };
}

Deno.test("test_renderTrickView_shows_played_cards", () => {
  const snap = makeSnapshot();
  const el = renderTrickView(snap);
  const trickCards = el.querySelectorAll(".trick-card");
  assertEquals(trickCards.length >= 1, true);
});

Deno.test("test_renderTrickView_empty_trick", () => {
  const snap = makeSnapshot({ trick: null });
  const el = renderTrickView(snap);
  const trickCards = el.querySelectorAll(".trick-card");
  assertEquals(trickCards.length, 0);
});

Deno.test("test_renderTrickView_player_labels", () => {
  const snap = makeSnapshot();
  const el = renderTrickView(snap);
  const labels = el.querySelectorAll(".player-label");
  assertEquals(labels.length >= 1, true);
});
