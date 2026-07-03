import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  chooseFirstActionHint,
  computeLegalCardIds,
  computeStirButtonState,
  isSelectionStillLegal,
} from "../engine/ui-state-computer.ts";
import type { Card, StateSnapshot } from "../core/types.ts";

function makeSnapshot(
  overrides: Partial<StateSnapshot> = {},
): StateSnapshot {
  return {
    phase: "PLAYING",
    player_hand: [
      { id: "D1-hearts-5", suit: "hearts", rank: "5" },
      { id: "D1-spades-4", suit: "spades", rank: "4" },
    ],
    bottom_cards: [],
    trump_rank: "4",
    trump_suit: null,
    declarer_team: 0,
    declarer_player: 2,
    defender_points: 0,
    action_hints: [[{ id: "D1-spades-4", suit: "spades", rank: "4" }]],
    trick: {
      lead_player: 0,
      slots: [{
        player: 0,
        cards: [{ id: "D1-diamonds-A", suit: "diamonds", rank: "A" }],
      }],
      current_player: 2,
      failed_throw: null,
    },
    last_completed_trick: null,
    defender_point_cards: [],
    bid_events: [],
    bid_winner: null,
    stir_events: [],
    own_initial_bottom_exchange: null,
    awaiting_action: "play",
    stirring_state: null,
    scoring: null,
    winning_team: null,
    team0_level: "4",
    team1_level: "2",
    player_hand_counts: [10, 10, 10, 10],
    next_round_confirmed: [],
    ...overrides,
  };
}

Deno.test("isSelectionStillLegal keeps non-hint play selections", () => {
  const snap = makeSnapshot();
  const selectedCardIds = new Set(["D1-hearts-5"]);

  assertEquals(isSelectionStillLegal(snap, selectedCardIds), true);
});

Deno.test("isSelectionStillLegal rejects selections no longer in hand", () => {
  const snap = makeSnapshot();
  const selectedCardIds = new Set(["D1-clubs-9"]);

  assertEquals(isSelectionStillLegal(snap, selectedCardIds), false);
});

Deno.test("computeLegalCardIds keeps hints advisory in play mode", () => {
  const snap = makeSnapshot();
  const legalCardIds = computeLegalCardIds(snap, "play");

  assertEquals(legalCardIds.has("D1-spades-4"), true);
  assertEquals(legalCardIds.has("D1-hearts-5"), false);
});

Deno.test("chooseFirstActionHint preserves server hint order", () => {
  const firstHint: Card[] = [
    { id: "D1-spades-A", suit: "spades", rank: "A" },
  ];
  const secondHint: Card[] = [
    { id: "D1-diamonds-3", suit: "diamonds", rank: "3" },
  ];
  const snap = makeSnapshot({
    trump_rank: "4",
    trump_suit: "hearts",
    action_hints: [firstHint, secondHint],
  });

  const result = chooseFirstActionHint(snap);

  assertEquals(result?.map((card) => card.id), ["D1-spades-A"]);
});

Deno.test("computeStirButtonState enables selected legal hint", () => {
  const pair: Card[] = [
    { id: "D1-diamonds-5", suit: "diamonds", rank: "5" },
    { id: "D2-diamonds-5", suit: "diamonds", rank: "5" },
  ];
  const snap = makeSnapshot({
    phase: "STIRRING",
    trump_rank: "5",
    player_hand: pair,
    action_hints: [pair],
    awaiting_action: "stir",
  });

  const result = computeStirButtonState(
    snap,
    new Set(pair.map((card) => card.id)),
  );

  assertEquals(result.disabled, false);
});

Deno.test("computeStirButtonState disables pair outside legal hints", () => {
  const pair: Card[] = [
    { id: "D1-diamonds-5", suit: "diamonds", rank: "5" },
    { id: "D2-diamonds-5", suit: "diamonds", rank: "5" },
  ];
  const snap = makeSnapshot({
    phase: "STIRRING",
    trump_rank: "5",
    player_hand: pair,
    action_hints: [],
    awaiting_action: "stir",
  });

  const result = computeStirButtonState(
    snap,
    new Set(pair.map((card) => card.id)),
  );

  assertEquals(result.disabled, true);
  assertEquals(result.title, "没有可反的对子");
});
