import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  handlePlayAction,
  handleStirAction,
} from "../engine/action-handler.ts";
import type { Card, StateSnapshot } from "../core/types.ts";

function makeSnapshot(
  overrides: Partial<StateSnapshot> = {},
): StateSnapshot {
  return {
    phase: "PLAYING",
    player_hand: [
      { id: "D1-spades-K", suit: "spades", rank: "K" },
      { id: "D1-spades-Q", suit: "spades", rank: "Q" },
    ],
    bottom_cards: [],
    trump_rank: "2",
    trump_suit: "hearts",
    declarer_team: 0,
    declarer_player: 2,
    defender_points: 0,
    action_hints: [[{ id: "D1-spades-K", suit: "spades", rank: "K" }]],
    trick: null,
    last_completed_trick: null,
    defender_point_cards: [],
    failed_throw: null,
    bid_events: [],
    bid_winner: null,
    stir_events: [],
    own_bottom_exchange_events: [],
    awaiting_action: "play",
    stirring_state: null,
    scoring: null,
    winning_team: null,
    team0_level: "2",
    team1_level: "2",
    player_hand_counts: [13, 13, 13, 13],
    next_round_confirmed: [],
    ...overrides,
  };
}

Deno.test("handlePlayAction sends selected throw even when hints do not match", () => {
  const result = handlePlayAction(
    makeSnapshot(),
    new Set(["D1-spades-Q", "D1-spades-K"]),
    12,
  );

  assertEquals(result.success, true);
  assertNotEquals(result.action, undefined);
  if (result.action === undefined || result.action.type !== "play") {
    throw new Error("expected play action");
  }
  assertEquals(result.action.seq, 12);
  assertEquals(result.action.cards, ["D1-spades-Q", "D1-spades-K"]);
});

Deno.test("handleStirAction accepts selected cards from hints", () => {
  const pair: Card[] = [
    { id: "D1-diamonds-5", suit: "diamonds", rank: "5" },
    { id: "D2-diamonds-5", suit: "diamonds", rank: "5" },
  ];
  const result = handleStirAction(
    makeSnapshot({
      phase: "STIRRING",
      trump_rank: "5",
      player_hand: pair,
      action_hints: [pair],
      awaiting_action: "stir",
    }),
    pair.map((card) => card.id),
    18,
  );

  assertEquals(result.success, true);
  assertNotEquals(result.action, undefined);
  if (result.action === undefined || result.action.type !== "stir") {
    throw new Error("expected stir action");
  }
  assertEquals(result.action.seq, 18);
  assertEquals(result.action.cards, ["D1-diamonds-5", "D2-diamonds-5"]);
});

Deno.test("handleStirAction rejects valid-looking pair outside hints", () => {
  const pair: Card[] = [
    { id: "D1-diamonds-5", suit: "diamonds", rank: "5" },
    { id: "D2-diamonds-5", suit: "diamonds", rank: "5" },
  ];
  const result = handleStirAction(
    makeSnapshot({
      phase: "STIRRING",
      trump_rank: "5",
      player_hand: pair,
      action_hints: [],
      awaiting_action: "stir",
    }),
    pair.map((card) => card.id),
    18,
  );

  assertEquals(result.success, false);
  assertEquals(result.error, "优先级不足，不能反主");
});
