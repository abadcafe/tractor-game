import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import type { Card, Rank, Suit } from "../core/types.ts";
import {
  computeBidOptionsFromHints,
  computeBidPriority,
  computeDealBidAction,
} from "../engine/bid-logic.ts";

function card(id: string, suit: Suit, rank: Rank): Card {
  return { id, suit, rank };
}

Deno.test("computeBidPriority rejects single jokers", () => {
  assertEquals(
    computeBidPriority([card("D1-joker-BJ", "joker", "BJ")], "2"),
    0,
  );
  assertEquals(
    computeBidPriority([card("D1-joker-SJ", "joker", "SJ")], "2"),
    0,
  );
});

Deno.test("computeBidOptionsFromHints formats backend logical hints", () => {
  const hints: Card[][] = [
    [card("D1-diamonds-2", "diamonds", "2")],
    [card("D1-clubs-2", "clubs", "2")],
    [card("D1-hearts-2", "hearts", "2")],
    [card("D1-spades-2", "spades", "2")],
    [
      card("D1-diamonds-2", "diamonds", "2"),
      card("D2-diamonds-2", "diamonds", "2"),
    ],
    [
      card("D1-clubs-2", "clubs", "2"),
      card("D2-clubs-2", "clubs", "2"),
    ],
    [
      card("D1-hearts-2", "hearts", "2"),
      card("D2-hearts-2", "hearts", "2"),
    ],
    [
      card("D1-spades-2", "spades", "2"),
      card("D2-spades-2", "spades", "2"),
    ],
    [
      card("D1-joker-SJ", "joker", "SJ"),
      card("D2-joker-SJ", "joker", "SJ"),
    ],
    [
      card("D1-joker-BJ", "joker", "BJ"),
      card("D2-joker-BJ", "joker", "BJ"),
    ],
  ];

  const options = computeBidOptionsFromHints(hints, "2");

  assertEquals(options.map((option) => option.label), [
    "♦2",
    "♣2",
    "♥2",
    "♠2",
    "♦♦2",
    "♣♣2",
    "♥♥2",
    "♠♠2",
    "小王对",
    "大王对",
  ]);
  assertEquals(options.length, 10);
});

Deno.test("computeDealBidAction auto-passes when hints exist but no pending intent", () => {
  const hint = [card("D1-spades-2", "spades", "2")];
  const decision = computeDealBidAction([hint], null, 9);

  assertEquals(decision.action, { type: "bid", seq: 9, pass: true });
  assertEquals(decision.matchedPending, false);
  assertEquals(decision.stalePending, false);
});

Deno.test("computeDealBidAction sends pending intent only when current hints match", () => {
  const hint = [card("D1-spades-2", "spades", "2")];
  const decision = computeDealBidAction([hint], {
    cardIds: ["D1-spades-2"],
    label: "♠2",
    trumpSuit: "spades",
    priority: 103,
  }, 10);

  assertEquals(decision.action, {
    type: "bid",
    seq: 10,
    cards: ["D1-spades-2"],
  });
  assertEquals(decision.matchedPending, true);
  assertEquals(decision.stalePending, false);
});

Deno.test("computeDealBidAction consumes pending intent even when hints differ", () => {
  const hint = [card("D1-hearts-2", "hearts", "2")];
  const decision = computeDealBidAction([hint], {
    cardIds: ["D1-spades-2"],
    label: "♠2",
    trumpSuit: "spades",
    priority: 103,
  }, 11);

  assertEquals(decision.action, {
    type: "bid",
    seq: 11,
    cards: ["D1-spades-2"],
  });
  assertEquals(decision.matchedPending, true);
  assertEquals(decision.stalePending, false);
});
