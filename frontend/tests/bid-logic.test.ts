import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import type { Card } from "../core/types.ts";
import {
  computeBidOptionsFromHints,
  computeBidPriority,
} from "../engine/bid-logic.ts";

function card(id: string, suit: string, rank: string): Card {
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
    "♦2对",
    "♣2对",
    "♥2对",
    "♠2对",
    "小王对",
    "大王对",
  ]);
  assertEquals(options.length, 10);
});
