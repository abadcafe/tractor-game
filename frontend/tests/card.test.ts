import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  cardDisplay,
  cardPoints,
  isBigJoker,
  isJoker,
  isPointCard,
  isTrumpRank,
  suitSymbol,
} from "../core/card.ts";
import type { Card } from "../core/types.ts";

function makeCard(suit: string, rank: string): Card {
  return { id: `D1-${suit}-${rank}`, suit, rank };
}

// --- cardDisplay ---

Deno.test("test_cardDisplay_hearts_5", () => {
  assertEquals(cardDisplay(makeCard("hearts", "5")), "♥5");
});

Deno.test("test_cardDisplay_spades_A", () => {
  assertEquals(cardDisplay(makeCard("spades", "A")), "♠A");
});

Deno.test("test_cardDisplay_diamonds_10", () => {
  assertEquals(cardDisplay(makeCard("diamonds", "10")), "♦10");
});

Deno.test("test_cardDisplay_clubs_K", () => {
  assertEquals(cardDisplay(makeCard("clubs", "K")), "♣K");
});

Deno.test("test_cardDisplay_small_joker", () => {
  assertEquals(cardDisplay(makeCard("joker", "SJ")), "🃏小王");
});

Deno.test("test_cardDisplay_big_joker", () => {
  assertEquals(cardDisplay(makeCard("joker", "BJ")), "🃏大王");
});

// --- isJoker ---

Deno.test("test_isJoker_small_joker", () => {
  assertEquals(isJoker(makeCard("joker", "SJ")), true);
});

Deno.test("test_isJoker_big_joker", () => {
  assertEquals(isJoker(makeCard("joker", "BJ")), true);
});

Deno.test("test_isJoker_normal_card", () => {
  assertEquals(isJoker(makeCard("hearts", "2")), false);
});

// --- isBigJoker ---

Deno.test("test_isBigJoker_big_joker", () => {
  assertEquals(isBigJoker(makeCard("joker", "BJ")), true);
});

Deno.test("test_isBigJoker_small_joker", () => {
  assertEquals(isBigJoker(makeCard("joker", "SJ")), false);
});

Deno.test("test_isBigJoker_normal_card", () => {
  assertEquals(isBigJoker(makeCard("spades", "A")), false);
});

// --- isTrumpRank ---

Deno.test("test_isTrumpRank_matching_rank", () => {
  assertEquals(isTrumpRank(makeCard("hearts", "2"), "2"), true);
});

Deno.test("test_isTrumpRank_non_matching_rank", () => {
  assertEquals(isTrumpRank(makeCard("hearts", "3"), "2"), false);
});

Deno.test("test_isTrumpRank_joker_is_not_trump_rank", () => {
  assertEquals(isTrumpRank(makeCard("joker", "SJ"), "2"), false);
});

// --- isPointCard ---

Deno.test("test_isPointCard_5", () => {
  assertEquals(isPointCard(makeCard("hearts", "5")), true);
});

Deno.test("test_isPointCard_10", () => {
  assertEquals(isPointCard(makeCard("diamonds", "10")), true);
});

Deno.test("test_isPointCard_K", () => {
  assertEquals(isPointCard(makeCard("spades", "K")), true);
});

Deno.test("test_isPointCard_non_point", () => {
  assertEquals(isPointCard(makeCard("clubs", "3")), false);
});

Deno.test("test_isPointCard_joker", () => {
  assertEquals(isPointCard(makeCard("joker", "BJ")), false);
});

// --- cardPoints ---

Deno.test("test_cardPoints_5", () => {
  assertEquals(cardPoints(makeCard("hearts", "5")), 5);
});

Deno.test("test_cardPoints_10", () => {
  assertEquals(cardPoints(makeCard("diamonds", "10")), 10);
});

Deno.test("test_cardPoints_K", () => {
  assertEquals(cardPoints(makeCard("spades", "K")), 10);
});

Deno.test("test_cardPoints_non_point", () => {
  assertEquals(cardPoints(makeCard("clubs", "3")), 0);
});

Deno.test("test_cardPoints_joker", () => {
  assertEquals(cardPoints(makeCard("joker", "BJ")), 0);
});

// --- suitSymbol ---

Deno.test("test_suitSymbol_hearts", () => {
  assertEquals(suitSymbol("hearts"), "♥");
});

Deno.test("test_suitSymbol_spades", () => {
  assertEquals(suitSymbol("spades"), "♠");
});

Deno.test("test_suitSymbol_diamonds", () => {
  assertEquals(suitSymbol("diamonds"), "♦");
});

Deno.test("test_suitSymbol_clubs", () => {
  assertEquals(suitSymbol("clubs"), "♣");
});

Deno.test("test_suitSymbol_joker", () => {
  assertEquals(suitSymbol("joker"), "🃏");
});
