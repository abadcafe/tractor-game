import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  validateBidCards,
  validateDiscard,
  validatePlay,
} from "../engine/input-validator.ts";
import type { Card, Rank, Suit } from "../core/types.ts";

function makeCard(id: string, suit: Suit, rank: Rank): Card {
  return { id, suit, rank };
}

const H5 = makeCard("D1-hearts-5", "hearts", "5");
const H6 = makeCard("D1-hearts-6", "hearts", "6");
const S2 = makeCard("D1-spades-2", "spades", "2");
const BJ = makeCard("D2-joker-BJ", "joker", "BJ");
const BJ2 = makeCard("D1-joker-BJ", "joker", "BJ");
const SJ = makeCard("D2-joker-SJ", "joker", "SJ");
const SJ2 = makeCard("D1-joker-SJ", "joker", "SJ");

// --- validatePlay ---

Deno.test("test_validatePlay_matching_single", () => {
  const legal: Card[][] = [[H5]];
  const result = validatePlay([H5], legal);
  assertEquals(result !== null, true);
  assertEquals(result!.length, 1);
  assertEquals(result![0].id, H5.id);
});

Deno.test("test_validatePlay_matching_pair", () => {
  const pairCard = makeCard("D2-hearts-5", "hearts", "5");
  const legal: Card[][] = [[H5, pairCard]];
  // Selecting only 1 card of a 2-card pair should NOT match
  // (exact-size match required to prevent hand imbalance).
  const result = validatePlay([H5], legal);
  assertEquals(result, null);
  // Selecting both cards should match
  const result2 = validatePlay([H5, pairCard], legal);
  assertEquals(result2 !== null, true);
  assertEquals(result2!.length, 2);
});

Deno.test("test_validatePlay_no_match", () => {
  const legal: Card[][] = [[H6]];
  const result = validatePlay([H5], legal);
  assertEquals(result, null);
});

Deno.test("test_validatePlay_empty_legal", () => {
  const result = validatePlay([H5], []);
  assertEquals(result, null);
});

Deno.test("test_validatePlay_empty_selected", () => {
  const legal: Card[][] = [[H5]];
  const result = validatePlay([], legal);
  assertEquals(result, null);
});

Deno.test("test_validatePlay_multiple_legal_first_match", () => {
  const legal: Card[][] = [[H6], [H5]];
  const result = validatePlay([H5], legal);
  assertEquals(result !== null, true);
  assertEquals(result!.length, 1);
  assertEquals(result![0].id, H5.id);
});

Deno.test("test_validatePlay_tractor_match", () => {
  const legal: Card[][] = [[H5, H6]];
  const result = validatePlay([H5, H6], legal);
  assertEquals(result !== null, true);
  assertEquals(result!.length, 2);
});

// --- validateDiscard ---

Deno.test("test_validateDiscard_correct_count", () => {
  assertEquals(validateDiscard([H5, H6], 2), true);
});

Deno.test("test_validateDiscard_wrong_count", () => {
  assertEquals(validateDiscard([H5], 2), false);
});

Deno.test("test_validateDiscard_empty_selection", () => {
  assertEquals(validateDiscard([], 0), true);
});

Deno.test("test_validateDiscard_more_than_expected", () => {
  assertEquals(validateDiscard([H5, H6, S2], 2), false);
});

// --- validateBidCards ---

Deno.test("test_validateBidCards_trump_rank", () => {
  assertEquals(validateBidCards([S2], "2"), true);
});

Deno.test("test_validateBidCards_single_big_joker_rejected", () => {
  assertEquals(validateBidCards([BJ], "2"), false);
});

Deno.test("test_validateBidCards_single_small_joker_rejected", () => {
  assertEquals(validateBidCards([SJ], "2"), false);
});

Deno.test("test_validateBidCards_joker_pair", () => {
  assertEquals(validateBidCards([BJ, BJ2], "2"), true);
  assertEquals(validateBidCards([SJ, SJ2], "2"), true);
});

Deno.test("test_validateBidCards_mixed_joker_pair_rejected", () => {
  assertEquals(validateBidCards([BJ, SJ], "2"), false);
});

Deno.test("test_validateBidCards_non_trump_non_joker", () => {
  assertEquals(validateBidCards([H5], "2"), false);
});

Deno.test("test_validateBidCards_empty", () => {
  assertEquals(validateBidCards([], "2"), false);
});

Deno.test("test_validateBidCards_mixed_valid_invalid", () => {
  // All selected cards must be trump rank or joker
  assertEquals(validateBidCards([S2, H5], "2"), false);
});

Deno.test("test_validateBidCards_multiple_trump_rank", () => {
  assertEquals(
    validateBidCards([S2, makeCard("D2-spades-2", "spades", "2")], "2"),
    true,
  );
});
