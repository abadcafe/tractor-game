import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { validatePlay, validateDiscard, validateBidCards } from "../engine/input-validator.ts";
import type { Card, PlayAction } from "../core/types.ts";

function makeCard(id: string, suit: string, rank: string): Card {
  return { id, suit, rank };
}

const H5 = makeCard("D1-hearts-5", "hearts", "5");
const H6 = makeCard("D1-hearts-6", "hearts", "6");
const S2 = makeCard("D1-spades-2", "spades", "2");
const BJ = makeCard("D2-joker-BJ", "joker", "BJ");
const SJ = makeCard("D2-joker-SJ", "joker", "SJ");

function makeLegalAction(type: string, cards: Card[]): PlayAction {
  return { type, cards };
}

// --- validatePlay ---

Deno.test("test_validatePlay_matching_single", () => {
  const legal = [makeLegalAction("single", [H5])];
  const result = validatePlay([H5], legal);
  assertEquals(result !== null, true);
  assertEquals(result!.type, "single");
});

Deno.test("test_validatePlay_matching_pair", () => {
  const legal = [makeLegalAction("pair", [H5, makeCard("D2-hearts-5", "hearts", "5")])];
  const result = validatePlay([H5], legal);
  // selected card IDs match one of the legal actions' card IDs
  assertEquals(result !== null, true);
});

Deno.test("test_validatePlay_no_match", () => {
  const legal = [makeLegalAction("single", [H6])];
  const result = validatePlay([H5], legal);
  assertEquals(result, null);
});

Deno.test("test_validatePlay_empty_legal", () => {
  const result = validatePlay([H5], []);
  assertEquals(result, null);
});

Deno.test("test_validatePlay_empty_selected", () => {
  const legal = [makeLegalAction("single", [H5])];
  const result = validatePlay([], legal);
  assertEquals(result, null);
});

Deno.test("test_validatePlay_multiple_legal_first_match", () => {
  const legal = [
    makeLegalAction("single", [H6]),
    makeLegalAction("single", [H5]),
  ];
  const result = validatePlay([H5], legal);
  assertEquals(result !== null, true);
  assertEquals(result!.type, "single");
});

Deno.test("test_validatePlay_tractor_match", () => {
  const legal = [makeLegalAction("tractor", [H5, H6])];
  const result = validatePlay([H5, H6], legal);
  assertEquals(result !== null, true);
  assertEquals(result!.type, "tractor");
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

Deno.test("test_validateBidCards_joker", () => {
  assertEquals(validateBidCards([BJ], "2"), true);
});

Deno.test("test_validateBidCards_small_joker", () => {
  assertEquals(validateBidCards([SJ], "2"), true);
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
  assertEquals(validateBidCards([S2, makeCard("D2-spades-2", "spades", "2")], "2"), true);
});
