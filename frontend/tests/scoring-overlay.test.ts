import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderScoringOverlay } from "../ui/components/scoring-overlay.ts";
import type { StateSnapshot } from "../core/types.ts";

const doc = new DOMParser().parseFromString(
  `<html><body><div id="app"></div></body></html>`,
  "text/html",
);
// @ts-ignore test setup
globalThis.document = doc;

function makeSnapshot(
  overrides: Partial<StateSnapshot> = {},
): StateSnapshot {
  return {
    phase: "WAITING",
    player_hand: [],
    bottom_cards: [],
    trump_rank: "2",
    trump_suit: null,
    declarer_team: 0,
    declarer_player: 2,
    defender_points: 30,
    action_hints: [],
    trick: null,
    last_completed_trick: null,
    defender_point_cards: [],
    failed_throw: null,
    bid_events: [],
    bid_winner: null,
    awaiting_action: "next_round",
    stirring_state: null,
    scoring: {
      declarer_team: 0,
      defender_points: 30,
      total_defender_points: 30,
      bottom_card_bonus: 0,
      bottom_cards: [],
    },
    winning_team: null,
    team0_level: "2",
    team1_level: "2",
    player_hand_counts: [13, 13, 13, 13],
    next_round_confirmed: [],
    ...overrides,
  };
}

Deno.test("test_renderScoringOverlay_shows_scoring", () => {
  const snap = makeSnapshot();
  const el = renderScoringOverlay(snap, "next_round");
  const text = el.textContent ?? "";
  assertEquals(text.includes("30"), true);
});

Deno.test("test_renderScoringOverlay_next_round_button", () => {
  const snap = makeSnapshot();
  const el = renderScoringOverlay(snap, "next_round");
  const buttons = el.querySelectorAll("button");
  const actionButton = el.querySelector(
    ".scoring-overlay__actions .scoring-overlay__next-round",
  );
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("下一轮"), true);
  assertNotEquals(actionButton, null);
});

Deno.test("test_renderScoringOverlay_bottom_cards_are_prominent_cards", () => {
  const snap = makeSnapshot({
    scoring: {
      declarer_team: 0,
      defender_points: 30,
      total_defender_points: 70,
      bottom_card_bonus: 40,
      bottom_cards: [
        { id: "D1-hearts-5", suit: "hearts", rank: "5" },
        { id: "D1-diamonds-10", suit: "diamonds", rank: "10" },
        { id: "D1-spades-K", suit: "spades", rank: "K" },
      ],
    },
  });
  const el = renderScoringOverlay(snap, "next_round");
  const bottom = el.querySelector(".scoring-overlay__bottom");
  const bottomCards = el.querySelectorAll(".scoring-bottom-card");
  const text = bottom?.textContent ?? "";

  assertNotEquals(bottom, null);
  assertEquals(text.includes("底牌"), true);
  assertEquals(bottomCards.length, 3);
});

Deno.test("test_renderScoringOverlay_eight_bottom_cards_use_centered_layout", () => {
  const bottomCards = Array.from({ length: 8 }, (_, index) => ({
    id: `D${index}-hearts-5`,
    suit: "hearts" as const,
    rank: "5" as const,
  }));
  const snap = makeSnapshot({
    scoring: {
      declarer_team: 0,
      defender_points: 120,
      total_defender_points: 240,
      bottom_card_bonus: 120,
      bottom_cards: bottomCards,
    },
  });
  const el = renderScoringOverlay(snap, "next_round");
  const cardsWrap = el.querySelector(".scoring-overlay__bottom-cards");
  const cards = el.querySelectorAll(".scoring-bottom-card");

  assertEquals(cards.length, 8);
  assertEquals(
    cardsWrap?.classList.contains(
      "scoring-overlay__bottom-cards--many",
    ),
    false,
  );
});

Deno.test("test_renderScoringOverlay_no_button_when_human_ready", () => {
  const snap = makeSnapshot({ next_round_confirmed: [2] });
  const el = renderScoringOverlay(snap, null);
  const buttons = el.querySelectorAll("button");
  assertEquals(buttons.length, 0);
});

Deno.test("test_renderScoringOverlay_next_round_callback", () => {
  const snap = makeSnapshot();
  let nextRoundCalled = false;
  const onNextRound = () => {
    nextRoundCalled = true;
  };
  const el = renderScoringOverlay(snap, "next_round", onNextRound);
  const buttons = el.querySelectorAll("button");
  const nextButton = Array.from(buttons).find((b) =>
    b.textContent === "下一轮"
  );
  assertNotEquals(nextButton, undefined);
  nextButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(nextRoundCalled, true);
});

Deno.test("test_renderScoringOverlay_null_scoring", () => {
  const snap = makeSnapshot({ scoring: null });
  const el = renderScoringOverlay(snap, "next_round");
  // Should not throw; overlay rendered with no scoring details
  const text = el.textContent ?? "";
  assertEquals(text.includes("Defender Points"), false);
  assertEquals(text.includes("Declarer Team"), false);
  assertEquals(text.includes("Bottom Cards"), false);
  // Button should still appear since interactionMode is "next_round"
  const buttons = el.querySelectorAll("button");
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("下一轮"), true);
});

Deno.test("test_renderScoringOverlay_next_round_button_no_callback", () => {
  const snap = makeSnapshot();
  // Provide "next_round" mode but no callback
  const el = renderScoringOverlay(snap, "next_round");
  const buttons = el.querySelectorAll("button");
  const nextButton = Array.from(buttons).find((b) =>
    b.textContent === "下一轮"
  );
  assertNotEquals(nextButton, undefined);
  // Clicking should not throw
  nextButton!.dispatchEvent(new Event("click", { bubbles: true }));
});
