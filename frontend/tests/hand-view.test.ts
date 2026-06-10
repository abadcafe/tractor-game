import { assertEquals, assertNotEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderHandView } from "../ui/components/hand-view.ts";
import type { StateSnapshot, InteractionMode } from "../core/types.ts";

const doc = new DOMParser().parseFromString(
  `<html><body><div id="app"></div></body></html>`,
  "text/html",
);
// @ts-ignore test setup
globalThis.document = doc;

function makeSnapshot(overrides: Partial<StateSnapshot> = {}): StateSnapshot {
  return {
    phase: "PLAYING",
    player_hand: [
      { id: "D1-hearts-5", suit: "hearts", rank: "5" },
      { id: "D1-spades-2", suit: "spades", rank: "2" },
    ],
    bottom_cards: [],
    trump_rank: "2",
    trump_suit: "hearts",
    declarer_team: 0,
    declarer_player: 3,
    current_player: 3,
    defender_points: 0,
    legal_actions: [[{ id: "D1-hearts-5", suit: "hearts", rank: "5" }]],
    trick: null,
    trick_history: [],
    bid_events: [],
    bid_winner: null,
    awaiting_action: "play",
    stirring_state: null,
    exchange_state: null,
    scoring: null,
    winning_team: null,
    team0_level: "2",
    team1_level: "2",
    player_hand_counts: [13, 13, 13, 13],
    next_round_confirmed: [],
    ...overrides,
  };
}

Deno.test("test_renderHandView_displays_cards", () => {
  const snap = makeSnapshot();
  const el = renderHandView(snap, "play");
  const cards = el.querySelectorAll(".card");
  assertEquals(cards.length, 2);
});

Deno.test("test_renderHandView_selected_card", () => {
  const snap = makeSnapshot();
  const selectedIds = new Set(["D1-hearts-5"]);
  const el = renderHandView(snap, "play", selectedIds);
  const cards = el.querySelectorAll(".card");
  // First card should be selected
  assertEquals(cards[0].classList.contains("selected"), true);
});

Deno.test("test_renderHandView_legal_highlight", () => {
  const snap = makeSnapshot();
  const el = renderHandView(snap, "play");
  // The legal card (hearts-5) should have the .legal class
  const legalCards = el.querySelectorAll(".card.legal");
  assertEquals(legalCards.length >= 1, true);
});

Deno.test("test_renderHandView_play_button", () => {
  const snap = makeSnapshot();
  const el = renderHandView(snap, "play");
  const buttons = el.querySelectorAll("button");
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("出牌"), true);
});

Deno.test("test_renderHandView_discard_button", () => {
  const snap = makeSnapshot({
    phase: "EXCHANGE",
    awaiting_action: "discard",
    exchange_state: { phase: "PICKED_UP", declarer_player: 3, count: 8 },
    legal_actions: [],
  });
  const el = renderHandView(snap, "discard");
  const buttons = el.querySelectorAll("button");
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("弃牌"), true);
});

Deno.test("test_renderHandView_no_button_when_spectating", () => {
  const snap = makeSnapshot();
  const el = renderHandView(snap, null);
  const buttons = el.querySelectorAll("button");
  assertEquals(buttons.length, 0);
});

Deno.test("test_renderHandView_card_click_callback", () => {
  const snap = makeSnapshot();
  let clickedCardId: string | null = null;
  const onCardClick = (cardId: string) => { clickedCardId = cardId; };
  const el = renderHandView(snap, "play", undefined, onCardClick);
  // Simulate clicking the first card
  const firstCard = el.querySelector(".card") as HTMLElement;
  assertNotEquals(firstCard, null);
  firstCard.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(clickedCardId, "D1-hearts-5");
});

Deno.test("test_renderHandView_action_button_callback", () => {
  const snap = makeSnapshot();
  let actionFired: string | null = null;
  const onAction = (action: string) => { actionFired = action; };
  const el = renderHandView(snap, "play", undefined, undefined, onAction);
  // Find the play button and click it
  const buttons = el.querySelectorAll("button");
  const playButton = Array.from(buttons).find((b) => b.textContent === "出牌");
  assertNotEquals(playButton, undefined);
  playButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(actionFired, "play");
});
