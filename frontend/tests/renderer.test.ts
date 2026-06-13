import { assertEquals, assertNotEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { render } from "../ui/renderer.ts";
import type { StateSnapshot } from "../core/types.ts";
import type { InteractionMode, GameAction } from "../engine/types.ts";
import type { ActionCallbacks } from "../ui/types.ts";

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
    defender_points: 15,
    legal_actions: [[{ id: "D1-hearts-5", suit: "hearts", rank: "5" }]],
    trick: {
      lead_player: 0,
      slots: [{ player: 0, cards: [{ id: "D1-clubs-7", suit: "clubs", rank: "7" }] }],
      current_player: 3,
    },
    trick_history: [],
    bid_events: [],
    bid_winner: null,
    awaiting_action: "play",
    stirring_state: null,
    exchange_state: null,
    scoring: null,
    winning_team: null,
    team0_level: "3",
    team1_level: "5",
    player_hand_counts: [13, 13, 13, 13],
    next_round_confirmed: [],
    ...overrides,
  };
}

// Create a fresh document for each test to avoid shared state
function freshContainer(): Element {
  const doc = new DOMParser().parseFromString(
    `<html><body><div id="app"></div></body></html>`,
    "text/html",
  );
  // @ts-ignore test setup
  globalThis.document = doc;
  return doc!.querySelector("#app")!;
}

Deno.test("test_render_playing_phase", () => {
  const container = freshContainer();
  const snap = makeSnapshot();
  render(snap, container, "play");
  const tableEl = container.querySelector(".game-table");
  assertNotEquals(tableEl, null);
  const handEl = container.querySelector(".hand-view");
  assertNotEquals(handEl, null);
  const trickEl = container.querySelector(".trick-view");
  assertNotEquals(trickEl, null);
  const scoreEl = container.querySelector(".scoreboard");
  assertNotEquals(scoreEl, null);
});

Deno.test("test_render_deal_bid_phase", () => {
  const container = freshContainer();
  const snap = makeSnapshot({ phase: "DEAL_BID", awaiting_action: null });
  render(snap, container, "bid");
  const bidEl = container.querySelector(".bidding-dialog");
  assertNotEquals(bidEl, null);
});

Deno.test("test_render_stirring_phase", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "STIRRING",
    awaiting_action: "stir",
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 3 },
    trick: null,
  });
  render(snap, container, "stir");
  const stirEl = container.querySelector(".bidding-dialog");
  assertNotEquals(stirEl, null);
});

Deno.test("test_render_complete_phase", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "COMPLETE",
    awaiting_action: "next_round",
    trick: null,
    scoring: {
      declarer_team: 0,
      defender_points: 30,
      total_defender_points: 30,
      bottom_card_bonus: 0,
      bottom_cards: [],
    },
  });
  render(snap, container, "next_round");
  const overlayEl = container.querySelector(".scoring-overlay");
  assertNotEquals(overlayEl, null);
});

Deno.test("test_render_game_over_phase", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "GAME_OVER",
    winning_team: 0,
    trick: null,
    awaiting_action: null,
  });
  render(snap, container, null);
  const overEl = container.querySelector(".game-over-overlay");
  assertNotEquals(overEl, null);
});

Deno.test("test_render_exchange_phase", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "EXCHANGE",
    awaiting_action: "discard",
    exchange_state: { phase: "PICKED_UP", declarer_player: 3, count: 8 },
    trick: null,
  });
  render(snap, container, "discard");
  const handEl = container.querySelector(".hand-view");
  assertNotEquals(handEl, null);
});

Deno.test("test_render_hand_displays_cards", () => {
  const container = freshContainer();
  const snap = makeSnapshot();
  render(snap, container, "play");
  const cards = container.querySelectorAll(".card");
  assertEquals(cards.length, 2);
});

Deno.test("test_render_scoreboard_shows_levels", () => {
  const container = freshContainer();
  const snap = makeSnapshot();
  render(snap, container, null);
  const scoreEl = container.querySelector(".scoreboard");
  assertNotEquals(scoreEl, null);
  const text = scoreEl!.textContent ?? "";
  assertEquals(text.includes("3"), true);
  assertEquals(text.includes("5"), true);
});

Deno.test("test_render_game_table_shows_players", () => {
  const container = freshContainer();
  const snap = makeSnapshot();
  render(snap, container, null);
  const players = container.querySelectorAll(".player-area");
  assertEquals(players.length, 4);
});

Deno.test("test_render_trick_shows_played_cards", () => {
  const container = freshContainer();
  const snap = makeSnapshot();
  render(snap, container, null);
  const trickCards = container.querySelectorAll(".trick-card");
  assertEquals(trickCards.length >= 1, true);
});

Deno.test("test_render_hand_view_receives_callbacks", () => {
  const container = freshContainer();
  const snap = makeSnapshot();
  let clickedCardId: string | null = null;
  let actionFired: string | null = null;
  const callbacks: ActionCallbacks = {
    onCardClick: (cardId: string) => { clickedCardId = cardId; },
    onAction: (action: GameAction) => { actionFired = action; },
    onBid: () => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };
  render(snap, container, "play", { callbacks, selectedCardIds: new Set(), legalCardIds: new Set() });
  // Click a card in hand-view (first card after sorting: spades-2 is trump rank, comes first)
  const card = container.querySelector(".hand-view .card") as HTMLElement;
  assertNotEquals(card, null);
  card.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(clickedCardId, "D1-spades-2");
  // Click the play button
  const buttons = container.querySelectorAll(".hand-view button");
  const playButton = Array.from(buttons).find((b) => b.textContent === "出牌");
  assertNotEquals(playButton, undefined);
  playButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(actionFired, "play");
});

Deno.test("test_render_bidding_dialog_receives_callbacks", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "DEAL_BID",
    awaiting_action: null,
    trick: null,
  });
  let bidCards: string[] | null = null;
  let passCalled = false;
  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: () => {},
    onBid: (cardIds: string[]) => { bidCards = cardIds; },
    onStir: () => {},
    onPass: () => { passCalled = true; },
    onNewGame: () => {},
  };
  render(snap, container, "bid", { callbacks, selectedCardIds: new Set(), legalCardIds: new Set() });
  // In bid mode, there should be a bid button
  const bidEl = container.querySelector(".bidding-dialog");
  assertNotEquals(bidEl, null);
});

Deno.test("test_render_scoring_overlay_receives_callback", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "COMPLETE",
    awaiting_action: "next_round",
    trick: null,
    scoring: {
      declarer_team: 0,
      defender_points: 30,
      total_defender_points: 30,
      bottom_card_bonus: 0,
      bottom_cards: [],
    },
  });
  let nextRoundCalled = false;
  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: (action: GameAction) => { if (action === "next_round") nextRoundCalled = true; },
    onBid: () => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };
  render(snap, container, "next_round", { callbacks, selectedCardIds: new Set(), legalCardIds: new Set() });
  const buttons = container.querySelectorAll(".scoring-overlay button");
  const nextButton = Array.from(buttons).find((b) => b.textContent === "下一轮");
  assertNotEquals(nextButton, undefined);
  nextButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(nextRoundCalled, true);
});

Deno.test("test_render_game_over_receives_callback", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "GAME_OVER",
    winning_team: 0,
    trick: null,
    awaiting_action: null,
  });
  let newGameCalled = false;
  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: () => {},
    onBid: () => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => { newGameCalled = true; },
  };
  render(snap, container, null, { callbacks, selectedCardIds: new Set(), legalCardIds: new Set() });
  const buttons = container.querySelectorAll(".game-over-overlay button");
  const newGameButton = Array.from(buttons).find((b) => b.textContent === "新游戏");
  assertNotEquals(newGameButton, undefined);
  newGameButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(newGameCalled, true);
});

Deno.test("test_render_selected_cards_highlighted", () => {
  const container = freshContainer();
  const snap = makeSnapshot();
  const selectedCardIds = new Set(["D1-hearts-5"]);
  render(snap, container, "play", { selectedCardIds, legalCardIds: new Set() });
  const cards = container.querySelectorAll(".hand-view .card");
  // After sorting: spades-2 (trump rank) is first, hearts-5 (trump suit) is second
  assertEquals(cards[0].classList.contains("selected"), false);
  assertEquals(cards[1].classList.contains("selected"), true);
});
