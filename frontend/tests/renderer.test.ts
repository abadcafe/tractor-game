import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { render } from "../ui/renderer.ts";
import type { StateSnapshot } from "../core/types.ts";
import type {
  BidOption,
  GameAction,
  InteractionMode,
} from "../engine/types.ts";
import type { ActionCallbacks } from "../ui/types.ts";

function makeSnapshot(
  overrides: Partial<StateSnapshot> = {},
): StateSnapshot {
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
    defender_points: 15,
    action_hints: [[{ id: "D1-hearts-5", suit: "hearts", rank: "5" }]],
    trick: {
      lead_player: 0,
      slots: [{
        player: 0,
        cards: [{ id: "D1-clubs-7", suit: "clubs", rank: "7" }],
      }],
      current_player: 1,
    },
    last_completed_trick: null,
    defender_point_cards: [],
    failed_throw: null,
    bid_events: [],
    bid_winner: null,
    awaiting_action: "play",
    stirring_state: null,
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

function callbacksStub(): ActionCallbacks {
  return {
    onCardClick: () => {},
    onAction: () => {},
    onBidOptionSelect: (_option: BidOption) => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };
}

function countButtonsByText(container: Element, text: string): number {
  return Array.from(container.querySelectorAll("button"))
    .filter((button) => button.textContent === text)
    .length;
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
  const snap = makeSnapshot({
    phase: "DEAL_BID",
    awaiting_action: null,
  });
  render(snap, container, "bid");
  const bidEl = container.querySelector(".bidding-dialog");
  assertEquals(bidEl, null);
});

Deno.test("test_render_stirring_phase", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "STIRRING",
    awaiting_action: "stir",
    stirring_state: {
      phase: "WAITING",
      trump_suit: null,
      current_player: 3,
      declarer_player: 0,
      exchanging_player: null,
      exchange_count: null,
    },
    trick: null,
  });
  render(snap, container, "stir");
  const handEl = container.querySelector(".hand-view");
  assertNotEquals(handEl, null);
  const stirEl = container.querySelector(".bidding-dialog");
  assertEquals(stirEl, null);
});

Deno.test("test_render_complete_phase", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "WAITING",
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

Deno.test("test_render_complete_phase_keeps_last_trick_on_table", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "WAITING",
    awaiting_action: "next_round",
    trick: null,
    last_completed_trick: {
      lead_player: 0,
      winner: 1,
      points: 25,
      slots: [
        {
          player: 0,
          cards: [{ id: "D1-clubs-5", suit: "clubs", rank: "5" }],
        },
        {
          player: 1,
          cards: [{ id: "D1-hearts-10", suit: "hearts", rank: "10" }],
        },
        {
          player: 2,
          cards: [{ id: "D1-spades-K", suit: "spades", rank: "K" }],
        },
        {
          player: 3,
          cards: [{ id: "D1-diamonds-A", suit: "diamonds", rank: "A" }],
        },
      ],
    },
    scoring: {
      declarer_team: 0,
      defender_points: 75,
      total_defender_points: 100,
      bottom_card_bonus: 25,
      bottom_cards: [],
    },
  });
  render(snap, container, "next_round");

  assertNotEquals(container.querySelector(".scoring-overlay"), null);
  assertEquals(
    container.querySelectorAll(".trick-view .trick-card").length,
    4,
  );
  assertEquals(
    container.querySelectorAll(".trick-view .trick-slot.winner").length,
    1,
  );
  assertEquals(
    container.querySelectorAll(
      ".trick-view .trick-slot.lead .trick-lead-marker",
    )
      .length,
    1,
  );
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
    phase: "STIRRING",
    awaiting_action: "discard",
    stirring_state: {
      phase: "WAITING",
      trump_suit: null,
      current_player: 3,
      declarer_player: 0,
      exchanging_player: 3,
      exchange_count: 8,
    },
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
    onCardClick: (cardId: string) => {
      clickedCardId = cardId;
    },
    onAction: (action: GameAction) => {
      actionFired = action;
    },
    onBidOptionSelect: (_option: BidOption) => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };
  render(snap, container, "play", {
    callbacks,
    selectedCardIds: new Set(),
    legalCardIds: new Set(),
  });
  // Click a card in hand-view (first card after sorting: spades-2 is trump rank, comes first)
  const card = container.querySelector(
    ".hand-view .card",
  ) as HTMLElement;
  assertNotEquals(card, null);
  card.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(clickedCardId, "D1-spades-2");
  // Click the play button
  const buttons = container.querySelectorAll(".action-panel button");
  const playButton = Array.from(buttons).find((b) =>
    b.textContent === "出牌"
  );
  assertNotEquals(playButton, undefined);
  playButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(actionFired, "play");
});

Deno.test("test_render_bid_options_above_hand_receives_callbacks", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "DEAL_BID",
    awaiting_action: "bid",
    trick: null,
  });
  let selectedOption: BidOption | null = null;
  const bidOptions: BidOption[] = [{
    cardIds: ["D1-spades-2"],
    label: "♠2",
    trumpSuit: "spades",
    priority: 103,
  }];
  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: () => {},
    onBidOptionSelect: (option: BidOption) => {
      selectedOption = option;
    },
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };
  render(snap, container, "bid", {
    callbacks,
    selectedCardIds: new Set(),
    legalCardIds: new Set(),
    bidOptions,
  });
  const bidEl = container.querySelector(".bidding-dialog");
  const bidButton = container.querySelector(
    ".hand-actions .action-panel--bid button",
  );
  assertEquals(bidEl, null);
  assertNotEquals(bidButton, null);
  bidButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(selectedOption, bidOptions[0]);
});

Deno.test("test_render_scoring_overlay_receives_callback", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "WAITING",
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
    onAction: (action: GameAction) => {
      if (action === "next_round") nextRoundCalled = true;
    },
    onBidOptionSelect: (_option: BidOption) => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };
  render(snap, container, "next_round", {
    callbacks,
    selectedCardIds: new Set(),
    legalCardIds: new Set(),
  });
  const buttons = container.querySelectorAll(".scoring-overlay button");
  const nextButton = Array.from(buttons).find((b) =>
    b.textContent === "下一轮"
  );
  assertNotEquals(nextButton, undefined);
  nextButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(nextRoundCalled, true);
});

Deno.test("test_render_waiting_has_single_next_round_button", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "WAITING",
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
  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: () => {},
    onBidOptionSelect: (_option: BidOption) => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };

  render(snap, container, "next_round", {
    callbacks,
    selectedCardIds: new Set(),
    legalCardIds: new Set(),
  });

  const nextRoundButtons = Array.from(
    container.querySelectorAll("button"),
  )
    .filter((button) => button.textContent === "下一轮");

  assertEquals(nextRoundButtons.length, 1);
  assertNotEquals(
    container.querySelector(".scoring-overlay__next-round"),
    null,
  );
  assertEquals(
    container.querySelector(".hand-actions .hand-action-button"),
    null,
  );
});

Deno.test("test_render_primary_actions_are_not_duplicated", () => {
  const callbacks = callbacksStub();

  const playContainer = freshContainer();
  render(makeSnapshot(), playContainer, "play", {
    callbacks,
    selectedCardIds: new Set(["D1-hearts-5"]),
    legalCardIds: new Set(["D1-hearts-5"]),
  });
  assertEquals(countButtonsByText(playContainer, "出牌"), 1);

  const discardContainer = freshContainer();
  render(
    makeSnapshot({
      phase: "STIRRING",
      awaiting_action: "discard",
      stirring_state: {
        phase: "EXCHANGING",
        trump_suit: "spades",
        current_player: 3,
        declarer_player: 3,
        exchanging_player: 3,
        exchange_count: 2,
      },
    }),
    discardContainer,
    "discard",
    {
      callbacks,
      selectedCardIds: new Set(["D1-hearts-5", "D1-spades-2"]),
      legalCardIds: new Set(["D1-hearts-5", "D1-spades-2"]),
    },
  );
  assertEquals(countButtonsByText(discardContainer, "换底牌"), 1);

  const stirContainer = freshContainer();
  render(
    makeSnapshot({
      phase: "STIRRING",
      awaiting_action: "stir",
      trick: null,
      stirring_state: {
        phase: "WAITING",
        trump_suit: "spades",
        current_player: 3,
        declarer_player: 0,
        exchanging_player: null,
        exchange_count: null,
      },
    }),
    stirContainer,
    "stir",
    {
      callbacks,
      selectedCardIds: new Set(["D1-spades-2"]),
      legalCardIds: new Set(["D1-spades-2"]),
      stirButtonState: { disabled: false },
    },
  );
  assertEquals(countButtonsByText(stirContainer, "反主"), 1);
  assertEquals(countButtonsByText(stirContainer, "不反"), 1);

  const waitingContainer = freshContainer();
  render(
    makeSnapshot({
      phase: "WAITING",
      awaiting_action: "next_round",
      trick: null,
      scoring: {
        declarer_team: 0,
        defender_points: 30,
        total_defender_points: 30,
        bottom_card_bonus: 0,
        bottom_cards: [],
      },
    }),
    waitingContainer,
    "next_round",
    {
      callbacks,
      selectedCardIds: new Set(),
      legalCardIds: new Set(),
    },
  );
  assertEquals(countButtonsByText(waitingContainer, "下一轮"), 1);

  const gameOverContainer = freshContainer();
  render(
    makeSnapshot({
      phase: "GAME_OVER",
      awaiting_action: null,
      winning_team: 0,
    }),
    gameOverContainer,
    null,
    {
      callbacks,
      selectedCardIds: new Set(),
      legalCardIds: new Set(),
    },
  );
  assertEquals(countButtonsByText(gameOverContainer, "新游戏"), 1);
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
    onBidOptionSelect: (_option: BidOption) => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {
      newGameCalled = true;
    },
  };
  render(snap, container, null, {
    callbacks,
    selectedCardIds: new Set(),
    legalCardIds: new Set(),
  });
  const buttons = container.querySelectorAll(
    ".game-over-overlay button",
  );
  const newGameButton = Array.from(buttons).find((b) =>
    b.textContent === "新游戏"
  );
  assertNotEquals(newGameButton, undefined);
  newGameButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(newGameCalled, true);
});

Deno.test("test_render_selected_cards_highlighted", () => {
  const container = freshContainer();
  const snap = makeSnapshot();
  const selectedCardIds = new Set(["D1-hearts-5"]);
  render(snap, container, "play", {
    selectedCardIds,
    legalCardIds: new Set(),
  });
  const cards = container.querySelectorAll(".hand-view .card");
  // After sorting: spades-2 (trump rank) is first, hearts-5 (trump suit) is second
  assertEquals(cards[0].classList.contains("selected"), false);
  assertEquals(cards[1].classList.contains("selected"), true);
});
