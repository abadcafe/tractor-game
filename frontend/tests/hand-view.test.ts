import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderHandView } from "../ui/components/hand-view.ts";
import type { StateSnapshot } from "../core/types.ts";
import type { BidOption, InteractionMode } from "../engine/types.ts";

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
    defender_points: 0,
    action_hints: [[{ id: "D1-hearts-5", suit: "hearts", rank: "5" }]],
    trick: null,
    trick_history: [],
    failed_throw: null,
    bid_events: [],
    bid_winner: null,
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

Deno.test("test_renderHandView_displays_cards", () => {
  const snap = makeSnapshot();
  const el = renderHandView(snap, "play");
  const cards = el.querySelectorAll(".card");
  assertEquals(cards.length, 2);
});

Deno.test("test_renderHandView_selected_card", () => {
  const snap = makeSnapshot();
  const selectedIds = new Set(["D1-hearts-5"]);
  const legalCardIds = new Set(["D1-hearts-5"]);
  const el = renderHandView(snap, "play", selectedIds, legalCardIds);
  const cards = el.querySelectorAll(".card");
  // After sorting: spades-2 (trump rank) is first, hearts-5 (trump suit) is second
  // hearts-5 should be selected
  const selectedCard = Array.from(cards).find((c) =>
    c.classList.contains("selected")
  );
  assertNotEquals(selectedCard, undefined);
});

Deno.test("test_renderHandView_legal_highlight", () => {
  const snap = makeSnapshot();
  const legalCardIds = new Set(["D1-hearts-5"]);
  const el = renderHandView(snap, "play", undefined, legalCardIds);
  // The legal card (hearts-5) should have the .legal class
  const legalCards = el.querySelectorAll(".card.legal");
  assertEquals(legalCards.length >= 1, true);
});

Deno.test("test_renderHandView_play_button", () => {
  const snap = makeSnapshot();
  const onAction = (_action: string) => {};
  const el = renderHandView(
    snap,
    "play",
    undefined,
    undefined,
    undefined,
    onAction,
  );
  const buttons = el.querySelectorAll("button");
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("出牌"), true);
});

Deno.test("test_renderHandView_previous_trick_button_above_hand", () => {
  const snap = makeSnapshot({
    trick_history: [{
      lead_player: 0,
      winner: 3,
      points: 10,
      slots: [
        {
          player: 0,
          cards: [{ id: "D1-diamonds-3", suit: "diamonds", rank: "3" }],
        },
        {
          player: 1,
          cards: [{ id: "D1-diamonds-4", suit: "diamonds", rank: "4" }],
        },
        {
          player: 2,
          cards: [{ id: "D2-diamonds-K", suit: "diamonds", rank: "K" }],
        },
        {
          player: 3,
          cards: [{ id: "D1-diamonds-K", suit: "diamonds", rank: "K" }],
        },
      ],
    }],
  });
  let called = false;
  const el = renderHandView(
    snap,
    null,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    false,
    undefined,
    undefined,
    undefined,
    () => {
      called = true;
    },
  );

  const previousPanel = el.querySelector(
    ".hand-actions .hand-panel__previous-trick",
  );
  const previousButton = Array.from(el.querySelectorAll("button")).find(
    (
      button,
    ) => button.textContent === "上一墩",
  );

  assertNotEquals(previousPanel, null);
  assertNotEquals(previousButton, undefined);

  previousButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(called, true);
});

Deno.test("test_renderHandView_previous_trick_button_hidden_without_history", () => {
  const snap = makeSnapshot();
  const el = renderHandView(
    snap,
    null,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    false,
    undefined,
    undefined,
    undefined,
    () => {},
  );

  const previousButton = Array.from(el.querySelectorAll("button")).find(
    (
      button,
    ) => button.textContent === "上一墩",
  );

  assertEquals(previousButton, undefined);
});

Deno.test("test_renderHandView_actions_and_hint_tools_are_above_hand", () => {
  const snap = makeSnapshot();
  const el = renderHandView(
    snap,
    "play",
    new Set(["D1-hearts-5"]),
    undefined,
    undefined,
    () => {},
    () => {},
    () => {},
  );

  const handActions = el.querySelector(".hand-actions");
  const handPanelButton = el.querySelector(".hand-panel button");
  const summaryText = el.textContent ?? "";
  const actionTools = el.querySelector(
    ".hand-actions .hand-panel__tools",
  );
  const actionPanel = el.querySelector(
    ".hand-actions .action-panel",
  );
  const actionPanelButtons = Array.from(
    el.querySelectorAll(".action-panel button"),
  ).map((button) => button.textContent);

  assertNotEquals(handActions, null);
  assertEquals(handPanelButton, null);
  assertNotEquals(actionTools, null);
  assertNotEquals(actionPanel, null);
  assertEquals(summaryText.includes("自由出牌"), false);
  assertEquals(summaryText.includes("服务器校验"), false);
  assertEquals(actionTools?.textContent?.includes("提示"), true);
  assertEquals(actionTools?.textContent?.includes("清牌"), true);
  assertEquals(actionPanelButtons, ["出牌"]);
});

Deno.test("test_renderHandView_does_not_show_compact_sort_button", () => {
  const snap = makeSnapshot();
  const onAction = (_action: string) => {};
  const el = renderHandView(
    snap,
    "play",
    undefined,
    undefined,
    undefined,
    onAction,
    () => {},
    () => {},
    () => {},
  );
  const buttonTexts = Array.from(el.querySelectorAll("button")).map((
    b,
  ) => b.textContent);
  assertEquals(buttonTexts.includes("理牌"), false);
  assertEquals(buttonTexts.includes("展开"), false);
});

Deno.test("test_renderHandView_score_pile_shows_captured_point_cards", () => {
  const snap = makeSnapshot({
    defender_points: 15,
    trick_history: [{
      lead_player: 0,
      winner: 1,
      points: 15,
      slots: [
        {
          player: 0,
          cards: [{ id: "D1-clubs-5", suit: "clubs", rank: "5" }],
        },
        {
          player: 1,
          cards: [{ id: "D1-spades-2", suit: "spades", rank: "2" }],
        },
        {
          player: 2,
          cards: [{ id: "D1-hearts-K", suit: "hearts", rank: "K" }],
        },
        {
          player: 3,
          cards: [{ id: "D1-diamonds-4", suit: "diamonds", rank: "4" }],
        },
      ],
    }],
  });
  const el = renderHandView(snap, "play");
  const text = el.querySelector(".score-pile__label")?.textContent ??
    "";
  const scoreCards = el.querySelectorAll(".score-pile-card");
  assertEquals(text.includes("捡分 15"), true);
  assertEquals(scoreCards.length, 2);
});

Deno.test("test_renderHandView_discard_button", () => {
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
  });
  const onAction = (_action: string) => {};
  const el = renderHandView(
    snap,
    "discard",
    undefined,
    undefined,
    undefined,
    onAction,
  );
  const buttons = el.querySelectorAll("button");
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("换底牌"), true);
});

Deno.test("test_renderHandView_stir_buttons_are_above_hand", () => {
  const snap = makeSnapshot({
    phase: "STIRRING",
    awaiting_action: "stir",
    player_hand: [
      { id: "D1-spades-2", suit: "spades", rank: "2" },
      { id: "D2-spades-2", suit: "spades", rank: "2" },
    ],
    stirring_state: {
      phase: "WAITING",
      trump_suit: null,
      current_player: 3,
      declarer_player: 0,
      exchanging_player: null,
      exchange_count: null,
    },
  });
  let stirCardIds: string[] | null = null;
  let passCalled = false;
  const el = renderHandView(
    snap,
    "stir",
    new Set(["D1-spades-2", "D2-spades-2"]),
    new Set(["D1-spades-2", "D2-spades-2"]),
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    false,
    (cardIds: string[]) => {
      stirCardIds = cardIds;
    },
    () => {
      passCalled = true;
    },
    { disabled: false },
  );

  const summaryActionPanel = el.querySelector(
    ".hand-actions .action-panel--stir",
  );
  assertNotEquals(summaryActionPanel, null);

  const stirButton = Array.from(el.querySelectorAll("button")).find((
    button,
  ) => button.textContent === "反主");
  const passButton = Array.from(el.querySelectorAll("button")).find((
    button,
  ) => button.textContent === "不反");
  assertNotEquals(stirButton, undefined);
  assertNotEquals(passButton, undefined);

  stirButton!.dispatchEvent(new Event("click", { bubbles: true }));
  passButton!.dispatchEvent(new Event("click", { bubbles: true }));

  assertEquals(stirCardIds, ["D1-spades-2", "D2-spades-2"]);
  assertEquals(passCalled, true);
});

Deno.test("test_renderHandView_bid_options_are_above_hand", () => {
  const snap = makeSnapshot({
    phase: "DEAL_BID",
    awaiting_action: "bid",
    action_hints: [],
    trump_suit: null,
  });
  const bidOptions: BidOption[] = [
    {
      cardIds: ["D1-spades-2"],
      label: "♠2",
      trumpSuit: "spades",
      priority: 103,
    },
    {
      cardIds: ["D1-hearts-2"],
      label: "♥2",
      trumpSuit: "hearts",
      priority: 102,
    },
  ];
  let selectedOption: BidOption | null = null;
  const el = renderHandView(
    snap,
    "bid",
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    false,
    undefined,
    undefined,
    undefined,
    undefined,
    bidOptions,
    bidOptions[1],
    (option: BidOption) => {
      selectedOption = option;
    },
  );

  const bidPanel = el.querySelector(".hand-actions .action-panel--bid");
  const handPanelButton = el.querySelector(".hand-panel button");
  const buttons = Array.from(
    el.querySelectorAll(".action-panel--bid button"),
  );

  assertNotEquals(bidPanel, null);
  assertEquals(el.classList.contains("has-bid-actions"), true);
  assertEquals(handPanelButton, null);
  assertEquals(buttons.map((button) => button.textContent), [
    "♠2",
    "♥2",
  ]);
  assertEquals(buttons[1].classList.contains("selected"), true);

  buttons[0].dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(selectedOption, bidOptions[0]);
});

Deno.test("test_renderHandView_bid_option_colors_match_suits_and_jokers", () => {
  const snap = makeSnapshot({
    phase: "DEAL_BID",
    awaiting_action: "bid",
    action_hints: [],
    trump_suit: null,
  });
  const bidOptions: BidOption[] = [
    {
      cardIds: ["D1-joker-BJ", "D2-joker-BJ"],
      label: "大王对",
      trumpSuit: null,
      priority: 205,
    },
    {
      cardIds: ["D1-joker-SJ", "D2-joker-SJ"],
      label: "小王对",
      trumpSuit: null,
      priority: 204,
    },
    {
      cardIds: ["D1-hearts-2"],
      label: "♥2",
      trumpSuit: "hearts",
      priority: 102,
    },
    {
      cardIds: ["D1-diamonds-2"],
      label: "♦2",
      trumpSuit: "diamonds",
      priority: 100,
    },
    {
      cardIds: ["D1-spades-2"],
      label: "♠2",
      trumpSuit: "spades",
      priority: 103,
    },
    {
      cardIds: ["D1-clubs-2"],
      label: "♣2",
      trumpSuit: "clubs",
      priority: 101,
    },
  ];

  const el = renderHandView(
    snap,
    "bid",
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    false,
    undefined,
    undefined,
    undefined,
    undefined,
    bidOptions,
    null,
    () => {},
  );
  const buttons = Array.from(
    el.querySelectorAll(".action-panel--bid button"),
  );

  assertEquals(
    buttons.map((button) =>
      button.classList.contains("bid-button-red")
    ),
    [
      true,
      false,
      true,
      true,
      false,
      false,
    ],
  );
  assertEquals(
    buttons.map((button) =>
      button.classList.contains("bid-button-black")
    ),
    [
      false,
      true,
      false,
      false,
      true,
      true,
    ],
  );
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
  const onCardClick = (cardId: string) => {
    clickedCardId = cardId;
  };
  const el = renderHandView(
    snap,
    "play",
    undefined,
    undefined,
    onCardClick,
  );
  // Simulate clicking the first card (spades-2 after sorting)
  const firstCard = el.querySelector(".card") as HTMLElement;
  assertNotEquals(firstCard, null);
  firstCard.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(clickedCardId, "D1-spades-2");
});

Deno.test("test_renderHandView_play_mode_hint_cards_do_not_block_other_cards", () => {
  const snap = makeSnapshot();
  let clickedCardId: string | null = null;
  const onCardClick = (cardId: string) => {
    clickedCardId = cardId;
  };
  const legalCardIds = new Set(["D1-hearts-5"]);
  const el = renderHandView(
    snap,
    "play",
    undefined,
    legalCardIds,
    onCardClick,
  );

  const nonHintCard = Array.from(el.querySelectorAll(".card")).find(
    (card) => card.getAttribute("data-card-id") === "D1-spades-2",
  );
  assertNotEquals(nonHintCard, undefined);
  nonHintCard!.dispatchEvent(new Event("click", { bubbles: true }));

  assertEquals(clickedCardId, "D1-spades-2");
});

Deno.test("test_renderHandView_action_button_callback", () => {
  const snap = makeSnapshot();
  let actionFired: string | null = null;
  const onAction = (action: string) => {
    actionFired = action;
  };
  const el = renderHandView(
    snap,
    "play",
    undefined,
    undefined,
    undefined,
    onAction,
  );
  // Find the play button and click it
  const buttons = el.querySelectorAll("button");
  const playButton = Array.from(buttons).find((b) =>
    b.textContent === "出牌"
  );
  assertNotEquals(playButton, undefined);
  playButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(actionFired, "play");
});
