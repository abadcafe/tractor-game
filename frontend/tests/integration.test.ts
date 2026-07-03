import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { StateManager } from "../core/state.ts";
import { GameLoop } from "../engine/game-loop.ts";
import { handlePlayAction } from "../engine/action-handler.ts";
import { validateBidCards } from "../engine/input-validator.ts";
import { render } from "../ui/renderer.ts";
import type { StateSnapshot } from "../core/types.ts";
import type { ClientAction, ServerMessage } from "../core/protocol.ts";
import type {
  BidOption,
  GameAction,
  InteractionMode,
} from "../engine/types.ts";
import type { ActionCallbacks, RenderContext } from "../ui/types.ts";
const HUMAN_PLAYER_INDEX = 2;

// deno-dom's Element is not structurally compatible with the DOM Element type
// expected by render() and GameLoop. Use this helper to create a properly-typed container.
function freshContainer(): Element {
  const doc = new DOMParser().parseFromString(
    `<html><body><div id="app"></div></body></html>`,
    "text/html",
  );
  // @ts-ignore test setup
  globalThis.document = doc;
  return doc!.querySelector("#app")! as unknown as Element;
}

function makeSnapshot(
  overrides: Partial<StateSnapshot> = {},
): StateSnapshot {
  return {
    phase: "PLAYING",
    player_hand: [
      { id: "D1-hearts-5", suit: "hearts", rank: "5" },
      { id: "D1-spades-2", suit: "spades", rank: "2" },
      { id: "D1-clubs-3", suit: "clubs", rank: "3" },
    ],
    bottom_cards: [],
    trump_rank: "2",
    trump_suit: "hearts",
    declarer_team: 0,
    declarer_player: 2,
    defender_points: 15,
    action_hints: [[{ id: "D1-hearts-5", suit: "hearts", rank: "5" }]],
    trick: null,
    last_completed_trick: null,
    defender_point_cards: [],
    bid_events: [],
    bid_winner: null,
    stir_events: [],
    own_initial_bottom_exchange: null,
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

// Track rendered state
let lastRenderedMode: InteractionMode = null;
let lastRenderedSnapshot: StateSnapshot | null = null;
let lastRenderedCtx: RenderContext | undefined = undefined;

function resetTrackingState(): void {
  lastRenderedMode = null;
  lastRenderedSnapshot = null;
  lastRenderedCtx = undefined;
}

function trackingRender(
  snapshot: StateSnapshot,
  container: Element,
  interactionMode: InteractionMode,
): void {
  lastRenderedMode = interactionMode;
  lastRenderedSnapshot = snapshot;
  render(snapshot, container, interactionMode, lastRenderedCtx);
}

Deno.test("test_integration_ws_to_render", () => {
  // Reset shared state
  resetTrackingState();
  const container = freshContainer();
  const stateManager = new StateManager();
  const gameLoop = new GameLoop(
    stateManager,
    trackingRender,
    container,
    3,
  );

  // Set up render context with callbacks so action buttons are rendered
  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: () => {},
    onBidOptionSelect: (_option: BidOption) => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };
  lastRenderedCtx = {
    callbacks,
    selectedCardIds: new Set(),
    legalCardIds: new Set(),
  };

  // Simulate WS message for PLAYING phase with user turn
  const msg: ServerMessage = {
    type: "state",
    seq: 1,
    state: makeSnapshot({ phase: "PLAYING", awaiting_action: "play" }),
  };
  gameLoop.handleMessage(msg);

  // Verify: state manager was updated
  assertEquals(stateManager.get()!.phase, "PLAYING");

  // Verify: renderer was called with correct interactionMode
  assertEquals(lastRenderedMode, "play");
  assertEquals(lastRenderedSnapshot!.phase, "PLAYING");

  // Verify: DOM was updated
  const handEl = container.querySelector(".hand-view");
  assertNotEquals(handEl, null);
  const playButton = container.querySelector(".action-panel button");
  assertNotEquals(playButton, null);
  assertEquals(playButton!.textContent, "出牌");
});

Deno.test("test_integration_play_action", () => {
  const container = freshContainer();
  const snap = makeSnapshot();

  // Step 1: Render with play mode and callbacks
  const selectedCardIds = new Set<string>();
  let sentAction: ClientAction | null = null;
  const callbacks: ActionCallbacks = {
    onCardClick: (cardId: string) => {
      if (selectedCardIds.has(cardId)) selectedCardIds.delete(cardId);
      else selectedCardIds.add(cardId);
    },
    onAction: (action: GameAction) => {
      if (action === "play") {
        const result = handlePlayAction(snap, selectedCardIds, 1);
        if (result.success && result.action) {
          sentAction = result.action;
        }
      }
    },
    onBidOptionSelect: (_option: BidOption) => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };

  render(snap, container, "play", {
    callbacks,
    selectedCardIds,
    legalCardIds: new Set(),
  });

  // Step 2: User selects a card by clicking the legal card (hearts-5)
  // Note: cards are sorted by trump rank first, so spades-2 comes before hearts-5
  const allCards = container.querySelectorAll(".hand-view .card");
  const card = Array.from(allCards).find((c) =>
    c.textContent?.includes("♥")
  ) as unknown as HTMLElement;
  assertNotEquals(card, undefined);
  card.dispatchEvent(new Event("click", { bubbles: true })); // triggers onCardClick -> adds to selectedCardIds

  // Step 3: User clicks play button
  const playButton = Array.from(
    container.querySelectorAll(".action-panel button"),
  )
    .find((b) => b.textContent === "出牌");
  assertNotEquals(playButton, undefined);
  playButton!.dispatchEvent(new Event("click", { bubbles: true })); // triggers onAction("play")

  // Step 4: Verify the action was constructed correctly
  assertNotEquals(sentAction, null);
  const action = sentAction as unknown as {
    type: "play";
    cards: string[];
  };
  assertEquals(action.type, "play");
  assertEquals(action.cards, ["D1-hearts-5"]);
});

Deno.test("test_integration_bid_action", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "DEAL_BID",
    awaiting_action: null,
    player_hand: [
      { id: "D1-spades-2", suit: "spades", rank: "2" },
      { id: "D1-hearts-5", suit: "hearts", rank: "5" },
    ],
  });

  // Step 1: Render with bid mode
  let selectedBidOption: BidOption | null = null;
  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: () => {},
    onBidOptionSelect: (option: BidOption) => {
      selectedBidOption = option;
    },
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };

  const bidOptions: BidOption[] = [
    {
      cardIds: ["D1-spades-2"],
      label: "♠2",
      trumpSuit: "spades",
      priority: 1,
    },
  ];

  render(snap, container, "bid", {
    callbacks,
    selectedCardIds: new Set(),
    legalCardIds: new Set(),
    bidOptions,
  });

  // Step 2: Verify the old bottom bidding dialog is not shown
  const bidEl = container.querySelector(".bidding-dialog");
  assertEquals(bidEl, null);

  // Step 3: Verify bid options are rendered as hand action buttons
  const buttons = container.querySelectorAll(
    ".hand-actions .action-panel--bid button",
  );
  assertEquals(buttons.length, 1);

  // Step 4: Click the bid option button to select it
  buttons[0].dispatchEvent(new Event("click", { bubbles: true }));
  assertNotEquals(selectedBidOption, null);
  assertEquals(selectedBidOption!.cardIds, ["D1-spades-2"]);
  assertEquals(selectedBidOption!.label, "♠2");
});

Deno.test("test_integration_stir_action", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "STIRRING",
    awaiting_action: "stir",
    stirring_state: {
      phase: "WAITING",
      trump_suit: null,
      current_player: HUMAN_PLAYER_INDEX,
      declarer_player: 0,
      exchanging_player: null,
      exchange_count: null,
    },
    player_hand: [
      { id: "D1-spades-2", suit: "spades", rank: "2" },
      { id: "D2-spades-2", suit: "spades", rank: "2" },
      { id: "D1-hearts-5", suit: "hearts", rank: "5" },
    ],
  });

  // Step 1: Render with stir mode
  let stirCardIds: string[] | null = null;
  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: () => {},
    onBidOptionSelect: (_option: BidOption) => {},
    onStir: (cardIds: string[]) => {
      stirCardIds = cardIds;
    },
    onPass: () => {},
    onNewGame: () => {},
  };

  const selectedCards = snap.player_hand.filter((c) => c.rank === "2");
  render(snap, container, "stir", {
    callbacks,
    selectedCardIds: new Set(selectedCards.map((card) => card.id)),
    legalCardIds: new Set(selectedCards.map((card) => card.id)),
    stirButtonState: { disabled: false },
  });

  // Step 2: Verify stir actions are rendered above the hand, not in the dialog
  const bidEl = container.querySelector(".bidding-dialog");
  assertEquals(bidEl, null);
  const stirButton = Array.from(
    container.querySelectorAll(".hand-actions button"),
  ).find((button) => button.textContent === "反主");
  assertNotEquals(stirButton, undefined);

  // Step 3: Validate
  const valid = validateBidCards(selectedCards, snap.trump_rank);
  assertEquals(valid, true);

  // Step 4: Click the hand-level stir button.
  stirButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(stirCardIds, ["D1-spades-2", "D2-spades-2"]);
});

Deno.test("test_integration_error_message", () => {
  // Reset shared state
  resetTrackingState();
  const container = freshContainer();
  const stateManager = new StateManager();
  let errorReceived: string | null = null;
  const gameLoop = new GameLoop(
    stateManager,
    trackingRender,
    container,
    3,
    undefined,
    (message) => {
      errorReceived = message;
    },
  );

  // First, establish a state
  const stateMsg: ServerMessage = {
    type: "state",
    seq: 1,
    state: makeSnapshot(),
  };
  gameLoop.handleMessage(stateMsg);
  const stateBefore = stateManager.get();
  assertNotEquals(stateBefore, null);

  // Now send a state message with an error
  const errMsg: ServerMessage = {
    type: "state",
    seq: 1,
    state: makeSnapshot(),
    error: "无效的出牌",
  };
  gameLoop.handleMessage(errMsg);

  // Verify: state was updated (state messages always update, even with errors)
  assertNotEquals(stateManager.get(), null);

  // Verify: error callback was called
  assertEquals(errorReceived, "无效的出牌");
});

Deno.test("test_integration_stir_not_human_ignored", () => {
  // Reset shared state
  resetTrackingState();
  const container = freshContainer();
  const stateManager = new StateManager();
  const gameLoop = new GameLoop(
    stateManager,
    trackingRender,
    container,
    3,
  );

  // STIRRING phase, but NOT user's turn.
  const msg: ServerMessage = {
    type: "state",
    seq: 1,
    state: makeSnapshot({
      phase: "STIRRING",
      awaiting_action: null,
      stirring_state: {
        phase: "WAITING",
        trump_suit: null,
        current_player: 1,
        declarer_player: 0,
        exchanging_player: null,
        exchange_count: null,
      },
    }),
  };
  gameLoop.handleMessage(msg);

  // Verify: interactionMode is null (spectator mode)
  assertEquals(lastRenderedMode, null);

  // Verify: no interactive buttons in the rendered output
  render(stateManager.get()!, container, null, undefined);
  // In spectator mode, the action-panel should have no buttons
  const handEl = container.querySelector(".hand-view");
  if (handEl) {
    const buttons = handEl.querySelectorAll("button");
    assertEquals(buttons.length, 0);
  }
});

Deno.test("test_integration_card_selection_persists_across_renders", () => {
  const container = freshContainer();
  const snap = makeSnapshot();
  const selectedCardIds = new Set<string>();

  const callbacks: ActionCallbacks = {
    onCardClick: (cardId: string) => {
      if (selectedCardIds.has(cardId)) selectedCardIds.delete(cardId);
      else selectedCardIds.add(cardId);
    },
    onAction: () => {},
    onBidOptionSelect: (_option: BidOption) => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };

  // First render -- no cards selected
  render(snap, container, "play", {
    callbacks,
    selectedCardIds,
    legalCardIds: new Set(),
  });
  let cards = container.querySelectorAll(".hand-view .card");
  // Cards are sorted by trump rank first: spades-2 (trump rank), hearts-5 (trump suit), clubs-3
  const hearts5Card = Array.from(cards).find((c) =>
    c.textContent?.includes("♥")
  )!;
  assertEquals(hearts5Card.classList.contains("selected"), false);

  // Simulate clicking hearts-5 card
  callbacks.onCardClick("D1-hearts-5");
  assertEquals(selectedCardIds.has("D1-hearts-5"), true);

  // Re-render with same selectedCardIds -- selection should persist
  render(snap, container, "play", {
    callbacks,
    selectedCardIds,
    legalCardIds: new Set(),
  });
  cards = container.querySelectorAll(".hand-view .card");
  const selectedCard = Array.from(cards).find((c) =>
    c.classList.contains("selected")
  );
  assertNotEquals(selectedCard, undefined);
  assertEquals(selectedCard!.textContent?.includes("♥"), true);
  // Verify only one card is selected
  const allSelected = Array.from(cards).filter((c) =>
    c.classList.contains("selected")
  );
  assertEquals(allSelected.length, 1);
});

Deno.test("test_integration_callback_triggers_send", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "WAITING",
    awaiting_action: "next_round",
    trick: null,
    scoring: {
      round_winning_team: 0,
      defender_points: 30,
      total_defender_points: 30,
      bottom_card_bonus: 0,
      bottom_cards: [],
    },
  });

  let sentAction: ClientAction | null = null;
  const mockSend = (action: ClientAction) => {
    sentAction = action;
  };

  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: (action: GameAction) => {
      if (action === "next_round") {
        mockSend({ type: "next_round", seq: 1 });
      }
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

  // Click the "next round" button
  const nextButton = Array.from(
    container.querySelectorAll(".scoring-overlay button"),
  )
    .find((b) => b.textContent === "下一轮");
  assertNotEquals(nextButton, undefined);
  nextButton!.dispatchEvent(new Event("click", { bubbles: true }));

  // Verify the action was sent
  assertNotEquals(sentAction, null);
  assertEquals(sentAction!.type, "next_round");
});
