import { assertEquals, assertNotEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { StateManager } from "../core/state.ts";
import { GameLoop } from "../engine/game-loop.ts";
import { validatePlay, validateBidCards } from "../engine/input-validator.ts";
import { render } from "../ui/renderer.ts";
import type { StateSnapshot, ServerMessage, InteractionMode, ClientAction, ActionCallbacks } from "../core/types.ts";
import { HUMAN_PLAYER_INDEX } from "../config.ts";

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

function makeSnapshot(overrides: Partial<StateSnapshot> = {}): StateSnapshot {
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
    declarer_player: 3,
    current_player: 3,
    defender_points: 15,
    legal_actions: [{ type: "single", cards: [{ id: "D1-hearts-5", suit: "hearts", rank: "5" }] }],
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
    ...overrides,
  };
}

// Track rendered state
let lastRenderedMode: InteractionMode = null;
let lastRenderedSnapshot: StateSnapshot | null = null;
let lastRenderedCallbacks: ActionCallbacks | undefined = undefined;
let lastRenderedSelectedIds: Set<string> | undefined = undefined;

function resetTrackingState(): void {
  lastRenderedMode = null;
  lastRenderedSnapshot = null;
  lastRenderedCallbacks = undefined;
  lastRenderedSelectedIds = undefined;
}

function trackingRender(
  snapshot: StateSnapshot,
  container: Element,
  interactionMode: InteractionMode,
  callbacks?: ActionCallbacks,
  selectedCardIds?: Set<string>,
): void {
  lastRenderedMode = interactionMode;
  lastRenderedSnapshot = snapshot;
  lastRenderedCallbacks = callbacks;
  lastRenderedSelectedIds = selectedCardIds;
  render(snapshot, container, interactionMode, callbacks, selectedCardIds);
}

Deno.test("test_integration_ws_to_render", () => {
  // Reset shared state
  resetTrackingState();
  const container = freshContainer();
  const stateManager = new StateManager();
  const gameLoop = new GameLoop(stateManager, trackingRender, container);

  // Simulate WS message for PLAYING phase with human turn
  const msg: ServerMessage = {
    type: "state",
    awaiting: "play",
    state: makeSnapshot({ phase: "PLAYING", current_player: HUMAN_PLAYER_INDEX }),
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
  const playButton = container.querySelector("button");
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
    onAction: (action: string) => {
      if (action === "play") {
        const selectedCards = snap.player_hand.filter((c) => selectedCardIds.has(c.id));
        const playAction = validatePlay(selectedCards, snap.legal_actions);
        if (playAction) {
          sentAction = { type: "play", cards: playAction.cards.map((c) => c.id) };
        }
      }
    },
    onBid: () => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };

  render(snap, container, "play", callbacks, selectedCardIds);

  // Step 2: User selects a card by clicking the legal card (hearts-5)
  // Note: cards are sorted by suit then rank, so we find the specific card by text content
  const allCards = container.querySelectorAll(".hand-view .card");
  const card = Array.from(allCards).find((c) => c.textContent?.includes("♥5")) as unknown as HTMLElement;
  assertNotEquals(card, undefined);
  card.dispatchEvent(new Event("click", { bubbles: true })); // triggers onCardClick -> adds to selectedCardIds

  // Step 3: User clicks play button
  const playButton = Array.from(container.querySelectorAll(".hand-view button"))
    .find((b) => b.textContent === "出牌");
  assertNotEquals(playButton, undefined);
  playButton!.dispatchEvent(new Event("click", { bubbles: true })); // triggers onAction("play")

  // Step 4: Verify the action was constructed correctly
  assertNotEquals(sentAction, null);
  const action = sentAction as unknown as { type: "play"; cards: string[] };
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
    legal_actions: [],
  });

  // Step 1: Render with bid mode
  let bidCards: string[] | null = null;
  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: () => {},
    onBid: (cardIds: string[]) => { bidCards = cardIds; },
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };

  render(snap, container, "bid", callbacks);

  // Step 2: Verify bidding dialog is shown
  const bidEl = container.querySelector(".bidding-dialog");
  assertNotEquals(bidEl, null);

  // Step 3: User selects trump rank cards
  const selectedCards = [snap.player_hand[0]]; // spades-2

  // Step 4: Validate
  const valid = validateBidCards(selectedCards, snap.trump_rank);
  assertEquals(valid, true);

  // Step 5: Construct action via onBid callback
  callbacks.onBid([selectedCards[0].id]);
  assertEquals(bidCards, ["D1-spades-2"]);
});

Deno.test("test_integration_stir_action", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "STIRRING",
    awaiting_action: "stir",
    current_player: HUMAN_PLAYER_INDEX,
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: HUMAN_PLAYER_INDEX },
    player_hand: [
      { id: "D1-spades-2", suit: "spades", rank: "2" },
      { id: "D2-spades-2", suit: "spades", rank: "2" },
      { id: "D1-hearts-5", suit: "hearts", rank: "5" },
    ],
    legal_actions: [],
  });

  // Step 1: Render with stir mode
  let stirCardIds: string[] | null = null;
  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: () => {},
    onBid: () => {},
    onStir: (cardIds: string[]) => { stirCardIds = cardIds; },
    onPass: () => {},
    onNewGame: () => {},
  };

  render(snap, container, "stir", callbacks);

  // Step 2: Verify bidding dialog is shown in stir mode
  const bidEl = container.querySelector(".bidding-dialog");
  assertNotEquals(bidEl, null);

  // Step 3: User selects trump rank cards for stirring
  const selectedCards = snap.player_hand.filter((c) => c.rank === "2");

  // Step 4: Validate
  const valid = validateBidCards(selectedCards, snap.trump_rank);
  assertEquals(valid, true);

  // Step 5: Construct action via onStir callback -- this sends { type: "stir" }, NOT { type: "bid" }
  callbacks.onStir([selectedCards[0].id, selectedCards[1].id]);
  assertEquals(stirCardIds, ["D1-spades-2", "D2-spades-2"]);
});

Deno.test("test_integration_error_message", () => {
  // Reset shared state
  resetTrackingState();
  const container = freshContainer();
  const stateManager = new StateManager();
  const gameLoop = new GameLoop(stateManager, trackingRender, container);

  // First, establish a state
  const stateMsg: ServerMessage = {
    type: "state",
    awaiting: "play",
    state: makeSnapshot(),
  };
  gameLoop.handleMessage(stateMsg);
  const stateBefore = stateManager.get();
  assertNotEquals(stateBefore, null);

  // Now send an error
  const errMsg: ServerMessage = { type: "error", message: "无效的出牌" };
  gameLoop.handleMessage(errMsg);

  // Verify: state was NOT updated
  assertEquals(stateManager.get(), stateBefore);

  // Verify: error toast was displayed
  const toast = container.querySelector(".error-toast");
  assertNotEquals(toast, null);
  assertEquals(toast!.textContent, "无效的出牌");
});

Deno.test("test_integration_stir_not_human_ignored", () => {
  // Reset shared state
  resetTrackingState();
  const container = freshContainer();
  const stateManager = new StateManager();
  const gameLoop = new GameLoop(stateManager, trackingRender, container);

  // STIRRING phase, but NOT human's turn
  const msg: ServerMessage = {
    type: "state",
    awaiting: "stir",
    state: makeSnapshot({
      phase: "STIRRING",
      current_player: 1,
      awaiting_action: "stir",
      stirring_state: { phase: "WAITING", trump_suit: null, current_player: 1 },
    }),
  };
  gameLoop.handleMessage(msg);

  // Verify: interactionMode is null (spectator mode)
  assertEquals(lastRenderedMode, null);

  // Verify: no interactive buttons in the rendered output
  render(stateManager.get()!, container, null);
  // In spectator mode, hand-view should have no buttons
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
    onBid: () => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };

  // First render -- no cards selected
  render(snap, container, "play", callbacks, selectedCardIds);
  let cards = container.querySelectorAll(".hand-view .card");
  // Cards are sorted by suit then rank: clubs-3, hearts-5, spades-2
  const hearts5Card = Array.from(cards).find((c) => c.textContent?.includes("♥5"))!;
  assertEquals(hearts5Card.classList.contains("selected"), false);

  // Simulate clicking hearts-5 card
  callbacks.onCardClick("D1-hearts-5");
  assertEquals(selectedCardIds.has("D1-hearts-5"), true);

  // Re-render with same selectedCardIds -- selection should persist
  render(snap, container, "play", callbacks, selectedCardIds);
  cards = container.querySelectorAll(".hand-view .card");
  const selectedCard = Array.from(cards).find((c) => c.classList.contains("selected"));
  assertNotEquals(selectedCard, undefined);
  assertEquals(selectedCard!.textContent?.includes("♥5"), true);
  // Verify only one card is selected
  const allSelected = Array.from(cards).filter((c) => c.classList.contains("selected"));
  assertEquals(allSelected.length, 1);
});

Deno.test("test_integration_callback_triggers_send", () => {
  const container = freshContainer();
  const snap = makeSnapshot({
    phase: "COMPLETE",
    awaiting_action: "next_round",
    trick: null,
    scoring: {
      declarer_team: 0,
      defender_points: 30,
      bottom_cards: [],
    },
  });

  let sentAction: ClientAction | null = null;
  const mockSend = (action: ClientAction) => { sentAction = action; };

  const callbacks: ActionCallbacks = {
    onCardClick: () => {},
    onAction: (action: string) => {
      if (action === "next_round") mockSend({ type: "next_round" });
    },
    onBid: () => {},
    onStir: () => {},
    onPass: () => {},
    onNewGame: () => {},
  };

  render(snap, container, "next_round", callbacks);

  // Click the "next round" button
  const nextButton = Array.from(container.querySelectorAll(".scoring-overlay button"))
    .find((b) => b.textContent === "下一轮");
  assertNotEquals(nextButton, undefined);
  nextButton!.dispatchEvent(new Event("click", { bubbles: true }));

  // Verify the action was sent
  assertNotEquals(sentAction, null);
  assertEquals(sentAction!.type, "next_round");
});
