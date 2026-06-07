import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { GameLoop } from "../engine/game-loop.ts";
import type { StateSnapshot, ServerMessage, InteractionMode } from "../core/types.ts";
import { StateManager } from "../core/state.ts";

function makeSnapshot(overrides: Partial<StateSnapshot> = {}): StateSnapshot {
  return {
    phase: "PLAYING",
    player_hand: [],
    bottom_cards: [],
    trump_rank: "2",
    trump_suit: null,
    declarer_team: null,
    declarer_player: null,
    current_player: 3,
    defender_points: 0,
    legal_actions: [],
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

function makeStateMsg(overrides: Partial<StateSnapshot> = {}, awaiting: string | null = null): ServerMessage {
  return { type: "state", awaiting, state: makeSnapshot(overrides) };
}

// Mock renderer that records what was rendered
let lastRenderedSnapshot: StateSnapshot | null = null;
let lastInteractionMode: InteractionMode = null;

function mockRender(snapshot: StateSnapshot, _container: Element, interactionMode: InteractionMode): void {
  lastRenderedSnapshot = snapshot;
  lastInteractionMode = interactionMode;
}

// Mock container
const mockContainer = {
  innerHTML: "",
  appendChild: () => {},
  querySelector: () => null,
  querySelectorAll: () => [],
} as unknown as Element;

Deno.test("test_handleMessage_deal_bid_shows_bidding", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({ phase: "DEAL_BID", awaiting_action: null }, null);
  loop.handleMessage(msg);
  assertEquals(lastRenderedSnapshot !== null, true);
  assertEquals(lastRenderedSnapshot!.phase, "DEAL_BID");
  assertEquals(lastInteractionMode, "bid");
});

Deno.test("test_handleMessage_stirring_human", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "STIRRING",
    awaiting_action: "stir",
    current_player: 3,
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 3 },
  }, "stir");
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, "stir");
});

Deno.test("test_handleMessage_stirring_not_human", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "STIRRING",
    awaiting_action: "stir",
    current_player: 1,
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 1 },
  }, "stir");
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, null);
});

Deno.test("test_handleMessage_exchange_human", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "EXCHANGE",
    awaiting_action: "discard",
    current_player: 3,
    exchange_state: { phase: "PICKED_UP", declarer_player: 3, count: 8 },
  }, "discard");
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, "discard");
});

Deno.test("test_handleMessage_playing_human", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "PLAYING",
    awaiting_action: "play",
    current_player: 3,
  }, "play");
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, "play");
});

Deno.test("test_handleMessage_playing_not_human", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "PLAYING",
    awaiting_action: "play",
    current_player: 1,
  }, "play");
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, null);
});

Deno.test("test_handleMessage_complete_human", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "COMPLETE",
    awaiting_action: "next_round",
    current_player: 3,
    scoring: { declarer_team: 0, defender_points: 30, bottom_cards: [] },
  }, "next_round");
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, "next_round");
});

Deno.test("test_handleMessage_game_over", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "GAME_OVER",
    winning_team: 0,
    awaiting_action: null,
  }, null);
  loop.handleMessage(msg);
  assertEquals(lastRenderedSnapshot!.phase, "GAME_OVER");
  assertEquals(lastInteractionMode, null);
});

Deno.test("test_handleMessage_error_does_not_update_state", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg: ServerMessage = { type: "error", message: "something went wrong" };
  loop.handleMessage(msg);
  // Error messages should not update state or re-render
  assertEquals(lastRenderedSnapshot, null);
});

Deno.test("test_handleMessage_updates_state_manager", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const snap = makeSnapshot({ phase: "PLAYING" });
  const msg: ServerMessage = { type: "state", awaiting: "play", state: snap };
  loop.handleMessage(msg);
  assertEquals(stateManager.get()!.phase, "PLAYING");
});

Deno.test("test_handleMessage_error_stores_error_message", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  assertEquals(loop.getLastError(), null);
  const msg: ServerMessage = { type: "error", message: "something went wrong" };
  loop.handleMessage(msg);
  assertEquals(loop.getLastError(), "something went wrong");
});

Deno.test("test_handleMessage_unknown_awaiting_returns_null", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "PLAYING",
    awaiting_action: "unknown_action",
    current_player: 3,
  }, "unknown_action");
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, null);
});
