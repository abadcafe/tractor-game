import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { GameLoop } from "../engine/game-loop.ts";
import type { StateSnapshot } from "../core/types.ts";
import type { ServerMessage } from "../core/protocol.ts";
import type { InteractionMode } from "../engine/types.ts";
import { StateManager } from "../core/state.ts";

type MalformedStateSnapshot = Omit<StateSnapshot, "awaiting_action"> & {
  awaiting_action: string | null;
};
type MalformedServerMessage = Omit<ServerMessage, "state"> & {
  state: MalformedStateSnapshot;
};

function makeSnapshot(
  overrides: Partial<StateSnapshot> = {},
): StateSnapshot {
  return {
    phase: "PLAYING",
    player_hand: [],
    bottom_cards: [],
    trump_rank: "2",
    trump_suit: null,
    declarer_team: null,
    declarer_player: null,
    defender_points: 0,
    action_hints: [],
    trick: null,
    last_completed_trick: null,
    defender_point_cards: [],
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

function makeStateMsg(
  overrides: Partial<StateSnapshot> = {},
): ServerMessage {
  return {
    type: "state",
    seq: 1,
    state: makeSnapshot(overrides),
  };
}

// Mock renderer that records what was rendered
let lastRenderedSnapshot: StateSnapshot | null = null;
let lastInteractionMode: InteractionMode = null;

function mockRender(
  snapshot: StateSnapshot,
  _container: Element,
  interactionMode: InteractionMode,
): void {
  lastRenderedSnapshot = snapshot;
  lastInteractionMode = interactionMode;
}

// Mock container
const mockContainer = {
  innerHTML: "",
  appendChild: () => {},
  querySelector: () => null,
  querySelectorAll: () => [],
  ownerDocument: {
    createElement: () => ({
      className: "",
      textContent: "",
      remove: () => {},
    }),
  },
} as unknown as Element;

Deno.test("test_handleMessage_deal_bid_our_turn_shows_bidding", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "DEAL_BID",
    awaiting_action: "bid",
  });
  loop.handleMessage(msg);
  assertEquals(lastRenderedSnapshot !== null, true);
  assertEquals(lastRenderedSnapshot!.phase, "DEAL_BID");
  assertEquals(lastInteractionMode, "bid");
});

Deno.test("test_handleMessage_deal_bid_not_our_turn_shows_null", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({ phase: "DEAL_BID", awaiting_action: null });
  loop.handleMessage(msg);
  assertEquals(lastRenderedSnapshot !== null, true);
  assertEquals(lastInteractionMode, null);
});

Deno.test("test_handleMessage_stirring_our_turn", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "STIRRING",
    awaiting_action: "stir",
    stirring_state: {
      phase: "WAITING",
      trump_suit: null,
      current_player: 2,
      declarer_player: 0,
      exchanging_player: null,
      exchange_count: null,
    },
  });
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, "stir");
});

Deno.test("test_handleMessage_stirring_not_our_turn", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
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
  });
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, null);
});

Deno.test("test_handleMessage_exchange_our_turn", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "STIRRING",
    awaiting_action: "discard",
    stirring_state: {
      phase: "EXCHANGING",
      trump_suit: null,
      current_player: 2,
      declarer_player: 0,
      exchanging_player: 2,
      exchange_count: 8,
    },
  });
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, "discard");
});

Deno.test("test_handleMessage_playing_our_turn", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "PLAYING",
    awaiting_action: "play",
  });
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, "play");
});

Deno.test("test_handleMessage_playing_not_our_turn", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "PLAYING",
    awaiting_action: null,
  });
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, null);
});

Deno.test("test_handleMessage_waiting_next_round", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "WAITING",
    awaiting_action: "next_round",
    scoring: {
      declarer_team: 0,
      defender_points: 30,
      total_defender_points: 30,
      bottom_card_bonus: 0,
      bottom_cards: [],
    },
  });
  loop.handleMessage(msg);
  assertEquals(lastInteractionMode, "next_round");
});

Deno.test("test_handleMessage_game_over_no_interaction", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg = makeStateMsg({
    phase: "WAITING",
    winning_team: 0,
    awaiting_action: null,
  });
  loop.handleMessage(msg);
  assertEquals(lastRenderedSnapshot!.winning_team, 0);
  // Game over has no awaiting action -> null interaction mode
  assertEquals(lastInteractionMode, null);
});

Deno.test("test_handleMessage_error_shows_error_and_updates_state", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  let errorReceived: string | null = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(
    stateManager,
    mockRender,
    mockContainer,
    undefined,
    undefined,
    (message) => {
      errorReceived = message;
    },
  );
  const msg: ServerMessage = {
    type: "state",
    seq: 1,
    state: makeSnapshot({ phase: "PLAYING" }),
    error: "something went wrong",
  };
  loop.handleMessage(msg);
  // State messages with errors still update state and re-render
  assertEquals(lastRenderedSnapshot !== null, true);
  // onError callback should have been called
  assertEquals(errorReceived, "something went wrong");
});

Deno.test("test_handleMessage_updates_state_manager", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const snap = makeSnapshot({ phase: "PLAYING" });
  const msg: ServerMessage = {
    type: "state",
    seq: 1,
    state: snap,
  };
  loop.handleMessage(msg);
  assertEquals(stateManager.get()!.phase, "PLAYING");
});

Deno.test("test_handleMessage_error_stores_error_message", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  assertEquals(loop.getLastError(), null);
  const msg: ServerMessage = {
    type: "state",
    seq: 1,
    state: makeSnapshot({ phase: "PLAYING" }),
    error: "something went wrong",
  };
  loop.handleMessage(msg);
  assertEquals(loop.getLastError(), "something went wrong");
});

Deno.test("test_handleMessage_unknown_awaiting_returns_null", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const malformedMsg: MalformedServerMessage = {
    type: "state",
    seq: 1,
    state: {
      ...makeSnapshot({ phase: "PLAYING" }),
      awaiting_action: "unknown_action",
    },
  };
  loop.handleMessage(malformedMsg as unknown as ServerMessage);
  assertEquals(lastInteractionMode, null);
});

Deno.test("test_handleMessage_reconnecting_disables_interaction", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  let reconnecting = true;
  const loop = new GameLoop(
    stateManager,
    mockRender,
    mockContainer,
    undefined,
    () => reconnecting,
  );
  const msg = makeStateMsg({
    phase: "PLAYING",
    awaiting_action: "play",
  });
  loop.handleMessage(msg);
  // While reconnecting, interaction should be disabled
  assertEquals(lastInteractionMode, null);

  // After reconnection, interaction should be enabled
  reconnecting = false;
  const msg2 = makeStateMsg({
    phase: "PLAYING",
    awaiting_action: "play",
  });
  loop.handleMessage(msg2);
  assertEquals(lastInteractionMode, "play");
});

Deno.test("test_handleMessage_seq_stored_in_state_manager", () => {
  lastRenderedSnapshot = null;
  lastInteractionMode = null;
  const stateManager = new StateManager();
  const loop = new GameLoop(stateManager, mockRender, mockContainer);
  const msg: ServerMessage = {
    type: "state",
    seq: 42,
    state: makeSnapshot({ phase: "PLAYING" }),
  };
  loop.handleMessage(msg);
  assertEquals(stateManager.seq, 42);
});
