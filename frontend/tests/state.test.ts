import { assertEquals, assertNotEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { StateManager } from "../core/state.ts";
import type { StateSnapshot } from "../core/types.ts";

function makeSnapshot(phase: string): StateSnapshot {
  return {
    phase: phase as StateSnapshot["phase"],
    player_hand: [],
    bottom_cards: [],
    trump_rank: "2",
    trump_suit: null,
    declarer_team: null,
    declarer_player: null,
    current_player: 0,
    defender_points: 0,
    legal_actions: [],
    trick: null,
    trick_history: [],
    bid_events: [],
    bid_winner: null,
    awaiting_action: null,
    stirring_state: null,
    exchange_state: null,
    scoring: null,
    winning_team: null,
    team0_level: "2",
    team1_level: "2",
    player_hand_counts: [13, 13, 13, 13],
    next_round_confirmed: [],
  };
}

Deno.test("test_get_returns_null_initially", () => {
  const mgr = new StateManager();
  assertEquals(mgr.get(), null);
});

Deno.test("test_update_stores_snapshot", () => {
  const mgr = new StateManager();
  const snap = makeSnapshot("DEAL_BID");
  mgr.update(snap);
  assertEquals(mgr.get(), snap);
});

Deno.test("test_get_returns_latest", () => {
  const mgr = new StateManager();
  mgr.update(makeSnapshot("DEAL_BID"));
  mgr.update(makeSnapshot("STIRRING"));
  assertEquals(mgr.get()!.phase, "STIRRING");
});

Deno.test("test_onChange_called_on_update", () => {
  const mgr = new StateManager();
  let called = false;
  let received: StateSnapshot | null = null;
  mgr.onChange((snap) => {
    called = true;
    received = snap;
  });
  const snap = makeSnapshot("PLAYING");
  mgr.update(snap);
  assertEquals(called, true);
  assertEquals(received, snap);
});

Deno.test("test_onChange_unsubscribe", () => {
  const mgr = new StateManager();
  let callCount = 0;
  const unsub = mgr.onChange(() => { callCount++; });
  unsub();
  mgr.update(makeSnapshot("DEAL_BID"));
  assertEquals(callCount, 0);
});

Deno.test("test_onChange_multiple_subscribers", () => {
  const mgr = new StateManager();
  let count1 = 0;
  let count2 = 0;
  mgr.onChange(() => { count1++; });
  mgr.onChange(() => { count2++; });
  mgr.update(makeSnapshot("DEAL_BID"));
  assertEquals(count1, 1);
  assertEquals(count2, 1);
});

Deno.test("test_onChange_not_called_on_unsubscribed", () => {
  const mgr = new StateManager();
  let count1 = 0;
  let count2 = 0;
  const unsub1 = mgr.onChange(() => { count1++; });
  mgr.onChange(() => { count2++; });
  unsub1();
  mgr.update(makeSnapshot("DEAL_BID"));
  assertEquals(count1, 0);
  assertEquals(count2, 1);
});

Deno.test("test_update_replaces_previous_snapshot", () => {
  const mgr = new StateManager();
  mgr.update(makeSnapshot("DEAL_BID"));
  mgr.update(makeSnapshot("STIRRING"));
  const result = mgr.get()!;
  assertEquals(result.phase, "STIRRING");
});

Deno.test("test_reset_clears_state", () => {
  const mgr = new StateManager();
  mgr.update(makeSnapshot("PLAYING"));
  assertNotEquals(mgr.get(), null);
  mgr.reset();
  assertEquals(mgr.get(), null);
});

Deno.test("test_reset_returns_get_to_null", () => {
  const mgr = new StateManager();
  mgr.update(makeSnapshot("DEAL_BID"));
  mgr.update(makeSnapshot("STIRRING"));
  mgr.reset();
  assertEquals(mgr.get(), null);
});
