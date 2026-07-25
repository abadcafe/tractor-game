import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { StateManager } from "../core/state.ts";
import type { RoundPhase, StateSnapshot } from "../core/types.ts";

function makeSnapshot(phase: RoundPhase): StateSnapshot {
  return {
    phase,
    round_number: 1,
    hand: [],
    bottom_cards: [],
    trump: { kind: "no_trump", rank: "2" },
    declarer: null,
    defender_points: 0,
    action_hints: [],
    trick: null,
    last_completed_trick: null,
    defender_point_cards: [],
    bid_events: [],
    bid_winner: null,
    stir_events: [],
    own_initial_bottom_exchange: null,
    awaiting_action: null,
    stirring_state: null,
    scoring: null,
    winning_partnership: null,
    partnership_levels: { first: "2", second: "2" },
    remaining_cards: { a: 13, b: 13, c: 13, d: 13 },
    mandatory_levels: ["A"],
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
  mgr.update(snap, 0);
  assertEquals(mgr.get(), snap);
});

Deno.test("test_get_returns_latest", () => {
  const mgr = new StateManager();
  mgr.update(makeSnapshot("DEAL_BID"), 0);
  mgr.update(makeSnapshot("STIRRING"), 1);
  assertEquals(mgr.get()!.phase, "STIRRING");
});

Deno.test("test_update_replaces_previous_snapshot", () => {
  const mgr = new StateManager();
  mgr.update(makeSnapshot("DEAL_BID"), 0);
  mgr.update(makeSnapshot("STIRRING"), 1);
  const result = mgr.get()!;
  assertEquals(result.phase, "STIRRING");
});

Deno.test("test_reset_clears_state", () => {
  const mgr = new StateManager();
  mgr.update(makeSnapshot("PLAYING"), 0);
  assertNotEquals(mgr.get(), null);
  mgr.reset();
  assertEquals(mgr.get(), null);
});

Deno.test("test_reset_returns_get_to_null", () => {
  const mgr = new StateManager();
  mgr.update(makeSnapshot("DEAL_BID"), 0);
  mgr.update(makeSnapshot("STIRRING"), 1);
  mgr.reset();
  assertEquals(mgr.get(), null);
});
