import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { StateManager } from "../core/state.ts";
import type { PublicGamePhase, StateSnapshot } from "../core/types.ts";

function makeSnapshot(phase: PublicGamePhase): StateSnapshot {
  return {
    phase,
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
    awaiting_action: null,
    stirring_state: null,
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
