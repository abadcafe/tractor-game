import { assertEquals, assertNotEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderBiddingDialog } from "../ui/components/bidding-dialog.ts";
import type { StateSnapshot, InteractionMode } from "../core/types.ts";

const doc = new DOMParser().parseFromString(
  `<html><body><div id="app"></div></body></html>`,
  "text/html",
);
// @ts-ignore test setup
globalThis.document = doc;

function makeSnapshot(overrides: Partial<StateSnapshot> = {}): StateSnapshot {
  return {
    phase: "DEAL_BID",
    player_hand: [
      { id: "D1-spades-2", suit: "spades", rank: "2" },
    ],
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
    ...overrides,
  };
}

Deno.test("test_renderBiddingDialog_deal_bid_phase", () => {
  const snap = makeSnapshot({ phase: "DEAL_BID" });
  const el = renderBiddingDialog(snap, "bid");
  assertNotEquals(el, null);
  const text = el.textContent ?? "";
  assertEquals(text.includes("叫牌"), true);
});

Deno.test("test_renderBiddingDialog_stirring_human", () => {
  const snap = makeSnapshot({
    phase: "STIRRING",
    current_player: 3,
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 3 },
  });
  const el = renderBiddingDialog(snap, "stir");
  const buttons = el.querySelectorAll("button");
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("不反"), true);
});

Deno.test("test_renderBiddingDialog_stirring_not_human", () => {
  const snap = makeSnapshot({
    phase: "STIRRING",
    current_player: 1,
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 1 },
  });
  // When interactionMode is null (not human's turn), no action buttons
  const el = renderBiddingDialog(snap, null);
  const buttons = el.querySelectorAll("button");
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("不反"), false);
});

Deno.test("test_renderBiddingDialog_bid_events_displayed", () => {
  const snap = makeSnapshot({
    bid_events: [{
      player: 1,
      cards: [{ id: "D1-hearts-2", suit: "hearts", rank: "2" }],
      kind: "trump_rank",
      suit: "hearts",
      joker_type: null,
      count: 1,
    }],
  });
  const el = renderBiddingDialog(snap, "bid");
  const events = el.querySelectorAll(".bid-event");
  assertEquals(events.length, 1);
});

Deno.test("test_renderBiddingDialog_pass_callback", () => {
  const snap = makeSnapshot({
    phase: "STIRRING",
    current_player: 3,
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 3 },
  });
  let passCalled = false;
  const onPass = () => { passCalled = true; };
  const el = renderBiddingDialog(snap, "stir", undefined, undefined, onPass);
  // Find and click the "不反" button
  const buttons = el.querySelectorAll("button");
  const passButton = Array.from(buttons).find((b) => b.textContent === "不反");
  assertNotEquals(passButton, undefined);
  passButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertEquals(passCalled, true);
});

Deno.test("test_renderBiddingDialog_bid_callback", () => {
  const snap = makeSnapshot({ phase: "DEAL_BID" });
  let bidCardIds: string[] | null = null;
  const onBid = (cardIds: string[]) => { bidCardIds = cardIds; };
  const el = renderBiddingDialog(snap, "bid", onBid);
  // In DEAL_BID, there should be a bid button
  const bidButton = Array.from(el.querySelectorAll("button")).find((b) => b.textContent === "叫牌");
  assertNotEquals(bidButton, undefined);
  // Clicking the bid button should call onBid with the currently selected trump rank cards
  // (In the component, the bid button triggers onBid with the player's trump rank cards)
  bidButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertNotEquals(bidCardIds, null);
});

Deno.test("test_renderBiddingDialog_stir_callback", () => {
  const snap = makeSnapshot({
    phase: "STIRRING",
    current_player: 3,
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 3 },
    player_hand: [
      { id: "D1-spades-2", suit: "spades", rank: "2" },
      { id: "D1-hearts-2", suit: "hearts", rank: "2" },
    ],
  });
  let stirCardIds: string[] | null = null;
  const onStir = (cardIds: string[]) => { stirCardIds = cardIds; };
  const el = renderBiddingDialog(snap, "stir", undefined, onStir);
  // In STIRRING, there should be a "反主" button that calls onStir
  const stirButton = Array.from(el.querySelectorAll("button")).find((b) => b.textContent === "反主");
  assertNotEquals(stirButton, undefined);
  stirButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertNotEquals(stirCardIds, null);
});

// --- Edge-case tests (CQ-001) ---

Deno.test("test_renderBiddingDialog_other_phase_empty", () => {
  const snap = makeSnapshot({ phase: "PLAYING" });
  const el = renderBiddingDialog(snap, null);
  assertNotEquals(el, null);
  assertEquals(el.classList.contains("bidding-dialog"), true);
  const buttons = el.querySelectorAll("button");
  assertEquals(buttons.length, 0);
});

Deno.test("test_renderBiddingDialog_empty_hand_bid", () => {
  const snap = makeSnapshot({ phase: "DEAL_BID", player_hand: [] });
  let bidCardIds: string[] | null = null;
  const onBid = (cardIds: string[]) => { bidCardIds = cardIds; };
  const el = renderBiddingDialog(snap, "bid", onBid);
  const bidButton = Array.from(el.querySelectorAll("button")).find((b) => b.textContent === "叫牌");
  assertNotEquals(bidButton, undefined);
  bidButton!.dispatchEvent(new Event("click", { bubbles: true }));
  assertNotEquals(bidCardIds, null);
  assertEquals(bidCardIds!.length, 0);
});

Deno.test("test_renderBiddingDialog_multiple_bid_events", () => {
  const snap = makeSnapshot({
    bid_events: [
      {
        player: 1,
        cards: [{ id: "D1-hearts-2", suit: "hearts", rank: "2" }],
        kind: "trump_rank",
        suit: "hearts",
        joker_type: null,
        count: 1,
      },
      {
        player: 2,
        cards: [{ id: "D1-spades-2", suit: "spades", rank: "2" }],
        kind: "trump_rank",
        suit: "spades",
        joker_type: null,
        count: 1,
      },
      {
        player: 0,
        cards: [{ id: "D1-spades-BJ", suit: "joker", rank: "BJ" }],
        kind: "joker",
        suit: null,
        joker_type: "big",
        count: 1,
      },
    ],
  });
  const el = renderBiddingDialog(snap, "bid");
  const events = el.querySelectorAll(".bid-event");
  assertEquals(events.length, 3);
});
