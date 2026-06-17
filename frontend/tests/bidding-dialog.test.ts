import { assertEquals, assertNotEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderBiddingDialog } from "../ui/components/bidding-dialog.ts";
import type { StateSnapshot } from "../core/types.ts";
import type { BidOption, StirButtonState } from "../engine/types.ts";

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
    defender_points: 0,
    legal_actions: [],
    bid_legal_actions: null,
    trick: null,
    trick_history: [],
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
    ...overrides,
  };
}

/** Convenience bid options for tests. */
const sampleBidOptions: BidOption[] = [
  { cardIds: ["D1-spades-2"], label: "♠2", trumpSuit: "spades", priority: 1 },
  { cardIds: ["D1-hearts-2"], label: "♥2", trumpSuit: "hearts", priority: 1 },
];

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
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 3, declarer_player: 0, legal_actions: [], exchanging_player: null, exchange_count: null },
  });
  const el = renderBiddingDialog(snap, "stir");
  const buttons = el.querySelectorAll("button");
  const buttonTexts = Array.from(buttons).map((b) => b.textContent);
  assertEquals(buttonTexts.includes("不反"), true);
});

Deno.test("test_renderBiddingDialog_stirring_not_human", () => {
  const snap = makeSnapshot({
    phase: "STIRRING",
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 1, declarer_player: 0, legal_actions: [], exchanging_player: null, exchange_count: null },
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
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 3, declarer_player: 0, legal_actions: [], exchanging_player: null, exchange_count: null },
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

Deno.test("test_renderBiddingDialog_stir_callback", () => {
  const snap = makeSnapshot({
    phase: "STIRRING",
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 3, declarer_player: 0, legal_actions: [], exchanging_player: null, exchange_count: null },
    player_hand: [
      { id: "D1-spades-2", suit: "spades", rank: "2" },
      { id: "D2-spades-2", suit: "spades", rank: "2" },
    ],
  });
  let stirCardIds: string[] | null = null;
  const onStir = (cardIds: string[]) => { stirCardIds = cardIds; };
  // The stir dialog reads selected cards from .hand-view .card.selected in the DOM.
  // We need to add mock elements to the document for the click handler to find them.
  const handView = document.createElement("div");
  handView.classList.add("hand-view");
  for (const id of ["D1-spades-2", "D2-spades-2"]) {
    const cardEl = document.createElement("div");
    cardEl.classList.add("card", "selected");
    cardEl.setAttribute("data-card-id", id);
    handView.appendChild(cardEl);
  }
  document.body.appendChild(handView);

  try {
    const el = renderBiddingDialog(snap, "stir", undefined, onStir, undefined);
    // In STIRRING, there should be a "反主" button that calls onStir
    const stirButton = Array.from(el.querySelectorAll("button")).find((b) => b.textContent === "反主");
    assertNotEquals(stirButton, undefined);
    stirButton!.dispatchEvent(new Event("click", { bubbles: true }));
    assertNotEquals(stirCardIds, null);
    assertEquals(stirCardIds, ["D1-spades-2", "D2-spades-2"]);
  } finally {
    document.body.removeChild(handView);
  }
});

// --- Edge-case tests ---

Deno.test("test_renderBiddingDialog_other_phase_empty", () => {
  const snap = makeSnapshot({ phase: "PLAYING" });
  const el = renderBiddingDialog(snap, null);
  assertNotEquals(el, null);
  // Other phases render an empty div
  assertEquals(el.childNodes.length, 0);
});

Deno.test("test_renderBiddingDialog_no_bid_options_shows_hint", () => {
  // When no bid options are available, a hint is shown instead of bid option pills
  const snap = makeSnapshot({ phase: "DEAL_BID", player_hand: [] });
  const el = renderBiddingDialog(snap, "bid", undefined, undefined, undefined, undefined, undefined, undefined, []);
  const hint = el.querySelector(".bidding-dialog-hint");
  assertNotEquals(hint, null);
  assertEquals(hint!.textContent, "当前无可用叫牌选项");
  // No bid option pills
  const pills = el.querySelectorAll(".bid-option");
  assertEquals(pills.length, 0);
});

Deno.test("test_renderBiddingDialog_bid_options_displayed_as_pills", () => {
  const snap = makeSnapshot({ phase: "DEAL_BID" });
  const el = renderBiddingDialog(
    snap, "bid",
    undefined, undefined, undefined, undefined, undefined,
    undefined,
    sampleBidOptions,
  );
  const pills = el.querySelectorAll(".bid-option");
  assertEquals(pills.length, 2);
  // First pill should show the label with trump suit
  const text0 = pills[0].textContent ?? "";
  assertEquals(text0.includes("♠2"), true);
});

Deno.test("test_renderBiddingDialog_clicking_bid_option_calls_onBidOptionSelect", () => {
  const snap = makeSnapshot({ phase: "DEAL_BID" });
  let selectedOption: BidOption | null = null;
  const onBidOptionSelect = (option: BidOption) => { selectedOption = option; };
  const el = renderBiddingDialog(
    snap, "bid",
    undefined, undefined, undefined, undefined, undefined,
    undefined,
    sampleBidOptions,
    null,
    onBidOptionSelect,
  );
  const pills = el.querySelectorAll(".bid-option");
  assertEquals(pills.length, 2);
  // Click the first bid option pill
  pills[0].dispatchEvent(new Event("click", { bubbles: true }));
  assertNotEquals(selectedOption, null);
  assertEquals(selectedOption!.cardIds, ["D1-spades-2"]);
  assertEquals(selectedOption!.label, "♠2");
});

Deno.test("test_renderBiddingDialog_selected_bid_option_has_class", () => {
  const snap = makeSnapshot({ phase: "DEAL_BID" });
  // Set pendingBidIntent to match the first bid option
  const pendingBidIntent: BidOption = sampleBidOptions[0];
  const el = renderBiddingDialog(
    snap, "bid",
    undefined, undefined, undefined, undefined, undefined,
    undefined,
    sampleBidOptions,
    pendingBidIntent,
  );
  const pills = el.querySelectorAll(".bid-option");
  assertEquals(pills.length, 2);
  // First pill should have the .selected class since it matches pendingBidIntent
  assertEquals(pills[0].classList.contains("selected"), true);
  // Second pill should not
  assertEquals(pills[1].classList.contains("selected"), false);
});

Deno.test("test_renderBiddingDialog_pending_intent_hint_shown", () => {
  const snap = makeSnapshot({ phase: "DEAL_BID" });
  const pendingBidIntent: BidOption = sampleBidOptions[0];
  const el = renderBiddingDialog(
    snap, "bid",
    undefined, undefined, undefined, undefined, undefined,
    undefined,
    sampleBidOptions,
    pendingBidIntent,
  );
  const hint = el.querySelector(".bid-intent-hint");
  assertNotEquals(hint, null);
  const text = hint!.textContent ?? "";
  assertEquals(text.includes("待叫"), true);
  assertEquals(text.includes("♠2"), true);
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

Deno.test("test_renderBiddingDialog_stir_button_disabled", () => {
  const snap = makeSnapshot({
    phase: "STIRRING",
    stirring_state: { phase: "WAITING", trump_suit: null, current_player: 3, declarer_player: 0, legal_actions: [], exchanging_player: null, exchange_count: null },
  });
  const stirButtonState: StirButtonState = { disabled: true, title: "请选择对子" };
  const el = renderBiddingDialog(
    snap, "stir",
    undefined, undefined, undefined, undefined, undefined,
    stirButtonState,
  );
  const stirButton = Array.from(el.querySelectorAll("button")).find((b) => b.textContent === "反主");
  assertNotEquals(stirButton, undefined);
  assertEquals(stirButton!.disabled, true);
  assertEquals(stirButton!.title, "请选择对子");
});
