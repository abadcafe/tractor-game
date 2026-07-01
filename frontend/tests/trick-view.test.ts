import {
  assert,
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderTrickView } from "../ui/components/trick-view.ts";
import type { CompletedTrick, StateSnapshot } from "../core/types.ts";

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
    player_hand: [],
    bottom_cards: [],
    trump_rank: "2",
    trump_suit: null,
    declarer_team: null,
    declarer_player: null,
    defender_points: 0,
    action_hints: [],
    trick: {
      lead_player: 0,
      slots: [{
        player: 0,
        cards: [{ id: "D1-clubs-7", suit: "clubs", rank: "7" }],
      }],
      current_player: 1,
    },
    last_completed_trick: null,
    defender_point_cards: [],
    failed_throw: null,
    bid_events: [],
    bid_winner: null,
    stir_events: [],
    own_bottom_exchange_events: [],
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

function renderedRanks(el: Element, selector: string): string[] {
  return Array.from(el.querySelectorAll(selector)).map((card) =>
    card.getAttribute("data-rank") ?? ""
  );
}

Deno.test("test_renderTrickView_shows_played_cards", () => {
  const snap = makeSnapshot();
  const el = renderTrickView(snap);
  const trickCards = el.querySelectorAll(".trick-card");
  assertEquals(trickCards.length, 1);
  assertEquals(el.querySelectorAll(".trick-lead-marker").length, 1);
  assertEquals(
    el.querySelector(".trick-lead-marker")?.textContent ?? "",
    "先出",
  );
});

Deno.test("test_renderTrickView_empty_trick", () => {
  const snap = makeSnapshot({ trick: null });
  const el = renderTrickView(snap);
  const trickCards = el.querySelectorAll(".trick-card");
  assertEquals(trickCards.length, 0);
});

Deno.test("test_renderTrickView_waiting_scoring_keeps_last_trick", () => {
  const snap = makeSnapshot({
    phase: "WAITING",
    trick: null,
    last_completed_trick: {
      lead_player: 1,
      winner: 3,
      points: 20,
      slots: [
        {
          player: 0,
          cards: [{ id: "D1-clubs-5", suit: "clubs", rank: "5" }],
        },
        {
          player: 1,
          cards: [{ id: "D1-hearts-10", suit: "hearts", rank: "10" }],
        },
        {
          player: 2,
          cards: [{ id: "D1-spades-K", suit: "spades", rank: "K" }],
        },
        {
          player: 3,
          cards: [{ id: "D1-diamonds-A", suit: "diamonds", rank: "A" }],
        },
      ],
    },
    scoring: {
      declarer_team: 0,
      defender_points: 80,
      total_defender_points: 100,
      bottom_card_bonus: 20,
      bottom_cards: [],
    },
  });
  const el = renderTrickView(snap);

  assertEquals(el.classList.contains("showing-previous"), true);
  assertEquals(el.querySelectorAll(".trick-card").length, 4);
  assertEquals(el.querySelectorAll(".trick-slot.winner").length, 1);
  assertEquals(
    el.querySelectorAll(".trick-slot.lead .trick-lead-marker").length,
    1,
  );
  assertEquals((el.textContent ?? "").includes("10"), true);
});

Deno.test("test_renderTrickView_previous_trick_preview_shows_four_players", () => {
  const snap = makeSnapshot();
  const el = renderTrickView(snap, {
    lead_player: 0,
    winner: 2,
    points: 15,
    slots: [
      {
        player: 0,
        cards: [{ id: "D1-clubs-5", suit: "clubs", rank: "5" }],
      },
      {
        player: 1,
        cards: [{ id: "D1-hearts-9", suit: "hearts", rank: "9" }],
      },
      {
        player: 2,
        cards: [{ id: "D1-spades-K", suit: "spades", rank: "K" }],
      },
      {
        player: 3,
        cards: [{ id: "D1-diamonds-A", suit: "diamonds", rank: "A" }],
      },
    ],
  });

  assertEquals((el.textContent ?? "").includes("上一墩 15 分"), false);
  assertEquals(el.querySelectorAll(".trick-slot").length, 4);
  assertEquals(el.querySelectorAll(".trick-card").length, 4);
  assertEquals(el.querySelectorAll(".trick-slot.winner").length, 1);
  assertEquals(
    el.querySelectorAll(".trick-slot.lead .trick-lead-marker").length,
    1,
  );
  assertEquals(
    el.querySelectorAll(".trick-card-placeholder").length,
    0,
  );
});

Deno.test("test_renderTrickView_failed_throw_preview_shows_attempted_and_forced_cards", () => {
  const snap = makeSnapshot({ trump_suit: "spades", trump_rank: "2" });
  const previousTrick: CompletedTrick = {
    lead_player: 0,
    winner: 0,
    points: 0,
    slots: [
      {
        player: 0,
        cards: [{ id: "D1-clubs-5", suit: "clubs", rank: "5" }],
      },
      { player: 1, cards: [] },
      { player: 2, cards: [] },
      { player: 3, cards: [] },
    ],
  };
  const el = renderTrickView(snap, previousTrick, {
    player: 0,
    attempted_cards: [
      { id: "D1-spades-Q", suit: "spades", rank: "Q" },
      { id: "D1-joker-SJ", suit: "joker", rank: "SJ" },
      { id: "D1-spades-K", suit: "spades", rank: "K" },
    ],
    forced_cards: [
      { id: "D1-spades-Q", suit: "spades", rank: "Q" },
      { id: "D1-spades-A", suit: "spades", rank: "A" },
      { id: "D1-spades-K", suit: "spades", rank: "K" },
    ],
  });

  const text = el.textContent ?? "";
  assertEquals(text.includes("甩牌失败，捡小"), false);
  assertEquals(text.includes("玩家 0甩牌失败"), true);
  assertEquals(text.includes("暴露"), true);
  assertEquals(text.includes("捡小"), true);
  assertEquals(text.includes("上一墩"), false);
  assertEquals(el.querySelectorAll(".failed-throw-preview").length, 1);
  assertEquals(
    el.querySelectorAll(".failed-throw-preview__cards .trick-card")
      .length,
    6,
  );
  const rows = Array.from(
    el.querySelectorAll(".failed-throw-preview__row"),
  );
  assertEquals(rows.length, 2);
  const exposedRow = rows[0];
  const forcedRow = rows[1];
  assert(exposedRow !== undefined);
  assert(forcedRow !== undefined);
  assertEquals(renderedRanks(exposedRow, ".trick-card"), [
    "SJ",
    "K",
    "Q",
  ]);
  assertEquals(renderedRanks(forcedRow, ".trick-card"), [
    "A",
    "K",
    "Q",
  ]);
});

Deno.test("test_renderTrickView_player_labels", () => {
  const snap = makeSnapshot();
  const el = renderTrickView(snap);
  const labels = el.querySelectorAll(".trick-player-label");
  assertEquals(labels.length, 1);
});

Deno.test("test_renderTrickView_multiple_slots", () => {
  const snap = makeSnapshot({
    trick: {
      lead_player: 0,
      slots: [
        {
          player: 0,
          cards: [{ id: "D1-clubs-7", suit: "clubs", rank: "7" }],
        },
        {
          player: 1,
          cards: [{ id: "D2-hearts-9", suit: "hearts", rank: "9" }],
        },
        {
          player: 2,
          cards: [{ id: "D3-spades-J", suit: "spades", rank: "J" }],
        },
      ],
      current_player: 3,
    },
  });
  const el = renderTrickView(snap);
  const slots = el.querySelectorAll(".trick-slot");
  assertEquals(slots.length, 3);
  const trickCards = el.querySelectorAll(".trick-card");
  assertEquals(trickCards.length, 3);
  const labels = el.querySelectorAll(".trick-player-label");
  assertEquals(labels.length, 3);
});

Deno.test("test_renderTrickView_sorts_current_trick_cards_like_hand", () => {
  const snap = makeSnapshot({
    trump_rank: "4",
    trump_suit: null,
    trick: {
      lead_player: 3,
      slots: [
        {
          player: 0,
          cards: [
            { id: "D1-diamonds-6", suit: "diamonds", rank: "6" },
            { id: "D1-diamonds-3", suit: "diamonds", rank: "3" },
            { id: "D1-diamonds-10", suit: "diamonds", rank: "10" },
          ],
        },
        {
          player: 1,
          cards: [
            { id: "D1-diamonds-7", suit: "diamonds", rank: "7" },
            { id: "D1-diamonds-5", suit: "diamonds", rank: "5" },
            { id: "D1-diamonds-3", suit: "diamonds", rank: "3" },
          ],
        },
        {
          player: 2,
          cards: [
            { id: "D1-diamonds-2", suit: "diamonds", rank: "2" },
            { id: "D2-diamonds-2", suit: "diamonds", rank: "2" },
            { id: "D1-diamonds-6", suit: "diamonds", rank: "6" },
          ],
        },
        {
          player: 3,
          cards: [
            { id: "D1-diamonds-K", suit: "diamonds", rank: "K" },
            { id: "D1-diamonds-A", suit: "diamonds", rank: "A" },
            { id: "D2-diamonds-A", suit: "diamonds", rank: "A" },
          ],
        },
      ],
      current_player: 1,
    },
  });

  const el = renderTrickView(snap);

  assertEquals(
    renderedRanks(el, ".trick-slot-south .trick-card"),
    ["6", "2", "2"],
  );
  assertEquals(
    renderedRanks(el, ".trick-slot-north .trick-card"),
    ["10", "6", "3"],
  );
  assertEquals(
    renderedRanks(el, ".trick-slot-west .trick-card"),
    ["7", "5", "3"],
  );
  assertEquals(
    renderedRanks(el, ".trick-slot-east .trick-card"),
    ["A", "A", "K"],
  );
});

Deno.test("test_renderTrickView_sorts_previous_trick_cards_like_hand", () => {
  const snap = makeSnapshot({
    trump_rank: "4",
    trump_suit: null,
  });
  const el = renderTrickView(snap, {
    lead_player: 3,
    winner: 3,
    points: 25,
    slots: [
      {
        player: 0,
        cards: [
          { id: "D1-diamonds-6", suit: "diamonds", rank: "6" },
          { id: "D1-diamonds-3", suit: "diamonds", rank: "3" },
          { id: "D1-diamonds-10", suit: "diamonds", rank: "10" },
        ],
      },
      {
        player: 1,
        cards: [
          { id: "D1-diamonds-7", suit: "diamonds", rank: "7" },
          { id: "D1-diamonds-5", suit: "diamonds", rank: "5" },
          { id: "D1-diamonds-3", suit: "diamonds", rank: "3" },
        ],
      },
      {
        player: 2,
        cards: [
          { id: "D1-diamonds-2", suit: "diamonds", rank: "2" },
          { id: "D2-diamonds-2", suit: "diamonds", rank: "2" },
          { id: "D1-diamonds-6", suit: "diamonds", rank: "6" },
        ],
      },
      {
        player: 3,
        cards: [
          { id: "D1-diamonds-K", suit: "diamonds", rank: "K" },
          { id: "D1-diamonds-A", suit: "diamonds", rank: "A" },
          { id: "D2-diamonds-A", suit: "diamonds", rank: "A" },
        ],
      },
    ],
  });

  assertEquals(
    renderedRanks(el, ".trick-slot-north .trick-card"),
    ["10", "6", "3"],
  );
  assertEquals(
    renderedRanks(el, ".trick-slot-west .trick-card"),
    ["7", "5", "3"],
  );
  assertEquals(
    renderedRanks(el, ".trick-slot-east .trick-card"),
    ["A", "A", "K"],
  );
  assertEquals(
    renderedRanks(el, ".trick-slot-south .trick-card"),
    ["6", "2", "2"],
  );
});

Deno.test("test_renderTrickView_slot_with_empty_cards", () => {
  const snap = makeSnapshot({
    trick: {
      lead_player: 0,
      slots: [
        { player: 0, cards: [] },
      ],
      current_player: 0,
    },
  });
  const el = renderTrickView(snap);
  const slots = el.querySelectorAll(".trick-slot");
  assertEquals(slots.length, 1);
  const trickCards = el.querySelectorAll(".trick-card");
  assertEquals(trickCards.length, 0);
  const labels = el.querySelectorAll(".trick-player-label");
  assertEquals(labels.length, 0);
  assertEquals(el.querySelectorAll(".trick-lead-marker").length, 0);
});

Deno.test("test_renderTrickView_current_player_highlight", () => {
  const snap = makeSnapshot({
    trick: {
      lead_player: 0,
      slots: [
        {
          player: 0,
          cards: [{ id: "D1-clubs-7", suit: "clubs", rank: "7" }],
        },
        {
          player: 1,
          cards: [{ id: "D2-hearts-9", suit: "hearts", rank: "9" }],
        },
      ],
      current_player: 1,
    },
  });
  const el = renderTrickView(snap);
  const slots = el.querySelectorAll(".trick-slot");
  assertEquals(slots.length, 2);
  // Slot for player 0 should NOT have 'current' class
  assertEquals(slots[0].classList.contains("current"), false);
  // Slot for player 1 should have 'current' class
  assertEquals(slots[1].classList.contains("current"), true);
});
