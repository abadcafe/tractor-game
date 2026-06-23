import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { renderGameTable } from "../ui/components/game-table.ts";
import type { StateSnapshot } from "../core/types.ts";

// Set up global document
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
    trump_suit: "hearts",
    declarer_team: 0,
    declarer_player: 2,
    defender_points: 15,
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
    team0_level: "3",
    team1_level: "5",
    player_hand_counts: [13, 13, 13, 13],
    next_round_confirmed: [],
    ...overrides,
  };
}

Deno.test("test_renderGameTable_shows_four_players", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap);
  const players = el.querySelectorAll(".player-area");
  assertEquals(players.length, 4);
});

Deno.test("test_renderGameTable_debug_avatars_use_seat_labels_not_ai_type", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap, null, null, "game-1");
  const avatars = Array.from(el.querySelectorAll(".player-avatar"));
  assertEquals(avatars.length, 4);
  assertEquals(avatars.map((avatar) => avatar.textContent), [
    "同",
    "左",
    "你",
    "右",
  ]);
  assertEquals(
    avatars.every((avatar) => avatar.textContent !== "ai"),
    true,
  );
});

Deno.test("test_renderGameTable_declarer_in_status_badge", () => {
  const snap = makeSnapshot({ declarer_player: 2 });
  const el = renderGameTable(snap);
  const southStatus = el.querySelector(
    '.player-area[data-position="南"] .player-status-badge',
  );
  assertEquals(southStatus !== null, true);
  assertEquals((southStatus?.textContent ?? "").includes("庄"), true);
});

Deno.test("test_renderGameTable_deal_bid_can_show_fixed_declarer_separate_from_bid_winner", () => {
  const snap = makeSnapshot({
    phase: "DEAL_BID",
    trump_rank: "3",
    trump_suit: null,
    declarer_team: 1,
    declarer_player: 3,
    bid_winner: {
      player: 1,
      cards: [{ id: "D1-spades-3", suit: "spades", rank: "3" }],
      kind: "trump_rank",
      suit: "spades",
      joker_type: null,
      count: 1,
    },
  });
  const el = renderGameTable(snap);
  const eastText = el.querySelector('.player-area[data-position="东"]')
    ?.textContent ?? "";
  const westText = el.querySelector('.player-area[data-position="西"]')
    ?.textContent ?? "";

  assertEquals(eastText.includes("庄"), true);
  assertEquals(eastText.includes("♠3"), false);
  assertEquals(westText.includes("♠3"), true);
  assertEquals(westText.includes("♠主"), true);
  assertEquals(westText.includes("庄"), false);
});

Deno.test("test_renderGameTable_current_player_highlight", () => {
  const snap = makeSnapshot({
    awaiting_action: "play",
    trick: { lead_player: 0, slots: [], current_player: 1 },
  });
  const el = renderGameTable(snap);
  const current = el.querySelectorAll(".player-area.current");
  assertEquals(current.length, 1);
});

Deno.test("test_renderGameTable_player_status_badges_are_grouped", () => {
  const snap = makeSnapshot({
    phase: "WAITING",
    declarer_player: 1,
    next_round_confirmed: [1],
    player_hand_counts: [0, 0, 0, 0],
    bid_winner: {
      player: 1,
      cards: [{ id: "D1-spades-2", suit: "spades", rank: "2" }],
      kind: "trump_rank",
      suit: "spades",
      joker_type: null,
      count: 1,
    },
  });
  const el = renderGameTable(snap);
  const west = el.querySelector('.player-area[data-position="西"]');
  const badges = west?.querySelector(".player-badges");
  const text = badges?.textContent ?? "";

  assertEquals(badges !== null, true);
  assertEquals(text.includes("♠2"), true);
  assertEquals(text.includes("0张"), true);
  assertEquals(text.includes("庄"), true);
  assertEquals(text.includes("OK"), true);
});

Deno.test("test_renderGameTable_player_labels", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap);
  const labels = el.querySelectorAll(".player-label");
  const labelTexts = Array.from(labels).map((l) => l.textContent);
  assertEquals(labelTexts.includes("你"), true);
  assertEquals(labelTexts.includes("同伴"), true);
  assertEquals(labelTexts.includes("左家"), true);
  assertEquals(labelTexts.includes("右家"), true);
});

Deno.test("test_renderGameTable_global_info_bar_in_table", () => {
  const snap = makeSnapshot({
    phase: "PLAYING",
    trump_suit: "spades",
    trump_rank: "2",
  });
  const el = renderGameTable(snap);
  const infoBar = el.querySelector(".info-bar");
  const trickInfo = el.querySelector(".trick-view .info-bar");
  const text = infoBar?.textContent ?? "";

  assertEquals(infoBar !== null, true);
  assertEquals(trickInfo, null);
  assertEquals(text.includes("主:"), true);
  assertEquals(text.includes("♠"), true);
  assertEquals(text.includes("级牌 2"), true);
  assertEquals(text.includes("出牌阶段"), true);
});

Deno.test("test_renderGameTable_status_notice_in_top_right_not_bottom_bar", () => {
  const snap = makeSnapshot({
    awaiting_action: "play",
    defender_points: 25,
    trick: { lead_player: 0, slots: [], current_player: 2 },
  });
  const el = renderGameTable(snap);
  const notice = el.querySelector(".table-notice");
  const bottomStatus = el.querySelector(".table-status");
  const text = notice?.textContent ?? "";

  assertEquals(notice !== null, true);
  assertEquals(bottomStatus, null);
  assertEquals(text.includes("轮到你出牌"), true);
  assertEquals(text.includes("捡分 25"), true);
});

Deno.test("test_renderGameTable_previous_trick_label_in_top_right_notice", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap, {
    lead_player: 0,
    winner: 2,
    points: 5,
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
  const notice = el.querySelector(".table-notice");
  const trickText = el.querySelector(".trick-view")?.textContent ?? "";
  const noticeText = notice?.textContent ?? "";

  assertEquals(noticeText.includes("上一墩 5 分"), true);
  assertEquals(trickText.includes("上一墩"), false);
});

Deno.test("test_renderGameTable_failed_throw_label_in_top_right_notice", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap, null, {
    player: 3,
    attempted_cards: [
      { id: "D1-spades-K", suit: "spades", rank: "K" },
      { id: "D1-spades-Q", suit: "spades", rank: "Q" },
    ],
    forced_cards: [
      { id: "D1-spades-Q", suit: "spades", rank: "Q" },
    ],
  });
  const noticeText = el.querySelector(".table-notice")?.textContent ??
    "";
  const trickText = el.querySelector(".trick-view")?.textContent ?? "";

  assertEquals(noticeText.includes("甩牌失败，捡小"), true);
  assertEquals(trickText.includes("甩牌失败，捡小"), false);
});

Deno.test("test_renderGameTable_shows_only_current_bid_winner_under_avatar", () => {
  const snap = makeSnapshot({
    trump_suit: "spades",
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
    ],
    bid_winner: {
      player: 2,
      cards: [{ id: "D1-spades-2", suit: "spades", rank: "2" }],
      kind: "trump_rank",
      suit: "spades",
      joker_type: null,
      count: 1,
    },
  });
  const el = renderGameTable(snap);
  const markers = el.querySelectorAll(".player-bid-marker");
  const markerText = markers[0]?.textContent ?? "";
  assertEquals(markers.length, 1);
  assertEquals(markerText.includes("♠2"), true);
  assertEquals(markerText.includes("♠主"), true);
});

Deno.test("test_renderGameTable_uses_updated_bid_winner_after_stir", () => {
  const snap = makeSnapshot({
    phase: "WAITING",
    trump_suit: "clubs",
    trump_rank: "2",
    bid_events: [
      {
        player: 2,
        cards: [{ id: "D1-spades-2", suit: "spades", rank: "2" }],
        kind: "trump_rank",
        suit: "spades",
        joker_type: null,
        count: 1,
      },
    ],
    bid_winner: {
      player: 1,
      cards: [
        { id: "D1-clubs-2", suit: "clubs", rank: "2" },
        { id: "D2-clubs-2", suit: "clubs", rank: "2" },
      ],
      kind: "trump_rank",
      suit: "clubs",
      joker_type: null,
      count: 2,
    },
  });
  const el = renderGameTable(snap);
  const markers = el.querySelectorAll(".player-bid-marker");
  const markerText = markers[0]?.textContent ?? "";
  const tableText = el.textContent ?? "";

  assertEquals(markers.length, 1);
  assertEquals(markerText.includes("♣2 ♣2"), true);
  assertEquals(markerText.includes("♣主"), true);
  assertEquals(tableText.includes("♠2"), false);
});
