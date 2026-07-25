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
    round_number: 1,
    hand: [],
    bottom_cards: [],
    trump: { kind: "suited", rank: "2", suit: "hearts" },
    declarer: "c",
    defender_points: 15,
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
    partnership_levels: { first: "3", second: "5" },
    remaining_cards: { a: 13, b: 13, c: 13, d: 13 },
    mandatory_levels: ["A"],
    next_round_confirmed: [],
    ...overrides,
  };
}

Deno.test("test_renderGameTable_shows_four_players", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap, "c");
  const players = el.querySelectorAll(".player-area");
  assertEquals(players.length, 4);
});

Deno.test("test_renderGameTable_debug_avatars_use_player_labels_not_ai_type", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap, "c", null, null, "game-1");
  const avatars = Array.from(el.querySelectorAll(".player-avatar"));
  assertEquals(avatars.length, 4);
  assertEquals(avatars.map((avatar) => avatar.textContent), [
    "A",
    "B",
    "C",
    "D",
  ]);
  assertEquals(
    avatars.every((avatar) => avatar.textContent !== "ai"),
    true,
  );
});

Deno.test("test_renderGameTable_declarer_after_team_chip", () => {
  const snap = makeSnapshot({ declarer: "c" });
  const el = renderGameTable(snap, "c");
  const viewerTeamRow = el.querySelector(
    '.player-area[data-position="bottom"] .player-area__team-row',
  );
  const viewerDeclarer = viewerTeamRow?.querySelector(".declarer-text");
  const viewerStatus = el.querySelector(
    '.player-area[data-position="bottom"] .player-status-badge',
  );

  assertEquals(viewerTeamRow?.textContent, "我方庄");
  assertEquals(viewerDeclarer?.textContent, "庄");
  assertEquals((viewerStatus?.textContent ?? "").includes("庄"), false);
});

Deno.test("test_renderGameTable_deal_bid_can_show_fixed_declarer_separate_from_bid_winner", () => {
  const snap = makeSnapshot({
    phase: "DEAL_BID",
    trump: { kind: "no_trump", rank: "3" },
    declarer: "d",
    bid_winner: {
      actor: "b",
      cards: [{ id: "D1-spades-3", suit: "spades", rank: "3" }],
      kind: "trump_rank",
      suit: "spades",
      joker_type: null,
      count: 1,
      deal_ordinal: 1,
    },
  });
  const el = renderGameTable(snap, "c");
  const nextText =
    el.querySelector('.player-area[data-position="right"]')
      ?.textContent ?? "";
  const previousText =
    el.querySelector('.player-area[data-position="left"]')
      ?.textContent ?? "";

  assertEquals(nextText.includes("庄"), true);
  assertEquals(nextText.includes("♠3"), false);
  assertEquals(previousText.includes("♠3"), true);
  assertEquals(previousText.includes("♠主"), true);
  assertEquals(previousText.includes("庄"), false);
});

Deno.test("test_renderGameTable_current_actor_highlight", () => {
  const snap = makeSnapshot({
    awaiting_action: "play",
    trick: {
      lead_actor: "a",
      slots: [],
      current_actor: "b",
      failed_throw: null,
    },
  });
  const el = renderGameTable(snap, "c");
  const current = el.querySelectorAll(".player-area.current");
  assertEquals(current.length, 1);
});

Deno.test("test_renderGameTable_player_status_badges_are_grouped", () => {
  const snap = makeSnapshot({
    phase: "WAITING",
    declarer: "b",
    mandatory_levels: ["A"],
    next_round_confirmed: ["b"],
    remaining_cards: { a: 0, b: 0, c: 0, d: 0 },
    bid_winner: {
      actor: "b",
      cards: [{ id: "D1-spades-2", suit: "spades", rank: "2" }],
      kind: "trump_rank",
      suit: "spades",
      joker_type: null,
      count: 1,
      deal_ordinal: 1,
    },
  });
  const el = renderGameTable(snap, "c");
  const previous = el.querySelector(
    '.player-area[data-position="left"]',
  );
  const badges = previous?.querySelector(".player-badges");
  const text = badges?.textContent ?? "";

  assertEquals(badges !== null, true);
  assertEquals(text.includes("♠2"), true);
  assertEquals(text.includes("0张"), true);
  assertEquals(text.includes("庄"), false);
  assertEquals(text.includes("OK"), true);
  assertEquals(
    previous?.querySelector(".player-area__team-row")?.textContent,
    "对方庄",
  );
});

Deno.test("test_renderGameTable_player_labels", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap, "c");
  const labels = el.querySelectorAll(".player-label");
  const labelTexts = Array.from(labels).map((l) => l.textContent);
  assertEquals(labelTexts.includes("座位 c / 你"), true);
  assertEquals(labelTexts.includes("座位 a"), true);
  assertEquals(labelTexts.includes("座位 b"), true);
  assertEquals(labelTexts.includes("座位 d"), true);
});

Deno.test("test_renderGameTable_orients_viewer_at_bottom", () => {
  const snap = makeSnapshot();
  const el = renderGameTable(snap, "b", null, null, "game-1");
  const bottomText =
    el.querySelector('.player-area[data-position="bottom"]')
      ?.textContent ?? "";
  const topText = el.querySelector('.player-area[data-position="top"]')
    ?.textContent ?? "";

  assertEquals(bottomText.includes("座位 b / 你"), true);
  assertEquals(topText.includes("座位 d"), true);
});

Deno.test("test_renderGameTable_global_info_bar_in_table", () => {
  const snap = makeSnapshot({
    phase: "PLAYING",
    trump: { kind: "suited", rank: "2", suit: "spades" },
  });
  const el = renderGameTable(snap, "c");
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
    trick: {
      lead_actor: "a",
      slots: [],
      current_actor: "c",
      failed_throw: null,
    },
  });
  const el = renderGameTable(snap, "c");
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
  const el = renderGameTable(snap, "c", {
    lead_actor: "a",
    winner: "c",
    points: 5,
    failed_throw: null,
    slots: [
      {
        actor: "a",
        cards: [{ id: "D1-clubs-5", suit: "clubs", rank: "5" }],
      },
      {
        actor: "b",
        cards: [{ id: "D1-hearts-9", suit: "hearts", rank: "9" }],
      },
      {
        actor: "c",
        cards: [{ id: "D1-spades-K", suit: "spades", rank: "K" }],
      },
      {
        actor: "d",
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
  const el = renderGameTable(snap, "c", null, {
    actor: "d",
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
    trump: { kind: "suited", rank: "2", suit: "spades" },
    bid_events: [
      {
        actor: "b",
        cards: [{ id: "D1-hearts-2", suit: "hearts", rank: "2" }],
        kind: "trump_rank",
        suit: "hearts",
        joker_type: null,
        count: 1,
        deal_ordinal: 1,
      },
      {
        actor: "c",
        cards: [{ id: "D1-spades-2", suit: "spades", rank: "2" }],
        kind: "trump_rank",
        suit: "spades",
        joker_type: null,
        count: 1,
        deal_ordinal: 1,
      },
    ],
    bid_winner: {
      actor: "c",
      cards: [{ id: "D1-spades-2", suit: "spades", rank: "2" }],
      kind: "trump_rank",
      suit: "spades",
      joker_type: null,
      count: 1,
      deal_ordinal: 1,
    },
  });
  const el = renderGameTable(snap, "c");
  const markers = el.querySelectorAll(".player-bid-marker");
  const markerText = markers[0]?.textContent ?? "";
  assertEquals(markers.length, 1);
  assertEquals(markerText.includes("♠2"), true);
  assertEquals(markerText.includes("♠主"), true);
});

Deno.test("test_renderGameTable_uses_updated_bid_winner_after_stir", () => {
  const snap = makeSnapshot({
    phase: "WAITING",
    trump: { kind: "suited", rank: "2", suit: "clubs" },
    bid_events: [
      {
        actor: "c",
        cards: [{ id: "D1-spades-2", suit: "spades", rank: "2" }],
        kind: "trump_rank",
        suit: "spades",
        joker_type: null,
        count: 1,
        deal_ordinal: 1,
      },
    ],
    bid_winner: {
      actor: "b",
      cards: [
        { id: "D1-clubs-2", suit: "clubs", rank: "2" },
        { id: "D2-clubs-2", suit: "clubs", rank: "2" },
      ],
      kind: "trump_rank",
      suit: "clubs",
      joker_type: null,
      count: 2,
      deal_ordinal: 1,
    },
  });
  const el = renderGameTable(snap, "c");
  const markers = el.querySelectorAll(".player-bid-marker");
  const markerText = markers[0]?.textContent ?? "";
  const tableText = el.textContent ?? "";

  assertEquals(markers.length, 1);
  assertEquals(markerText.includes("♣2 ♣2"), true);
  assertEquals(markerText.includes("♣主"), true);
  assertEquals(tableText.includes("♠2"), false);
});
