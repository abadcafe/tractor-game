import type {
  BidEvent,
  CompletedTrick,
  FailedThrow,
  StateSnapshot,
} from "../../core/types.ts";
import { el } from "../dom.ts";
import { SEAT_MAP } from "../../config.ts";
import { renderTrickView } from "./trick-view.ts";
import { suitSymbol } from "../../core/card.ts";

/**
 * Determine which player is currently active based on awaiting_action
 * and phase-specific state.
 */
function getCurrentPlayer(snapshot: StateSnapshot): number | null {
  if (snapshot.awaiting_action === "play" && snapshot.trick) {
    return snapshot.trick.current_player;
  }
  if (snapshot.awaiting_action === "stir" && snapshot.stirring_state) {
    return snapshot.stirring_state.current_player;
  }
  if (
    snapshot.awaiting_action === "discard" && snapshot.stirring_state
  ) {
    return snapshot.stirring_state.exchanging_player;
  }
  return null;
}

function actionText(snapshot: StateSnapshot): string {
  switch (snapshot.awaiting_action) {
    case "bid":
      return "轮到你叫牌";
    case "stir":
      return "轮到你反主";
    case "discard":
      return `请换底牌 ${
        snapshot.stirring_state?.exchange_count ?? ""
      } 张`.trim();
    case "play":
      return "轮到你出牌";
    case "next_round":
      return "请确认下一轮";
  }

  if (snapshot.phase === "DEAL_BID") return "发牌与叫牌进行中";
  if (snapshot.phase === "STIRRING" && snapshot.stirring_state) {
    const player = snapshot.stirring_state.phase === "EXCHANGING"
      ? snapshot.stirring_state.exchanging_player
      : snapshot.stirring_state.current_player;
    const seat = player !== null && player !== undefined
      ? SEAT_MAP[player]
      : null;
    return seat ? `等待${seat.label}` : "等待反主";
  }
  if (snapshot.phase === "PLAYING" && snapshot.trick) {
    const seat = SEAT_MAP[snapshot.trick.current_player];
    return seat ? `等待${seat.label}出牌` : "等待出牌";
  }
  if (snapshot.phase === "WAITING") return "等待确认下一轮";
  return "观察牌局";
}

/** Phase labels in Chinese. */
const PHASE_LABELS: Record<string, string> = {
  DEAL_BID: "叫牌阶段",
  STIRRING: "反主阶段",
  PLAYING: "出牌阶段",
  WAITING: "结算中",
  GAME_OVER: "游戏结束",
};

/**
 * Render the game table with four player areas, trump/phase info bar, and trick view.
 */
export function renderGameTable(
  snapshot: StateSnapshot,
  previousTrickPreview?: CompletedTrick | null,
  failedThrowPreview?: FailedThrow | null,
): HTMLElement {
  const table = el("div", { class: "game-table" });
  const currentPlayer = getCurrentPlayer(snapshot);

  for (let i = 0; i < 4; i++) {
    const seat = SEAT_MAP[i];
    const attrs: Record<string, string> = {
      class: "player-area",
      "data-position": seat.position,
    };

    if (currentPlayer === i) {
      attrs.class += " current";
    }

    const area = el("div", attrs);

    const header = el("div", { class: "player-area__header" });
    header.appendChild(
      el(
        "span",
        { class: `player-avatar team${seat.team}` },
        seat.label.slice(0, 1),
      ),
    );
    const labelClass = `player-label team${seat.team}`;
    header.appendChild(el("span", { class: labelClass }, seat.label));
    header.appendChild(
      el(
        "span",
        { class: `team-chip team${seat.team}` },
        seat.team === 0 ? "我方" : "对方",
      ),
    );
    area.appendChild(header);

    const badges = el("div", { class: "player-badges" });

    const bidMarker = renderBidMarker(snapshot.bid_winner, i);
    if (bidMarker !== null) {
      badges.appendChild(bidMarker);
    }

    badges.appendChild(renderStatusBadge(snapshot, i));

    if (snapshot.phase === "WAITING") {
      const isReady = snapshot.next_round_confirmed.includes(i);
      const status = badges.querySelector(".player-status-badge");
      status?.appendChild(
        el("span", { class: "status-separator" }, "·"),
      );
      status?.appendChild(
        el(
          "span",
          { class: `ready-text ${isReady ? "ready" : "pending"}` },
          isReady ? "OK" : "WAIT",
        ),
      );
    }

    area.appendChild(badges);

    table.appendChild(area);
  }

  table.appendChild(renderInfoBar(snapshot));
  table.appendChild(
    renderTableNotice(
      snapshot,
      previousTrickPreview,
      failedThrowPreview,
    ),
  );

  // Render trick view in center area.
  table.appendChild(
    renderTrickView(snapshot, previousTrickPreview, failedThrowPreview),
  );

  return table;
}

function renderStatusBadge(
  snapshot: StateSnapshot,
  player: number,
): HTMLElement {
  const count = snapshot.player_hand_counts?.[player] ?? 0;
  const badge = el("div", { class: "player-status-badge" });
  badge.appendChild(el("span", { class: "card-count" }, `${count}张`));
  if (snapshot.declarer_player === player) {
    badge.appendChild(el("span", { class: "status-separator" }, "·"));
    badge.appendChild(el("span", { class: "declarer-text" }, "庄"));
  }
  return badge;
}

function renderBidMarker(
  bidWinner: BidEvent | null,
  player: number,
): HTMLElement | null {
  if (bidWinner === null || bidWinner.player !== player) {
    return null;
  }
  const className = bidWinner.suit === null
    ? "player-bid-marker suit-joker"
    : `player-bid-marker suit-${bidWinner.suit}`;
  const marker = el("div", { class: className });
  marker.appendChild(
    el(
      "span",
      { class: "player-bid-marker__cards" },
      bidCardsText(bidWinner),
    ),
  );
  marker.appendChild(
    el(
      "span",
      { class: "player-bid-marker__label" },
      bidTrumpText(bidWinner),
    ),
  );
  return marker;
}

function bidCardsText(event: BidEvent): string {
  return event.cards.map((card) => {
    if (card.suit === "joker") {
      return card.rank === "BJ" ? "大王" : "小王";
    }
    return `${suitSymbol(card.suit)}${card.rank}`;
  }).join(" ");
}

function bidTrumpText(event: BidEvent): string {
  if (event.kind === "joker") {
    return "无主";
  }
  if (event.suit !== null) {
    return `${suitSymbol(event.suit)}主`;
  }
  return "亮主";
}

/**
 * Render a small info bar showing trump suit/rank and current phase.
 */
export function renderInfoBar(snapshot: StateSnapshot): HTMLElement {
  const bar = el("div", { class: "info-bar" });

  // Trump display
  const trumpDiv = el("div", { class: "info-bar__trump" });
  trumpDiv.appendChild(el("span", {}, "主:"));

  if (snapshot.trump_suit) {
    const suitSpan = el("span", {
      class: `trump-suit suit-${snapshot.trump_suit}`,
    });
    suitSpan.textContent = suitSymbol(snapshot.trump_suit);
    trumpDiv.appendChild(suitSpan);
  } else if (
    snapshot.phase === "DEAL_BID" || snapshot.phase === "STIRRING"
  ) {
    trumpDiv.appendChild(el("span", {}, "待定"));
  } else {
    trumpDiv.appendChild(el("span", {}, "无主"));
  }

  trumpDiv.appendChild(el("span", {}, `级牌 ${snapshot.trump_rank}`));
  bar.appendChild(trumpDiv);

  // Phase display
  const phaseLabel = PHASE_LABELS[snapshot.phase] ?? snapshot.phase;
  bar.appendChild(el("div", { class: "info-bar__phase" }, phaseLabel));

  return bar;
}

function renderTableNotice(
  snapshot: StateSnapshot,
  previousTrickPreview?: CompletedTrick | null,
  failedThrowPreview?: FailedThrow | null,
): HTMLElement {
  const notice = el("div", { class: "table-notice" });
  const primary = failedThrowPreview !== null &&
      failedThrowPreview !== undefined
    ? "甩牌失败，捡小"
    : previousTrickPreview !== null &&
        previousTrickPreview !== undefined
    ? `上一墩 ${previousTrickPreview.points} 分`
    : actionText(snapshot);

  notice.appendChild(
    el("div", { class: "table-notice__primary" }, primary),
  );
  notice.appendChild(
    el(
      "div",
      { class: "table-notice__secondary" },
      `捡分 ${snapshot.defender_points}`,
    ),
  );
  return notice;
}
