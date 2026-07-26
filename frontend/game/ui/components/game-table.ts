import type {
  BidEvent,
  CompletedTrick,
  FailedThrow,
  RoundPhase,
  StateSnapshot,
} from "../../core/types.ts";
import { trumpRank, trumpSuit } from "../../core/types.ts";
import { el } from "../dom.ts";
import {
  type PartnershipId,
  SEAT_IDS,
  type SeatId,
} from "../../config.ts";
import { seatView } from "../seat-view.ts";
import { renderTrickView } from "./trick-view.ts";
import { suitSymbol } from "../../core/card.ts";

/**
 * Determine which player is currently active based on awaiting_action
 * and phase-specific state.
 */
function getCurrentSeat(snapshot: StateSnapshot): SeatId | null {
  if (snapshot.awaiting_action === "play" && snapshot.trick) {
    return snapshot.trick.current_actor;
  }
  if (snapshot.awaiting_action === "stir" && snapshot.stirring_state) {
    return snapshot.stirring_state.current_actor;
  }
  if (
    snapshot.awaiting_action === "discard" && snapshot.stirring_state
  ) {
    return snapshot.stirring_state.exchanging_actor;
  }
  return null;
}

function actionText(
  snapshot: StateSnapshot,
  viewerSeat: SeatId,
): string {
  if (snapshot.winning_partnership !== null) return "游戏结束";

  switch (snapshot.awaiting_action) {
    case "bid":
      return "轮到你抢主";
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

  if (snapshot.phase === "DEAL_BID") return "发牌与抢主进行中";
  if (snapshot.phase === "STIRRING" && snapshot.stirring_state) {
    const seat = snapshot.stirring_state.phase === "EXCHANGING"
      ? snapshot.stirring_state.exchanging_actor
      : snapshot.stirring_state.current_actor;
    return seat !== null
      ? `等待${seatView(seat, viewerSeat).label}`
      : "等待反主";
  }
  if (snapshot.phase === "PLAYING" && snapshot.trick) {
    return `等待${
      seatView(snapshot.trick.current_actor, viewerSeat).label
    }出牌`;
  }
  if (snapshot.phase === "WAITING") return "等待确认下一轮";
  return "观察牌局";
}

/** Phase labels in Chinese. */
const PHASE_LABELS: Record<RoundPhase, string> = {
  DEAL_BID: "抢主阶段",
  STIRRING: "反主阶段",
  PLAYING: "出牌阶段",
  WAITING: "结算中",
};

/**
 * Render the game table with four player areas, trump/phase info bar, and trick view.
 */
export function renderGameTable(
  snapshot: StateSnapshot,
  viewerSeat: SeatId,
  previousTrickPreview?: CompletedTrick | null,
  failedThrowPreview?: FailedThrow | null,
  gameId?: string | null,
): HTMLElement {
  const table = el("div", { class: "game-table" });
  const currentSeat = getCurrentSeat(snapshot);

  for (const seatId of SEAT_IDS) {
    const view = seatView(seatId, viewerSeat);
    const attrs: Record<string, string> = {
      class: "player-area",
      "data-position": view.slot,
      "data-seat": seatId,
    };

    if (currentSeat === seatId) {
      attrs.class += " current";
    }

    const area = el("div", attrs);

    const header = el("div", { class: "player-area__header" });
    header.appendChild(
      renderDebugAvatar(
        seatId,
        view.partnership,
        view.avatarText,
        gameId,
      ),
    );
    const labelClass = `player-label partnership-${view.partnership}`;
    header.appendChild(el("span", { class: labelClass }, view.label));
    const teamRow = el("span", { class: "player-area__team-row" });
    teamRow.appendChild(
      el(
        "span",
        { class: `team-chip partnership-${view.partnership}` },
        view.partnershipLabel,
      ),
    );
    const declarerBadge = renderDeclarerBadge(snapshot, seatId);
    if (declarerBadge !== null) {
      teamRow.appendChild(declarerBadge);
    }
    header.appendChild(teamRow);
    area.appendChild(header);

    const badges = el("div", { class: "player-badges" });

    const bidMarker = renderBidMarker(snapshot.bid_winner, seatId);
    if (bidMarker !== null) {
      badges.appendChild(bidMarker);
    }

    badges.appendChild(renderStatusBadge(snapshot, seatId));

    if (snapshot.phase === "WAITING") {
      const isReady = snapshot.next_round_confirmed.includes(
        seatId,
      );
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
      viewerSeat,
    ),
  );

  // Render trick view in center area.
  table.appendChild(
    renderTrickView(
      snapshot,
      viewerSeat,
      previousTrickPreview,
      failedThrowPreview,
    ),
  );

  return table;
}

function renderDebugAvatar(
  seat: SeatId,
  partnership: PartnershipId,
  avatarText: string,
  gameId?: string | null,
): HTMLElement {
  const attrs: Record<string, string> = {
    class:
      `player-avatar player-avatar--debug partnership-${partnership}`,
  };
  if (gameId === null || gameId === undefined || gameId.length === 0) {
    return el("span", attrs, avatarText);
  }
  return el(
    "a",
    {
      ...attrs,
      href: `/debug/llm/${encodeURIComponent(gameId)}?seat=${seat}`,
      target: "_blank",
      rel: "noreferrer",
      title: `LLM transcript seat ${seat}`,
    },
    avatarText,
  );
}

function renderStatusBadge(
  snapshot: StateSnapshot,
  seat: SeatId,
): HTMLElement {
  const count = snapshot.remaining_cards[seat];
  const badge = el("div", { class: "player-status-badge" });
  badge.appendChild(el("span", { class: "card-count" }, `${count}张`));
  return badge;
}

function renderDeclarerBadge(
  snapshot: StateSnapshot,
  seat: SeatId,
): HTMLElement | null {
  if (snapshot.declarer === seat) {
    return el("span", { class: "declarer-text" }, "庄");
  }
  return null;
}

function renderBidMarker(
  bidWinner: BidEvent | null,
  seat: SeatId,
): HTMLElement | null {
  if (bidWinner === null || bidWinner.actor !== seat) {
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
  return "抢主";
}

/**
 * Render a small info bar showing trump suit/rank and current phase.
 */
export function renderInfoBar(snapshot: StateSnapshot): HTMLElement {
  const bar = el("div", { class: "info-bar" });

  // Trump display
  const trumpDiv = el("div", { class: "info-bar__trump" });
  trumpDiv.appendChild(el("span", {}, "主:"));

  const suit = trumpSuit(snapshot);
  if (suit !== null) {
    const suitSpan = el("span", {
      class: `trump-suit suit-${suit}`,
    });
    suitSpan.textContent = suitSymbol(suit);
    trumpDiv.appendChild(suitSpan);
  } else if (
    snapshot.phase === "DEAL_BID" || snapshot.phase === "STIRRING"
  ) {
    trumpDiv.appendChild(el("span", {}, "待定"));
  } else {
    trumpDiv.appendChild(el("span", {}, "无主"));
  }

  trumpDiv.appendChild(el("span", {}, `级牌 ${trumpRank(snapshot)}`));
  bar.appendChild(trumpDiv);

  // Phase display
  const phaseLabel = snapshot.winning_partnership !== null
    ? "游戏结束"
    : PHASE_LABELS[snapshot.phase] ?? snapshot.phase;
  bar.appendChild(el("div", { class: "info-bar__phase" }, phaseLabel));

  return bar;
}

function renderTableNotice(
  snapshot: StateSnapshot,
  previousTrickPreview: CompletedTrick | null | undefined,
  failedThrowPreview: FailedThrow | null | undefined,
  viewerSeat: SeatId,
): HTMLElement {
  const notice = el("div", { class: "table-notice" });
  const primary = failedThrowPreview !== null &&
      failedThrowPreview !== undefined
    ? "甩牌失败，捡小"
    : previousTrickPreview !== null &&
        previousTrickPreview !== undefined
    ? `上一墩 ${previousTrickPreview.points} 分`
    : actionText(snapshot, viewerSeat);

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
