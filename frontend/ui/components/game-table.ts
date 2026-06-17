import type { StateSnapshot } from "../../core/types.ts";
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
  if (snapshot.awaiting_action === "discard" && snapshot.stirring_state) {
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
      return `请换底牌 ${snapshot.stirring_state?.exchange_count ?? ""} 张`.trim();
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
    const seat = player !== null && player !== undefined ? SEAT_MAP[player] : null;
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
export function renderGameTable(snapshot: StateSnapshot): HTMLElement {
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
    const labelClass = `player-label team${seat.team}`;
    header.appendChild(el("span", { class: labelClass }, seat.label));
    header.appendChild(
      el("span", { class: `team-chip team${seat.team}` }, seat.team === 0 ? "我方" : "对方"),
    );
    area.appendChild(header);

    const count = snapshot.player_hand_counts?.[i] ?? 0;
    const countWrap = el("div", { class: "card-count-wrap" });
    countWrap.appendChild(el("span", { class: "card-count" }, `${count}`));
    countWrap.appendChild(el("span", { class: "card-count-label" }, "张"));
    area.appendChild(countWrap);

    // Declarer marker
    if (snapshot.declarer_player === i) {
      const marker = el("span", { class: "declarer-marker" }, "庄");
      area.appendChild(marker);
    }

    table.appendChild(area);
  }

  // Render trick view in center area (includes trump info)
  table.appendChild(renderTrickView(snapshot));

  const status = el("div", { class: "table-status" });
  status.appendChild(el("span", { class: "table-status__dot" }));
  status.appendChild(el("span", { class: "table-status__text" }, actionText(snapshot)));
  table.appendChild(status);

  return table;
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
    const suitSpan = el("span", { class: `trump-suit suit-${snapshot.trump_suit}` });
    suitSpan.textContent = suitSymbol(snapshot.trump_suit);
    trumpDiv.appendChild(suitSpan);
  }

  trumpDiv.appendChild(el("span", {}, `级牌 ${snapshot.trump_rank}`));
  bar.appendChild(trumpDiv);

  // Phase display
  const phaseLabel = PHASE_LABELS[snapshot.phase] ?? snapshot.phase;
  bar.appendChild(el("div", { class: "info-bar__phase" }, phaseLabel));

  return bar;
}
