import type { StateSnapshot } from "../../core/types.ts";
import type { Card } from "../../core/types.ts";
import type {
  InteractionMode,
  LevelChangeInfo,
} from "../../engine/types.ts";
import { el } from "../dom.ts";
import {
  type PartnershipId,
  partnershipOf,
  SEAT_IDS,
  type SeatId,
} from "../../config.ts";
import {
  partnershipLabelForViewer,
  seatView,
  viewerPartnership,
} from "../seat-view.ts";
import { cardDisplay, suitSymbol } from "../../core/card.ts";

/**
 * Render a round scoring overlay showing scoring details, player confirmation
 * status, and optionally a "下一轮" button.
 */
export function renderScoringOverlay(
  snapshot: StateSnapshot,
  viewerSeat: SeatId,
  _interactionMode: InteractionMode,
  onNextRound?: () => void,
  levelChange?: LevelChangeInfo,
): HTMLElement {
  const overlay = el("div", { class: "scoring-overlay" });
  const card = el("div", { class: "scoring-overlay__card" });

  const confirmedSet = new Set(snapshot.next_round_confirmed);

  if (snapshot.scoring) {
    card.appendChild(
      renderScoringSummary(snapshot, levelChange, viewerSeat),
    );
    card.appendChild(renderBottomCards(snapshot.scoring.bottom_cards));
  } else {
    card.appendChild(
      el(
        "div",
        { class: "scoring-overlay__summary" },
        el("div", { class: "scoring-overlay__title" }, "本轮结算"),
      ),
    );
    card.appendChild(renderBottomCards([]));
  }

  card.appendChild(
    renderScoringActions(confirmedSet, onNextRound, viewerSeat),
  );

  overlay.appendChild(card);
  return overlay;
}

function renderScoringSummary(
  snapshot: StateSnapshot,
  levelChange: LevelChangeInfo | undefined,
  viewerSeat: SeatId,
): HTMLElement {
  const scoring = snapshot.scoring;
  const summary = el("div", { class: "scoring-overlay__summary" });
  summary.appendChild(
    el("div", { class: "scoring-overlay__title" }, "本轮结算"),
  );
  if (scoring === null) {
    return summary;
  }

  const roundWinnerLabel = partnershipLabelForViewer(
    scoring.winning_partnership,
    viewerSeat,
  );
  summary.appendChild(
    el(
      "div",
      { class: "scoring-overlay__score" },
      `${scoring.total_defender_points} 分`,
    ),
  );
  summary.appendChild(
    el(
      "div",
      { class: "scoring-overlay__meta" },
      `牌分 ${scoring.defender_points} / 抠底 ${scoring.bottom_card_bonus}`,
    ),
  );
  summary.appendChild(
    el(
      "div",
      { class: "scoring-overlay__meta" },
      `本轮胜者 ${roundWinnerLabel}`,
    ),
  );

  const resultText = scoringResultText(
    snapshot.declarer === null
      ? null
      : partnershipOf(snapshot.declarer),
    levelChange,
    viewerSeat,
  );
  if (resultText !== null) {
    summary.appendChild(
      el("div", { class: "scoring-overlay__result" }, resultText),
    );
  }
  return summary;
}

function scoringResultText(
  declarerPartnership: PartnershipId | null,
  levelChange: LevelChangeInfo | undefined,
  viewerSeat: SeatId,
): string | null {
  if (levelChange === undefined) {
    return null;
  }
  const isViewerDeclarer = declarerPartnership ===
    viewerPartnership(viewerSeat);
  if (levelChange.switched) {
    const loser = isViewerDeclarer ? "我方" : "对方";
    const winner = isViewerDeclarer ? "对方" : "我方";
    const gainText = levelChange.defenderDelta > 0
      ? ` / ${winner}升${levelChange.defenderDelta}级`
      : "";
    return `${loser}下庄${gainText}`;
  }
  const who = isViewerDeclarer ? "我方" : "对方";
  return `${who}升${levelChange.declarerDelta}级`;
}

function renderBottomCards(cards: Card[]): HTMLElement {
  const bottom = el("div", { class: "scoring-overlay__bottom" });
  bottom.appendChild(
    el("div", { class: "scoring-overlay__bottom-title" }, "底牌"),
  );
  const cardsEl = el("div", { class: "scoring-overlay__bottom-cards" });
  if (cards.length === 0) {
    cardsEl.appendChild(
      el("span", { class: "scoring-overlay__empty" }, "无底牌"),
    );
  }
  for (const card of cards) {
    cardsEl.appendChild(renderBottomCard(card));
  }
  bottom.appendChild(cardsEl);
  return bottom;
}

function renderBottomCard(card: Card): HTMLElement {
  return el("span", {
    class: `scoring-bottom-card trick-card suit-${card.suit}`,
    "data-rank": card.rank,
    "data-suit-symbol": suitSymbol(card.suit),
  }, cardDisplay(card));
}

function renderScoringActions(
  confirmedSet: Set<SeatId>,
  onNextRound: (() => void) | undefined,
  viewerSeat: SeatId,
): HTMLElement {
  const actions = el("div", { class: "scoring-overlay__actions" });
  const confirmGrid = el("div", { class: "confirm-grid" });
  for (const seatId of SEAT_IDS) {
    const view = seatView(seatId, viewerSeat);
    const isReady = confirmedSet.has(seatId);
    const slotClass = `confirm-slot ${isReady ? "ready" : "pending"}`;
    const slot = el("div", { class: slotClass });
    slot.appendChild(
      el("span", { class: "confirm-slot__name" }, view.label),
    );
    slot.appendChild(
      el(
        "span",
        { class: "confirm-slot__status" },
        isReady ? "✓" : "⋯",
      ),
    );
    confirmGrid.appendChild(slot);
  }
  actions.appendChild(confirmGrid);

  const viewerReady = confirmedSet.has(
    viewerSeat,
  );
  if (!viewerReady) {
    const button = el("button", {
      class: "btn-primary scoring-overlay__next-round",
    }, "下一轮");
    if (onNextRound) {
      button.addEventListener("click", () => onNextRound());
    }
    actions.appendChild(button);
  }

  return actions;
}
