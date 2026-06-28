import type { StateSnapshot } from "../../core/types.ts";
import type { Card } from "../../core/types.ts";
import type {
  InteractionMode,
  LevelChangeInfo,
} from "../../engine/types.ts";
import { el } from "../dom.ts";
import {
  DEFAULT_VIEWER_PLAYER,
  PLAYER_INDEXES,
  type PlayerIndex,
} from "../../config.ts";
import {
  playerView,
  teamLabelForViewer,
  viewerTeam,
} from "../player-view.ts";
import { cardDisplay, suitSymbol } from "../../core/card.ts";

/**
 * Render a round scoring overlay showing scoring details, player confirmation
 * status, and optionally a "下一轮" button.
 */
export function renderScoringOverlay(
  snapshot: StateSnapshot,
  _interactionMode: InteractionMode,
  onNextRound?: () => void,
  levelChange?: LevelChangeInfo,
  viewerPlayer?: PlayerIndex | null,
): HTMLElement {
  const overlay = el("div", { class: "scoring-overlay" });
  const card = el("div", { class: "scoring-overlay__card" });

  const confirmedSet = new Set(snapshot.next_round_confirmed);

  if (snapshot.scoring) {
    card.appendChild(
      renderScoringSummary(snapshot, levelChange, viewerPlayer),
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
    renderScoringActions(confirmedSet, onNextRound, viewerPlayer),
  );

  overlay.appendChild(card);
  return overlay;
}

function renderScoringSummary(
  snapshot: StateSnapshot,
  levelChange?: LevelChangeInfo,
  viewerPlayer?: PlayerIndex | null,
): HTMLElement {
  const scoring = snapshot.scoring;
  const summary = el("div", { class: "scoring-overlay__summary" });
  summary.appendChild(
    el("div", { class: "scoring-overlay__title" }, "本轮结算"),
  );
  if (scoring === null) {
    return summary;
  }

  const declarerLabel = scoring.declarer_team !== null
    ? teamLabelForViewer(scoring.declarer_team, viewerPlayer)
    : "—";
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
      `庄家 ${declarerLabel}`,
    ),
  );

  const resultText = scoringResultText(
    scoring.declarer_team,
    levelChange,
    viewerPlayer,
  );
  if (resultText !== null) {
    summary.appendChild(
      el("div", { class: "scoring-overlay__result" }, resultText),
    );
  }
  return summary;
}

function scoringResultText(
  declarerTeam: number | null,
  levelChange?: LevelChangeInfo,
  viewerPlayer?: PlayerIndex | null,
): string | null {
  if (levelChange === undefined) {
    return null;
  }
  const isViewerDeclarer = declarerTeam === viewerTeam(viewerPlayer);
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
  confirmedSet: Set<number>,
  onNextRound?: () => void,
  viewerPlayer?: PlayerIndex | null,
): HTMLElement {
  const actions = el("div", { class: "scoring-overlay__actions" });
  const confirmGrid = el("div", { class: "confirm-grid" });
  for (const playerIndex of PLAYER_INDEXES) {
    const view = playerView(playerIndex, viewerPlayer);
    const isReady = confirmedSet.has(playerIndex);
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
    viewerPlayer ?? DEFAULT_VIEWER_PLAYER,
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
