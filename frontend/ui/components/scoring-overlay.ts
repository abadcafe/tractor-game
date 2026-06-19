import type { StateSnapshot } from "../../core/types.ts";
import type { Card } from "../../core/types.ts";
import type {
  InteractionMode,
  LevelChangeInfo,
} from "../../engine/types.ts";
import { el } from "../dom.ts";
import {
  HUMAN_SEAT,
  HUMAN_TEAM,
  SEAT_MAP,
  TEAM_LABELS,
} from "../../config.ts";
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
): HTMLElement {
  const overlay = el("div", { class: "scoring-overlay" });
  const card = el("div", { class: "scoring-overlay__card" });

  const confirmedSet = new Set(snapshot.next_round_confirmed);

  if (snapshot.scoring) {
    card.appendChild(renderScoringSummary(snapshot, levelChange));
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
    renderScoringActions(confirmedSet, onNextRound),
  );

  overlay.appendChild(card);
  return overlay;
}

function renderScoringSummary(
  snapshot: StateSnapshot,
  levelChange?: LevelChangeInfo,
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
    ? TEAM_LABELS[scoring.declarer_team] ??
      `队伍${scoring.declarer_team}`
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
      `牌分 ${scoring.defender_points} / 底牌 ${scoring.bottom_card_bonus}`,
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
): string | null {
  if (levelChange === undefined) {
    return null;
  }
  const isHumanDeclarer = declarerTeam === HUMAN_TEAM;
  if (levelChange.switched) {
    const loser = isHumanDeclarer ? TEAM_LABELS[0] : TEAM_LABELS[1];
    const winner = isHumanDeclarer ? TEAM_LABELS[1] : TEAM_LABELS[0];
    const gainText = levelChange.defenderDelta > 0
      ? ` / ${winner}升${levelChange.defenderDelta}级`
      : "";
    return `${loser}下庄${gainText}`;
  }
  const who = isHumanDeclarer ? TEAM_LABELS[0] : TEAM_LABELS[1];
  return `${who}升${levelChange.declarerDelta}级`;
}

function renderBottomCards(cards: Card[]): HTMLElement {
  const bottom = el("div", { class: "scoring-overlay__bottom" });
  bottom.appendChild(
    el("div", { class: "scoring-overlay__bottom-title" }, "底牌"),
  );
  const cardsClass = cards.length > 12
    ? "scoring-overlay__bottom-cards scoring-overlay__bottom-cards--many"
    : "scoring-overlay__bottom-cards";
  const cardsEl = el("div", { class: cardsClass });
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
): HTMLElement {
  const actions = el("div", { class: "scoring-overlay__actions" });
  // Confirmation status grid
  const confirmGrid = el("div", { class: "confirm-grid" });
  for (let i = 0; i < 4; i++) {
    const seat = SEAT_MAP[i];
    const isReady = confirmedSet.has(i);
    const slotClass = `confirm-slot ${isReady ? "ready" : "pending"}`;
    const slot = el("div", { class: slotClass });
    slot.appendChild(
      el("span", { class: "confirm-slot__name" }, seat.label),
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

  // Next round button (shown when human hasn't confirmed yet)
  const humanReady = confirmedSet.has(HUMAN_SEAT);
  if (!humanReady) {
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
