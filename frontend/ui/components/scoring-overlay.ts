import type { StateSnapshot } from "../../core/types.ts";
import type { InteractionMode, LevelChangeInfo } from "../../engine/types.ts";
import { el } from "../dom.ts";
import { HUMAN_SEAT, HUMAN_TEAM, TEAM_LABELS, SEAT_MAP } from "../../config.ts";
import { suitSymbol } from "../../core/card.ts";

/**
 * Render a round scoring overlay showing scoring details, player confirmation
 * status, and optionally a "下一轮" button.
 */
export function renderScoringOverlay(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  onNextRound?: () => void,
  levelChange?: LevelChangeInfo,
): HTMLElement {
  const overlay = el("div", { class: "scoring-overlay" });
  const card = el("div", { class: "scoring-overlay__card" });

  card.appendChild(el("div", { class: "scoring-overlay__title" }, "本轮结算"));

  if (snapshot.scoring) {
    const trickPts = snapshot.scoring.defender_points;
    const bonus = snapshot.scoring.bottom_card_bonus;
    const total = snapshot.scoring.total_defender_points;

    // Points breakdown
    if (bonus > 0) {
      card.appendChild(
        el("div", { class: "scoring-overlay__row" },
          `防守方得牌分: ${trickPts}  +  底牌加分: ${bonus}`),
      );
    }
    card.appendChild(
      el("div", { class: "scoring-overlay__row scoring-overlay__row--highlight" },
        `防守方总分: ${total}`),
    );

    // Declarer team
    const declarerLabel = snapshot.scoring.declarer_team !== null
      ? TEAM_LABELS[snapshot.scoring.declarer_team] ?? `队伍${snapshot.scoring.declarer_team}`
      : "—";
    card.appendChild(
      el("div", { class: "scoring-overlay__row" }, `庄家: ${declarerLabel}`),
    );

    // Level change info
    if (levelChange) {
      const isHumanDeclarer = snapshot.scoring.declarer_team === HUMAN_TEAM;

      if (levelChange.switched) {
        const loser = isHumanDeclarer ? TEAM_LABELS[0] : TEAM_LABELS[1];
        const winner = isHumanDeclarer ? TEAM_LABELS[1] : TEAM_LABELS[0];
        const gainText = levelChange.defenderDelta > 0
          ? `，${winner}升 ${levelChange.defenderDelta} 级`
          : "";
        card.appendChild(
          el("div", { class: "scoring-overlay__row scoring-overlay__row--highlight" },
            `${loser}下庄${gainText}`),
        );
      } else {
        const who = isHumanDeclarer ? TEAM_LABELS[0] : TEAM_LABELS[1];
        card.appendChild(
          el("div", { class: "scoring-overlay__row scoring-overlay__row--highlight" },
            `${who}升 ${levelChange.declarerDelta} 级`),
        );
      }
    }

    // Bottom cards
    if (snapshot.scoring.bottom_cards.length > 0) {
      const cardTexts = snapshot.scoring.bottom_cards.map((c) => `${suitSymbol(c.suit)}${c.rank}`).join("  ");
      card.appendChild(
        el("div", { class: "scoring-overlay__bottom-cards" },
          `底牌: ${cardTexts}`),
      );
    }
  }

  // Confirmation status grid
  const confirmedSet = new Set(snapshot.next_round_confirmed);
  const confirmGrid = el("div", { class: "confirm-grid" });
  for (let i = 0; i < 4; i++) {
    const seat = SEAT_MAP[i];
    const isReady = confirmedSet.has(i);
    const slotClass = `confirm-slot ${isReady ? "ready" : "pending"}`;
    const slot = el("div", { class: slotClass });
    slot.appendChild(el("span", { class: "confirm-slot__name" }, seat.label));
    slot.appendChild(el("span", { class: "confirm-slot__status" }, isReady ? "✓" : "⋯"));
    confirmGrid.appendChild(slot);
  }
  card.appendChild(confirmGrid);

  // Next round button (shown when human hasn't confirmed yet)
  const humanReady = confirmedSet.has(HUMAN_SEAT);
  if (!humanReady) {
    const button = el("button", { class: "btn-primary scoring-overlay__next-round" }, "下一轮");
    if (onNextRound) {
      button.addEventListener("click", () => onNextRound());
    }
    card.appendChild(button);
  }

  overlay.appendChild(card);
  return overlay;
}
