import type { StateSnapshot } from "../../core/types.ts";
import type { InteractionMode, LevelChangeInfo } from "../../engine/types.ts";
import { el } from "../dom.ts";
import { HUMAN_TEAM } from "../../config.ts";
import { suitSymbol } from "../../core/card.ts";

/**
 * Render a round scoring overlay showing scoring details and optionally
 * a "下一轮" (next round) button when the human declarer needs to acknowledge.
 *
 * @param snapshot - Current game state snapshot
 * @param interactionMode - Current interaction mode; "next_round" shows the button
 * @param onNextRound - Optional callback invoked when the next-round button is clicked
 * @param levelChange - Pre-computed level change info from engine layer
 * @returns An HTMLElement containing the scoring overlay
 */
export function renderScoringOverlay(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  onNextRound?: () => void,
  levelChange?: LevelChangeInfo,
): HTMLElement {
  const overlay = el("div", { class: "scoring-overlay" });

  if (snapshot.scoring) {
    const trickPts = snapshot.scoring.defender_points;
    const bonus = snapshot.scoring.bottom_card_bonus;
    const total = snapshot.scoring.total_defender_points;

    if (bonus > 0) {
      overlay.appendChild(
        el("div", { class: "scoring-overlay__defender-points" },
          `敌方得牌分: ${trickPts} + 底牌加分: ${bonus} = 总分: ${total}`),
      );
    } else {
      overlay.appendChild(
        el("div", { class: "scoring-overlay__defender-points" },
          `敌方得分: ${total}`),
      );
    }

    overlay.appendChild(
      el("div", { class: "scoring-overlay__declarer-team" },
        `庄家队伍: ${snapshot.scoring.declarer_team === 0 ? "队伍0" : "队伍1"}`),
    );

    // Level change info - pre-computed by engine layer
    if (levelChange) {
      const isHumanDeclarer = snapshot.scoring.declarer_team === HUMAN_TEAM;

      if (levelChange.switched) {
        // Declarer lost — new declarer (old defender) gains levels
        const loser = isHumanDeclarer ? "我们" : "对方";
        const winner = isHumanDeclarer ? "对方" : "我们";
        const gainText = levelChange.defenderDelta > 0
          ? `，${winner}升 ${levelChange.defenderDelta} 级`
          : "";
        overlay.appendChild(
          el("div", { class: "scoring-overlay__level-change" },
            `${loser}下庄${gainText}`),
        );
      } else {
        const who = isHumanDeclarer ? "我们" : "对方";
        overlay.appendChild(
          el("div", { class: "scoring-overlay__level-change" },
            `${who}升 ${levelChange.declarerDelta} 级`),
        );
      }
    }

    if (snapshot.scoring.bottom_cards.length > 0) {
      const cardTexts = snapshot.scoring.bottom_cards.map((c) => `${suitSymbol(c.suit)}${c.rank}`).join(", ");
      overlay.appendChild(
        el("div", { class: "scoring-overlay__bottom-cards" },
          `底牌: ${cardTexts}`),
      );
    }
  }

  if (interactionMode === "next_round") {
    const button = el("button", { class: "scoring-overlay__next-round" }, "下一轮");
    if (onNextRound) {
      button.addEventListener("click", () => onNextRound());
    }
    overlay.appendChild(button);
  }

  return overlay;
}
