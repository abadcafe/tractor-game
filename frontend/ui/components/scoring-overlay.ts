import type { StateSnapshot, InteractionMode } from "../../core/types.ts";
import { el } from "../dom.ts";

const SUIT_NAMES: Record<string, string> = {
  spades: "♠",
  hearts: "♥",
  clubs: "♣",
  diamonds: "♦",
  joker: "🃏",
};

function suitName(s: string): string {
  return SUIT_NAMES[s] ?? s;
}

/**
 * Render a round scoring overlay showing scoring details and optionally
 * a "下一轮" (next round) button when the human declarer needs to acknowledge.
 *
 * @param snapshot - Current game state snapshot
 * @param interactionMode - Current interaction mode; "next_round" shows the button
 * @param onNextRound - Optional callback invoked when the next-round button is clicked
 * @returns An HTMLElement containing the scoring overlay
 */
export function renderScoringOverlay(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  onNextRound?: () => void,
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

    // Level change info based on scoring rules
    const humanTeam = 0;
    const isHumanDeclarer = snapshot.scoring.declarer_team === humanTeam;
    let levelDelta: number;
    let switched: boolean;
    if (total < 40) {
      levelDelta = total < 5 ? 3 : 2;
      switched = false;
    } else if (total < 80) {
      levelDelta = 1;
      switched = false;
    } else if (total < 120) {
      levelDelta = 1;
      switched = true;
    } else if (total < 160) {
      levelDelta = 2;
      switched = true;
    } else if (total < 200) {
      levelDelta = 3;
      switched = true;
    } else {
      levelDelta = 4;
      switched = true;
    }

    if (switched) {
      // Declarer lost — new declarer gains levels
      const loser = isHumanDeclarer ? "我们" : "对方";
      const winner = isHumanDeclarer ? "对方" : "我们";
      overlay.appendChild(
        el("div", { class: "scoring-overlay__level-change" },
          `${loser}下庄，${winner}升 ${levelDelta} 级`),
      );
    } else {
      const who = isHumanDeclarer ? "我们" : "对方";
      overlay.appendChild(
        el("div", { class: "scoring-overlay__level-change" },
          `${who}升 ${levelDelta} 级`),
      );
    }

    if (snapshot.scoring.bottom_cards.length > 0) {
      const cardTexts = snapshot.scoring.bottom_cards.map((c) => `${suitName(c.suit)}${c.rank}`).join(", ");
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
