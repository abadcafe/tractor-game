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
    overlay.appendChild(
      el("div", { class: "scoring-overlay__defender-points" },
        `敌方得分: ${snapshot.scoring.defender_points}`),
    );
    overlay.appendChild(
      el("div", { class: "scoring-overlay__declarer-team" },
        `庄家队伍: ${snapshot.scoring.declarer_team === 0 ? "队伍0" : "队伍1"}`),
    );

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
