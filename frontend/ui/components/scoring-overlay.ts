import type { StateSnapshot, InteractionMode } from "../../core/types.ts";
import { el } from "../dom.ts";

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
        `Defender Points: ${snapshot.scoring.defender_points}`),
    );
    overlay.appendChild(
      el("div", { class: "scoring-overlay__declarer-team" },
        `Declarer Team: ${snapshot.scoring.declarer_team}`),
    );

    if (snapshot.scoring.bottom_cards.length > 0) {
      const cardTexts = snapshot.scoring.bottom_cards.map((c) => `${c.rank}${c.suit}`).join(", ");
      overlay.appendChild(
        el("div", { class: "scoring-overlay__bottom-cards" },
          `Bottom Cards: ${cardTexts}`),
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
