import { cardDisplay, suitSymbol } from "../../../core/card.ts";
import type { Card, StateSnapshot } from "../../../core/types.ts";
import { el } from "../../dom.ts";
import { isPointCard } from "./cards.ts";

export function renderScorePile(snapshot: StateSnapshot): HTMLElement {
  const scorePile = el("div", { class: "score-pile" });
  scorePile.appendChild(
    el(
      "span",
      { class: "score-pile__label" },
      `捡分 ${snapshot.defender_points}`,
    ),
  );

  const cardsWrap = el("div", { class: "score-pile__cards" });
  for (const card of snapshot.defender_point_cards) {
    cardsWrap.appendChild(
      el("span", {
        class: scorePileCardClass(card),
        "data-rank": card.rank,
        "data-suit-symbol": suitSymbol(card.suit),
      }, cardDisplay(card)),
    );
  }
  scorePile.appendChild(cardsWrap);
  return scorePile;
}

function scorePileCardClass(card: Card): string {
  let className = `score-pile-card trick-card suit-${card.suit}`;
  if (isPointCard(card)) className += " point-card";
  return className;
}
