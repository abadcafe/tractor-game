import type { StateSnapshot, Card } from "../../core/types.ts";
import type { InteractionMode, GameAction } from "../../engine/types.ts";
import { cardDisplay, sortHand, isJoker, isTrumpRank } from "../../core/card.ts";
import { el } from "../dom.ts";

/**
 * Render the human player's hand with card display, click selection,
 * legal action highlighting, trump card highlighting, and action buttons.
 */
export function renderHandView(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  selectedCardIds?: Set<string>,
  legalCardIds?: Set<string>,
  onCardClick?: (cardId: string) => void,
  onAction?: (action: GameAction) => void,
): HTMLElement {
  const container = el("div", { class: "hand-panel" });

  const summary = el("div", { class: "hand-panel__summary" });
  summary.appendChild(el("span", { class: "hand-panel__title" }, "手牌"));
  summary.appendChild(el("span", { class: "hand-panel__count" }, `${snapshot.player_hand.length} 张`));
  const hint = interactionHint(interactionMode, snapshot.action_hints ?? []);
  if (hint) {
    summary.appendChild(el("span", { class: "hand-panel__hint" }, hint));
  }
  container.appendChild(summary);

  const handView = el("div", { class: "hand-view" });

  // Sort hand per spec
  const sortedHand = sortHand(
    snapshot.player_hand,
    snapshot.trump_suit,
    snapshot.trump_rank,
  );

  // Render each card
  for (const card of sortedHand) {
    const cardSpan = el("span", { class: `card suit-${card.suit}` });
    cardSpan.textContent = cardDisplay(card);

    // Highlight legal cards
    if (legalCardIds?.has(card.id)) {
      cardSpan.classList.add("legal");
    }

    // Highlight trump cards
    if (isTrumpCard(card, snapshot.trump_suit, snapshot.trump_rank)) {
      cardSpan.classList.add("trump-card");
    }

    // Selected state
    if (selectedCardIds?.has(card.id)) {
      cardSpan.classList.add("selected");
    }

    const canSelect = canSelectCard(card.id, interactionMode, legalCardIds);
    if (!canSelect) {
      cardSpan.classList.add("not-selectable");
    }

    if (onCardClick && canSelect) {
      cardSpan.addEventListener("click", () => onCardClick(card.id));
    }

    handView.appendChild(cardSpan);
  }

  container.appendChild(handView);

  // Action buttons below hand
  const needsButton = interactionMode === "play" || interactionMode === "discard" || interactionMode === "next_round";
  if (needsButton && onAction) {
    const panel = el("div", { class: "action-panel" });
    const selectedCount = selectedCardIds?.size ?? 0;

    if (interactionMode === "play") {
      const button = el("button", { class: "btn-primary" }, "出牌");
      if (selectedCount === 0) {
        button.setAttribute("disabled", "true");
        button.setAttribute("title", "请先选择要出的牌");
      }
      button.addEventListener("click", () => onAction("play"));
      panel.appendChild(button);
    } else if (interactionMode === "discard") {
      const button = el("button", { class: "btn-warning" }, "换底牌");
      const count = snapshot.stirring_state?.exchange_count ?? 0;
      if (count > 0 && selectedCount !== count) {
        button.setAttribute("disabled", "true");
        button.setAttribute("title", `请选择 ${count} 张牌`);
      }
      button.addEventListener("click", () => onAction("discard"));
      panel.appendChild(button);
    } else if (interactionMode === "next_round") {
      const button = el("button", { class: "btn-primary" }, "下一轮");
      button.addEventListener("click", () => onAction("next_round"));
      panel.appendChild(button);
    }

    container.appendChild(panel);
  }

  return container;
}

function interactionHint(interactionMode: InteractionMode, actionHints: Card[][]): string {
  switch (interactionMode) {
    case "bid":
      return "选择级牌或王叫牌";
    case "stir":
      return "选择对子反主";
    case "discard":
      return "选择要放入底牌的牌";
    case "play":
      return actionHints.length > 0 ? "绿色边框为提示出牌" : "自由出牌，服务器校验";
    case "next_round":
      return "本轮结束";
    default:
      return "等待其他玩家";
  }
}

function canSelectCard(
  cardId: string,
  interactionMode: InteractionMode,
  legalCardIds?: Set<string>,
): boolean {
  if (interactionMode === null || interactionMode === "next_round") {
    return false;
  }
  if (interactionMode === "discard") {
    return true;
  }
  if (!legalCardIds || legalCardIds.size === 0) {
    return true;
  }
  return legalCardIds.has(cardId);
}

/** Check if a card is a trump card (for visual highlighting). */
function isTrumpCard(c: Card, trumpSuit: string | null, trumpRank: string): boolean {
  if (isJoker(c)) return true;
  if (isTrumpRank(c, trumpRank)) return true;
  if (trumpSuit !== null && c.suit === trumpSuit) return true;
  return false;
}
