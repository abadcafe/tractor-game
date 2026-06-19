import type { Card, StateSnapshot } from "../../core/types.ts";
import type {
  BidOption,
  GameAction,
  InteractionMode,
  StirButtonState,
} from "../../engine/types.ts";
import {
  cardDisplay,
  isJoker,
  isTrumpRank,
  sortHand,
  suitSymbol,
} from "../../core/card.ts";
import { SEAT_MAP } from "../../config.ts";
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
  onClearSelection?: () => void,
  onUseHint?: () => void,
  onToggleCompact?: () => void,
  compactHand?: boolean,
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
  stirButtonState?: StirButtonState,
  onShowPreviousTrick?: () => void,
  bidOptions?: BidOption[],
  pendingBidIntent?: BidOption | null,
  onBidOptionSelect?: (option: BidOption) => void,
): HTMLElement {
  const actionHints = snapshot.action_hints ?? [];
  const selectedCount = selectedCardIds?.size ?? 0;
  const sortedHand = sortHand(
    snapshot.player_hand,
    snapshot.trump_suit,
    snapshot.trump_rank,
  );
  const selectedStirCardIds = sortedHand
    .filter((card) => selectedCardIds?.has(card.id) ?? false)
    .map((card) => card.id);
  const showTools = interactionMode === "play" ||
    interactionMode === "discard" || interactionMode === "stir";
  const needsButton = interactionMode === "play" ||
    interactionMode === "discard";
  const needsStirButtons = interactionMode === "stir" &&
    (onStir !== undefined || onPass !== undefined);
  const needsBidButtons = snapshot.phase === "DEAL_BID" &&
    bidOptions !== undefined &&
    bidOptions.length > 0 &&
    onBidOptionSelect !== undefined;
  const canShowPreviousTrick = snapshot.trick_history.length > 0 &&
    onShowPreviousTrick !== undefined;
  const hasControls = (needsButton && onAction !== undefined) ||
    needsStirButtons || needsBidButtons || showTools ||
    canShowPreviousTrick;

  const container = el("div", {
    class: `hand-area${compactHand ? " compact" : ""}${
      hasControls ? " has-actions" : ""
    }${needsBidButtons ? " has-bid-actions" : ""}`,
  });

  if (hasControls) {
    const controls = el("div", { class: "hand-actions" });
    if (canShowPreviousTrick) {
      controls.appendChild(
        renderPreviousTrickButton(onShowPreviousTrick),
      );
    }
    if (needsButton && onAction) {
      controls.appendChild(
        renderActionPanel(
          snapshot,
          interactionMode,
          selectedCount,
          onAction,
        ),
      );
    }
    if (needsStirButtons) {
      controls.appendChild(
        renderStirActionPanel(
          selectedStirCardIds,
          onStir,
          onPass,
          stirButtonState,
        ),
      );
    }
    if (needsBidButtons) {
      controls.appendChild(
        renderBidActionPanel(
          bidOptions,
          pendingBidIntent ?? null,
          onBidOptionSelect,
        ),
      );
    }
    if (showTools) {
      controls.appendChild(
        renderHandTools(
          selectedCount,
          actionHints.length > 0,
          onClearSelection,
          onUseHint,
        ),
      );
    }
    container.appendChild(controls);
  }

  const panel = el("div", {
    class: `hand-panel${compactHand ? " compact" : ""}`,
  });

  const handView = el("div", {
    class: `hand-view${compactHand ? " compact" : ""}`,
  });
  const hasActionHints = actionHints.length > 0;
  const constrainedCardIds = hasActionHints ? legalCardIds : undefined;

  // Render each card
  for (const card of sortedHand) {
    const cardSpan = el("span", {
      class: handCardClass(
        card,
        snapshot.trump_suit,
        snapshot.trump_rank,
      ),
      "data-card-id": card.id,
      "data-rank": card.rank,
      "data-suit-symbol": suitSymbol(card.suit),
    });
    cardSpan.textContent = cardDisplay(card);

    // Highlight legal cards
    if (constrainedCardIds?.has(card.id)) {
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

    const canSelect = canSelectCard(
      card.id,
      interactionMode,
      constrainedCardIds,
    );
    if (!canSelect) {
      cardSpan.classList.add("not-selectable");
    }

    if (onCardClick && canSelect) {
      cardSpan.addEventListener("click", () => onCardClick(card.id));
    }

    handView.appendChild(cardSpan);
  }

  panel.appendChild(handView);

  panel.appendChild(renderScorePile(snapshot));
  container.appendChild(panel);

  return container;
}

function renderBidActionPanel(
  bidOptions: BidOption[],
  pendingBidIntent: BidOption | null,
  onBidOptionSelect: (option: BidOption) => void,
): HTMLElement {
  const panel = el("div", { class: "action-panel action-panel--bid" });
  const hasPendingBid = pendingBidIntent !== null;
  for (const option of bidOptions) {
    const className = [
      isSameBidOption(option, pendingBidIntent)
        ? "btn-warning hand-action-button selected"
        : "btn-secondary hand-action-button",
      bidOptionColorClass(option),
    ].join(" ");
    const attrs: Record<string, string> = {
      class: className,
      title: bidOptionTitle(option),
    };
    if (hasPendingBid) {
      attrs.disabled = "true";
    }
    const button = el("button", attrs, option.label);
    if (!hasPendingBid) {
      button.addEventListener("click", () => onBidOptionSelect(option));
    }
    panel.appendChild(button);
  }
  return panel;
}

function bidOptionColorClass(option: BidOption): string {
  if (
    option.trumpSuit === "hearts" || option.trumpSuit === "diamonds"
  ) {
    return "bid-button-red";
  }
  if (option.trumpSuit === "spades" || option.trumpSuit === "clubs") {
    return "bid-button-black";
  }
  return option.label.startsWith("大王")
    ? "bid-button-red"
    : "bid-button-black";
}

function isSameBidOption(
  option: BidOption,
  pendingBidIntent: BidOption | null,
): boolean {
  return pendingBidIntent !== null &&
    pendingBidIntent.cardIds.join(",") === option.cardIds.join(",");
}

function bidOptionTitle(option: BidOption): string {
  return option.trumpSuit === null
    ? `${option.label}，无主`
    : `${option.label}，抢${suitSymbol(option.trumpSuit)}主`;
}

function renderPreviousTrickButton(
  onShowPreviousTrick: () => void,
): HTMLElement {
  const panel = el("div", { class: "hand-panel__previous-trick" });
  const button = el("button", {
    class: "btn-secondary hand-action-button",
    title: "查看上一墩",
  }, "上一墩");
  button.addEventListener("click", () => onShowPreviousTrick());
  panel.appendChild(button);
  return panel;
}

function renderActionPanel(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  selectedCount: number,
  onAction: (action: GameAction) => void,
): HTMLElement {
  const panel = el("div", { class: "action-panel" });

  if (interactionMode === "play") {
    const button = el("button", {
      class: "btn-primary hand-action-button",
    }, "出牌");
    if (selectedCount === 0) {
      button.setAttribute("disabled", "true");
      button.setAttribute("title", "请先选择要出的牌");
    }
    button.addEventListener("click", () => onAction("play"));
    panel.appendChild(button);
  } else if (interactionMode === "discard") {
    const button = el("button", {
      class: "btn-warning hand-action-button",
    }, "换底牌");
    const count = snapshot.stirring_state?.exchange_count ?? 0;
    if (count > 0 && selectedCount !== count) {
      button.setAttribute("disabled", "true");
      button.setAttribute("title", `请选择 ${count} 张牌`);
    }
    button.addEventListener("click", () => onAction("discard"));
    panel.appendChild(button);
  }

  return panel;
}

function renderStirActionPanel(
  selectedCardIds: string[],
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
  stirButtonState?: StirButtonState,
): HTMLElement {
  const panel = el("div", { class: "action-panel action-panel--stir" });

  const stirButton = el("button", {
    class: "btn-warning hand-action-button",
  }, "反主");
  const stirDisabled = stirButtonState?.disabled ?? true;
  if (stirDisabled || onStir === undefined) {
    stirButton.setAttribute("disabled", "true");
  }
  if (stirButtonState?.title) {
    stirButton.setAttribute("title", stirButtonState.title);
  }
  stirButton.addEventListener("click", () => {
    if (!stirDisabled && onStir !== undefined) {
      onStir(selectedCardIds);
    }
  });
  panel.appendChild(stirButton);

  const passButton = el("button", {
    class: "btn-secondary hand-action-button",
  }, "不反");
  if (onPass === undefined) {
    passButton.setAttribute("disabled", "true");
  }
  passButton.addEventListener("click", () => onPass?.());
  panel.appendChild(passButton);

  return panel;
}

function renderHandTools(
  selectedCount: number,
  hasActionHints: boolean,
  onClearSelection?: () => void,
  onUseHint?: () => void,
): HTMLElement {
  const tools = el("div", { class: "hand-panel__tools" });

  const hintButton = el("button", {
    class: "btn-info hand-action-button",
  }, "提示");
  if (!hasActionHints || !onUseHint) {
    hintButton.setAttribute("disabled", "true");
  }
  hintButton.addEventListener("click", () => onUseHint?.());
  tools.appendChild(hintButton);

  const clearButton = el("button", {
    class: "btn-secondary hand-action-button",
  }, "清牌");
  if (selectedCount === 0 || !onClearSelection) {
    clearButton.setAttribute("disabled", "true");
  }
  clearButton.addEventListener("click", () => onClearSelection?.());
  tools.appendChild(clearButton);

  return tools;
}

function renderScorePile(snapshot: StateSnapshot): HTMLElement {
  const scorePile = el("div", { class: "score-pile" });
  scorePile.appendChild(
    el(
      "span",
      { class: "score-pile__label" },
      `捡分 ${snapshot.defender_points}`,
    ),
  );

  const cardsWrap = el("div", { class: "score-pile__cards" });
  for (const card of defenderPointCards(snapshot)) {
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

function defenderPointCards(snapshot: StateSnapshot): Card[] {
  if (snapshot.declarer_team === null) {
    return [];
  }
  const cards: Card[] = [];
  for (const trick of snapshot.trick_history) {
    const winnerTeam = SEAT_MAP[trick.winner]?.team;
    if (
      winnerTeam === undefined || winnerTeam === snapshot.declarer_team
    ) {
      continue;
    }
    for (const slot of trick.slots) {
      for (const card of slot.cards) {
        if (isPointCard(card)) {
          cards.push(card);
        }
      }
    }
  }
  return cards;
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
  if (interactionMode === "play") {
    return true;
  }
  if (!legalCardIds || legalCardIds.size === 0) {
    return true;
  }
  return legalCardIds.has(cardId);
}

/** Check if a card is a trump card (for visual highlighting). */
function isTrumpCard(
  c: Card,
  trumpSuit: string | null,
  trumpRank: string,
): boolean {
  if (isJoker(c)) return true;
  if (isTrumpRank(c, trumpRank)) return true;
  if (trumpSuit !== null && c.suit === trumpSuit) return true;
  return false;
}

function handCardClass(
  card: Card,
  trumpSuit: string | null,
  trumpRank: string,
): string {
  let className = `card suit-${card.suit}`;
  if (isPointCard(card)) className += " point-card";
  if (isTrumpCard(card, trumpSuit, trumpRank)) {
    className += " trump-card";
  }
  return className;
}

function isPointCard(card: Card): boolean {
  return card.rank === "5" || card.rank === "10" || card.rank === "K";
}

function scorePileCardClass(card: Card): string {
  let className = `score-pile-card trick-card suit-${card.suit}`;
  if (isPointCard(card)) className += " point-card";
  return className;
}
