import type { StateSnapshot, InteractionMode, Card } from "../../core/types.ts";
import { cardDisplay, isTrump, isTrumpRank } from "../../core/card.ts";

/** Rank order for sorting (higher = stronger). */
const RANK_ORDER: Record<string, number> = {
  "2": 0,
  "3": 1,
  "4": 2,
  "5": 3,
  "6": 4,
  "7": 5,
  "8": 6,
  "9": 7,
  "10": 8,
  "J": 9,
  "Q": 10,
  "K": 11,
  "A": 12,
  "SJ": 13,
  "BJ": 14,
};

/** Suit order: 黑桃 > 红桃 > 梅花 > 方块. */
const SUIT_ORDER: Record<string, number> = {
  spades: 0,
  hearts: 1,
  clubs: 2,
  diamonds: 3,
  joker: -1,
};

/** Compute sort priority within trump cards.
 *  Lower number = earlier in hand (stronger / more important).
 */
function _trumpSortPriority(
  c: Card,
  trumpSuit: string | null,
  trumpRank: string,
): number {
  if (c.rank === "BJ") return 0; // 大王
  if (c.rank === "SJ") return 1; // 小王
  if (isTrumpRank(c, trumpRank)) {
    // 级牌
    if (trumpSuit !== null && c.suit === trumpSuit) return 2; // 主级牌
    return 3; // 副级牌（其他花色的级牌）
  }
  // 其他主牌（主花色的非级牌）
  return 4;
}

/** Compare two cards that are both trump. */
function _compareTrump(
  a: Card,
  b: Card,
  trumpSuit: string | null,
  trumpRank: string,
): number {
  const pa = _trumpSortPriority(a, trumpSuit, trumpRank);
  const pb = _trumpSortPriority(b, trumpSuit, trumpRank);
  if (pa !== pb) return pa - pb;

  // Same priority group:
  if (pa === 3) {
    // 副级牌：按花色 黑桃>红桃>梅花>方块
    return (SUIT_ORDER[a.suit] ?? 99) - (SUIT_ORDER[b.suit] ?? 99);
  }

  // 主级牌 / 其他主牌：按点数从大到小
  return (RANK_ORDER[b.rank] ?? -1) - (RANK_ORDER[a.rank] ?? -1);
}

/** Compare two non-trump cards. */
function _compareNonTrump(a: Card, b: Card): number {
  // 先按花色 黑桃>红桃>梅花>方块
  const suitDiff = (SUIT_ORDER[a.suit] ?? 99) - (SUIT_ORDER[b.suit] ?? 99);
  if (suitDiff !== 0) return suitDiff;
  // 同花色按点数从大到小
  return (RANK_ORDER[b.rank] ?? -1) - (RANK_ORDER[a.rank] ?? -1);
}

/** Sort hand: trump first (大王→小王→主级牌→副级牌→其他主牌),
 *  then non-trump by suit (黑桃→红桃→梅花→方块) and rank.
 */
function sortHand(
  hand: Card[],
  trumpSuit: string | null,
  trumpRank: string,
): Card[] {
  return [...hand].sort((a, b) => {
    const aTrump = isTrump(a, trumpSuit, trumpRank);
    const bTrump = isTrump(b, trumpSuit, trumpRank);
    if (aTrump && !bTrump) return -1;
    if (!aTrump && bTrump) return 1;
    if (aTrump && bTrump) {
      return _compareTrump(a, b, trumpSuit, trumpRank);
    }
    return _compareNonTrump(a, b);
  });
}

/**
 * Render the human player's hand with card display, click selection,
 * legal action highlighting, and action buttons.
 *
 * @param snapshot - current game state snapshot
 * @param interactionMode - "play", "discard", or null (spectator)
 * @param selectedCardIds - set of currently selected card IDs (managed by parent)
 * @param onCardClick - callback when a card is clicked
 * @param onAction - callback when an action button is clicked
 */
export function renderHandView(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  selectedCardIds?: Set<string>,
  onCardClick?: (cardId: string) => void,
  onAction?: (action: string) => void,
): HTMLElement {
  const handView = document.createElement("div");
  handView.classList.add("hand-view");

  // Build a set of legal card IDs for highlighting
  const legalCardIds = new Set<string>();
  for (const cards of snapshot.legal_actions) {
    for (const card of cards) {
      legalCardIds.add(card.id);
    }
  }

  // Sort hand per spec
  const sortedHand = sortHand(
    snapshot.player_hand,
    snapshot.trump_suit,
    snapshot.trump_rank,
  );

  // Render each card
  for (const card of sortedHand) {
    const cardSpan = document.createElement("span");
    cardSpan.classList.add("card", `suit-${card.suit}`);
    cardSpan.textContent = cardDisplay(card);

    if (legalCardIds.has(card.id)) {
      cardSpan.classList.add("legal");
    }

    if (selectedCardIds?.has(card.id)) {
      cardSpan.classList.add("selected");
    }

    if (onCardClick) {
      cardSpan.addEventListener("click", () => onCardClick(card.id));
    }

    handView.appendChild(cardSpan);
  }

  // Render action buttons based on interaction mode
  if (interactionMode === "play") {
    const button = document.createElement("button");
    button.textContent = "出牌";
    if (onAction) {
      button.addEventListener("click", () => onAction("play"));
    }
    handView.appendChild(button);
  } else if (interactionMode === "discard") {
    const button = document.createElement("button");
    button.textContent = "弃牌";
    if (onAction) {
      button.addEventListener("click", () => onAction("discard"));
    }
    handView.appendChild(button);
  }
  // null interactionMode: no buttons (spectator)

  return handView;
}
