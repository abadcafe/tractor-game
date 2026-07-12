import type { Card, Rank, Suit } from "./types.ts";

/** Map of suit name to display symbol. */
const SUIT_SYMBOLS: Record<Suit, string> = {
  hearts: "♥",
  spades: "♠",
  diamonds: "♦",
  clubs: "♣",
  joker: "🃏",
};

/** Returns the Unicode symbol for a given suit name. */
export function suitSymbol(suit: Suit): string {
  return SUIT_SYMBOLS[suit];
}

/** Returns a display string for a card (symbol + rank, or joker label).
 *  Uses newline to stack suit above rank for vertical card layout.
 */
export function cardDisplay(c: Card): string {
  if (c.suit === "joker") {
    if (c.rank === "SJ") return "小\n王";
    if (c.rank === "BJ") return "大\n王";
  }
  return suitSymbol(c.suit) + "\n" + c.rank;
}

/** Returns true if the card is a joker (small or big). */
export function isJoker(c: Card): boolean {
  return c.rank === "SJ" || c.rank === "BJ";
}

/** Returns true if the card matches the given trump rank (jokers never match). */
export function isTrumpRank(c: Card, rank: Rank): boolean {
  return !isJoker(c) && c.rank === rank;
}

/** Returns true if the card is a trump card. */
function isTrump(
  c: Card,
  trumpSuit: Suit | null,
  trumpRank: Rank,
): boolean {
  if (isJoker(c)) return true;
  if (isTrumpRank(c, trumpRank)) return true;
  if (trumpSuit !== null && c.suit === trumpSuit) return true;
  return false;
}

// --- Card sorting ---

/** Rank order for sorting (higher = stronger). */
const RANK_ORDER: Record<Rank, number> = {
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
const SUIT_ORDER: Record<Suit, number> = {
  spades: 0,
  hearts: 1,
  clubs: 2,
  diamonds: 3,
  joker: -1,
};

/** Compute sort priority within trump cards.
 *  Lower number = earlier in hand (stronger / more important).
 */
function trumpSortPriority(
  c: Card,
  trumpSuit: Suit | null,
  trumpRank: Rank,
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
function compareTrump(
  a: Card,
  b: Card,
  trumpSuit: Suit | null,
  trumpRank: Rank,
): number {
  const pa = trumpSortPriority(a, trumpSuit, trumpRank);
  const pb = trumpSortPriority(b, trumpSuit, trumpRank);
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
function compareNonTrump(a: Card, b: Card): number {
  // 先按花色 黑桃>红桃>梅花>方块
  const suitDiff = (SUIT_ORDER[a.suit] ?? 99) -
    (SUIT_ORDER[b.suit] ?? 99);
  if (suitDiff !== 0) return suitDiff;
  // 同花色按点数从大到小
  return (RANK_ORDER[b.rank] ?? -1) - (RANK_ORDER[a.rank] ?? -1);
}

/** Sort hand: trump first (大王→小王→主级牌→副级牌→其他主牌),
 *  then non-trump by suit (黑桃→红桃→梅花→方块) and rank.
 */
export function sortHand(
  hand: Card[],
  trumpSuit: Suit | null,
  trumpRank: Rank,
): Card[] {
  return [...hand].sort((a, b) => {
    const aTrump = isTrump(a, trumpSuit, trumpRank);
    const bTrump = isTrump(b, trumpSuit, trumpRank);
    if (aTrump && !bTrump) return -1;
    if (!aTrump && bTrump) return 1;
    if (aTrump && bTrump) {
      return compareTrump(a, b, trumpSuit, trumpRank);
    }
    return compareNonTrump(a, b);
  });
}
