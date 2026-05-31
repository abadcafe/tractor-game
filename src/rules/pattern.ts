/**
 * Pattern detection for 升级: single, pair, tractor, throw.
 *
 * Pure functions — no side effects, no game state dependency beyond
 * trumpSuit and trumpRank.
 */

import { Card, Suit, Rank, RANK_ORDER } from '../core/card';
import { PlayType, type PlayAction } from '../core/types';
import { trumpOrder, effectiveSuit, sortByTrumpOrder } from './comparator';

/**
 * Detect all singles in a hand.
 */
export function detectSingles(hand: Card[]): PlayAction[] {
  return hand.map(c => ({
    type: PlayType.SINGLE,
    cards: [c],
  }));
}

/**
 * Detect all pairs in a hand.
 * A pair = two identical cards (same suit + same rank).
 * Since we have 2 decks, each specific card appears at most twice.
 */
export function detectPairs(hand: Card[]): PlayAction[] {
  const count = countByIdentical(hand);
  const pairs: PlayAction[] = [];

  for (const [, cards] of count) {
    if (cards.length >= 2) {
      pairs.push({
        type: PlayType.PAIR,
        cards: cards.slice(0, 2),
      });
    }
  }

  return pairs;
}

/**
 * Detect all tractor combinations in a hand.
 *
 * A tractor = 2+ consecutive pairs of the SAME effective suit.
 *
 * In trump: consecutive pairs in trump ordering (joker pairs, trump-rank pairs, trump-suit pairs).
 * In non-trump: consecutive pairs in the natural rank ordering of that suit (excluding trump rank).
 */
export function detectTractors(
  hand: Card[],
  trumpSuit: Suit,
  trumpRank: Rank,
): PlayAction[] {
  const tractors: PlayAction[] = [];

  // Partition hand into trump and non-trump groups
  const trumpCards: Card[] = [];
  const nonTrumpBySuit: Map<Suit, Card[]> = new Map();

  for (const c of hand) {
    const eff = effectiveSuit(c, trumpSuit, trumpRank);
    if (eff === 'trump') {
      trumpCards.push(c);
    } else {
      const arr = nonTrumpBySuit.get(eff) ?? [];
      arr.push(c);
      nonTrumpBySuit.set(eff, arr);
    }
  }

  // Find tractors in trump
  tractors.push(...findTractorsInGroup(trumpCards, trumpSuit, trumpRank, true));

  // Find tractors in each non-trump suit
  for (const [, cards] of nonTrumpBySuit) {
    tractors.push(...findTractorsInGroup(cards, trumpSuit, trumpRank, false));
  }

  // Deduplicate by card IDs
  return deduplicateByCardIds(tractors);
}

/**
 * Find all tractor combinations within a group of cards (all same effective suit).
 */
function findTractorsInGroup(
  cards: Card[],
  trumpSuit: Suit,
  trumpRank: Rank,
  isTrump: boolean,
): PlayAction[] {
  if (cards.length < 4) return []; // Minimum tractor = 2 pairs = 4 cards

  // Sort cards by ordering
  const sorted = isTrump
    ? sortByTrumpOrder(cards, trumpSuit, trumpRank)
    : [...cards].sort((a, b) => RANK_ORDER[b.rank] - RANK_ORDER[a.rank]);

  // Find pairs at each rank/order level
  const pairLevels: Map<number, Card[]> = new Map();

  for (const c of sorted) {
    const order = isTrump
      ? trumpOrder(c, trumpSuit, trumpRank)
      : RANK_ORDER[c.rank];

    const existing = pairLevels.get(order);
    if (existing) {
      if (existing.length < 2) {
        existing.push(c);
      }
    } else {
      pairLevels.set(order, [c]);
    }
  }

  // Filter to only levels that have pairs (2 cards at the same order)
  const pairEntries: { order: number; cards: Card[] }[] = [];
  for (const [order, pairCards] of pairLevels) {
    if (pairCards.length >= 2) {
      pairEntries.push({ order, cards: pairCards.slice(0, 2) });
    }
  }

  // Sort pair entries by order (descending)
  pairEntries.sort((a, b) => b.order - a.order);

  // Find consecutive runs of pairs
  const tractors: PlayAction[] = [];

  for (let i = 0; i < pairEntries.length; i++) {
    let run = pairEntries[i].cards;
    let j = i + 1;

    while (j < pairEntries.length && isConsecutive(pairEntries[j - 1].order, pairEntries[j].order, isTrump)) {
      run = [...run, ...pairEntries[j].cards];
      j++;
    }

    const pairCount = j - i;
    if (pairCount >= 2) {
      // Found a tractor with `pairCount` consecutive pairs
      // Also emit sub-tractors (e.g., from 3 consecutive pairs, emit the 3-pair tractor
      // and all 2-pair sub-tractors)
      for (let start = i; start < j - 1; start++) {
        for (let end = start + 2; end <= j; end++) {
          const tractorCards: Card[] = [];
          for (let k = start; k < end; k++) {
            tractorCards.push(...pairEntries[k].cards);
          }
          tractors.push({
            type: PlayType.TRACTOR,
            cards: tractorCards,
          });
        }
      }
    }

    // Skip to before j (loop increment will move to j)
    i = j - 1;
  }

  return tractors;
}

/**
 * Check if two order values are consecutive in the given context.
 *
 * In trump: the order scale has gaps (100, 90, 80, 70+offsets, 60+rank...).
 * We check if there's nothing between them in the same effective suit group.
 *
 * In non-trump: simply check if rank difference is 1 (natural consecutive ranks).
 */
function isConsecutive(orderA: number, orderB: number, isTrump: boolean): boolean {
  if (!isTrump) {
    // Natural ranks: consecutive if order diff is 1
    return orderA - orderB === 1;
  }

  // Trump: check if they're in adjacent trump levels
  // The trump order groups: 100(BJ), 90(SJ), 80(主牌), 70+offset(副级牌), 60+rank(trump suit)
  return orderA - orderB === getTrumpOrderStep(orderA);
}

/**
 * Get the step size between consecutive trump order levels.
 * Different ranges have different gaps.
 */
function getTrumpOrderStep(currentOrder: number): number {
  if (currentOrder === 100) return 10;  // BJ → SJ
  if (currentOrder === 90) return 10;   // SJ → 主牌
  if (currentOrder === 80) return 1;    // 主牌 → 副级牌 (70+3...70+0)
  if (currentOrder >= 70 && currentOrder < 80) {
    // 副级牌 → next 副级牌 or → trump suit
    if (currentOrder === 70) return 1;  // Last 副级牌 → top trump suit (60+14)
    return 1; // 副级牌 of adjacent suits
  }
  if (currentOrder >= 60 && currentOrder < 70) {
    // Within trump suit ranks: consecutive ranks differ by 1
    return 1;
  }
  return 0; // Not consecutive
}

/**
 * Detect valid throw (甩牌) combinations from a hand for a given suit.
 *
 * A throw = multiple singles of the SAME non-trump suit,
 * where each card is among the highest remaining cards of that suit.
 *
 * This means: no other player can have a higher card of that suit
 * (excluding trump, since trump can always beat non-trump).
 *
 * Since we don't know remaining cards at the rules layer, we return ALL
 * possible throw candidates. The engine layer filters based on remaining cards.
 */
export function detectThrowCandidates(
  hand: Card[],
  suit: Suit,
  trumpSuit: Suit,
  trumpRank: Rank,
): PlayAction[] {
  // Only non-trump suits can be thrown
  if (suit === Suit.JOKER || suit === trumpSuit) return [];

  const suitCards = hand.filter(c =>
    c.suit === suit && c.rank !== trumpRank
  );

  if (suitCards.length < 2) return [];

  // Generate all subsets of size 2+
  const candidates: PlayAction[] = [];
  const sorted = [...suitCards].sort((a, b) => RANK_ORDER[b.rank] - RANK_ORDER[a.rank]);

  for (let size = 2; size <= sorted.length; size++) {
    // Take the top `size` cards as a throw candidate
    candidates.push({
      type: PlayType.THROW,
      cards: sorted.slice(0, size),
    });
  }

  return candidates;
}

// ---- Helpers ----

/**
 * Count identical cards. Key = "suit-rank".
 */
function countByIdentical(hand: Card[]): Map<string, Card[]> {
  const map = new Map<string, Card[]>();
  for (const c of hand) {
    const key = `${c.suit}-${c.rank}`;
    const arr = map.get(key) ?? [];
    arr.push(c);
    map.set(key, arr);
  }
  return map;
}

/**
 * Deduplicate play actions by sorted card ID concatenation.
 */
function deduplicateByCardIds(actions: PlayAction[]): PlayAction[] {
  const seen = new Set<string>();
  const result: PlayAction[] = [];

  for (const action of actions) {
    const key = action.cards.map(c => c.id).sort().join(',');
    if (!seen.has(key)) {
      seen.add(key);
      result.push(action);
    }
  }

  return result;
}

/**
 * Get a human-readable description of a play action.
 */
export function describePlay(action: PlayAction): string {
  const cardStrs = action.cards.map(c => {
    if (c.isJoker) return c.isBigJoker ? '大王' : '小王';
    const suits: Record<string, string> = { hearts: '♥', spades: '♠', diamonds: '♦', clubs: '♣' };
    return `${suits[c.suit]}${c.rank}`;
  });

  switch (action.type) {
    case PlayType.SINGLE: return `单张 ${cardStrs[0]}`;
    case PlayType.PAIR: return `对子 ${cardStrs[0]}${cardStrs[1]}`;
    case PlayType.TRACTOR:
      return `拖拉机 ${cardStrs.join(' ')} (${action.cards.length / 2}对)`;
    case PlayType.THROW:
      return `甩牌 ${cardStrs.join(' ')}`;
  }
}
