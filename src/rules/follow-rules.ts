/**
 * Following rules for 升级 trick-taking.
 *
 * When a player leads, they can play anything.
 * When following:
 *   1. Must match the lead pattern type (single→single, pair→pair, etc.)
 *   2. Must follow suit if possible (for the lead suit)
 *   3. If can't follow suit with the required pattern, can play anything
 */

import { Card, Suit, Rank } from '../core/card';
import { PlayType, type PlayAction } from '../core/types';
import { effectiveSuit, trumpOrder } from './comparator';

/**
 * Determine the lead suit of a trick.
 * For the first card played, its effective suit is the lead suit.
 * For trump leads, the lead suit is 'trump'.
 */
export function getLeadSuit(leadAction: PlayAction, trumpSuit: Suit, trumpRank: Rank): Suit | 'trump' {
  if (leadAction.cards.length === 0) return Suit.JOKER;
  return effectiveSuit(leadAction.cards[0], trumpSuit, trumpRank);
}

/**
 * Check if a player CAN follow the lead pattern with their hand.
 *
 * @returns true if the player has cards that can follow the lead.
 */
export function canFollow(
  hand: Card[],
  leadAction: PlayAction,
  trumpSuit: Suit,
  trumpRank: Rank,
): boolean {
  switch (leadAction.type) {
    case PlayType.SINGLE:
      return canFollowSingle(hand, leadAction.cards[0], trumpSuit, trumpRank);
    case PlayType.PAIR:
      return canFollowPair(hand, leadAction.cards[0], trumpSuit, trumpRank);
    case PlayType.TRACTOR:
      return canFollowTractor(hand, leadAction, trumpSuit, trumpRank);
    case PlayType.THROW:
      return canFollowThrow(hand, leadAction, trumpSuit, trumpRank);
  }
}

/**
 * Get all legal follow plays for a given lead.
 * Returns an array of all valid card combinations the player can play.
 */
export function getLegalFollows(
  hand: Card[],
  leadAction: PlayAction,
  trumpSuit: Suit,
  trumpRank: Rank,
): PlayAction[] {
  switch (leadAction.type) {
    case PlayType.SINGLE:
      return getFollowSingles(hand, leadAction.cards[0], trumpSuit, trumpRank);
    case PlayType.PAIR:
      return getFollowPairs(hand, leadAction.cards[0], trumpSuit, trumpRank);
    case PlayType.TRACTOR:
      return getFollowTractors(hand, leadAction, trumpSuit, trumpRank);
    case PlayType.THROW:
      return getFollowThrows(hand, leadAction, trumpSuit, trumpRank);
  }
}

// ---- Single following ----

function canFollowSingle(hand: Card[], leadCard: Card, trumpSuit: Suit, trumpRank: Rank): boolean {
  const effSuit = effectiveSuit(leadCard, trumpSuit, trumpRank);

  if (effSuit === 'trump') {
    // Must follow with trump — any trump card works
    return hand.some(c => effectiveSuit(c, trumpSuit, trumpRank) === 'trump');
  }

  // Must follow with same suit (non-trump)
  return hand.some(c =>
    effectiveSuit(c, trumpSuit, trumpRank) === effSuit &&
    c.rank !== trumpRank
  );
}

function getFollowSingles(hand: Card[], leadCard: Card, trumpSuit: Suit, trumpRank: Rank): PlayAction[] {
  const effSuit = effectiveSuit(leadCard, trumpSuit, trumpRank);
  const canFollow = canFollowSingle(hand, leadCard, trumpSuit, trumpRank);

  if (canFollow) {
    // Must follow with matching suit
    return hand
      .filter(c => {
        if (effSuit === 'trump') {
          return effectiveSuit(c, trumpSuit, trumpRank) === 'trump';
        }
        return effectiveSuit(c, trumpSuit, trumpRank) === effSuit && c.rank !== trumpRank;
      })
      .map(c => ({ type: PlayType.SINGLE, cards: [c] }));
  }

  // Can't follow suit — can play any single
  return hand.map(c => ({ type: PlayType.SINGLE, cards: [c] }));
}

// ---- Pair following ----

function canFollowPair(hand: Card[], leadCard: Card, trumpSuit: Suit, trumpRank: Rank): boolean {
  const effSuit = effectiveSuit(leadCard, trumpSuit, trumpRank);
  return hasPairInEffectiveSuit(hand, effSuit, trumpSuit, trumpRank);
}

function getFollowPairs(hand: Card[], leadCard: Card, trumpSuit: Suit, trumpRank: Rank): PlayAction[] {
  const effSuit = effectiveSuit(leadCard, trumpSuit, trumpRank);
  const pairs = findPairsInEffectiveSuit(hand, effSuit, trumpSuit, trumpRank);

  if (pairs.length > 0) {
    // Must follow with a pair of matching suit
    return pairs.map(cards => ({ type: PlayType.PAIR, cards: cards.slice(0, 2) }));
  }

  // Can't follow — can play any pair or any two singles (treated as discards)
  // Actually, if you can't follow a pair, you play any two cards
  return generateAnyTwoCards(hand);
}

// ---- Tractor following ----

function canFollowTractor(hand: Card[], leadAction: PlayAction, trumpSuit: Suit, trumpRank: Rank): boolean {
  const effSuit = effectiveSuit(leadAction.cards[0], trumpSuit, trumpRank);
  const pairCount = leadAction.cards.length / 2;
  return hasTractorInEffectiveSuit(hand, effSuit, pairCount, trumpSuit, trumpRank);
}

function getFollowTractors(hand: Card[], leadAction: PlayAction, trumpSuit: Suit, trumpRank: Rank): PlayAction[] {
  const effSuit = effectiveSuit(leadAction.cards[0], trumpSuit, trumpRank);
  const pairCount = leadAction.cards.length / 2;
  const tractors = findTractorsInEffectiveSuit(hand, effSuit, pairCount, trumpSuit, trumpRank);

  if (tractors.length > 0) {
    // Must follow with a tractor of same suit and length
    return tractors.map(cards => ({
      type: PlayType.TRACTOR,
      cards: cards.slice(0, pairCount * 2),
    }));
  }

  // Can't follow tractor — can play any (pairCount * 2) cards
  // If can follow with some pairs (but not consecutive), play those pairs
  const pairs = findPairsInEffectiveSuit(hand, effSuit, trumpSuit, trumpRank);
  const followCards: Card[] = [];
  for (const pair of pairs) {
    followCards.push(...pair.slice(0, 2));
  }
  // Fill remaining with any cards from same suit or discards
  if (followCards.length >= pairCount * 2) {
    return [{
      type: PlayType.TRACTOR,
      cards: followCards.slice(0, pairCount * 2),
    }];
  }
  // Pad with any cards
  const remaining = hand.filter(c => !followCards.includes(c));
  const all = [...followCards, ...remaining];
  if (all.length >= pairCount * 2) {
    return [{
      type: PlayType.TRACTOR,
      cards: all.slice(0, pairCount * 2),
    }];
  }
  return [];
}

// ---- Throw following ----

function canFollowThrow(hand: Card[], leadAction: PlayAction, trumpSuit: Suit, trumpRank: Rank): boolean {
  const suit = leadAction.cards[0].suit;
  if (suit === Suit.JOKER) return false;

  // Must follow with cards of the same suit
  const matchingHand = hand.filter(c => c.suit === suit && c.rank !== trumpRank);
  return matchingHand.length >= leadAction.cards.length;
}

function getFollowThrows(hand: Card[], leadAction: PlayAction, trumpSuit: Suit, trumpRank: Rank): PlayAction[] {
  const suit = leadAction.cards[0].suit;
  if (suit === Suit.JOKER) return [];

  const matchingHand = hand.filter(c => c.suit === suit && c.rank !== trumpRank);
  const required = leadAction.cards.length;

  if (matchingHand.length >= required) {
    // Must follow with `required` cards of the same suit
    // Generate combinations (simplified: take highest `required` cards)
    return [{
      type: PlayType.THROW,
      cards: matchingHand.slice(0, required),
    }];
  }

  // Can't fully follow — can discard any `required` cards
  // But we must play all matching cards first
  const discards = hand
    .filter(c => !matchingHand.includes(c))
    .slice(0, required - matchingHand.length);
  return [{
    type: PlayType.THROW,
    cards: [...matchingHand, ...discards],
  }];
}

// ---- Helper functions ----

/**
 * Check if there's at least one pair in the given effective suit.
 */
function hasPairInEffectiveSuit(
  hand: Card[],
  effSuit: Suit | 'trump',
  trumpSuit: Suit,
  trumpRank: Rank,
): boolean {
  const group = groupByEffectiveOrder(hand, effSuit, trumpSuit, trumpRank);
  for (const cards of group.values()) {
    if (cards.length >= 2) return true;
  }
  return false;
}

/**
 * Find all pairs in the given effective suit.
 */
function findPairsInEffectiveSuit(
  hand: Card[],
  effSuit: Suit | 'trump',
  trumpSuit: Suit,
  trumpRank: Rank,
): Card[][] {
  const group = groupByEffectiveOrder(hand, effSuit, trumpSuit, trumpRank);
  const pairs: Card[][] = [];
  for (const cards of group.values()) {
    if (cards.length >= 2) {
      pairs.push(cards.slice(0, 2));
    }
  }
  return pairs;
}

/**
 * Check if there's a tractor of given length in the effective suit.
 */
function hasTractorInEffectiveSuit(
  hand: Card[],
  effSuit: Suit | 'trump',
  pairCount: number,
  trumpSuit: Suit,
  trumpRank: Rank,
): boolean {
  const pairs = findConsecutivePairsInGroup(hand, effSuit, trumpSuit, trumpRank);
  return findConsecutiveRun(pairs, pairCount) !== null;
}

/**
 * Find all tractors of given length in the effective suit.
 */
function findTractorsInEffectiveSuit(
  hand: Card[],
  effSuit: Suit | 'trump',
  pairCount: number,
  trumpSuit: Suit,
  trumpRank: Rank,
): Card[][] {
  const pairs = findConsecutivePairsInGroup(hand, effSuit, trumpSuit, trumpRank);
  if (pairs.length < pairCount) return [];

  const results: Card[][] = [];
  for (let i = 0; i <= pairs.length - pairCount; i++) {
    let isConsecutive = true;
    for (let j = i + 1; j < i + pairCount; j++) {
      if (pairs[j].order !== pairs[j - 1].order - getTrumpStep(pairs[j - 1].order, effSuit)) {
        isConsecutive = false;
        break;
      }
    }
    if (isConsecutive) {
      const cards: Card[] = [];
      for (let j = i; j < i + pairCount; j++) {
        cards.push(...pairs[j].cards.slice(0, 2));
      }
      results.push(cards);
    }
  }

  return results;
}

interface OrderGroup {
  order: number;
  cards: Card[];
}

function findConsecutivePairsInGroup(
  hand: Card[],
  effSuit: Suit | 'trump',
  trumpSuit: Suit,
  trumpRank: Rank,
): OrderGroup[] {
  const group = groupByEffectiveOrder(hand, effSuit, trumpSuit, trumpRank);

  // Filter to orders that have pairs, sorted by order descending
  const pairs: OrderGroup[] = [];
  for (const [orderStr, cards] of group) {
    const order = parseInt(orderStr);
    if (cards.length >= 2) {
      pairs.push({ order, cards });
    }
  }
  pairs.sort((a, b) => b.order - a.order);
  return pairs;
}

function findConsecutiveRun(pairs: OrderGroup[], length: number): OrderGroup[] | null {
  if (pairs.length < length) return null;
  for (let i = 0; i <= pairs.length - length; i++) {
    let ok = true;
    for (let j = i + 1; j < i + length; j++) {
      const step = pairs[j - 1].order >= 70 ? 1 : 1; // Simplified
      if (pairs[j].order !== pairs[j - 1].order - step) {
        ok = false;
        break;
      }
    }
    if (ok) return pairs.slice(i, i + length);
  }
  return null;
}

/**
 * Group cards by their effective order in the given suit context.
 */
function groupByEffectiveOrder(
  hand: Card[],
  effSuit: Suit | 'trump',
  trumpSuit: Suit,
  trumpRank: Rank,
): Map<string, Card[]> {
  const map = new Map<string, Card[]>();

  for (const c of hand) {
    const eff = effectiveSuit(c, trumpSuit, trumpRank);
    if (eff !== effSuit) continue;

    const order = trumpOrder(c, trumpSuit, trumpRank);
    const key = String(order);
    const arr = map.get(key) ?? [];
    arr.push(c);
    map.set(key, arr);
  }

  return map;
}

function getTrumpStep(order: number, effSuit: Suit | 'trump'): number {
  if (effSuit === 'trump') {
    if (order === 100) return 10;
    if (order === 90) return 10;
    return 1;
  }
  return 1;
}

/**
 * Generate all ways to play any two cards (for when you can't follow a pair).
 */
function generateAnyTwoCards(hand: Card[]): PlayAction[] {
  if (hand.length < 2) return [];
  // Return one option: play the two lowest cards
  return [{
    type: PlayType.PAIR,
    cards: [hand[hand.length - 2], hand[hand.length - 1]],
  }];
}
