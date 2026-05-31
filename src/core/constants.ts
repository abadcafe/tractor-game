/**
 * Game constants for 升级 (Shengji).
 */

import { Rank } from './card';

// ---- Deck Configuration ----

/** Number of decks used. */
export const DECK_COUNT = 2;

/** Total cards in play. */
export const TOTAL_CARDS = 108;

/** Number of players. */
export const PLAYER_COUNT = 4;

/** Number of bottom cards (底牌). */
export const BOTTOM_CARD_COUNT = 8;

/** Cards per player. */
export const CARDS_PER_PLAYER = (TOTAL_CARDS - BOTTOM_CARD_COUNT) / PLAYER_COUNT; // 25

/** Total points in the game. */
export const TOTAL_POINTS = 200;

// ---- Level Progression ----

/** All levels in order. */
export const LEVELS: Rank[] = [
  Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE,
  Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE,
  Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
];

/** Starting level. */
export const START_LEVEL = Rank.TWO;

/** Target level (game ends when a team reaches/passes this). */
export const TARGET_LEVEL = Rank.ACE;

// ---- Scoring Thresholds ----

/**
 * Scoring table: [defender points threshold, level change for declarer]
 * Level change: positive = declarer goes up, negative = defender goes up.
 * "Switch" means the defending team becomes the next declarer.
 */
export interface ScoreThreshold {
  maxPoints: number;      // Upper bound (exclusive) of defender points
  declarerChange: number; // Levels declarer gains (>0) or loses (negative means defender gains)
  switchDeclarer: boolean; // True if the defending team becomes declarer next round
}

export const SCORE_TABLE: ScoreThreshold[] = [
  { maxPoints: 0,   declarerChange: 3,  switchDeclarer: false },  // 0 pts → declarer +3
  { maxPoints: 40,  declarerChange: 2,  switchDeclarer: false },  // 5-35 → declarer +2
  { maxPoints: 80,  declarerChange: 1,  switchDeclarer: false },  // 40-75 → declarer +1
  { maxPoints: 120, declarerChange: 0,  switchDeclarer: true  },  // 80-115 → switch
  { maxPoints: 160, declarerChange: -1, switchDeclarer: true  },  // 120-155 → defender +1
  { maxPoints: 200, declarerChange: -2, switchDeclarer: true  },  // 160-195 → defender +2
  // 200 → defender +3 (handled specially)
];

// ---- Player Positioning ----

/**
 * Player indices and seating:
 *   0 = North (human's partner, AI)
 *   1 = West  (opponent, AI)
 *   2 = East  (opponent, AI)
 *   3 = South (human)
 *
 * Teams: {0,3} vs {1,2}
 *   Team 0: North (AI) + South (Human)
 *   Team 1: West (AI) + East (AI)
 */

export const HUMAN_PLAYER_INDEX = 3;

export const TEAM_0 = [0, 3]; // North + South
export const TEAM_1 = [1, 2]; // West + East

/**
 * Clockwise next-player lookup.
 *
 * Table layout:
 *        North(0)
 *   West(1)    East(2)
 *        South(3)
 *
 * Clockwise: N(0)→E(2)→S(3)→W(1)→N(0)
 */
export const NEXT_PLAYER: Record<number, number> = {
  0: 2,  // North → East
  1: 0,  // West  → North
  2: 3,  // East  → South
  3: 1,  // South → West
};

/** Advance to the next player clockwise. */
export function nextPlayer(current: number): number {
  return NEXT_PLAYER[current];
}

/**
 * Number of clockwise steps from `from` to `to`.
 * E.g. clockwiseDistance(0, 3) = 2  (N→E→S)
 *      clockwiseDistance(0, 0) = 0  (same seat)
 *      clockwiseDistance(3, 1) = 1  (S→W)
 */
export function clockwiseDistance(from: number, to: number): number {
  let steps = 0;
  let cur = from;
  while (cur !== to && steps < 4) {
    cur = NEXT_PLAYER[cur];
    steps++;
  }
  return steps;
}

export function getTeamIndex(playerIndex: number): number {
  return (playerIndex === 0 || playerIndex === 3) ? 0 : 1;
}

export function getPartnerIndex(playerIndex: number): number {
  return (playerIndex + 2) % 4;
}

// ---- Default Settings ----

export const DEFAULT_SETTINGS = {
  model: 'gpt-4o',
  baseUrl: 'https://api.openai.com/v1',
  targetLevel: Rank.ACE,
  bottomCardCount: BOTTOM_CARD_COUNT,
};

// ---- OpenAI ----

export const OPENAI_DEFAULT_BASE_URL = 'https://api.openai.com/v1';
export const OPENAI_DEFAULT_MODEL = 'gpt-4o';
