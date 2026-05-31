/**
 * Shared type definitions for the 升级 game engine.
 */

import type { Card } from './card';
import { Suit, Rank } from './card';

// ---- Re-export for convenience ----
export { Suit, Rank };
export type { Card };

// ---- Game Phases ----

export enum Phase {
  /** Initial dealing */
  DEALING = 'dealing',
  /** Bidding for declarer rights (叫牌) */
  BIDDING = 'bidding',
  /** 炒地皮 — stirring round after initial bid */
  STIRRING = 'stirring',
  /** Declarer picks up bottom cards and discards */
  EXCHANGE = 'exchange',
  /** Main trick-taking phase */
  PLAYING = 'playing',
  /** End of round scoring */
  SCORING = 'scoring',
  /** Game over */
  GAME_OVER = 'game_over',
}

// ---- Play Patterns ----

export enum PlayType {
  SINGLE = 'single',
  PAIR = 'pair',
  TRACTOR = 'tractor',
  THROW = 'throw',
}

/** A validated play action. */
export interface PlayAction {
  type: PlayType;
  cards: Card[];
}

// ---- Bidding ----

export interface BidAction {
  /** The player who bid. */
  playerIndex: number;
  /** The level bid, or null if passing. */
  level: Rank | null;
  /** True if this is a pass. */
  pass: boolean;
}

/** A stir action (炒地皮). */
export interface StirAction {
  playerIndex: number;
  /** The new trump suit (must differ from current if same level). */
  newTrumpSuit: Suit;
  /** The level being stirred at (>= current bid level). */
  level: Rank;
}

// ---- Trick ----

export interface TrickSlot {
  playerIndex: number;
  cards: Card[] | null; // null if hasn't played yet
}

export interface CompletedTrick {
  leadPlayerIndex: number;
  leadType: PlayType;
  slots: { playerIndex: number; cards: Card[] }[];
  winnerIndex: number;
  points: number; // points collected in this trick
}

// ---- Player & Team ----

export interface PlayerState {
  index: number;       // 0-3
  name: string;
  hand: Card[];
  teamIndex: number;   // 0 or 1
  isHuman: boolean;
  isDeclarer: boolean;
}

export interface TeamState {
  index: number;       // 0 or 1
  tricks: CompletedTrick[];
  /** Current level (starting from 2, up to A). */
  currentLevel: Rank;
}

// ---- Game Configuration ----

export interface GameSettings {
  /** OpenAI API key (stored in localStorage). */
  apiKey: string;
  /** Model name, e.g. "gpt-4o". */
  model: string;
  /** API base URL. */
  baseUrl: string;
  /** Target level to reach — game ends when a team reaches this. */
  targetLevel: Rank;
  /** Number of bottom cards (usually 8 for 108-card game). */
  bottomCardCount: number;
}

// ---- Main Game State ----

export interface GameState {
  phase: Phase;

  /** The level being played this round. */
  currentLevel: Rank;

  /** 4 players. Indices: 0=S(N)  human's partner, 1=W(L) opponent, 2=E(R) opponent, 3=S(B) human */
  players: PlayerState[];

  /** 2 teams. Index 0 vs 1. */
  teams: [TeamState, TeamState];

  /** Whose turn it is (playerIndex 0-3). */
  currentPlayerIndex: number;

  /** Trump suit (null until bidding resolves). */
  trumpSuit: Suit | null;

  /** Trump rank = currentLevel. */
  trumpRank: Rank;

  /** Which team (0 or 1) is the declarer this round. */
  declarerTeamIndex: number;

  /** Current trick slots: 4 slots, null if not yet played. */
  currentTrick: TrickSlot[];

  /** Who led the current trick. */
  leadPlayerIndex: number;

  /** The pattern type of the lead play (null if no cards played yet). */
  leadPlayType: PlayType | null;

  /** The 8 bottom cards (底牌). */
  bottomCards: Card[];

  /** All completed tricks this round. */
  trickHistory: CompletedTrick[];

  /** The most recently completed trick (kept for display during the pause). */
  lastCompletedTrick: CompletedTrick | null;

  /** Bidding history for this round. */
  biddingHistory: BidAction[];

  /** Stirring history for this round. */
  stirHistory: StirAction[];

  /** Points collected by the defender team so far this round. */
  defenderPoints: number;

  /** Settings. */
  settings: GameSettings;
}

// ---- AI Decision Types ----

/** Request from frontend to backend AI. */
export interface AIDecisionRequest {
  playerIndex: number;
  phase: 'bidding' | 'stirring' | 'exchange' | 'playing';
  gameState: GameState;
  /** Enumerated legal actions for this player. */
  legalActions: AIAction[];
  /** Session-specific data the backend needs. */
  sessionData?: Record<string, unknown>;
}

/** A legal action presented to the AI. */
export interface AIAction {
  actionType: 'play' | 'bid' | 'stir' | 'discard' | 'pass';
  cards?: Card[];
  bidLevel?: Rank;
  stirLevel?: Rank;
  stirTrumpSuit?: Suit;
  /** Human-readable description. */
  description: string;
}

/** Response from backend AI. */
export interface AIDecisionResponse {
  actionType: 'play' | 'bid' | 'stir' | 'discard' | 'pass';
  cardIds: string[];
  reasoning: string;
}
