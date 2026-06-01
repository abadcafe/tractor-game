/**
 * Type definitions for the thin TS client.
 *
 * These types match the JSON shape returned by the Python backend.
 * All game logic lives server-side.
 */

import type { Card } from './card';
import { Suit, Rank } from './card';

// ---- Re-export for convenience ----
export { Suit, Rank };
export type { Card };

// ---- Game Phases ----

export enum Phase {
  DEALING = 'dealing',
  BIDDING = 'bidding',
  STIRRING = 'stirring',
  EXCHANGE = 'exchange',
  PLAYING = 'playing',
  SCORING = 'scoring',
  GAME_OVER = 'game_over',
}

// ---- Play Patterns ----

export enum PlayType {
  SINGLE = 'single',
  PAIR = 'pair',
  TRACTOR = 'tractor',
  THROW = 'throw',
}

// ---- Bidding ----

export interface BidAction {
  playerIndex: number;
  level: Rank | null;
  pass: boolean;
}

// ---- Trick ----

export interface TrickSlot {
  playerIndex: number;
  cards: Card[] | null;
}

export interface CompletedTrick {
  leadPlayerIndex: number;
  leadType: PlayType;
  slots: { playerIndex: number; cards: Card[] }[];
  winnerIndex: number;
  points: number;
}

// ---- Player & Team ----

export interface PlayerState {
  index: number;
  name: string;
  hand: Card[];
  teamIndex: number;
  isHuman: boolean;
  isDeclarer: boolean;
}

export interface TeamState {
  index: number;
  tricks: CompletedTrick[];
  currentLevel: Rank;
}

// ---- Game Configuration ----

export interface GameSettings {
  apiKey: string;
  model: string;
  baseUrl: string;
  targetLevel: Rank;
  bottomCardCount: number;
}

// ---- Main Game State ----

export interface GameState {
  phase: Phase;
  currentLevel: Rank;
  players: PlayerState[];
  teams: [TeamState, TeamState];
  currentPlayerIndex: number;
  trumpSuit: Suit | null;
  trumpRank: Rank;
  declarerTeamIndex: number;
  currentTrick: TrickSlot[];
  leadPlayerIndex: number;
  leadPlayType: PlayType | null;
  bottomCards: Card[];
  trickHistory: CompletedTrick[];
  lastCompletedTrick: CompletedTrick | null;
  biddingHistory: BidAction[];
  stirHistory: unknown[];
  defenderPoints: number;
  settings: GameSettings;
}
