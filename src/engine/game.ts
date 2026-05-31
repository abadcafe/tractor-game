/**
 * Game state machine — orchestrates phase transitions and action dispatch.
 *
 * The Game class is the central controller. It holds the current GameState
 * and provides methods to transition between phases.
 *
 * The engine calls Rules layer functions for validation; it does NOT
 * contain rule logic itself.
 */

import { Card, Suit, Rank, sortHand } from '../core/card';
import { Phase, PlayType, type PlayAction, type GameState, type BidAction, type StirAction, type GameSettings, type PlayerState } from '../core/types';
import { PLAYER_COUNT, HUMAN_PLAYER_INDEX } from '../core/constants';
import { getTeamIndex, nextPlayer, clockwiseDistance } from '../core/constants';
import {
  createInitialState,
  dealCards,
  recordBid,
  setDeclarer,
  recordStir,
  pickupBottomCards,
  discardCards,
  playCards,
  advanceRound,
  serializeForAI,
} from './state';
import { calculateScore, isGameOver } from './scoring';
import { isValidBid, getValidBidLevels, isBiddingOver, getWinningBid, isValidStir, getValidStirOptions } from '../rules/bidding';
import { getLegalPlays } from '../rules/validator';

export type GameEventCallback = (state: GameState) => void;

export class Game {
  state: GameState;
  private listeners: Set<GameEventCallback> = new Set();
  private static STORAGE_KEY = 'tractor-game-state';

  constructor(settings?: Partial<GameSettings>) {
    // Try restoring saved state first
    const saved = Game.loadState();
    if (saved) {
      this.state = { ...saved, settings: { ...saved.settings, ...settings } };
    } else {
      this.state = createInitialState(settings);
    }
  }

  /** Subscribe to state changes. */
  onChange(cb: GameEventCallback): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  private emit(): void {
    // Sort player hands before notifying
    this.state = {
      ...this.state,
      players: this.state.players.map(p => ({
        ...p,
        hand: sortHand(p.hand),
      })) as [PlayerState, PlayerState, PlayerState, PlayerState],
    };

    // Persist to localStorage
    Game.saveState(this.state);

    for (const cb of this.listeners) {
      cb(this.state);
    }
  }

  /** Save game state to localStorage. */
  private static saveState(state: GameState): void {
    try {
      localStorage.setItem(Game.STORAGE_KEY, JSON.stringify(state));
    } catch {
      // localStorage full or unavailable — just skip
    }
  }

  /** Load game state from localStorage. Returns null if none or corrupt. */
  private static loadState(): GameState | null {
    try {
      const raw = localStorage.getItem(Game.STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      // Basic sanity check
      if (parsed && parsed.phase && parsed.players && parsed.players.length === 4) {
        return parsed as GameState;
      }
    } catch {
      // Corrupt data — ignore
    }
    return null;
  }

  /** Clear saved game state. */
  static clearSavedState(): void {
    localStorage.removeItem(Game.STORAGE_KEY);
  }

  // ---- Game Flow ----

  /** Start a new game. */
  startNewGame(): void {
    Game.clearSavedState();
    this.state = createInitialState(this.state.settings);
    this.emit();
  }

  /** Deal cards and begin bidding. */
  startRound(): void {
    this.state = dealCards(this.state);
    this.emit();
  }

  // ---- Bidding ----

  /** Submit a bid for the current player. */
  submitBid(playerIndex: number, level: Rank | null, pass: boolean): boolean {
    if (this.state.phase !== Phase.BIDDING) return false;
    if (playerIndex !== this.state.currentPlayerIndex) return false;

    const highestBid = getWinningBid(this.state.biddingHistory);
    const highestLevel = highestBid?.level ?? null;

    if (!isValidBid(level, pass, highestLevel, this.state.currentLevel, this.state.biddingHistory.length === 0)) {
      return false;
    }

    const bid: BidAction = { playerIndex, level, pass };
    this.state = recordBid(this.state, bid);

    // Check if bidding is over
    if (isBiddingOver(this.state.biddingHistory, PLAYER_COUNT)) {
      const winner = getWinningBid(this.state.biddingHistory);
      if (winner && winner.level) {
        // Winner must choose trump suit
        // If human: wait for UI input -> handled by setTrumpAfterBid
        // If AI: handled externally
        this.emit();
      } else {
        // No one bid — redeal
        this.startRound();
      }
    } else {
      this.emit();
    }

    return true;
  }

  /** Set trump suit after winning the bid. */
  setTrumpAfterBid(playerIndex: number, trumpSuit: Suit): boolean {
    const winner = getWinningBid(this.state.biddingHistory);
    if (!winner || winner.playerIndex !== playerIndex) return false;
    if (this.state.phase !== Phase.BIDDING) return false;

    const bidLevel = winner.level!;
    this.state = setDeclarer(this.state, playerIndex, trumpSuit, bidLevel);
    this.emit();
    return true;
  }

  /** Get valid bid levels for the current player. */
  getValidBids(): Rank[] {
    const highestBid = getWinningBid(this.state.biddingHistory);
    return getValidBidLevels(highestBid?.level ?? null, this.state.currentLevel);
  }

  // ---- Stirring (炒地皮) ----

  /** Submit a stir or pass. */
  submitStir(playerIndex: number, stir: StirAction | null): boolean {
    if (this.state.phase !== Phase.STIRRING) return false;
    if (playerIndex !== this.state.currentPlayerIndex) return false;

    if (stir === null) {
      // Pass on stirring
      // Move to next player
      this.state = {
        ...this.state,
        currentPlayerIndex: nextPlayer(playerIndex),
      };

      // Check if stirring round is over (all passed since last stir)
      const lastStirIdx = this.state.stirHistory.length > 0
        ? this.state.stirHistory[this.state.stirHistory.length - 1].playerIndex
        : getWinningBid(this.state.biddingHistory)!.playerIndex;

      // Count clockwise steps from lastStirIdx to current player
      const playersSinceStir = clockwiseDistance(lastStirIdx, this.state.currentPlayerIndex);
      if (playersSinceStir === 0) {
        // Everyone passed — stirring over, move to exchange
        this.state = pickupBottomCards(this.state);
      }
    } else {
      if (!isValidStir(stir, this.state.trumpSuit!, this.state.trumpRank, this.state.stirHistory, playerIndex)) {
        return false;
      }
      this.state = recordStir(this.state, stir);
    }

    this.emit();
    return true;
  }

  /** Complete stirring phase (when all pass) and begin exchange. */
  completeStirring(): void {
    if (this.state.phase !== Phase.STIRRING) return;
    this.state = pickupBottomCards(this.state);
    this.emit();
  }

  /** Get valid stir options for the current player. */
  getValidStirs(): StirAction[] {
    return getValidStirOptions(
      this.state.trumpSuit!,
      this.state.trumpRank,
      this.state.currentPlayerIndex,
      this.state.stirHistory,
    );
  }

  // ---- Exchange (扣底) ----

  /** Declarer discards cards after picking up bottom cards. */
  submitDiscard(playerIndex: number, cards: Card[]): boolean {
    if (this.state.phase !== Phase.EXCHANGE) return false;
    if (playerIndex !== this.state.currentPlayerIndex) return false;

    const player = this.state.players[playerIndex];
    if (cards.length !== this.state.settings.bottomCardCount) return false;

    // Verify all cards are in player's hand
    const handIds = new Set(player.hand.map(c => c.id));
    for (const card of cards) {
      if (!handIds.has(card.id)) return false;
    }

    this.state = discardCards(this.state, playerIndex, cards);
    this.emit();
    return true;
  }

  // ---- Playing ----

  /** Play cards from a player's hand. */
  submitPlay(playerIndex: number, cards: Card[]): boolean {
    if (this.state.phase !== Phase.PLAYING) return false;
    if (playerIndex !== this.state.currentPlayerIndex) return false;

    // Determine the play type
    const legalPlays = this.getLegalPlays(playerIndex);
    const action = this.matchPlayAction(cards, legalPlays);

    if (!action) return false;

    this.state = playCards(this.state, playerIndex, action);
    this.emit();
    return true;
  }

  /** Get all legal plays for a player. */
  getLegalPlays(playerIndex: number): PlayAction[] {
    const player = this.state.players[playerIndex];
    const isLeading = this.state.currentTrick.every(s => s.cards === null);

    // Get lead action if following
    let leadAction: PlayAction | null = null;
    if (!isLeading) {
      const leadSlot = this.state.currentTrick.find(s => s.cards !== null);
      if (leadSlot && leadSlot.cards) {
        leadAction = {
          type: this.state.leadPlayType ?? PlayType.SINGLE,
          cards: leadSlot.cards,
        };
      }
    }

    return getLegalPlays(
      player.hand,
      this.state.currentTrick,
      this.state.trumpSuit!,
      this.state.trumpRank,
      isLeading,
      leadAction,
    );
  }

  // ---- Scoring ----

  /** Calculate scores for the round. */
  calculateRoundScore() {
    if (this.state.phase !== Phase.SCORING) throw new Error('Not in scoring phase');

    const lastTrick = this.state.trickHistory[this.state.trickHistory.length - 1];
    const lastTrickWinnerTeam = getTeamIndex(lastTrick.winnerIndex);

    return calculateScore(
      this.state.defenderPoints,
      this.state.bottomCards,
      lastTrickWinnerTeam,
      this.state.declarerTeamIndex,
      this.state.currentLevel,
    );
  }

  /** Advance to the next round. */
  nextRound(): void {
    const result = this.calculateRoundScore();

    if (isGameOver(result.nextLevel, this.state.settings.targetLevel)) {
      this.state = { ...this.state, phase: Phase.GAME_OVER };
      this.emit();
      return;
    }

    this.state = advanceRound(this.state, result.nextLevel, result.nextDeclarerTeam);
    this.emit();
  }

  // ---- Helpers ----

  /**
   * Clear the current trick display after the post-trick pause.
   * Resets currentTrick slots to empty for the next trick.
   */
  clearTrick(): void {
    this.state = {
      ...this.state,
      currentTrick: this.state.currentTrick.map(s => ({ ...s, cards: null })),
    };
    this.emit();
  }

  /** Check if it's the human player's turn. */
  isHumanTurn(): boolean {
    return this.state.currentPlayerIndex === HUMAN_PLAYER_INDEX;
  }

  /** Get the human player's hand. */
  getHumanHand(): Card[] {
    return this.state.players[HUMAN_PLAYER_INDEX].hand;
  }

  /** Get serialized state for AI prompts. */
  serializeForAI(): Record<string, unknown> {
    return serializeForAI(this.state);
  }

  /** Get state for a specific player (for AI). */
  getPlayerState(playerIndex: number) {
    const player = this.state.players[playerIndex];
    return {
      index: playerIndex,
      name: player.name,
      hand: player.hand,
      teamIndex: player.teamIndex,
      isDeclarer: player.isDeclarer,
      isHuman: player.isHuman,
    };
  }

  /** Try to match played cards to a legal play action. */
  private matchPlayAction(cards: Card[], legalPlays: PlayAction[]): PlayAction | null {
    const cardIds = new Set(cards.map(c => c.id));

    for (const play of legalPlays) {
      const playIds = new Set(play.cards.map(c => c.id));
      if (cardIds.size !== playIds.size) continue;

      let match = true;
      for (const id of cardIds) {
        if (!playIds.has(id)) {
          match = false;
          break;
        }
      }
      if (match) return play;
    }

    return null;
  }

  /** Update settings. */
  updateSettings(settings: Partial<GameSettings>): void {
    this.state = {
      ...this.state,
      settings: { ...this.state.settings, ...settings },
    };
    this.emit();
  }

  /** Begin the exchange phase after stirring ends. */
  beginExchange(): void {
    if (this.state.phase !== Phase.STIRRING) return;
    this.state = pickupBottomCards(this.state);
    this.emit();
  }
}
