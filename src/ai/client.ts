/**
 * AI client — calls the Python backend for AI player decisions.
 *
 * Falls back to random valid play when backend is unavailable (dev mode).
 */

import type { Card } from '../core/card';
import type { GameState } from '../core/types';

export interface AIClientConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

export class AIClient {
  private config: AIClientConfig;

  constructor(config: AIClientConfig) {
    this.config = config;
  }

  updateConfig(config: Partial<AIClientConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Request an AI decision for a player.
   *
   * @param playerIndex - Which AI player (0, 1, or 2).
   * @param phase - Current game phase.
   * @param gameState - Current game state.
   * @param legalActions - Descriptions of legal actions (for prompt).
   * @param hand - AI player's hand.
   */
  async decide(
    playerIndex: number,
    phase: string,
    gameState: GameState,
    legalActions: string[],
    hand: Card[],
  ): Promise<{ actionType: string; cardIds: string[]; reasoning: string }> {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/ai/decide`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify({
          player_index: playerIndex,
          phase,
          game_state: this.serializeState(gameState, playerIndex),
          hand: hand.map(c => ({
            id: c.id,
            suit: c.suit,
            rank: c.rank,
            is_joker: c.isJoker,
            is_big_joker: c.isBigJoker,
            points: c.points,
            display: this.cardDisplayStr(c),
          })),
          legal_actions: legalActions,
          model: this.config.model,
        }),
      });

      if (!response.ok) {
        throw new Error(`AI backend error: ${response.status}`);
      }

      const data = await response.json();
      return {
        actionType: data.action_type,
        cardIds: data.card_ids ?? [],
        reasoning: data.reasoning ?? '',
      };
    } catch (err) {
      console.warn(`AI backend unavailable for player ${playerIndex}, using fallback:`, err);
      return this.fallbackDecision(hand, legalActions);
    }
  }

  /** Fallback: pick a random valid action. */
  private fallbackDecision(
    hand: Card[],
    _legalActions: string[],
  ): { actionType: string; cardIds: string[]; reasoning: string } {
    if (hand.length === 0) {
      return { actionType: 'pass', cardIds: [], reasoning: 'No cards to play' };
    }
    // Play a random card
    const idx = Math.floor(Math.random() * hand.length);
    return {
      actionType: 'play',
      cardIds: [hand[idx].id],
      reasoning: 'Random fallback play',
    };
  }

  /** Serialize game state for the AI backend. */
  private serializeState(state: GameState, playerIndex: number): Record<string, unknown> {
    return {
      phase: state.phase,
      current_level: state.currentLevel,
      trump_suit: state.trumpSuit,
      trump_rank: state.trumpRank,
      declarer_team: state.declarerTeamIndex,
      defender_points: state.defenderPoints,
      current_trick: state.currentTrick
        .filter(s => s.cards !== null)
        .map(s => ({
          player: s.playerIndex,
          cards: s.cards!.map(c => this.cardDisplayStr(c)),
        })),
      lead_player: state.leadPlayerIndex,
      lead_play_type: state.leadPlayType,
      my_team_index: state.players[playerIndex].teamIndex,
      my_is_declarer: state.players[playerIndex].isDeclarer,
      trick_count: state.trickHistory.length,
    };
  }

  private cardDisplayStr(card: Card): string {
    if (card.isJoker) return card.isBigJoker ? '大王' : '小王';
    const s: Record<string, string> = {
      hearts: '♥', spades: '♠', diamonds: '♦', clubs: '♣',
    };
    return `${s[card.suit] ?? card.suit}${card.rank}`;
  }
}
