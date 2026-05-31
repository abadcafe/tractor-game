/**
 * Game table layout: updates player areas with card backs / info.
 */

import type { GameState } from '../core/types';
import { createCardBacks } from './card-view';

export class GameTable {
  private static playerAreaIds = ['cards-north', 'cards-west', 'cards-east', 'cards-south'];

  /** Update the table to reflect current game state. */
  static update(state: GameState): void {
    this.updatePlayerAreas(state);
    this.updatePlayerInfo(state);
    this.updateGameInfo(state);
  }

  private static updatePlayerAreas(state: GameState): void {
    for (const player of state.players) {
      const areaId = this.playerAreaIds[player.index];
      const area = document.getElementById(areaId);
      if (!area) continue;

      // Human hand is managed by HandView, so skip south
      if (player.index === 3) continue;

      // Show card backs for AI players
      area.innerHTML = '';
      const backs = createCardBacks(Math.min(player.hand.length, 25));
      for (const back of backs) {
        area.appendChild(back);
      }
    }
  }

  private static updatePlayerInfo(state: GameState): void {
    const nameElements = document.querySelectorAll('.player-name');
    for (const player of state.players) {
      const nameEl = nameElements[player.index] as HTMLElement;
      if (!nameEl) continue;

      // Clear existing badges
      const existingBadge = nameEl.parentElement?.querySelector('.player-declarer-badge');
      existingBadge?.remove();

      // Add declarer badge
      if (player.isDeclarer && state.trumpSuit) {
        const badge = document.createElement('span');
        badge.className = 'player-declarer-badge';
        badge.textContent = '庄';
        nameEl.parentElement?.appendChild(badge);
      }

      // Highlight current player
      const playerArea = document.getElementById(this.playerAreaIds[player.index])?.parentElement;
      if (playerArea) {
        if (player.index === state.currentPlayerIndex &&
            (state.phase === 'bidding' || state.phase === 'stirring' || state.phase === 'playing')) {
          playerArea.style.outline = '2px solid #ffd700';
        } else {
          playerArea.style.outline = '';
        }
      }
    }
  }

  private static updateGameInfo(state: GameState): void {
    const trumpInfo = document.getElementById('trump-info');
    const scoreInfo = document.getElementById('score-info');

    if (trumpInfo) {
      const suitSymbol: Record<string, string> = {
        hearts: '♥', spades: '♠', diamonds: '♦', clubs: '♣',
      };
      const trump = state.trumpSuit
        ? `${suitSymbol[state.trumpSuit] ?? state.trumpSuit} ${state.trumpRank}`
        : '未定';
      trumpInfo.textContent = `主牌: ${trump}`;
    }

    if (scoreInfo) {
      scoreInfo.textContent = `防守方得分: ${state.defenderPoints}`;
    }
  }
}
