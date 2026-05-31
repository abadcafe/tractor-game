/**
 * Scoreboard and game log display.
 *
 * The game log records each trick with who played what,
 * not just who won — so players can review the round.
 */

import type { GameState, CompletedTrick } from '../core/types';
import { cardDisplay } from '../core/card';

export class Scoreboard {
  /** Update scoreboard from game state. */
  static update(state: GameState): void {
    this.updateLevels(state);
    this.updateScores(state);
  }

  private static updateLevels(state: GameState): void {
    const levelDeclarer = document.getElementById('level-declarer');
    const levelDefender = document.getElementById('level-defender');

    if (levelDeclarer) levelDeclarer.textContent = state.teams[0].currentLevel;
    if (levelDefender) levelDefender.textContent = state.teams[1].currentLevel;
  }

  private static updateScores(state: GameState): void {
    const scoreInfo = document.getElementById('current-scores');
    if (!scoreInfo) return;

    scoreInfo.innerHTML = `
      <div>防守方得分: ${state.defenderPoints} / 200</div>
      <div>主牌: ${state.trumpSuit ? suitName(state.trumpSuit) : '未定'}</div>
      <div>级别: ${state.currentLevel}</div>
    `;
  }

  /** Add a log entry for a completed trick — shows who played what. */
  static logTrick(trick: CompletedTrick): void {
    const log = document.getElementById('log-entries');
    if (!log) return;

    const playerNames = ['同伴', '对手A', '对手B', '你'];

    // Build a line for each player's play
    const playLines = trick.slots.map(s => {
      const name = playerNames[s.playerIndex];
      const cards = s.cards.map(c => cardDisplay(c)).join(' ');
      return `${name}: ${cards}`;
    }).join(' | ');

    const winner = playerNames[trick.winnerIndex];

    const entry = document.createElement('div');
    entry.className = 'log-entry';
    if (trick.points > 0) {
      entry.classList.add('defender-point');
    }

    entry.textContent = playLines + ` → ${winner}赢` + (trick.points > 0 ? ` (+${trick.points}分)` : '');

    log.prepend(entry);

    // Keep max 50 entries
    while (log.children.length > 50) {
      log.lastChild?.remove();
    }
  }

  /** Add a custom log message. */
  static log(message: string): void {
    const log = document.getElementById('log-entries');
    if (!log) return;

    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.textContent = message;
    log.prepend(entry);

    while (log.children.length > 50) {
      log.lastChild?.remove();
    }
  }
}

function suitName(suit: string): string {
  switch (suit) {
    case 'hearts': return '♥';
    case 'spades': return '♠';
    case 'diamonds': return '♦';
    case 'clubs': return '♣';
    case 'joker': return '🃏';
    default: return suit;
  }
}
