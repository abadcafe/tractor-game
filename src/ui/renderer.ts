/**
 * Top-level UI renderer: orchestrates all UI component updates.
 */

import type { GameState, CompletedTrick } from '../core/types';
import { GameTable } from './game-table';
import { HandView } from './hand-view';
import { TrickView } from './trick-view';
import { Scoreboard } from './scoreboard';
import { BiddingView } from './bidding-view';

export class Renderer {
  handView: HandView;
  trickView: TrickView;

  constructor() {
    this.handView = new HandView('cards-south');
    this.trickView = new TrickView('trick-display');
    this.handView.initButtons();
  }

  /** Full render of the game state. */
  render(state: GameState): void {
    GameTable.update(state);
    Scoreboard.update(state);

    // Update trick display
    this.trickView.update(state.currentTrick);

    // Update human hand
    const humanPlayer = state.players[3];
    this.handView.setHand(humanPlayer.hand);

    // Log the latest trick
    if (state.trickHistory.length > 0) {
      const latest = state.trickHistory[state.trickHistory.length - 1];
      // Only log if we haven't already (simple dedup by checking if log has content)
      Scoreboard.logTrick(latest);
    }
  }

  /** Show the result of a completed trick (winner highlight + last-trick button). */
  showTrickResult(trick: CompletedTrick): void {
    this.trickView.showCompletedTrick(trick);
  }

  /** Show bidding UI for human player. */
  showBidding(
    validLevels: import('../core/card').Rank[],
    canPass: boolean,
    onBid: import('./bidding-view').BidCallback,
  ): void {
    BiddingView.showBidding(validLevels, canPass, onBid);
  }

  /** Show trump selection for human. */
  showTrumpSelection(onSelect: import('./bidding-view').TrumpCallback): void {
    BiddingView.showTrumpSelection(onSelect);
  }

  /** Show stirring panel. */
  showStirring(canPass: boolean, onStir: import('./bidding-view').StirCallback): void {
    BiddingView.showStirring(canPass, onStir);
  }

  /** Show scoring overlay. */
  showScoring(message: string, details: string, onNext: () => void): void {
    BiddingView.showScoring(message, details, onNext);
  }

  /** Show game over screen. */
  showGameOver(winningTeam: number): void {
    const message = winningTeam === 0 ? '🎉 恭喜！你们赢了！' : '😞 对手获胜';
    BiddingView.showScoring(message, '', () => {
      // Reload to start fresh
      window.location.reload();
    });
  }

  /** Show thinking state for an AI player. */
  showThinking(playerIndex: number): void {
    this.trickView.showThinking(playerIndex);
  }

  /** Hide thinking state. */
  hideThinking(playerIndex: number): void {
    this.trickView.hideThinking(playerIndex);
  }

  /** Clear all UI panels. */
  clearPanels(): void {
    this.trickView.clear();
  }
}
