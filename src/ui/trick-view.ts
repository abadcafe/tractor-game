/**
 * Current trick display in the center area.
 *
 * Each slot shows a player label + their played cards,
 * so it's always clear who played what.
 *
 * After a trick completes, the cards stay visible during the
 * 3-second pause, and a "上一轮" button lets the player review
 * the most recently completed trick at any time.
 */

import { createCardElement } from './card-view';
import { cardDisplay } from '../core/card';
import type { TrickSlot, CompletedTrick } from '../core/types';

/** Player display names for trick slots. */
const PLAYER_LABELS = ['同伴', '对手A', '对手B', '你'];

export class TrickView {
  // Slot element IDs: north=top, west=left, east=right, south=bottom
  private slotIds = ['trick-north', 'trick-west', 'trick-east', 'trick-south'];

  /** The last completed trick, available for review. */
  private lastTrick: CompletedTrick | null = null;

  /** Whether we're currently showing the completed-trick review overlay. */
  private reviewingLastTrick = false;

  constructor(_containerId: string) {}

  /**
   * Update the trick display with current slots.
   * @param slots - Current trick state.
   */
  update(slots: TrickSlot[]): void {
    // Don't overwrite if the player is reviewing the last trick
    if (this.reviewingLastTrick) return;

    for (let i = 0; i < 4; i++) {
      const slotEl = document.getElementById(this.slotIds[i]);
      if (!slotEl) continue;

      slotEl.innerHTML = '';

      const slot = slots[i];

      // Player name label — always visible
      const label = document.createElement('span');
      label.className = 'trick-player-label';
      label.textContent = PLAYER_LABELS[i];
      slotEl.appendChild(label);

      if (!slot || !slot.cards) continue;

      // Create card elements for played cards
      for (const card of slot.cards) {
        const cardEl = createCardElement(card, true);
        cardEl.classList.add('playing');
        slotEl.appendChild(cardEl);
      }
    }

    this.updateLastTrickButton();
  }

  /**
   * Show a completed trick result — highlight the winner
   * and keep cards visible during the pause.
   */
  showCompletedTrick(trick: CompletedTrick): void {
    this.lastTrick = trick;

    // The trick cards are already displayed (resolveTrick keeps them).
    // Add a winner indicator.
    const winnerSlotId = this.slotIds[trick.winnerIndex];
    const winnerSlot = document.getElementById(winnerSlotId);
    if (winnerSlot) {
      const badge = document.createElement('span');
      badge.className = 'trick-winner-badge';
      badge.textContent = '✓ 赢';
      winnerSlot.appendChild(badge);
    }

    this.updateLastTrickButton();
  }

  /** Clear all trick slots. */
  clear(): void {
    this.reviewingLastTrick = false;
    for (const id of this.slotIds) {
      const el = document.getElementById(id);
      if (el) el.innerHTML = '';
    }
    this.removeLastTrickButton();
  }

  /** Show a thinking indicator on a player's slot. */
  showThinking(playerIndex: number): void {
    const slotEl = document.getElementById(this.slotIds[playerIndex]);
    if (slotEl) {
      // Keep the label, add thinking after it
      const label = slotEl.querySelector('.trick-player-label');
      if (label && !slotEl.querySelector('.thinking-indicator')) {
        const indicator = document.createElement('span');
        indicator.className = 'thinking-indicator';
        indicator.textContent = '思考中...';
        slotEl.appendChild(indicator);
      }
    }
  }

  /** Remove thinking indicator. */
  hideThinking(playerIndex: number): void {
    const slotEl = document.getElementById(this.slotIds[playerIndex]);
    if (slotEl) {
      const indicator = slotEl.querySelector('.thinking-indicator');
      indicator?.remove();
    }
  }

  /** Get slot element IDs for animation targeting. */
  getSlotIds(): string[] {
    return this.slotIds;
  }

  // ---- "上一轮" review button ----

  private updateLastTrickButton(): void {
    if (!this.lastTrick) {
      this.removeLastTrickButton();
      return;
    }

    let btn = document.getElementById('btn-last-trick');
    if (!btn) {
      btn = document.createElement('button');
      btn.id = 'btn-last-trick';
      btn.textContent = '上一轮';
      btn.title = '查看上一轮出牌';
      btn.addEventListener('click', () => this.toggleReview());
      // Place it near the trick display
      const center = document.getElementById('center-area');
      if (center) {
        center.appendChild(btn);
      }
    }
  }

  private removeLastTrickButton(): void {
    document.getElementById('btn-last-trick')?.remove();
  }

  /** Toggle the review overlay for the last completed trick. */
  private toggleReview(): void {
    if (this.reviewingLastTrick) {
      // Close review — restore the current trick display
      this.reviewingLastTrick = false;
      this.removeReviewOverlay();
      return;
    }

    if (!this.lastTrick) return;

    this.reviewingLastTrick = true;
    this.showReviewOverlay(this.lastTrick);
  }

  /** Show the review overlay with the last trick's cards. */
  private showReviewOverlay(trick: CompletedTrick): void {
    // Create a floating overlay
    let overlay = document.getElementById('trick-review-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'trick-review-overlay';
      document.getElementById('game-table')?.appendChild(overlay);
    }

    overlay.innerHTML = '';

    const title = document.createElement('div');
    title.className = 'review-title';
    title.textContent = '上一轮出牌';
    overlay.appendChild(title);

    // Player labels mapped to slot IDs
    const slotMap: Record<number, string> = {
      0: 'trick-north', 1: 'trick-west', 2: 'trick-east', 3: 'trick-south',
    };
    const names = ['同伴', '对手A', '对手B', '你'];

    // Show each player's cards
    for (const slot of trick.slots) {
      const row = document.createElement('div');
      row.className = 'review-row';
      if (slot.playerIndex === trick.winnerIndex) {
        row.classList.add('review-winner');
      }

      const nameLabel = document.createElement('span');
      nameLabel.className = 'review-name';
      nameLabel.textContent = names[slot.playerIndex];
      row.appendChild(nameLabel);

      const cardsEl = document.createElement('span');
      cardsEl.className = 'review-cards';
      cardsEl.textContent = slot.cards.map(c => cardDisplay(c)).join(' ');
      row.appendChild(cardsEl);

      if (slot.playerIndex === trick.winnerIndex) {
        const winBadge = document.createElement('span');
        winBadge.className = 'review-win-badge';
        winBadge.textContent = '赢';
        row.appendChild(winBadge);
      }

      overlay.appendChild(row);
    }

    // Points info
    if (trick.points > 0) {
      const pts = document.createElement('div');
      pts.className = 'review-points';
      pts.textContent = `得分: ${trick.points}`;
      overlay.appendChild(pts);
    }

    // Close button
    const closeBtn = document.createElement('button');
    closeBtn.className = 'review-close-btn';
    closeBtn.textContent = '关闭';
    closeBtn.addEventListener('click', () => {
      this.reviewingLastTrick = false;
      this.removeReviewOverlay();
    });
    overlay.appendChild(closeBtn);
  }

  private removeReviewOverlay(): void {
    document.getElementById('trick-review-overlay')?.remove();
  }
}
