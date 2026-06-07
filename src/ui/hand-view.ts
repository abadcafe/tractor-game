/**
 * Human player hand interaction: card selection and play.
 */

import { Card } from '../core/card';
import { createCardElement } from './card-view';

export type CardSelectCallback = (cards: Card[]) => void;
export type PlayCallback = (cards: Card[]) => void;

export class HandView {
  private container: HTMLElement;
  private hand: Card[] = [];
  private selectedIds: Set<string> = new Set();
  private onSelectChange?: CardSelectCallback;
  private onPlay?: PlayCallback;

  constructor(containerId: string) {
    this.container = document.getElementById(containerId)!;
  }

  /** Set the hand cards and re-render. */
  setHand(cards: Card[]): void {
    this.hand = cards;
    this.selectedIds.clear();
    this.render();
    this.updatePlayButton();
  }

  /** Subscribe to selection changes. */
  onSelectionChange(cb: CardSelectCallback): void {
    this.onSelectChange = cb;
  }

  /** Subscribe to play action. */
  onPlayAction(cb: PlayCallback): void {
    this.onPlay = cb;
  }

  /** Get currently selected cards. */
  getSelected(): Card[] {
    return this.hand.filter(c => this.selectedIds.has(c.id));
  }

  /** Clear selection. */
  clearSelection(): void {
    this.selectedIds.clear();
    this.render();
    this.updatePlayButton();
  }

  private render(): void {
    this.container.innerHTML = '';
    this.container.className = 'player-cards hand';

    for (const card of this.hand) {
      const el = createCardElement(card, true);
      if (this.selectedIds.has(card.id)) {
        el.classList.add('selected');
      }
      el.addEventListener('click', () => this.toggleCard(card));
      this.container.appendChild(el);
    }
  }

  private toggleCard(card: Card): void {
    if (this.selectedIds.has(card.id)) {
      this.selectedIds.delete(card.id);
    } else {
      this.selectedIds.add(card.id);
    }
    this.render();
    this.updatePlayButton();

    if (this.onSelectChange) {
      this.onSelectChange(this.getSelected());
    }
  }

  private updatePlayButton(): void {
    const btn = document.getElementById('btn-play') as HTMLButtonElement;
    if (btn) {
      btn.disabled = this.selectedIds.size === 0;
    }
  }

  /** Initialize play button handler. */
  initButtons(): void {
    const playBtn = document.getElementById('btn-play') as HTMLButtonElement;

    playBtn?.addEventListener('click', () => {
      const selected = this.getSelected();
      if (selected.length > 0 && this.onPlay) {
        this.onPlay(selected);
      }
    });
  }
}
