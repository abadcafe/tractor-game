/**
 * Bidding and stirring UI panels.
 */

import { Rank, Suit } from '../core/card';

export type BidCallback = (cards: any[], pass: boolean) => void;
export type StirCallback = (trumpSuit: Suit | null) => void; // null = pass
export type TrumpCallback = (trumpSuit: Suit) => void;

export class BiddingView {
  /** Show bidding panel for the human player (亮牌规则). */
  static showBidding(
    cards: any[],  // 级牌列表
    canPass: boolean,
    onBid: (cards: any[], pass: boolean) => void,
  ): void {
    this.removeExistingPanels();

    const panel = document.createElement('div');
    panel.id = 'bidding-panel';

    if (cards.length === 0) {
      panel.innerHTML = `<h3>亮牌</h3><p style="color:#aaa;font-size:12px;margin:6px 0;">没有级牌，跳过叫牌</p>`;
    } else {
      panel.innerHTML = `<h3>亮牌</h3><p style="color:#aaa;font-size:12px;margin:6px 0;">选择要亮的级牌（当前级别）</p>`;
    }

    const cardContainer = document.createElement('div');
    cardContainer.className = 'bid-cards';

    for (const card of cards) {
      const cardEl = document.createElement('button');
      cardEl.className = 'bid-card';
      cardEl.textContent = `${card.suit}${card.rank}`;
      cardEl.addEventListener('click', () => {
        panel.remove();
        onBid([card], false);
      });
      cardContainer.appendChild(cardEl);
    }

    if (canPass && cards.length > 0) {
      const passBtn = document.createElement('button');
      passBtn.className = 'pass-btn';
      passBtn.textContent = '不叫';
      passBtn.addEventListener('click', () => {
        panel.remove();
        onBid([], true);
      });
      cardContainer.appendChild(passBtn);
    }

    panel.appendChild(cardContainer);
    document.getElementById('game-table')?.appendChild(panel);
  }

  /** Show trump suit selection after winning bid. */
  static showTrumpSelection(onSelect: TrumpCallback): void {
    this.removeExistingPanels();

    const panel = document.createElement('div');
    panel.id = 'bidding-panel';
    panel.innerHTML = `<h3>选择主牌花色</h3>`;

    const suitOptions = document.createElement('div');
    suitOptions.className = 'suit-options';

    const suits: { suit: Suit; symbol: string; className: string }[] = [
      { suit: Suit.HEARTS, symbol: '♥', className: 'red' },
      { suit: Suit.SPADES, symbol: '♠', className: 'black' },
      { suit: Suit.DIAMONDS, symbol: '♦', className: 'red' },
      { suit: Suit.CLUBS, symbol: '♣', className: 'black' },
    ];

    for (const { suit, symbol, className } of suits) {
      const btn = document.createElement('button');
      btn.className = `suit-btn ${className}`;
      btn.textContent = symbol;
      btn.addEventListener('click', () => {
        panel.remove();
        onSelect(suit);
      });
      suitOptions.appendChild(btn);
    }

    panel.appendChild(suitOptions);
    document.getElementById('game-table')?.appendChild(panel);
  }

  /** Show stirring panel. */
  static showStirring(
    canPass: boolean,
    onStir: StirCallback,
  ): void {
    this.removeExistingPanels();

    const panel = document.createElement('div');
    panel.id = 'stirring-panel';
    panel.innerHTML = `<h3>是否炒地皮？</h3>`;

    const suitOptions = document.createElement('div');
    suitOptions.className = 'suit-options';

    const suits: { suit: Suit; symbol: string; className: string }[] = [
      { suit: Suit.HEARTS, symbol: '♥', className: 'red' },
      { suit: Suit.SPADES, symbol: '♠', className: 'black' },
      { suit: Suit.DIAMONDS, symbol: '♦', className: 'red' },
      { suit: Suit.CLUBS, symbol: '♣', className: 'black' },
    ];

    for (const { suit, symbol, className } of suits) {
      const btn = document.createElement('button');
      btn.className = `suit-btn ${className}`;
      btn.textContent = symbol;
      btn.title = `炒 ${symbol}`;
      btn.addEventListener('click', () => {
        panel.remove();
        onStir(suit);
      });
      suitOptions.appendChild(btn);
    }

    panel.appendChild(suitOptions);

    if (canPass) {
      const passRow = document.createElement('div');
      passRow.style.marginTop = '8px';
      const passBtn = document.createElement('button');
      passBtn.className = 'pass-btn';
      passBtn.textContent = '不炒';
      passBtn.addEventListener('click', () => {
        panel.remove();
        onStir(null);
      });
      passRow.appendChild(passBtn);
      panel.appendChild(passRow);
    }

    document.getElementById('game-table')?.appendChild(panel);
  }

  /** Show scoring overlay. */
  static showScoring(
    message: string,
    details: string,
    onNext: () => void,
  ): void {
    this.removeExistingPanels();

    const overlay = document.createElement('div');
    overlay.id = 'scoring-overlay';
    overlay.innerHTML = `
      <h3>${message}</h3>
      <div class="score-detail">${details}</div>
      <button id="btn-next-round">下一局</button>
    `;

    document.getElementById('game-table')?.appendChild(overlay);

    document.getElementById('btn-next-round')?.addEventListener('click', () => {
      overlay.remove();
      onNext();
    });
  }

  private static removeExistingPanels(): void {
    document.getElementById('bidding-panel')?.remove();
    document.getElementById('stirring-panel')?.remove();
    document.getElementById('scoring-overlay')?.remove();
  }
}
