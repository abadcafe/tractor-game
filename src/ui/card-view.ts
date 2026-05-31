/**
 * Card DOM element creation and styling.
 *
 * Cards display rank+suit in the TOP-LEFT corner (like real playing cards)
 * so that when cards overlap in a fan, the visible left portion always
 * shows enough to identify the card.
 */

import { Card, Suit } from '../core/card';

/**
 * Create a DOM element for a card.
 */
export function createCardElement(card: Card, faceUp: boolean = true): HTMLElement {
  if (!faceUp) {
    return createCardBack();
  }

  const el = document.createElement('div');
  el.className = 'card';
  el.dataset.cardId = card.id;

  // Color class
  if (card.suit === Suit.HEARTS || card.suit === Suit.DIAMONDS) {
    el.classList.add('red');
  } else if (card.suit === Suit.SPADES || card.suit === Suit.CLUBS) {
    el.classList.add('black');
  }

  // Joker special styling
  if (card.isJoker) {
    el.classList.add('joker');
    if (card.isBigJoker) {
      el.classList.add('big-joker');
    } else {
      el.classList.add('small-joker');
    }
  }

  // Point card marker
  if (card.points > 0) {
    el.classList.add('points-card');
  }

  // Top-left corner: rank above, suit below — always visible even when overlapped
  const corner = document.createElement('span');
  corner.className = 'card-corner';

  const rankEl = document.createElement('span');
  rankEl.className = 'card-rank';
  rankEl.textContent = getRankDisplay(card);

  const suitEl = document.createElement('span');
  suitEl.className = 'card-suit';
  suitEl.textContent = getSuitSymbol(card.suit);

  corner.appendChild(rankEl);
  corner.appendChild(suitEl);
  el.appendChild(corner);

  // Center pip (large suit symbol for visual flair — visible when card is fully shown)
  const centerEl = document.createElement('span');
  centerEl.className = 'card-center';
  centerEl.textContent = getSuitSymbol(card.suit);
  el.appendChild(centerEl);

  return el;
}

/**
 * Create a card back element.
 */
export function createCardBack(): HTMLElement {
  const el = document.createElement('div');
  el.className = 'card-back';
  return el;
}

/**
 * Create multiple card back elements.
 */
export function createCardBacks(count: number): HTMLElement[] {
  return Array.from({ length: count }, () => createCardBack());
}

function getSuitSymbol(suit: Suit): string {
  switch (suit) {
    case Suit.HEARTS: return '♥';
    case Suit.SPADES: return '♠';
    case Suit.DIAMONDS: return '♦';
    case Suit.CLUBS: return '♣';
    case Suit.JOKER: return '🃏';
  }
}

function getRankDisplay(card: Card): string {
  if (card.isJoker) {
    return card.isBigJoker ? '大' : '小';
  }
  return card.rank;
}

/**
 * Get the suit symbol for display.
 */
export function suitSymbol(suit: Suit): string {
  return getSuitSymbol(suit);
}
