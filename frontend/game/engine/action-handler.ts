import type { Card, StateSnapshot } from "../core/types.ts";
import type { ClientAction } from "../core/protocol.ts";
import {
  validateBidCards,
  validateDiscard,
  validatePlay,
  validateStirCards,
} from "./input-validator.ts";

/** Result of validating an action. */
interface ActionResult {
  success: boolean;
  action?: ClientAction;
  error?: string;
}

/** Handle a play action: validate and construct the client action. */
export function handlePlayAction(
  snap: StateSnapshot,
  selectedCardIds: Set<string>,
  seq: number,
): ActionResult {
  const selectedCards = selectedCardsInSelectionOrder(
    snap,
    selectedCardIds,
  );
  if (selectedCards.length === 0) {
    return { success: false, error: "请选择要出的牌" };
  }
  return {
    success: true,
    action: {
      type: "play",
      seq,
      cards: selectedCards.map((c) => c.id),
    },
  };
}

/** Handle a discard action: validate and construct the client action. */
export function handleDiscardAction(
  snap: StateSnapshot,
  selectedCardIds: Set<string>,
  seq: number,
): ActionResult {
  const selectedCards = selectedCardsInSelectionOrder(
    snap,
    selectedCardIds,
  );
  const count = snap.stirring_state?.exchange_count ?? 0;
  if (validateDiscard(selectedCards, count)) {
    return {
      success: true,
      action: {
        type: "discard",
        seq,
        cards: selectedCards.map((c) => c.id),
      },
    };
  }
  return { success: false, error: `请选择 ${count} 张牌弃掉` };
}

/** Handle a next_round action. */
export function handleNextRoundAction(seq: number): ActionResult {
  return { success: true, action: { type: "next_round", seq } };
}

/** Handle a bid action: validate and construct the client action. */
export function handleBidAction(
  snap: StateSnapshot,
  cardIds: string[],
  seq: number,
): ActionResult {
  const selectedCards = snap.player_hand.filter((c) =>
    cardIds.includes(c.id)
  );
  const hints = snap.action_hints ?? [];
  const matchedCards = hints.length > 0
    ? validatePlay(selectedCards, hints)
    : null;
  if (
    matchedCards ||
    (hints.length === 0 &&
      validateBidCards(selectedCards, snap.trump_rank))
  ) {
    return {
      success: true,
      action: { type: "bid", seq, cards: cardIds },
    };
  }
  return { success: false, error: "抢主牌张无效" };
}

/** Handle a skip-bid action (不抢). */
export function handleSkipBidAction(seq: number): ActionResult {
  return { success: true, action: { type: "bid", seq, pass: true } };
}

/** Handle a stir action: validate and construct the client action. */
export function handleStirAction(
  snap: StateSnapshot,
  cardIds: string[],
  seq: number,
): ActionResult {
  const selectedCards = snap.player_hand.filter((c) =>
    cardIds.includes(c.id)
  );
  const hints = snap.action_hints ?? [];
  const matchedCards = validatePlay(selectedCards, hints);
  if (matchedCards) {
    return {
      success: true,
      action: { type: "stir", seq, cards: cardIds },
    };
  }
  if (!validateStirCards(selectedCards, snap.trump_rank)) {
    return { success: false, error: "反主必须出对子" };
  }
  return { success: false, error: "优先级不足，不能反主" };
}

/** Handle a pass stir action (不反). */
export function handlePassStirAction(seq: number): ActionResult {
  return { success: true, action: { type: "stir", seq, pass: true } };
}

function selectedCardsInSelectionOrder(
  snap: StateSnapshot,
  selectedCardIds: Set<string>,
): Card[] {
  const cardsById = new Map(
    snap.player_hand.map((card) => [card.id, card]),
  );
  const selectedCards: Card[] = [];
  for (const cardId of selectedCardIds) {
    const card = cardsById.get(cardId);
    if (card !== undefined) {
      selectedCards.push(card);
    }
  }
  return selectedCards;
}
