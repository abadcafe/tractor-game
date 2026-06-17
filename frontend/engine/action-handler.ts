import type { StateSnapshot } from "../core/types.ts";
import type { GameAction } from "./types.ts";
import type { ClientAction } from "../core/protocol.ts";
import { validatePlay, validateDiscard, validateBidCards, validateStirCards } from "./input-validator.ts";

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
): ActionResult {
  const selectedCards = snap.player_hand.filter((c) => selectedCardIds.has(c.id));
  const matchedCards = validatePlay(selectedCards, snap.legal_actions);
  if (matchedCards) {
    return {
      success: true,
      action: { type: "play", cards: matchedCards.map((c) => c.id) },
    };
  }
  return { success: false, error: "无效的出牌组合" };
}

/** Handle a discard action: validate and construct the client action. */
export function handleDiscardAction(
  snap: StateSnapshot,
  selectedCardIds: Set<string>,
): ActionResult {
  const selectedCards = snap.player_hand.filter((c) => selectedCardIds.has(c.id));
  const count = snap.stirring_state?.exchange_count ?? 0;
  if (validateDiscard(selectedCards, count)) {
    return {
      success: true,
      action: { type: "discard", cards: selectedCards.map((c) => c.id) },
    };
  }
  return { success: false, error: `请选择 ${count} 张牌弃掉` };
}

/** Handle a next_round action. */
export function handleNextRoundAction(): ActionResult {
  return { success: true, action: { type: "next_round" } };
}

/** Handle a bid action: validate and construct the client action. */
export function handleBidAction(
  snap: StateSnapshot,
  cardIds: string[],
): ActionResult {
  const selectedCards = snap.player_hand.filter((c) => cardIds.includes(c.id));
  if (validateBidCards(selectedCards, snap.trump_rank)) {
    return { success: true, action: { type: "bid", cards: cardIds } };
  }
  return { success: false, error: "叫牌牌张无效" };
}

/** Handle a stir action: validate and construct the client action. */
export function handleStirAction(
  snap: StateSnapshot,
  cardIds: string[],
): ActionResult {
  const selectedCards = snap.player_hand.filter((c) => cardIds.includes(c.id));
  if (validateStirCards(selectedCards, snap.trump_rank)) {
    return { success: true, action: { type: "stir", cards: cardIds } };
  }
  return { success: false, error: "反主必须出对子" };
}

/** Handle a pass action. */
export function handlePassAction(): ActionResult {
  return { success: true, action: { type: "stir", pass: true } };
}
