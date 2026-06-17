import type { StateSnapshot, Card } from "../core/types.ts";
import type { InteractionMode, StirButtonState, LevelChangeInfo } from "./types.ts";
import { computeBidPriority } from "./bid-logic.ts";
import { computeLevelChange } from "./scoring-logic.ts";
import { validatePlay } from "./input-validator.ts";
import { isJoker, isTrumpRank } from "../core/card.ts";

/** Compute stir button state based on selected cards. */
export function computeStirButtonState(
  snap: StateSnapshot,
  selectedCardIds: Set<string>,
): StirButtonState {
  const selectedIds = [...selectedCardIds];
  const selectedCards = selectedIds
    .map((id) => snap.player_hand.find((c) => c.id === id))
    .filter((c): c is Card => c !== undefined);

  if (selectedCards.length === 0) {
    return { disabled: true, title: "请先选择要反主的对子" };
  }

  if (selectedCards.length !== 2) {
    return { disabled: true, title: "反主必须选择2张牌" };
  }

  const priority = computeBidPriority(selectedCards, snap.trump_rank);
  if (priority < 200) {
    return { disabled: true, title: "反主必须用对子" };
  }

  return { disabled: false };
}

/** Compute level change info for scoring overlay. */
export function computeLevelChangeInfo(totalPoints: number): LevelChangeInfo {
  return computeLevelChange(totalPoints);
}

/** Compute the set of legal card IDs for hand highlighting. */
export function computeLegalCardIds(
  snap: StateSnapshot,
  interactionMode: InteractionMode,
): Set<string> {
  const legalCardIds = new Set<string>();
  if (interactionMode === "stir") {
    // In stir mode, highlight all trump-rank cards and jokers as selectable
    for (const card of snap.player_hand) {
      if (isJoker(card) || isTrumpRank(card, snap.trump_rank)) {
        legalCardIds.add(card.id);
      }
    }
  } else if (interactionMode === "play" || interactionMode === "discard") {
    for (const cards of snap.legal_actions) {
      for (const card of cards) {
        legalCardIds.add(card.id);
      }
    }
  }
  // bid mode: no card selection needed (bid options are clicked instead)
  return legalCardIds;
}

/** Validate if current selection is still legal after state update. */
export function isSelectionStillLegal(
  snap: StateSnapshot,
  selectedCardIds: Set<string>,
): boolean {
  if (selectedCardIds.size === 0) return true;

  const handIds = new Set(snap.player_hand.map((c) => c.id));
  const allInHand = [...selectedCardIds].every((id) => handIds.has(id));
  if (!allInHand) return false;

  if (snap.phase === "PLAYING" && snap.legal_actions.length > 0) {
    const selectedCards = snap.player_hand.filter((c) => selectedCardIds.has(c.id));
    const matched = validatePlay(selectedCards, snap.legal_actions);
    if (!matched) return false;
  }

  return true;
}
