import type { Card, StateSnapshot } from "../core/types.ts";
import type {
  InteractionMode,
  LevelChangeInfo,
  StirButtonState,
} from "./types.ts";
import { computeBidPriority } from "./bid-logic.ts";
import { computeLevelChange } from "./scoring-logic.ts";

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

  const hints = snap.action_hints ?? [];
  if (hints.length === 0) {
    return { disabled: true, title: "没有可反的对子" };
  }
  if (!matchesHint(selectedCards, hints)) {
    return { disabled: true, title: "优先级不足，不能反主" };
  }

  return { disabled: false };
}

function matchesHint(selectedCards: Card[], hints: Card[][]): boolean {
  const selectedKey = cardIdsKey(selectedCards);
  return hints.some((hint) => cardIdsKey(hint) === selectedKey);
}

function cardIdsKey(cards: Card[]): string {
  return cards.map((card) => card.id).sort().join("\n");
}

/** Compute level change info for scoring overlay. */
export function computeLevelChangeInfo(
  totalPoints: number,
): LevelChangeInfo {
  return computeLevelChange(totalPoints);
}

/** Compute the set of legal card IDs for hand highlighting. */
export function computeLegalCardIds(
  snap: StateSnapshot,
  interactionMode: InteractionMode,
): Set<string> {
  const legalCardIds = new Set<string>();
  if (interactionMode === "stir") {
    const hints = snap.action_hints ?? [];
    if (hints.length > 0) {
      for (const cards of hints) {
        for (const card of cards) {
          legalCardIds.add(card.id);
        }
      }
    }
  } else if (
    interactionMode === "play" || interactionMode === "discard"
  ) {
    for (const cards of snap.action_hints ?? []) {
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

  return true;
}
