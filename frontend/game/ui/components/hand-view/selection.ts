import type { Card } from "../../../core/types.ts";
import type { InteractionMode } from "../../../engine/types.ts";

export interface DragSelectionController {
  bindCard: (cardEl: HTMLElement, cardId: string) => void;
  consumeSuppressedClick: () => boolean;
}

interface ActiveDragSelection {
  startId: string;
  moved: boolean;
  currentIds: string[];
  originalSelectedIds: string[];
}

export function createDragSelection(
  sortedHand: Card[],
  interactionMode: InteractionMode,
  constrainedCardIds: Set<string> | undefined,
  onCardRangeSelect: ((cardIds: string[]) => void) | undefined,
  handView: HTMLElement,
): DragSelectionController {
  let active: ActiveDragSelection | null = null;
  let suppressNextClick = false;
  const cardElements = new Map<string, HTMLElement>();

  function selectableRange(startId: string, endId: string): string[] {
    const startIndex = sortedHand.findIndex((card) =>
      card.id === startId
    );
    const endIndex = sortedHand.findIndex((card) => card.id === endId);
    if (startIndex < 0 || endIndex < 0) return [];
    const low = Math.min(startIndex, endIndex);
    const high = Math.max(startIndex, endIndex);
    return sortedHand
      .slice(low, high + 1)
      .filter((card) =>
        canSelectCard(card.id, interactionMode, constrainedCardIds)
      )
      .map((card) => card.id);
  }

  function preview(endId: string): void {
    if (active === null || onCardRangeSelect === undefined) return;
    if (endId === active.startId && !active.moved) return;
    const cardIds = selectableRange(active.startId, endId);
    if (cardIds.length === 0) return;
    active.moved = true;
    active.currentIds = cardIds;
    const selectedIds = new Set(cardIds);
    for (const [cardId, cardEl] of cardElements) {
      cardEl.classList.toggle("selected", selectedIds.has(cardId));
    }
  }

  function end(commit: boolean): void {
    const completed = active;
    if (active?.moved) {
      suppressNextClick = true;
      setTimeout(() => {
        suppressNextClick = false;
      }, 0);
    }
    active = null;
    handView.classList.remove("drag-selecting");
    if (!commit && completed !== null) {
      const selectedIds = new Set(completed.originalSelectedIds);
      for (const [cardId, cardEl] of cardElements) {
        cardEl.classList.toggle("selected", selectedIds.has(cardId));
      }
    }
    if (
      commit &&
      completed !== null &&
      completed.moved &&
      completed.currentIds.length > 0 &&
      onCardRangeSelect !== undefined
    ) {
      onCardRangeSelect(completed.currentIds);
    }
  }

  handView.addEventListener("pointerup", () => end(true));
  handView.addEventListener("pointercancel", () => end(false));

  return {
    bindCard(cardEl: HTMLElement, cardId: string): void {
      if (onCardRangeSelect === undefined) return;
      cardElements.set(cardId, cardEl);
      cardEl.addEventListener("pointerdown", () => {
        const originalSelectedIds = Array.from(cardElements.entries())
          .filter((entry) => entry[1].classList.contains("selected"))
          .map((entry) => entry[0]);
        active = {
          startId: cardId,
          moved: false,
          currentIds: [],
          originalSelectedIds,
        };
        handView.classList.add("drag-selecting");
      });
      cardEl.addEventListener("pointerenter", () => preview(cardId));
      cardEl.addEventListener("pointerup", () => preview(cardId));
    },
    consumeSuppressedClick(): boolean {
      if (!suppressNextClick) return false;
      suppressNextClick = false;
      return true;
    },
  };
}

export function canSelectCard(
  cardId: string,
  interactionMode: InteractionMode,
  legalCardIds?: Set<string>,
): boolean {
  if (interactionMode === null || interactionMode === "next_round") {
    return false;
  }
  if (interactionMode === "discard") {
    return true;
  }
  if (interactionMode === "play") {
    return true;
  }
  if (!legalCardIds || legalCardIds.size === 0) {
    return true;
  }
  return legalCardIds.has(cardId);
}
