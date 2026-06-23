import type { StateSnapshot } from "../../core/types.ts";
import type {
  BidOption,
  GameAction,
  InteractionMode,
  StirButtonState,
} from "../../engine/types.ts";
import { cardDisplay, sortHand, suitSymbol } from "../../core/card.ts";
import { el } from "../dom.ts";
import {
  renderActionPanel,
  renderBidActionPanel,
  renderHandTools,
  renderPreviousTrickButton,
  renderStirActionPanel,
} from "./hand-view/buttons.ts";
import { handCardClass, isTrumpCard } from "./hand-view/cards.ts";
import { renderScorePile } from "./hand-view/score-pile.ts";
import {
  canSelectCard,
  createDragSelection,
} from "./hand-view/selection.ts";

/**
 * Render the human player's hand with card display, click selection,
 * legal action highlighting, trump card highlighting, and action buttons.
 */
export function renderHandView(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  selectedCardIds?: Set<string>,
  legalCardIds?: Set<string>,
  onCardClick?: (cardId: string) => void,
  onAction?: (action: GameAction) => void,
  onClearSelection?: () => void,
  onUseHint?: () => void,
  _onToggleCompact?: () => void,
  compactHand?: boolean,
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
  stirButtonState?: StirButtonState,
  onShowPreviousTrick?: () => void,
  bidOptions?: BidOption[],
  pendingBidIntent?: BidOption | null,
  onBidOptionSelect?: (option: BidOption) => void,
  onCardRangeSelect?: (cardIds: string[]) => void,
): HTMLElement {
  const actionHints = snapshot.action_hints ?? [];
  const selectedCount = selectedCardIds?.size ?? 0;
  const sortedHand = sortHand(
    snapshot.player_hand,
    snapshot.trump_suit,
    snapshot.trump_rank,
  );
  const selectedStirCardIds = sortedHand
    .filter((card) => selectedCardIds?.has(card.id) ?? false)
    .map((card) => card.id);
  const showTools = interactionMode === "play" ||
    interactionMode === "discard" || interactionMode === "stir";
  const needsButton = interactionMode === "play" ||
    interactionMode === "discard";
  const needsStirButtons = interactionMode === "stir" &&
    (onStir !== undefined || onPass !== undefined);
  const needsBidButtons = snapshot.phase === "DEAL_BID" &&
    bidOptions !== undefined &&
    bidOptions.length > 0 &&
    onBidOptionSelect !== undefined;
  const canShowPreviousTrick = snapshot.last_completed_trick !== null &&
    onShowPreviousTrick !== undefined;
  const hasControls = (needsButton && onAction !== undefined) ||
    needsStirButtons || needsBidButtons || showTools ||
    canShowPreviousTrick;

  const container = el("div", {
    class: `hand-area${compactHand ? " compact" : ""}${
      hasControls ? " has-actions" : ""
    }${needsBidButtons ? " has-bid-actions" : ""}`,
  });

  if (hasControls) {
    const controls = el("div", { class: "hand-actions" });
    if (canShowPreviousTrick) {
      controls.appendChild(
        renderPreviousTrickButton(onShowPreviousTrick),
      );
    }
    if (needsButton && onAction) {
      controls.appendChild(
        renderActionPanel(
          snapshot,
          interactionMode,
          selectedCount,
          onAction,
        ),
      );
    }
    if (needsStirButtons) {
      controls.appendChild(
        renderStirActionPanel(
          selectedStirCardIds,
          onStir,
          onPass,
          stirButtonState,
        ),
      );
    }
    if (needsBidButtons) {
      controls.appendChild(
        renderBidActionPanel(
          bidOptions,
          pendingBidIntent ?? null,
          onBidOptionSelect,
        ),
      );
    }
    if (showTools) {
      controls.appendChild(
        renderHandTools(
          selectedCount,
          actionHints.length > 0,
          onClearSelection,
          onUseHint,
        ),
      );
    }
    container.appendChild(controls);
  }

  const panel = el("div", {
    class: `hand-panel${compactHand ? " compact" : ""}`,
  });

  const handView = el("div", {
    class: `hand-view${compactHand ? " compact" : ""}`,
  });
  const hasActionHints = actionHints.length > 0;
  const constrainedCardIds = hasActionHints ? legalCardIds : undefined;
  const dragSelection = createDragSelection(
    sortedHand,
    interactionMode,
    constrainedCardIds,
    onCardRangeSelect,
    handView,
  );

  // Render each card
  for (const card of sortedHand) {
    const cardSpan = el("span", {
      class: handCardClass(
        card,
        snapshot.trump_suit,
        snapshot.trump_rank,
      ),
      "data-card-id": card.id,
      "data-rank": card.rank,
      "data-suit-symbol": suitSymbol(card.suit),
    });
    cardSpan.textContent = cardDisplay(card);

    // Highlight legal cards
    if (constrainedCardIds?.has(card.id)) {
      cardSpan.classList.add("legal");
    }

    // Highlight trump cards
    if (isTrumpCard(card, snapshot.trump_suit, snapshot.trump_rank)) {
      cardSpan.classList.add("trump-card");
    }

    // Selected state
    if (selectedCardIds?.has(card.id)) {
      cardSpan.classList.add("selected");
    }

    const canSelect = canSelectCard(
      card.id,
      interactionMode,
      constrainedCardIds,
    );
    if (!canSelect) {
      cardSpan.classList.add("not-selectable");
    }

    if (onCardClick && canSelect) {
      cardSpan.addEventListener("click", () => {
        if (dragSelection.consumeSuppressedClick()) return;
        onCardClick(card.id);
      });
    }

    if (canSelect) {
      dragSelection.bindCard(cardSpan, card.id);
    }

    handView.appendChild(cardSpan);
  }

  panel.appendChild(handView);

  panel.appendChild(renderScorePile(snapshot));
  container.appendChild(panel);

  return container;
}
