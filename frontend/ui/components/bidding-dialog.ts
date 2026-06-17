import type { StateSnapshot, BidEvent } from "../../core/types.ts";
import type { InteractionMode, BidOption } from "../../engine/types.ts";
import { suitDisplayName } from "../../core/card.ts";
import { SEAT_MAP } from "../../config.ts";

/**
 * Render the bidding info panel for DEAL_BID phase.
 *
 * - Shows available bid options as clickable pills
 * - Highlights the currently selected pending intent
 * - Shows bid events history
 * - No action buttons — auto-bid handles everything
 *
 * For STIRRING phase, renders the traditional stir/pass dialog.
 */
export function renderBiddingDialog(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  _onBid?: (cardIds: string[]) => void,
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
  _selectedCardIds?: Set<string>,
  _bidButtonState?: unknown,
  stirButtonState?: { disabled: boolean; title?: string },
  bidOptions?: BidOption[],
  pendingBidIntent?: BidOption | null,
  onBidOptionSelect?: (option: BidOption) => void,
): HTMLElement {
  if (snapshot.phase === "DEAL_BID") {
    return renderBidOptionsPanel(snapshot, bidOptions ?? [], pendingBidIntent ?? null, onBidOptionSelect);
  }

  if (snapshot.phase === "STIRRING") {
    return renderStirDialog(snapshot, interactionMode, onStir, onPass, stirButtonState);
  }

  // Shouldn't be called for other phases
  return document.createElement("div");
}

/** Render the bid options panel for DEAL_BID phase. */
function renderBidOptionsPanel(
  snapshot: StateSnapshot,
  bidOptions: BidOption[],
  pendingBidIntent: BidOption | null,
  onBidOptionSelect?: (option: BidOption) => void,
): HTMLElement {
  const container = document.createElement("div");
  container.classList.add("bidding-dialog");

  // Title
  const title = document.createElement("div");
  title.classList.add("bidding-dialog-title");
  title.textContent = "叫牌";
  container.appendChild(title);

  // Bid options
  if (bidOptions.length > 0) {
    const optionsWrap = document.createElement("div");
    optionsWrap.classList.add("bid-options");

    for (const option of bidOptions) {
      const pill = document.createElement("span");
      pill.classList.add("bid-option");

      // Highlight if this is the pending intent
      if (pendingBidIntent && pendingBidIntent.cardIds.join(",") === option.cardIds.join(",")) {
        pill.classList.add("selected");
      }

      // Show what trump suit this would create
      const trumpLabel = option.trumpSuit
        ? ` (${suitDisplayName(option.trumpSuit)}主)`
        : "";
      pill.textContent = option.label + trumpLabel;

      if (onBidOptionSelect) {
        pill.addEventListener("click", () => onBidOptionSelect(option));
      }

      optionsWrap.appendChild(pill);
    }

    container.appendChild(optionsWrap);
  } else {
    const noOptions = document.createElement("div");
    noOptions.classList.add("bidding-dialog-hint");
    noOptions.textContent = "当前无可用叫牌选项";
    container.appendChild(noOptions);
  }

  // Pending intent indicator
  if (pendingBidIntent) {
    const intentHint = document.createElement("div");
    intentHint.classList.add("bid-intent-hint");
    const trumpLabel = pendingBidIntent.trumpSuit
      ? ` (${suitDisplayName(pendingBidIntent.trumpSuit)}主)`
      : "";
    intentHint.textContent = `待叫: ${pendingBidIntent.label}${trumpLabel}`;
    container.appendChild(intentHint);
  }

  // Bid events
  if (snapshot.bid_events.length > 0) {
    const eventsContainer = document.createElement("div");
    eventsContainer.classList.add("bid-events");
    for (const event of snapshot.bid_events) {
      const eventEl = document.createElement("div");
      eventEl.classList.add("bid-event");
      eventEl.textContent = formatBidEvent(event);
      eventsContainer.appendChild(eventEl);
    }
    container.appendChild(eventsContainer);
  }

  return container;
}

/** Render the stirring dialog for STIRRING phase. */
function renderStirDialog(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
  stirButtonState?: { disabled: boolean; title?: string },
): HTMLElement {
  const container = document.createElement("div");

  if (interactionMode === "stir") {
    container.classList.add("bidding-dialog");

    const title = document.createElement("div");
    title.classList.add("bidding-dialog-title");
    title.textContent = "反主";
    container.appendChild(title);

    const hint = document.createElement("div");
    hint.classList.add("bidding-dialog-hint");
    hint.textContent = "选择手牌中的对子来反主";
    container.appendChild(hint);

    const actions = document.createElement("div");
    actions.classList.add("bid-actions");

    const stirButton = document.createElement("button");
    stirButton.classList.add("btn-warning");
    stirButton.textContent = "反主";
    if (stirButtonState) {
      stirButton.disabled = stirButtonState.disabled;
      if (stirButtonState.title) stirButton.title = stirButtonState.title;
    }
    stirButton.addEventListener("click", () => {
      if (onStir && !stirButton.disabled) {
        // Get selected cards from the render context
        const selectedIds = document.querySelectorAll(".hand-view .card.selected");
        const cardIds: string[] = [];
        selectedIds.forEach((el) => {
          const cardId = el.getAttribute("data-card-id");
          if (cardId) cardIds.push(cardId);
        });
        if (cardIds.length > 0) onStir(cardIds);
      }
    });
    actions.appendChild(stirButton);

    const passButton = document.createElement("button");
    passButton.classList.add("btn-secondary");
    passButton.textContent = "不反";
    passButton.addEventListener("click", () => {
      if (onPass) onPass();
    });
    actions.appendChild(passButton);

    container.appendChild(actions);
  } else {
    // Not our turn — passive info bar
    container.classList.add("bid-info-bar");

    const hint = document.createElement("div");
    hint.classList.add("waiting-hint");
    const seat = snapshot.stirring_state
      ? SEAT_MAP[snapshot.stirring_state.current_player]
      : null;
    hint.textContent = seat ? `等待${seat.label}反主...` : "等待反主...";
    container.appendChild(hint);
  }

  return container;
}

/** Format a bid event for display. */
function formatBidEvent(event: BidEvent): string {
  const seat = SEAT_MAP[event.player];
  const name = seat ? seat.label : `玩家${event.player}`;
  const cardsStr = event.cards.map(compactCard).join(" ");
  if (event.kind === "trump_rank" && event.suit) {
    return `${name}: ${cardsStr} (${suitDisplayName(event.suit)}主)`;
  }
  if (event.kind === "joker" && event.joker_type) {
    return `${name}: ${cardsStr} (${event.joker_type === "big" ? "大" : "小"}王)`;
  }
  return `${name}: 不叫`;
}

/** Compact display for a card in bid events. */
function compactCard(c: { suit: string; rank: string }): string {
  if (c.suit === "joker") {
    return c.rank === "BJ" ? "大王" : "小王";
  }
  const symbols: Record<string, string> = { hearts: "♥", spades: "♠", diamonds: "♦", clubs: "♣" };
  return (symbols[c.suit] ?? "") + c.rank;
}
