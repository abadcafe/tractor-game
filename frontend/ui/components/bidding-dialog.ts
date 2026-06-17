import type { StateSnapshot, BidEvent } from "../../core/types.ts";
import type { InteractionMode, BidButtonState } from "../../engine/types.ts";
import { suitSymbol, suitDisplayName } from "../../core/card.ts";
import { SEAT_MAP } from "../../config.ts";

/**
 * Render the bidding/stirring dialog.
 *
 * - DEAL_BID + "bid" interaction: shows "叫牌" and "不叫" buttons
 * - STIRRING + "stir" interaction: shows "反主" and "不反" buttons
 * - When not our turn during DEAL_BID/STIRRING: shows a slim info bar with bid events
 */
export function renderBiddingDialog(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  onBid?: (cardIds: string[]) => void,
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
  selectedCardIds?: Set<string>,
  bidButtonState?: BidButtonState,
  stirButtonState?: BidButtonState,
): HTMLElement {
  // Get user-selected card IDs
  const selectedIds = selectedCardIds ? [...selectedCardIds] : [];

  const isOurTurn = interactionMode === "bid" || interactionMode === "stir";

  if (isOurTurn) {
    return renderActiveDialog(snapshot, interactionMode, onBid, onStir, onPass, selectedIds, bidButtonState, stirButtonState);
  }

  // Not our turn — show passive info bar
  return renderPassiveInfoBar(snapshot);
}

/** Render the active bidding/stirring dialog when it's our turn. */
function renderActiveDialog(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  onBid?: (cardIds: string[]) => void,
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
  selectedIds?: string[],
  bidButtonState?: BidButtonState,
  stirButtonState?: BidButtonState,
): HTMLElement {
  const container = document.createElement("div");
  container.classList.add("bidding-dialog");

  if (interactionMode === "bid") {
    // Title
    const title = document.createElement("div");
    title.classList.add("bidding-dialog-title");
    title.textContent = "叫牌";
    container.appendChild(title);

    // Hint
    const hint = document.createElement("div");
    hint.classList.add("bidding-dialog-hint");
    hint.textContent = selectedIds && selectedIds.length > 0
      ? `已选择 ${selectedIds.length} 张`
      : "选择手牌中的级牌或王来叫牌";
    container.appendChild(hint);

    // Action buttons
    const actions = document.createElement("div");
    actions.classList.add("bid-actions");

    // Bid button
    const bidButton = document.createElement("button");
    bidButton.classList.add("btn-primary");
    bidButton.textContent = "叫牌";
    if (bidButtonState) {
      bidButton.disabled = bidButtonState.disabled;
      if (bidButtonState.title) bidButton.title = bidButtonState.title;
    }
    bidButton.addEventListener("click", () => {
      if (onBid && !bidButton.disabled && selectedIds && selectedIds.length > 0) {
        onBid(selectedIds);
      }
    });
    actions.appendChild(bidButton);

    // Skip bid button (不叫)
    const skipButton = document.createElement("button");
    skipButton.classList.add("btn-secondary");
    skipButton.textContent = "不叫";
    skipButton.addEventListener("click", () => {
      if (onPass) {
        onPass();
      }
    });
    actions.appendChild(skipButton);

    container.appendChild(actions);

  } else if (interactionMode === "stir") {
    // Title
    const title = document.createElement("div");
    title.classList.add("bidding-dialog-title");
    title.textContent = "反主";
    container.appendChild(title);

    // Hint
    const hint = document.createElement("div");
    hint.classList.add("bidding-dialog-hint");
    hint.textContent = selectedIds && selectedIds.length > 0
      ? `已选择 ${selectedIds.length} 张`
      : "选择对子来反主，或选择不反";
    container.appendChild(hint);

    // Action buttons
    const actions = document.createElement("div");
    actions.classList.add("bid-actions");

    // Stir button
    const stirButton = document.createElement("button");
    stirButton.classList.add("btn-warning");
    stirButton.textContent = "反主";
    if (stirButtonState) {
      stirButton.disabled = stirButtonState.disabled;
      if (stirButtonState.title) stirButton.title = stirButtonState.title;
    }
    stirButton.addEventListener("click", () => {
      if (onStir && !stirButton.disabled && selectedIds && selectedIds.length > 0) {
        onStir(selectedIds);
      }
    });
    actions.appendChild(stirButton);

    // Pass button
    const passButton = document.createElement("button");
    passButton.classList.add("btn-secondary");
    passButton.textContent = "不反";
    passButton.addEventListener("click", () => {
      if (onPass) {
        onPass();
      }
    });
    actions.appendChild(passButton);

    container.appendChild(actions);
  }

  // Render bid events
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

/** Render a passive info bar when it's not our turn. */
function renderPassiveInfoBar(snapshot: StateSnapshot): HTMLElement {
  const bar = document.createElement("div");
  bar.classList.add("bid-info-bar");

  if (snapshot.bid_winner) {
    const winner = document.createElement("div");
    winner.classList.add("bid-current");
    winner.textContent = `当前叫牌 ${formatBidEvent(snapshot.bid_winner)}`;
    bar.appendChild(winner);
  }

  if (snapshot.bid_events.length > 0) {
    const eventsContainer = document.createElement("div");
    eventsContainer.classList.add("bid-events");
    for (const event of snapshot.bid_events) {
      const eventEl = document.createElement("div");
      eventEl.classList.add("bid-event");
      eventEl.textContent = formatBidEvent(event);
      eventsContainer.appendChild(eventEl);
    }
    bar.appendChild(eventsContainer);
  }

  let waitingFor = "";
  if (snapshot.phase === "DEAL_BID") {
    waitingFor = snapshot.awaiting_action === "bid" ? "等待你叫牌..." : "发牌与叫牌进行中...";
  } else if (snapshot.phase === "STIRRING" && snapshot.stirring_state) {
    const player = snapshot.stirring_state.phase === "EXCHANGING"
      ? snapshot.stirring_state.exchanging_player
      : snapshot.stirring_state.current_player;
    const seat = player !== null && player !== undefined ? SEAT_MAP[player] : null;
    const verb = snapshot.stirring_state.phase === "EXCHANGING" ? "换底牌" : "反主";
    waitingFor = seat ? `等待${seat.label}${verb}...` : `等待${verb}...`;
  } else {
    waitingFor = "等待其他玩家...";
  }

  const hint = document.createElement("div");
  hint.classList.add("waiting-hint");
  hint.textContent = waitingFor;
  bar.appendChild(hint);

  return bar;
}

/** Compact display for a card in bid events (horizontal, no newline). */
function _compactCard(c: { suit: string; rank: string }): string {
  if (c.suit === "joker") {
    return c.rank === "BJ" ? "大王" : "小王";
  }
  return suitSymbol(c.suit) + c.rank;
}

/** Format a bid event for display. */
function formatBidEvent(event: BidEvent): string {
  const seat = SEAT_MAP[event.player];
  const name = seat ? seat.label : `玩家${event.player}`;
  const cardsStr = event.cards.map(_compactCard).join(" ");
  if (event.kind === "trump_rank" && event.suit) {
    return `${name}: ${cardsStr} (${suitDisplayName(event.suit)}主)`;
  }
  if (event.kind === "joker" && event.joker_type) {
    return `${name}: ${cardsStr} (${event.joker_type === "big" ? "大" : "小"}王)`;
  }
  return `${name}: ${cardsStr}`;
}
