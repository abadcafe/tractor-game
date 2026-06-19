import type {
  Card,
  CompletedTrick,
  FailedThrow,
  StateSnapshot,
  TrickSlot,
} from "../../core/types.ts";
import { el } from "../dom.ts";
import { cardDisplay, suitSymbol } from "../../core/card.ts";
import { SEAT_MAP } from "../../config.ts";

/** Map player index to compass direction for trick grid positioning. */
const PLAYER_TO_DIRECTION: Record<number, string> = {
  0: "north",
  1: "west",
  2: "east",
  3: "south",
};

/**
 * Render the current trick area showing played cards with player labels,
 * previous-trick preview, and failed-throw preview.
 */
export function renderTrickView(
  snapshot: StateSnapshot,
  previousTrickPreview?: CompletedTrick | null,
  failedThrowPreview?: FailedThrow | null,
): HTMLElement {
  const showingFailedThrow = failedThrowPreview !== null &&
    failedThrowPreview !== undefined;
  const showingPrevious = !showingFailedThrow &&
    previousTrickPreview !== null &&
    previousTrickPreview !== undefined;
  const trickView = el("div", {
    class: trickViewClass(showingPrevious, showingFailedThrow),
  });

  const grid = showingPrevious
    ? renderCompletedTrickGrid(previousTrickPreview)
    : renderCurrentTrickGrid(snapshot);
  if (grid !== null) {
    trickView.appendChild(grid);
  }
  if (showingFailedThrow) {
    trickView.appendChild(renderFailedThrowPreview(failedThrowPreview));
  }

  return trickView;
}

function trickViewClass(
  showingPrevious: boolean,
  showingFailedThrow: boolean,
): string {
  if (showingFailedThrow) {
    return "trick-view showing-failed-throw";
  }
  if (showingPrevious) {
    return "trick-view showing-previous";
  }
  return "trick-view";
}

function renderCurrentTrickGrid(
  snapshot: StateSnapshot,
): HTMLElement | null {
  if (!snapshot.trick) {
    return null;
  }
  const grid = el("div", { class: "trick-grid" });
  const slotsByPlayer = new Map(
    snapshot.trick.slots.map((slot) => [slot.player, slot]),
  );

  for (const player of [0, 1, 2, 3]) {
    const direction = PLAYER_TO_DIRECTION[player] ?? "north";
    const isLead = player === snapshot.trick.lead_player;
    const isCurrent = player === snapshot.trick.current_player;
    const slot = slotsByPlayer.get(player);

    let slotClass = `${
      slot ? "trick-slot" : "trick-placeholder-slot"
    } trick-slot-${direction}`;
    if (isLead) slotClass += " lead";
    if (isCurrent && !isLead) slotClass += " current";
    if (!slot) slotClass += " empty";

    grid.appendChild(renderTrickSlot(player, slot, slotClass));
  }

  return grid;
}

function renderCompletedTrickGrid(trick: CompletedTrick): HTMLElement {
  const grid = el("div", { class: "trick-grid trick-grid--previous" });
  const slotsByPlayer = new Map(
    trick.slots.map((slot) => [slot.player, slot]),
  );

  for (const player of [0, 1, 2, 3]) {
    const direction = PLAYER_TO_DIRECTION[player] ?? "north";
    const slot = slotsByPlayer.get(player);
    let slotClass = `trick-slot trick-slot-${direction}`;
    if (player === trick.lead_player) slotClass += " lead";
    if (player === trick.winner) slotClass += " winner";
    grid.appendChild(renderTrickSlot(player, slot, slotClass));
  }

  return grid;
}

function renderTrickSlot(
  player: number,
  slot: TrickSlot | undefined,
  slotClass: string,
): HTMLElement {
  const slotEl = el("div", { class: slotClass });
  const seatInfo = SEAT_MAP[player];
  if (seatInfo && slot && slot.cards.length > 0) {
    slotEl.appendChild(
      el("span", { class: "trick-player-label" }, seatInfo.label),
    );
  }

  const cardsDiv = el("div", { class: "trick-cards" });
  for (const card of slot?.cards ?? []) {
    cardsDiv.appendChild(renderTrickCard(card));
  }
  slotEl.appendChild(cardsDiv);
  return slotEl;
}

function renderTrickCard(card: Card): HTMLElement {
  return el("span", {
    class: trickCardClass(card),
    "data-rank": card.rank,
    "data-suit-symbol": suitSymbol(card.suit),
  }, cardDisplay(card));
}

function trickCardClass(card: Card): string {
  let className = `trick-card suit-${card.suit}`;
  if (card.rank === "5" || card.rank === "10" || card.rank === "K") {
    className += " point-card";
  }
  return className;
}

function renderFailedThrowPreview(event: FailedThrow): HTMLElement {
  const seatInfo = SEAT_MAP[event.player];
  const playerLabel = seatInfo?.label ?? `玩家 ${event.player}`;
  const preview = el("div", { class: "failed-throw-preview" });
  preview.appendChild(
    el(
      "div",
      { class: "failed-throw-preview__title" },
      `${playerLabel}甩牌失败`,
    ),
  );
  preview.appendChild(
    renderFailedThrowRow("暴露", event.attempted_cards, false),
  );
  preview.appendChild(
    renderFailedThrowRow("捡小", event.forced_cards, true),
  );
  return preview;
}

function renderFailedThrowRow(
  label: string,
  cards: Card[],
  forced: boolean,
): HTMLElement {
  const row = el("div", {
    class: forced
      ? "failed-throw-preview__row failed-throw-preview__row--forced"
      : "failed-throw-preview__row",
  });
  row.appendChild(
    el("span", { class: "failed-throw-preview__label" }, label),
  );
  const cardsEl = el("div", { class: "failed-throw-preview__cards" });
  for (const card of cards) {
    cardsEl.appendChild(renderTrickCard(card));
  }
  row.appendChild(cardsEl);
  return row;
}
