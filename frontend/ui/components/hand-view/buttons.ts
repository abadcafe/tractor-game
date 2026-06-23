import { suitSymbol } from "../../../core/card.ts";
import type { StateSnapshot } from "../../../core/types.ts";
import type {
  BidOption,
  GameAction,
  InteractionMode,
  StirButtonState,
} from "../../../engine/types.ts";
import { el } from "../../dom.ts";

export function renderBidActionPanel(
  bidOptions: BidOption[],
  pendingBidIntent: BidOption | null,
  onBidOptionSelect: (option: BidOption) => void,
): HTMLElement {
  const panel = el("div", { class: "action-panel action-panel--bid" });
  const hasPendingBid = pendingBidIntent !== null;
  for (const option of bidOptions) {
    const className = [
      isSameBidOption(option, pendingBidIntent)
        ? "btn-warning hand-action-button selected"
        : "btn-secondary hand-action-button",
      bidOptionColorClass(option),
    ].join(" ");
    const attrs: Record<string, string> = {
      class: className,
      title: bidOptionTitle(option),
    };
    if (hasPendingBid) {
      attrs.disabled = "true";
    }
    const button = el("button", attrs, option.label);
    if (!hasPendingBid) {
      button.addEventListener("click", () => onBidOptionSelect(option));
    }
    panel.appendChild(button);
  }
  return panel;
}

export function renderPreviousTrickButton(
  onShowPreviousTrick: () => void,
): HTMLElement {
  const panel = el("div", { class: "hand-panel__previous-trick" });
  const button = el("button", {
    class: "btn-secondary hand-action-button",
    title: "查看上一墩",
  }, "上一墩");
  button.addEventListener("click", () => onShowPreviousTrick());
  panel.appendChild(button);
  return panel;
}

export function renderActionPanel(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  selectedCount: number,
  onAction: (action: GameAction) => void,
): HTMLElement {
  const panel = el("div", { class: "action-panel" });

  if (interactionMode === "play") {
    const button = el("button", {
      class: "btn-primary hand-action-button",
    }, "出牌");
    if (selectedCount === 0) {
      button.setAttribute("disabled", "true");
      button.setAttribute("title", "请先选择要出的牌");
    }
    button.addEventListener("click", () => onAction("play"));
    panel.appendChild(button);
  } else if (interactionMode === "discard") {
    const button = el("button", {
      class: "btn-warning hand-action-button",
    }, "换底牌");
    const count = snapshot.stirring_state?.exchange_count ?? 0;
    if (count > 0 && selectedCount !== count) {
      button.setAttribute("disabled", "true");
      button.setAttribute("title", `请选择 ${count} 张牌`);
    }
    button.addEventListener("click", () => onAction("discard"));
    panel.appendChild(button);
  }

  return panel;
}

export function renderStirActionPanel(
  selectedCardIds: string[],
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
  stirButtonState?: StirButtonState,
): HTMLElement {
  const panel = el("div", { class: "action-panel action-panel--stir" });

  const stirButton = el("button", {
    class: "btn-warning hand-action-button",
  }, "反主");
  const stirDisabled = stirButtonState?.disabled ?? true;
  if (stirDisabled || onStir === undefined) {
    stirButton.setAttribute("disabled", "true");
  }
  if (stirButtonState?.title) {
    stirButton.setAttribute("title", stirButtonState.title);
  }
  stirButton.addEventListener("click", () => {
    if (!stirDisabled && onStir !== undefined) {
      onStir(selectedCardIds);
    }
  });
  panel.appendChild(stirButton);

  const passButton = el("button", {
    class: "btn-secondary hand-action-button",
  }, "不反");
  if (onPass === undefined) {
    passButton.setAttribute("disabled", "true");
  }
  passButton.addEventListener("click", () => onPass?.());
  panel.appendChild(passButton);

  return panel;
}

export function renderHandTools(
  selectedCount: number,
  hasActionHints: boolean,
  onClearSelection?: () => void,
  onUseHint?: () => void,
): HTMLElement {
  const tools = el("div", { class: "hand-panel__tools" });

  const hintButton = el("button", {
    class: "btn-info hand-action-button",
  }, "提示");
  if (!hasActionHints || !onUseHint) {
    hintButton.setAttribute("disabled", "true");
  }
  hintButton.addEventListener("click", () => onUseHint?.());
  tools.appendChild(hintButton);

  const clearButton = el("button", {
    class: "btn-secondary hand-action-button",
  }, "清牌");
  if (selectedCount === 0 || !onClearSelection) {
    clearButton.setAttribute("disabled", "true");
  }
  clearButton.addEventListener("click", () => onClearSelection?.());
  tools.appendChild(clearButton);

  return tools;
}

function bidOptionColorClass(option: BidOption): string {
  if (
    option.trumpSuit === "hearts" || option.trumpSuit === "diamonds"
  ) {
    return "bid-button-red";
  }
  if (option.trumpSuit === "spades" || option.trumpSuit === "clubs") {
    return "bid-button-black";
  }
  return option.label.startsWith("大王")
    ? "bid-button-red"
    : "bid-button-black";
}

function isSameBidOption(
  option: BidOption,
  pendingBidIntent: BidOption | null,
): boolean {
  return pendingBidIntent !== null &&
    pendingBidIntent.cardIds.join(",") === option.cardIds.join(",");
}

function bidOptionTitle(option: BidOption): string {
  return option.trumpSuit === null
    ? `${option.label}，无主`
    : `${option.label}，抢${suitSymbol(option.trumpSuit)}主`;
}
