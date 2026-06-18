import { StateManager } from "./core/state.ts";
import type { StateSnapshot } from "./core/types.ts";
import type { ServerMessage, ClientAction } from "./core/protocol.ts";
import type { InteractionMode, GameAction, BidOption } from "./engine/types.ts";
import type { ActionCallbacks, RenderContext } from "./ui/types.ts";
import { createGame } from "./net/rest-client.ts";
import { WsClient } from "./net/ws-client.ts";
import { GameLoop } from "./engine/game-loop.ts";
import { render } from "./ui/renderer.ts";
import { showErrorToast } from "./ui/error-toast.ts";
import {
  handlePlayAction,
  handleDiscardAction,
  handleBidAction,
  handleStirAction,
  handlePassStirAction,
} from "./engine/action-handler.ts";
import {
  computeStirButtonState,
  computeLevelChangeInfo,
  computeLegalCardIds,
  isSelectionStillLegal,
} from "./engine/ui-state-computer.ts";
import { computeBidOptions, computeBidOptionsFromHints } from "./engine/bid-logic.ts";

function main() {
  const containerEl = document.querySelector("#app");
  if (!containerEl) {
    console.error("#app element not found");
    return;
  }
  const container = containerEl;

  const stateManager = new StateManager();
  const wsClient = new WsClient();

  // UI state: persists across re-renders
  const selectedCardIds = new Set<string>();
  let currentInteractionMode: InteractionMode = null;

  // Auto-bid state
  let pendingBidIntent: BidOption | null = null;
  let suppressBidError = false;

  // Shared render context
  const renderCtx: RenderContext = { selectedCardIds, legalCardIds: new Set() };

  /** Pre-compute all UI state and render. */
  function precomputeAndRender(snap: ReturnType<StateManager["get"]>) {
    if (!snap) return;
    renderCtx.legalCardIds = computeLegalCardIds(snap, currentInteractionMode);
    renderCtx.bidOptions = snap.phase === "DEAL_BID"
      ? computeBidOptions(snap.player_hand, snap.trump_rank, snap.bid_winner)
      : computeBidOptionsFromHints(snap.action_hints ?? [], snap.trump_rank);
    renderCtx.pendingBidIntent = pendingBidIntent;
    renderCtx.stirButtonState = computeStirButtonState(snap, selectedCardIds);
    renderCtx.levelChange = snap.scoring
      ? computeLevelChangeInfo(snap.scoring.total_defender_points)
      : undefined;
    render(snap, container, currentInteractionMode, renderCtx);
  }

  /** Re-render from current state (for selection changes). */
  function reRender() {
    const snap = stateManager.get();
    if (snap) precomputeAndRender(snap);
  }

  /** Get current seq for client actions. */
  function currentSeq(): number {
    return stateManager.seq;
  }

  /** Send action, clear selection, re-render. */
  function sendAndClear(action: ClientAction) {
    wsClient.send(action);
    selectedCardIds.clear();
    reRender();
  }

  /** Handle a validated action result. */
  function handleResult(result: { success: boolean; action?: ClientAction; error?: string }) {
    if (result.success && result.action) {
      sendAndClear(result.action);
    } else if (result.error) {
      showErrorToast(result.error, container);
    }
  }

  /** Auto-bid: send skip or pending intent when it's the human's turn to bid. */
  function handleAutoBid(snapshot: StateSnapshot, interactionMode: InteractionMode) {
    if (interactionMode !== "bid") return;
    const seq = currentSeq();

    if (pendingBidIntent) {
      const currentHintIds = (snapshot.action_hints ?? []).map((cards) =>
        cards.map((c) => c.id).sort().join(",")
      );
      const intentKey = [...pendingBidIntent.cardIds].sort().join(",");

      if (currentHintIds.includes(intentKey)) {
        suppressBidError = true;
        wsClient.send({ type: "bid", seq, cards: pendingBidIntent.cardIds });
        return;
      }
      pendingBidIntent = null;
    }

    // Auto-skip
    wsClient.send({ type: "bid", seq, pass: true });
  }

  // Action callbacks
  const callbacks: ActionCallbacks = {
    onCardClick(cardId: string) {
      if (selectedCardIds.has(cardId)) {
        selectedCardIds.delete(cardId);
      } else {
        selectedCardIds.add(cardId);
      }
      reRender();
    },

    onAction(action: GameAction) {
      const snap = stateManager.get();
      if (!snap) return;
      const seq = currentSeq();

      switch (action) {
        case "play":
          handleResult(handlePlayAction(snap, selectedCardIds, seq));
          break;
        case "discard":
          handleResult(handleDiscardAction(snap, selectedCardIds, seq));
          break;
        case "next_round": {
          const clientAction: ClientAction = { type: "next_round", seq };
          sendAndClear(clientAction);
          break;
        }
      }
    },

    onBidOptionSelect(option: BidOption) {
      pendingBidIntent = option;
      reRender();
    },

    onStir(cardIds: string[]) {
      const snap = stateManager.get();
      if (!snap) return;
      handleResult(handleStirAction(snap, cardIds, currentSeq()));
    },

    onPass() {
      handleResult(handlePassStirAction(currentSeq()));
    },

    onNewGame() {
      selectedCardIds.clear();
      pendingBidIntent = null;
      suppressBidError = false;
      stateManager.reset();
      container.innerHTML = "";
      startNewGame();
    },
  };
  renderCtx.callbacks = callbacks;

  // GameLoop with renderFn that injects callbacks + selectedCardIds
  const gameLoop = new GameLoop(
    stateManager,
    (snapshot, _containerEl, interactionMode) => {
      currentInteractionMode = interactionMode;
      if (!isSelectionStillLegal(snapshot, selectedCardIds)) {
        selectedCardIds.clear();
      }
      precomputeAndRender(snapshot);

      // Auto-bid after render
      handleAutoBid(snapshot, interactionMode);
    },
    container,
    undefined,
    () => wsClient.isReconnecting,
    (message) => {
      // If suppressing bid errors, silently clear intent and don't show toast
      if (suppressBidError) {
        suppressBidError = false;
        pendingBidIntent = null;
        reRender();
        return;
      }
      showErrorToast(message, container);
    },
  );

  // Register message handler BEFORE connecting
  wsClient.onMessage((msg: ServerMessage) => {
    gameLoop.handleMessage(msg);
  });

  wsClient.onDisconnect(() => {
    console.log("WebSocket disconnected");
  });

  wsClient.onReconnectFail(() => {
    showErrorToast("连接已断开，请刷新页面重试", container);
  });

  // Start game flow
  async function startNewGame() {
    try {
      wsClient.disconnect();
      const gameId = await createGame();
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = window.location.host;
      const wsHost = `${protocol}//${host}`;
      await wsClient.connect(gameId, wsHost);
    } catch (e) {
      console.error("Failed to start game:", e);
      showErrorToast("无法启动游戏，请刷新页面重试", container);
    }
  }

  startNewGame();
}

main();
