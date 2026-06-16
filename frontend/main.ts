import { StateManager } from "./core/state.ts";
import type { ServerMessage, ClientAction } from "./core/protocol.ts";
import type { InteractionMode, GameAction } from "./engine/types.ts";
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
} from "./engine/action-handler.ts";
import {
  computeBidButtonState,
  computeStirButtonState,
  computeLevelChangeInfo,
  computeLegalCardIds,
  isSelectionStillLegal,
} from "./engine/ui-state-computer.ts";

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

  // Shared render context
  const renderCtx: RenderContext = { selectedCardIds, legalCardIds: new Set() };

  /** Pre-compute all UI state and render. */
  function precomputeAndRender(snap: typeof stateManager.get extends () => infer R ? R extends null ? never : R : never) {
    renderCtx.legalCardIds = computeLegalCardIds(snap, currentInteractionMode);
    renderCtx.bidButtonState = computeBidButtonState(snap, selectedCardIds);
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

  // Action callbacks -- close over selectedCardIds, wsClient, stateManager
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

      switch (action) {
        case "play":
          handleResult(handlePlayAction(snap, selectedCardIds));
          break;
        case "discard":
          handleResult(handleDiscardAction(snap, selectedCardIds));
          break;
        case "next_round":
          sendAndClear({ type: "next_round" });
          break;
      }
    },

    onBid(cardIds: string[]) {
      const snap = stateManager.get();
      if (!snap) return;
      handleResult(handleBidAction(snap, cardIds));
    },

    onStir(cardIds: string[]) {
      const snap = stateManager.get();
      if (!snap) return;
      handleResult(handleStirAction(snap, cardIds));
    },

    onPass() {
      sendAndClear({ type: "stir", pass: true });
    },

    onNewGame() {
      selectedCardIds.clear();
      stateManager.reset();
      container.innerHTML = "";
      startNewGame();
    },
  };
  renderCtx.callbacks = callbacks;

  // GameLoop with renderFn that injects callbacks + selectedCardIds
  const gameLoop = new GameLoop(
    stateManager,
    (snapshot, containerEl, interactionMode) => {
      currentInteractionMode = interactionMode;
      if (!isSelectionStillLegal(snapshot, selectedCardIds)) {
        selectedCardIds.clear();
      }
      precomputeAndRender(snapshot);
    },
    container,
    undefined,  // humanPlayerIndex no longer needed
    () => wsClient.isReconnecting,
    (message) => showErrorToast(message, container),
  );

  // Register message handler BEFORE connecting (per spec: first state msg arrives immediately)
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
