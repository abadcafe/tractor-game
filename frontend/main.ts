import { StateManager } from "./core/state.ts";
import type { ClientAction, ServerMessage } from "./core/protocol.ts";
import type {
  BidOption,
  GameAction,
  InteractionMode,
} from "./engine/types.ts";
import type { ActionCallbacks, RenderContext } from "./ui/types.ts";
import { createGame } from "./net/rest-client.ts";
import { WsClient } from "./net/ws-client.ts";
import { GameLoop } from "./engine/game-loop.ts";
import { StatePlaybackQueue } from "./engine/state-playback-queue.ts";
import { render } from "./ui/renderer.ts";
import { showErrorToast } from "./ui/error-toast.ts";
import {
  handleDiscardAction,
  handlePassStirAction,
  handlePlayAction,
  handleStirAction,
} from "./engine/action-handler.ts";
import {
  chooseFirstActionHint,
  computeLegalCardIds,
  computeLevelChangeInfo,
  computeStirButtonState,
  isSelectionStillLegal,
} from "./engine/ui-state-computer.ts";
import { PendingBidController } from "./engine/pending-bid-controller.ts";
import { TrickPreviewController } from "./engine/trick-preview-controller.ts";

const GAME_ID_STORAGE_KEY = "tractor-game-id";
const DEAL_BID_PLAYBACK_INTERVAL_MS = 125;
const DEFAULT_PLAYBACK_INTERVAL_MS = 500;
const PREVIOUS_TRICK_PREVIEW_MS = 2000;
const FAILED_THROW_PREVIEW_MS = 5000;

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
  let compactHand = false;
  let actionPending = false;
  let playbackCaughtUp = true;
  let playbackQueue: StatePlaybackQueue<ServerMessage> | null = null;
  let currentGameId: string | null = null;
  const trickPreviewController = new TrickPreviewController(
    PREVIOUS_TRICK_PREVIEW_MS,
    FAILED_THROW_PREVIEW_MS,
    reRender,
  );
  const pendingBidController = new PendingBidController();

  // Shared render context
  const renderCtx: RenderContext = {
    selectedCardIds,
    legalCardIds: new Set(),
  };

  /** Pre-compute all UI state and render. */
  function precomputeAndRender(snap: ReturnType<StateManager["get"]>) {
    if (!snap) return;
    const effectiveInteractionMode = interactionBlocked()
      ? null
      : currentInteractionMode;
    renderCtx.legalCardIds = computeLegalCardIds(
      snap,
      effectiveInteractionMode,
    );
    const bidState = pendingBidController.updateRenderState(
      snap,
      effectiveInteractionMode,
    );
    renderCtx.bidOptions = bidState.bidOptions;
    renderCtx.pendingBidIntent = bidState.pendingBidIntent;
    renderCtx.stirButtonState = computeStirButtonState(
      snap,
      selectedCardIds,
    );
    renderCtx.compactHand = compactHand;
    renderCtx.gameId = currentGameId;
    renderCtx.previousTrickPreview =
      trickPreviewController.previousTrickPreview;
    renderCtx.failedThrowPreview =
      trickPreviewController.failedThrowPreview;
    renderCtx.levelChange = snap.scoring
      ? computeLevelChangeInfo(snap.scoring.total_defender_points)
      : undefined;
    render(snap, container, effectiveInteractionMode, renderCtx);
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

  /** Block actions while an action response or queued state playback is pending. */
  function interactionBlocked(): boolean {
    return actionPending || !playbackCaughtUp || !wsClient.isConnected;
  }

  /** Send an action and wait for the server's next state before accepting more input. */
  function sendAction(action: ClientAction): boolean {
    if (!wsClient.send(action)) {
      actionPending = false;
      showErrorToast("连接未就绪，请稍后重试", container);
      return false;
    }
    actionPending = true;
    return true;
  }

  /** Send action, clear selection, re-render. */
  function sendAndClear(action: ClientAction) {
    if (!sendAction(action)) {
      return;
    }
    selectedCardIds.clear();
    reRender();
  }

  /** Handle a validated action result. */
  function handleResult(
    result: { success: boolean; action?: ClientAction; error?: string },
  ) {
    if (result.success && result.action) {
      sendAndClear(result.action);
    } else if (result.error) {
      showErrorToast(result.error, container);
    }
  }

  /** Auto-bid: send skip or pending intent when it's the human's turn to bid. */
  function handleAutoBid(interactionMode: InteractionMode) {
    if (interactionMode !== "bid" || pendingBidController.isInFlight) {
      return;
    }
    const seq = currentSeq();

    const decision = pendingBidController.computeDealBidAction(seq);
    if (sendAction(decision.action)) {
      pendingBidController.markActionSent(decision);
      reRender();
    }
  }

  // Action callbacks
  const callbacks: ActionCallbacks = {
    onCardClick(cardId: string) {
      if (interactionBlocked()) return;
      if (selectedCardIds.has(cardId)) {
        selectedCardIds.delete(cardId);
      } else {
        selectedCardIds.add(cardId);
      }
      reRender();
    },

    onCardRangeSelect(cardIds: string[]) {
      if (interactionBlocked()) return;
      selectedCardIds.clear();
      for (const cardId of cardIds) {
        selectedCardIds.add(cardId);
      }
      reRender();
    },

    onClearSelection() {
      if (interactionBlocked()) return;
      selectedCardIds.clear();
      reRender();
    },

    onUseHint() {
      if (interactionBlocked()) return;
      const snap = stateManager.get();
      if (!snap) return;
      const hint = chooseFirstActionHint(snap);
      if (hint === null) return;
      selectedCardIds.clear();
      for (const card of hint) {
        selectedCardIds.add(card.id);
      }
      reRender();
    },

    onToggleHandCompact() {
      if (interactionBlocked()) return;
      compactHand = !compactHand;
      reRender();
    },

    onShowPreviousTrick() {
      const snap = stateManager.get();
      const latestTrick = snap?.last_completed_trick ?? null;
      if (latestTrick === null) return;
      trickPreviewController.showPreviousTrickPreview(
        latestTrick,
        true,
      );
    },

    onAction(action: GameAction) {
      if (interactionBlocked()) return;
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
          const clientAction: ClientAction = {
            type: "next_round",
            seq,
          };
          sendAndClear(clientAction);
          break;
        }
      }
    },

    onBidOptionSelect(option: BidOption) {
      const snap = stateManager.get();
      if (snap?.phase !== "DEAL_BID") {
        return;
      }
      if (pendingBidController.select(option)) {
        reRender();
      }
    },

    onStir(cardIds: string[]) {
      if (interactionBlocked()) return;
      const snap = stateManager.get();
      if (!snap) return;
      handleResult(handleStirAction(snap, cardIds, currentSeq()));
    },

    onPass() {
      if (interactionBlocked()) return;
      handleResult(handlePassStirAction(currentSeq()));
    },

    onNewGame() {
      selectedCardIds.clear();
      pendingBidController.reset();
      playbackQueue?.clear();
      actionPending = false;
      playbackCaughtUp = true;
      trickPreviewController.reset();
      stateManager.reset();
      localStorage.removeItem(GAME_ID_STORAGE_KEY);
      currentGameId = null;
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
      trickPreviewController.update(snapshot);
      precomputeAndRender(snapshot);

      // Auto-bid after render
      if (!interactionBlocked()) {
        handleAutoBid(interactionMode);
      }
    },
    container,
    undefined,
    () => wsClient.isReconnecting,
    (message) => {
      if (pendingBidController.consumeInFlightFailure()) {
        reRender();
        showErrorToast(`抢主失败：${message}`, container);
        return;
      }
      showErrorToast(message, container);
    },
  );

  playbackQueue = new StatePlaybackQueue<ServerMessage>(
    (msg) => {
      actionPending = false;
      pendingBidController.acknowledgeMessage(msg);
      gameLoop.handleMessage(msg);
    },
    {
      minFrameMsForMessage(msg) {
        return msg.state.phase === "DEAL_BID"
          ? DEAL_BID_PLAYBACK_INTERVAL_MS
          : DEFAULT_PLAYBACK_INTERVAL_MS;
      },
      onCaughtUpChange(caughtUp) {
        playbackCaughtUp = caughtUp;
        reRender();
      },
    },
  );

  // Register message handler BEFORE connecting
  wsClient.onMessage((msg: ServerMessage) => {
    playbackQueue?.enqueue(msg);
  });

  wsClient.onDisconnect(() => {
    console.log("WebSocket disconnected");
    actionPending = false;
    reRender();
  });

  wsClient.onReconnectFail(() => {
    actionPending = false;
    reRender();
    showErrorToast("连接已断开，请刷新页面重试", container);
  });

  function currentWsHost(): string {
    const protocol = window.location.protocol === "https:"
      ? "wss:"
      : "ws:";
    return `${protocol}//${window.location.host}`;
  }

  async function connectToGame(gameId: string): Promise<void> {
    await wsClient.connect(gameId, currentWsHost());
    currentGameId = gameId;
    localStorage.setItem(GAME_ID_STORAGE_KEY, gameId);
  }

  async function createAndConnectGame(): Promise<void> {
    const gameId = await createGame();
    await connectToGame(gameId);
  }

  async function resumeOrCreateGame(): Promise<void> {
    wsClient.disconnect();
    const savedGameId = localStorage.getItem(GAME_ID_STORAGE_KEY);
    if (savedGameId !== null && savedGameId.length > 0) {
      try {
        await connectToGame(savedGameId);
        return;
      } catch (e) {
        console.warn("Failed to resume saved game:", e);
        localStorage.removeItem(GAME_ID_STORAGE_KEY);
      }
    }
    await createAndConnectGame();
  }

  async function startNewGame(): Promise<void> {
    try {
      wsClient.disconnect();
      await createAndConnectGame();
    } catch (e) {
      console.error("Failed to start game:", e);
      showErrorToast("无法启动游戏，请刷新页面重试", container);
    }
  }

  resumeOrCreateGame().catch((e) => {
    console.error("Failed to resume or start game:", e);
    showErrorToast("无法启动游戏，请刷新页面重试", container);
  });
}

main();
