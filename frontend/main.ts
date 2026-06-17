import { StateManager } from "./core/state.ts";
import type { ClientAction, ServerMessage } from "./core/protocol.ts";
import type { GameAction, InteractionMode } from "./engine/types.ts";
import type { ActionCallbacks, RenderContext } from "./ui/types.ts";
import { createGame } from "./net/rest-client.ts";
import { WsClient } from "./net/ws-client.ts";
import { GameLoop } from "./engine/game-loop.ts";
import { StatePlaybackQueue } from "./engine/state-playback-queue.ts";
import { render } from "./ui/renderer.ts";
import { showErrorToast } from "./ui/error-toast.ts";
import {
  handleBidAction,
  handleDiscardAction,
  handlePassStirAction,
  handlePlayAction,
  handleSkipBidAction,
  handleStirAction,
} from "./engine/action-handler.ts";
import {
  computeBidButtonState,
  computeLegalCardIds,
  computeLevelChangeInfo,
  computeStirButtonState,
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
  let initialNextRoundPending = false;
  let actionPending = false;
  let playbackCaughtUp = true;
  let playbackQueue: StatePlaybackQueue<ServerMessage> | null = null;

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
    renderCtx.bidButtonState = computeBidButtonState(
      snap,
      selectedCardIds,
    );
    renderCtx.stirButtonState = computeStirButtonState(
      snap,
      selectedCardIds,
    );
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

  function interactionBlocked(): boolean {
    return actionPending || !playbackCaughtUp;
  }

  /** Send action, clear selection, re-render. */
  function sendAndClear(action: ClientAction) {
    actionPending = true;
    wsClient.send(action);
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

  // Action callbacks -- close over selectedCardIds, wsClient, stateManager
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
        case "skip_bid":
          handleResult(handleSkipBidAction(seq));
          break;
        case "next_round": {
          const result = {
            success: true,
            action: { type: "next_round" as const, seq },
          };
          sendAndClear(result.action);
          break;
        }
      }
    },

    onBid(cardIds: string[]) {
      if (interactionBlocked()) return;
      const snap = stateManager.get();
      if (!snap) return;
      handleResult(handleBidAction(snap, cardIds, currentSeq()));
    },

    onStir(cardIds: string[]) {
      if (interactionBlocked()) return;
      const snap = stateManager.get();
      if (!snap) return;
      handleResult(handleStirAction(snap, cardIds, currentSeq()));
    },

    onPass() {
      if (interactionBlocked()) return;
      const seq = currentSeq();
      // Determine if this is a bid pass or stir pass based on current mode
      if (currentInteractionMode === "bid") {
        handleResult(handleSkipBidAction(seq));
      } else {
        handleResult(handlePassStirAction(seq));
      }
    },

    onNewGame() {
      selectedCardIds.clear();
      stateManager.reset();
      playbackQueue?.clear();
      actionPending = false;
      playbackCaughtUp = true;
      container.innerHTML = "";
      startNewGame();
    },
  };
  renderCtx.callbacks = callbacks;

  container.innerHTML = `
    <div class="boot-screen">
      <div class="boot-screen__title">拖拉机</div>
      <div class="boot-screen__status">正在连接牌桌...</div>
    </div>
  `;

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
    undefined, // humanPlayerIndex no longer needed
    () => wsClient.isReconnecting,
    (message) => showErrorToast(message, container),
  );

  playbackQueue = new StatePlaybackQueue<ServerMessage>(
    (msg) => {
      const shouldRetryInitialNextRound = initialNextRoundPending &&
        msg.awaiting === "next_round";
      actionPending = false;
      gameLoop.handleMessage(
        shouldRetryInitialNextRound && msg.error
          ? { ...msg, error: undefined }
          : msg,
      );
      if (shouldRetryInitialNextRound) {
        actionPending = true;
        wsClient.send({ type: "next_round", seq: msg.seq });
        reRender();
        return;
      }
      initialNextRoundPending = false;
    },
    {
      minFrameMs: 500,
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
  });

  wsClient.onReconnectFail(() => {
    showErrorToast("连接已断开，请刷新页面重试", container);
  });

  // Start game flow
  async function startNewGame() {
    try {
      wsClient.disconnect();
      const gameId = await createGame();
      const protocol = window.location.protocol === "https:"
        ? "wss:"
        : "ws:";
      const host = window.location.host;
      const wsHost = `${protocol}//${host}`;
      await wsClient.connect(gameId, wsHost);
      initialNextRoundPending = true;
      actionPending = true;
      wsClient.send({ type: "next_round", seq: 0 });
    } catch (e) {
      console.error("Failed to start game:", e);
      initialNextRoundPending = false;
      actionPending = false;
      showErrorToast("无法启动游戏，请刷新页面重试", container);
    }
  }

  startNewGame();
}

main();
