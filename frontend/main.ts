import { StateManager } from "./core/state.ts";
import type { ClientAction, ServerMessage } from "./core/protocol.ts";
import type {
  BidOption,
  GameAction,
  InteractionMode,
} from "./engine/types.ts";
import type {
  ActionCallbacks,
  ConnectionStatus,
  RenderContext,
} from "./ui/types.ts";
import type { PlayerIndex } from "./config.ts";
import {
  type BotFillMode,
  createGame,
  deleteGame,
  fillBotPlayers,
  joinPlayer,
  leavePlayer,
  type ListedGame,
  listGames,
} from "./net/rest-client.ts";
import { WsClient } from "./net/ws-client.ts";
import {
  gamePlayerHref,
  type GamePlayerRoute,
  parseGamePlayerRoute,
} from "./routing.ts";
import {
  resolveLobbySelectedGameId,
  selectedGameHasEmptyPlayer,
} from "./lobby-selection.ts";
import { GameLoop } from "./engine/game-loop.ts";
import { StatePlaybackQueue } from "./engine/state-playback-queue.ts";
import { render } from "./ui/renderer.ts";
import {
  type LobbyCallbacks,
  type LobbyState,
  renderLobby,
} from "./ui/lobby.ts";
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
const USER_ID_STORAGE_KEY = "tractor-user-id";
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
  let connectionStatus: ConnectionStatus = "connecting";
  let lobbyGames: ListedGame[] = [];
  let lobbyLoading = false;
  let lobbyCreating = false;
  let lobbyPendingPlayerGameId: string | null = null;
  let lobbyPendingPlayerIndex: PlayerIndex | null = null;
  let lobbyDeletingGameId: string | null = null;
  let lobbySelectedGameId: string | null = localStorage.getItem(
    GAME_ID_STORAGE_KEY,
  );
  let lobbyBotFillMode: BotFillMode = "none";
  let currentPlayerIndex: PlayerIndex | null = null;
  let lobbyErrorMessage: string | null = null;
  let lobbyStatusMessage: string | null = null;
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
    connectionStatus,
  };

  function resetGameSession(): void {
    selectedCardIds.clear();
    pendingBidController.reset();
    playbackQueue?.clear();
    actionPending = false;
    playbackCaughtUp = true;
    currentInteractionMode = null;
    trickPreviewController.reset();
    stateManager.reset();
    currentGameId = null;
    currentPlayerIndex = null;
    renderCtx.viewerPlayer = null;
    connectionStatus = "connecting";
    renderCtx.connectionStatus = connectionStatus;
  }

  function updateConnectionStatus(status: ConnectionStatus): void {
    if (connectionStatus === status) {
      return;
    }
    connectionStatus = status;
    renderCtx.connectionStatus = status;
    reRender();
  }

  function lobbyState(): LobbyState {
    return {
      games: lobbyGames,
      loading: lobbyLoading,
      creating: lobbyCreating,
      pendingPlayerGameId: lobbyPendingPlayerGameId,
      pendingPlayerIndex: lobbyPendingPlayerIndex,
      deletingGameId: lobbyDeletingGameId,
      selectedGameId: lobbySelectedGameId,
      botFillMode: lobbyBotFillMode,
      errorMessage: lobbyErrorMessage,
      statusMessage: lobbyStatusMessage,
    };
  }

  const lobbyCallbacks: LobbyCallbacks = {
    onCreateGame() {
      void handleCreateGame();
    },
    onSelectGame(gameId: string) {
      lobbySelectedGameId = gameId;
      lobbyStatusMessage = null;
      lobbyErrorMessage = null;
      renderLobbyScreen();
    },
    onDeleteGame(gameId: string) {
      void handleDeleteGame(gameId);
    },
    onTogglePlayer(gameId, playerIndex) {
      void handleTogglePlayer(gameId, playerIndex);
    },
    onEnterPlayer(gameId, playerIndex) {
      void handleEnterPlayer(gameId, playerIndex);
    },
    enterPlayerHref(gameId, playerIndex) {
      return gamePlayerHref(gameId, playerIndex, ensureUserId());
    },
    onChangeBotFillMode(mode) {
      void handleChangeBotFillMode(mode);
    },
    onRefreshGames() {
      void refreshLobbyGames();
    },
  };

  function renderLobbyScreen(): void {
    container.classList.remove("game-shell", "game-shell--scoring");
    container.classList.add("lobby-shell");
    container.innerHTML = "";
    container.appendChild(renderLobby(lobbyState(), lobbyCallbacks));
  }

  function renderConnectingScreen(
    gameId: string,
    playerIndex: PlayerIndex,
  ): void {
    container.classList.remove(
      "lobby-shell",
      "game-shell",
      "game-shell--scoring",
    );
    container.innerHTML = "";
    container.appendChild(
      (() => {
        const screen = document.createElement("div");
        screen.className = "boot-screen";

        const title = document.createElement("div");
        title.className = "boot-screen__title";
        title.textContent = "正在进入牌局";

        const status = document.createElement("div");
        status.className = "boot-screen__status";
        status.textContent = `牌局 ${
          gameId.slice(0, 8)
        } · 玩家 ${playerIndex}`;

        screen.append(title, status);
        return screen;
      })(),
    );
  }

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
    renderCtx.viewerPlayer = currentPlayerIndex;
    renderCtx.connectionStatus = connectionStatus;
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

  /** Auto-bid: send skip or pending intent when it's the user's turn to bid. */
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
      wsClient.disconnect();
      resetGameSession();
      localStorage.removeItem(GAME_ID_STORAGE_KEY);
      globalThis.history.pushState(null, "", "/");
      lobbySelectedGameId = null;
      lobbyStatusMessage = null;
      lobbyErrorMessage = null;
      renderLobbyScreen();
      void handleCreateGame();
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
    updateConnectionStatus("connected");
    playbackQueue?.enqueue(msg);
  });

  wsClient.onDisconnect((event) => {
    console.log("WebSocket disconnected");
    actionPending = false;
    updateConnectionStatus(
      event.willReconnect ? "connecting" : "failed",
    );
  });

  wsClient.onReconnectFail(() => {
    actionPending = false;
    updateConnectionStatus("failed");
    showErrorToast("连接已断开，请刷新页面重试", container);
  });

  function currentWsHost(): string {
    const protocol = globalThis.location.protocol === "https:"
      ? "wss:"
      : "ws:";
    return `${protocol}//${globalThis.location.host}`;
  }

  async function connectToGame(route: GamePlayerRoute): Promise<void> {
    currentGameId = route.gameId;
    currentPlayerIndex = route.playerIndex;
    renderCtx.viewerPlayer = route.playerIndex;
    connectionStatus = "connecting";
    renderCtx.connectionStatus = connectionStatus;
    localStorage.setItem(GAME_ID_STORAGE_KEY, route.gameId);
    await wsClient.connect(
      {
        gameId: route.gameId,
        playerIndex: route.playerIndex,
        userId: route.userId,
      },
      currentWsHost(),
    );
  }

  async function connectFromRoute(
    route: GamePlayerRoute,
  ): Promise<void> {
    resetGameSession();
    renderConnectingScreen(route.gameId, route.playerIndex);
    try {
      await connectToGame(route);
    } catch (error: unknown) {
      console.error("Failed to enter player:", error);
      wsClient.disconnect();
      resetGameSession();
      globalThis.history.replaceState(null, "", "/");
      await refreshLobbyGames();
      lobbyErrorMessage = "无法进入该玩家";
      renderLobbyScreen();
    }
  }

  async function refreshLobbyGames(): Promise<void> {
    if (lobbyLoading) {
      return;
    }
    lobbyLoading = true;
    lobbyErrorMessage = null;
    renderLobbyScreen();
    try {
      lobbyGames = await listGames("", ensureUserId());
      lobbySelectedGameId = resolveLobbySelectedGameId(
        lobbyGames,
        lobbySelectedGameId,
      );
    } catch (error: unknown) {
      console.error("Failed to load games:", error);
      lobbyErrorMessage = "无法加载牌局";
    } finally {
      lobbyLoading = false;
      renderLobbyScreen();
    }
  }

  async function handleCreateGame(): Promise<void> {
    if (
      lobbyCreating || lobbyPendingPlayerGameId !== null ||
      lobbyDeletingGameId !== null
    ) {
      return;
    }
    lobbyCreating = true;
    lobbyErrorMessage = null;
    lobbyStatusMessage = null;
    renderLobbyScreen();
    try {
      const gameId = await createGame();
      lobbySelectedGameId = gameId;
      lobbyStatusMessage = `已创建牌局 ${gameId.slice(0, 8)}`;
      await refreshLobbyGames();
    } catch (error: unknown) {
      console.error("Failed to create game:", error);
      lobbyErrorMessage = "无法创建牌局";
    } finally {
      lobbyCreating = false;
      renderLobbyScreen();
    }
  }

  async function handleTogglePlayer(
    gameId: string,
    playerIndex: PlayerIndex,
  ): Promise<void> {
    if (
      lobbyPendingPlayerGameId !== null || lobbyDeletingGameId !== null
    ) {
      return;
    }
    lobbyPendingPlayerGameId = gameId;
    lobbyPendingPlayerIndex = playerIndex;
    lobbySelectedGameId = gameId;
    lobbyErrorMessage = null;
    lobbyStatusMessage = null;
    renderLobbyScreen();

    const userId = ensureUserId();
    const selectedGame =
      lobbyGames.find((game) => game.gameId === gameId) ??
        null;
    const selectedPlayer = selectedGame?.players.find(
      (player) => player.index === playerIndex,
    ) ?? null;
    const leavingPlayer = selectedPlayer?.mine === true;
    try {
      if (leavingPlayer) {
        await leavePlayer(gameId, playerIndex, userId);
        lobbyStatusMessage = `已离开玩家 ${playerIndex}`;
      } else {
        await joinPlayer(gameId, playerIndex, userId);
        lobbyStatusMessage = `已入座玩家 ${playerIndex}`;
      }
      await refreshLobbyGames();
    } catch (error: unknown) {
      console.error("Failed to update player:", error);
      lobbyErrorMessage = leavingPlayer
        ? "无法离开玩家"
        : "无法控制玩家";
    } finally {
      lobbyPendingPlayerGameId = null;
      lobbyPendingPlayerIndex = null;
      renderLobbyScreen();
    }
  }

  function handleEnterPlayer(
    gameId: string,
    playerIndex: PlayerIndex,
  ): void {
    lobbySelectedGameId = gameId;
    lobbyErrorMessage = null;
    markEnteredPlayer(playerIndex);
  }

  async function handleChangeBotFillMode(
    mode: BotFillMode,
  ): Promise<void> {
    if (
      lobbyPendingPlayerGameId !== null || lobbyDeletingGameId !== null
    ) {
      return;
    }
    lobbyBotFillMode = mode;
    lobbyErrorMessage = null;
    lobbyStatusMessage = null;

    if (mode === "none" || lobbySelectedGameId === null) {
      renderLobbyScreen();
      return;
    }

    const selectedGame =
      lobbyGames.find((game) => game.gameId === lobbySelectedGameId) ??
        null;
    if (
      selectedGame === null ||
      !selectedGame.players.some((player) => player.mine) ||
      !selectedGameHasEmptyPlayer(lobbyGames, selectedGame.gameId)
    ) {
      renderLobbyScreen();
      return;
    }

    lobbyPendingPlayerGameId = selectedGame.gameId;
    lobbyPendingPlayerIndex = null;
    lobbyStatusMessage = "正在填充 bot";
    renderLobbyScreen();

    try {
      await fillBotPlayers(selectedGame.gameId, mode, ensureUserId());
      lobbyStatusMessage = "已填充 bot";
      await refreshLobbyGames();
    } catch (error: unknown) {
      console.error("Failed to fill bot players:", error);
      lobbyErrorMessage = "无法填充 bot";
      lobbyStatusMessage = null;
    } finally {
      lobbyPendingPlayerGameId = null;
      renderLobbyScreen();
    }
  }

  async function handleDeleteGame(gameId: string): Promise<void> {
    if (
      lobbyPendingPlayerGameId !== null ||
      lobbyDeletingGameId !== null ||
      lobbyCreating
    ) {
      return;
    }
    lobbyDeletingGameId = gameId;
    lobbySelectedGameId = gameId;
    lobbyErrorMessage = null;
    lobbyStatusMessage = null;
    renderLobbyScreen();
    try {
      await deleteGame(gameId);
      if (localStorage.getItem(GAME_ID_STORAGE_KEY) === gameId) {
        localStorage.removeItem(GAME_ID_STORAGE_KEY);
      }
      if (lobbySelectedGameId === gameId) {
        lobbySelectedGameId = null;
      }
      lobbyStatusMessage = `已删除牌局 ${gameId.slice(0, 8)}`;
      await refreshLobbyGames();
    } catch (error: unknown) {
      console.error("Failed to delete game:", error);
      lobbyErrorMessage = "无法删除牌局";
      lobbyStatusMessage = null;
    } finally {
      lobbyDeletingGameId = null;
      renderLobbyScreen();
    }
  }

  function markEnteredPlayer(playerIndex: PlayerIndex): void {
    lobbyStatusMessage = `已打开玩家 ${playerIndex} 的牌桌页面`;
    globalThis.setTimeout(renderLobbyScreen, 0);
  }

  function ensureUserId(): string {
    const stored = localStorage.getItem(USER_ID_STORAGE_KEY);
    if (stored !== null && stored.trim() !== "") {
      return stored;
    }
    const generated = crypto.randomUUID();
    localStorage.setItem(USER_ID_STORAGE_KEY, generated);
    return generated;
  }

  const route = parseGamePlayerRoute(
    globalThis.location.pathname,
    globalThis.location.search,
  );
  if (route !== null) {
    void connectFromRoute(route);
  } else {
    resetGameSession();
    renderLobbyScreen();
    void refreshLobbyGames();
  }
}

main();
