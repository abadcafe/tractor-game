import { StateManager } from "./core/state.ts";
import type { ServerMessage, ClientAction, InteractionMode, ActionCallbacks } from "./core/types.ts";
import { createGame } from "./net/rest-client.ts";
import { WsClient } from "./net/ws-client.ts";
import { GameLoop } from "./engine/game-loop.ts";
import { render } from "./ui/renderer.ts";
import { validatePlay, validateDiscard, validateBidCards } from "./engine/input-validator.ts";
import { showErrorToast } from "./ui/error-toast.ts";
import { HUMAN_PLAYER_INDEX } from "./config.ts";

async function main() {
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

  // Re-render helper: reads current state and renders with current callbacks/selections
  function reRender() {
    const snap = stateManager.get();
    if (snap) {
      render(snap, container, currentInteractionMode, callbacks, selectedCardIds);
    }
  }

  // Action callbacks -- close over selectedCardIds, wsClient, stateManager
  const callbacks: ActionCallbacks = {
    onCardClick(cardId: string) {
      // Toggle selection
      if (selectedCardIds.has(cardId)) {
        selectedCardIds.delete(cardId);
      } else {
        selectedCardIds.add(cardId);
      }
      // Re-render to show updated selection
      reRender();
    },

    onAction(action: string) {
      const snap = stateManager.get();
      if (!snap) return;

      if (action === "play") {
        const selectedCards = snap.player_hand.filter((c) => selectedCardIds.has(c.id));
        const playAction = validatePlay(selectedCards, snap.legal_actions);
        if (playAction) {
          wsClient.send({ type: "play", cards: playAction.cards.map((c) => c.id) });
          selectedCardIds.clear();
        } else {
          showErrorToast("无效的出牌组合", container);
        }
      } else if (action === "discard") {
        const selectedCards = snap.player_hand.filter((c) => selectedCardIds.has(c.id));
        const count = snap.exchange_state?.count ?? 0;
        if (validateDiscard(selectedCards, count)) {
          wsClient.send({ type: "discard", cards: selectedCards.map((c) => c.id) });
          selectedCardIds.clear();
        } else {
          showErrorToast(`请选择 ${count} 张牌弃掉`, container);
        }
      } else if (action === "next_round") {
        wsClient.send({ type: "next_round" });
        selectedCardIds.clear();
      }
    },

    onBid(cardIds: string[]) {
      const snap = stateManager.get();
      if (!snap) return;
      const selectedCards = snap.player_hand.filter((c) => cardIds.includes(c.id));
      if (validateBidCards(selectedCards, snap.trump_rank)) {
        wsClient.send({ type: "bid", cards: cardIds });
        selectedCardIds.clear();
        reRender();
      } else {
        showErrorToast("叫牌牌张无效", container);
      }
    },

    onStir(cardIds: string[]) {
      const snap = stateManager.get();
      if (!snap) return;
      const selectedCards = snap.player_hand.filter((c) => cardIds.includes(c.id));
      if (validateBidCards(selectedCards, snap.trump_rank)) {
        wsClient.send({ type: "stir", cards: cardIds });
        selectedCardIds.clear();
        reRender();
      } else {
        showErrorToast("反主牌张无效", container);
      }
    },

    onPass() {
      wsClient.send({ type: "stir", pass: true });
      selectedCardIds.clear();
      reRender();
    },

    onNewGame() {
      selectedCardIds.clear();
      stateManager.reset();
      container.innerHTML = "";
      startNewGame();
    },
  };

  // GameLoop with renderFn that injects callbacks + selectedCardIds
  const gameLoop = new GameLoop(
    stateManager,
    (snapshot, containerEl, interactionMode) => {
      currentInteractionMode = interactionMode;
      render(snapshot, containerEl, interactionMode, callbacks, selectedCardIds);
    },
    container,
  );

  // Register message handler BEFORE connecting (per spec: first state msg arrives immediately)
  wsClient.onMessage((msg: ServerMessage) => {
    gameLoop.handleMessage(msg);
  });

  wsClient.onDisconnect(() => {
    console.log("WebSocket disconnected");
  });

  // Start game flow
  async function startNewGame() {
    try {
      wsClient.disconnect(); // Cancel any pending reconnection timers from previous game
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
