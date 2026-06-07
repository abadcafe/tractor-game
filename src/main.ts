/**
 * Thin API client for the Tractor game.
 *
 * All game logic (including AI decisions) lives on the Python backend.
 * This client fetches game state via REST API and delegates rendering
 * to existing UI components. The server handles AI turns automatically
 * and only returns awaitingAction when the human player needs to act.
 */

import type { Card } from './core/card';
import { Renderer } from './ui/renderer';
import { Scoreboard } from './ui/scoreboard';

// ---- Global error handler ----

(window as any).__jsErrors = [];
window.addEventListener('error', (event) => {
  (window as any).__jsErrors.push({
    message: event.message,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno,
    timestamp: new Date().toISOString(),
  });
});
window.addEventListener('unhandledrejection', (event) => {
  (window as any).__jsErrors.push({
    message: String(event.reason),
    timestamp: new Date().toISOString(),
  });
});

// ---- Constants ----

const HUMAN_PLAYER_INDEX = 3;
const TRICK_CLEAR_DELAY = 3000;

// ---- State ----

let gameId: string | null = null;
let currentState: any = null;

// ---- Bootstrap ----

const renderer = new Renderer();

// ---- New Game Button ----

document.getElementById('btn-new-game')?.addEventListener('click', () => {
  createGameAndDeal();
});

// ---- Card Play/Discard Button ----

renderer.handView.onPlayAction(async (cards: Card[]) => {
  if (!currentState || cards.length === 0) return;
  const cardIds = cards.map((c: Card) => c.id);
  const action = currentState.phase === 'exchange' ? '/discard' : '/play';
  try {
    const r = await apiCall(action, { playerIndex: HUMAN_PLAYER_INDEX, cardIds });
    handleResponse(r);
  } catch (err) {
    Scoreboard.log('操作失败，请重试');
    renderer.handView.clearSelection();
    console.error(err);
  }
});

// ---- API Layer ----

interface GameStateResponse {
  gameId: string;
  state: any;
  awaitingAction: string | null;
  legalActions?: { type: string; cards: string[] }[] | null;
  validBidLevels?: string[] | null;
  scoringMessage?: string | null;
  scoringDetails?: string | null;
  winningTeam?: number | null;
}

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(url: string, opts: RequestInit, retryLabel: string): Promise<Response> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const resp = await fetch(url, opts);
      return resp;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (err instanceof TypeError && attempt < MAX_RETRIES) {
        Scoreboard.log(`${retryLabel}网络错误，正在重试 (${attempt + 1}/${MAX_RETRIES})...`);
        await sleep(RETRY_DELAY_MS);
        continue;
      }
      if (err instanceof TypeError && attempt >= MAX_RETRIES) {
        Scoreboard.log('Connection lost. Please refresh the page.');
      }
      throw lastError;
    }
  }
  throw lastError ?? new Error('Unknown error');
}

async function apiCall(action: string, body?: Record<string, unknown>): Promise<GameStateResponse> {
  if (!gameId) throw new Error('No active game');
  const url = `/api/game/${gameId}${action}`;
  const opts: RequestInit = body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : { method: 'POST' };

  const resp = await fetchWithRetry(url, opts, '操作');
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? 'API error');
  }
  return resp.json();
}

// ---- Response Handler ----

function handleResponse(resp: GameStateResponse): void {
  gameId = resp.gameId;
  currentState = resp.state;
  renderer.render(currentState);

  if (!resp.awaitingAction) return;

  const humanActions = new Set(['bid', 'set_trump', 'stir', 'discard', 'play']);
  if (humanActions.has(resp.awaitingAction) && currentState.currentPlayerIndex !== HUMAN_PLAYER_INDEX) {
    Scoreboard.log('等待 AI 操作...');
    return;
  }

  switch (resp.awaitingAction) {
    case 'bid':
      handleBid(resp);
      break;
    case 'set_trump':
      handleSetTrump();
      break;
    case 'stir':
      handleStir();
      break;
    case 'discard':
      handleDiscard();
      break;
    case 'play':
      handlePlay();
      break;
    case 'clear_trick':
      handleClearTrick();
      break;
    case 'next_round':
      handleNextRound(resp);
      break;
    case 'game_over':
      handleGameOver(resp);
      break;
  }
}

// ---- Action Handlers (human input only, AI handled server-side) ----

function handleBid(resp: GameStateResponse): void {
  // 亮牌规则：显示玩家的级牌，让玩家选择亮牌
  const hand = currentState?.players[HUMAN_PLAYER_INDEX]?.hand ?? [];
  const trumpRank = currentState?.trumpRank ?? '2';

  // 找出级牌（当前级别的牌）
  const dominantCards = hand.filter((c: any) => c.rank === trumpRank);

  if (dominantCards.length === 0) {
    Scoreboard.log('没有级牌，跳过叫牌');
    apiCall('/bid', { playerIndex: HUMAN_PLAYER_INDEX, pass: true })
      .then(handleResponse)
      .catch((err) => {
        Scoreboard.log('自动跳过叫牌失败');
        console.error(err);
      });
    return;
  }

  // 显示级牌按钮
  renderer.showBidding(dominantCards as any, true, async (selectedCards: any, pass: boolean) => {
    try {
      if (pass) {
        const r = await apiCall('/bid', { playerIndex: HUMAN_PLAYER_INDEX, pass: true });
        handleResponse(r);
      } else {
        // 亮牌：使用第一张牌的花色作为主牌花色
        const cardIds = selectedCards.map((c: any) => c.id);
        const r = await apiCall('/bid', {
          playerIndex: HUMAN_PLAYER_INDEX,
          cardIds: cardIds,
          pass: false
        });
        handleResponse(r);
      }
    } catch (err) {
      Scoreboard.log('叫牌失败');
      console.error(err);
    }
  });
}

function handleSetTrump(): void {
  const winnerIndex = getBidWinnerIndex();
  if (winnerIndex !== HUMAN_PLAYER_INDEX) {
    Scoreboard.log('等待庄家选择主牌...');
    return;
  }

  renderer.showTrumpSelection(async (trumpSuit: any) => {
    try {
      const r = await apiCall('/set-trump', { playerIndex: HUMAN_PLAYER_INDEX, trumpSuit });
      handleResponse(r);
    } catch (err) {
      Scoreboard.log('设置主牌失败');
      console.error(err);
    }
  });
}

function getBidWinnerIndex(): number | null {
  const bids = currentState?.biddingHistory ?? [];
  for (let i = bids.length - 1; i >= 0; i--) {
    if (!bids[i].pass) {
      return bids[i].playerIndex;
    }
  }
  return null;
}

function handleStir(): void {
  renderer.showStirring(true, async (trumpSuit: any) => {
    try {
      const r = trumpSuit === null
        ? await apiCall('/stir', { playerIndex: HUMAN_PLAYER_INDEX, pass: true })
        : await apiCall('/stir', { playerIndex: HUMAN_PLAYER_INDEX, pass: false, newTrumpSuit: trumpSuit, level: currentState.trumpRank });
      handleResponse(r);
    } catch (err) {
      Scoreboard.log('炒地皮失败');
      console.error(err);
    }
  });
}

function handleDiscard(): void {
  Scoreboard.log('请选择要弃掉的牌');
}

function handlePlay(): void {
  Scoreboard.log('请选择要出的牌');
}

function handleClearTrick(): void {
  setTimeout(() => {
    apiCall('/clear-trick')
      .then(handleResponse)
      .catch(err => console.error('Clear trick error:', err));
  }, TRICK_CLEAR_DELAY);
}

function handleNextRound(resp: GameStateResponse): void {
  const message = resp.scoringMessage ?? '回合结束';
  const details = resp.scoringDetails ?? '';
  renderer.showScoring(message, details, async () => {
    try {
      const r = await apiCall('/next-round');
      handleResponse(r);
    } catch (err) {
      console.error('Next round error:', err);
    }
  });
}

function handleGameOver(resp: GameStateResponse): void {
  const winningTeam = resp.winningTeam ?? 0;
  renderer.showGameOver(winningTeam);
}

// ---- Create Game and Deal ----

async function createGameAndDeal(): Promise<void> {
  try {
    const createResp = await fetchWithRetry('/api/game', { method: 'POST' }, '创建游戏');
    if (!createResp.ok) throw new Error('Failed to create game');
    const created: GameStateResponse = await createResp.json();
    gameId = created.gameId;
    const dealResp = await apiCall('/deal');
    handleResponse(dealResp);
  } catch (err) {
    console.error('Create game error:', err);
    Scoreboard.log('创建游戏失败，请检查服务器是否运行');
  }
}

// ---- Auto-start ----

window.addEventListener('DOMContentLoaded', () => {
  createGameAndDeal();
});
