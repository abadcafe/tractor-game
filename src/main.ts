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
import { SettingsView, type SettingsData } from './ui/settings';
import { Scoreboard } from './ui/scoreboard';

// ---- Constants ----

const HUMAN_PLAYER_INDEX = 3;
const TRICK_CLEAR_DELAY = 3000;

// ---- State ----

let gameId: string | null = null;
let currentState: any = null;

// ---- Bootstrap ----

const renderer = new Renderer();

SettingsView.init((data: SettingsData) => {
  apiCallRaw('/api/config', {
    apiKey: data.apiKey,
    model: data.model,
    baseUrl: data.baseUrl,
  });
});

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
}

async function apiCall(action: string, body?: Record<string, unknown>): Promise<GameStateResponse> {
  if (!gameId) throw new Error('No active game');
  const url = `/api/game/${gameId}${action}`;
  const opts: RequestInit = body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : { method: 'POST' };
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? 'API error');
  }
  return resp.json();
}

async function apiCallRaw(path: string, body: unknown): Promise<void> {
  const resp = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    console.error(`Config API error: ${resp.status} ${resp.statusText}`);
  }
}

// ---- Response Handler ----

function handleResponse(resp: GameStateResponse): void {
  gameId = resp.gameId;
  currentState = resp.state;
  renderer.render(currentState);

  if (!resp.awaitingAction) return;

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
      handleGameOver();
      break;
  }
}

// ---- Action Handlers (human input only, AI handled server-side) ----

function handleBid(resp: GameStateResponse): void {
  const validLevels = resp.validBidLevels ?? [];
  renderer.showBidding(validLevels as any, true, async (level: any, pass: boolean) => {
    try {
      const r = await apiCall('/bid', { playerIndex: HUMAN_PLAYER_INDEX, level, pass });
      handleResponse(r);
    } catch (err) {
      Scoreboard.log('叫牌失败');
      console.error(err);
    }
  });
}

function handleSetTrump(): void {
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

function handleStir(): void {
  renderer.showStirring(true, async (trumpSuit: any) => {
    try {
      const r = trumpSuit === null
        ? await apiCall('/stir', { playerIndex: HUMAN_PLAYER_INDEX, pass: true })
        : await apiCall('/stir', { playerIndex: HUMAN_PLAYER_INDEX, pass: false, newTrumpSuit: trumpSuit });
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

function handleGameOver(): void {
  const winningTeam = currentState.teams[0].currentLevel > currentState.teams[1].currentLevel ? 0 : 1;
  renderer.showGameOver(winningTeam);
}

// ---- Create Game and Deal ----

async function createGameAndDeal(): Promise<void> {
  try {
    const createResp = await fetch('/api/game', { method: 'POST' });
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
