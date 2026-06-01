/**
 * Thin API client for the Tractor game.
 *
 * All game logic lives on the Python backend. This client fetches game
 * state via REST API and delegates rendering to existing UI components.
 */

import type { GameState } from './core/types';
import type { Card } from './core/card';
import { Renderer } from './ui/renderer';
import { SettingsView, type SettingsData } from './ui/settings';
import { Scoreboard } from './ui/scoreboard';

/** Delay between AI actions so the player can see them (ms). */
const AI_ACTION_DELAY = 1500;

/** Delay after a completed trick before clearing (ms). */
const TRICK_CLEAR_DELAY = 3000;

// ---- State ----

let gameId: string | null = null;
let currentState: GameState | null = null;
const HUMAN_PLAYER_INDEX = 3;

// ---- Bootstrap ----

const renderer = new Renderer();

SettingsView.init((data: SettingsData) => {
  apiCallRaw('/api/config', {
    apiKey: data.apiKey,
    model: data.model,
    baseUrl: data.baseUrl,
  });
});

// ---- New Game ----

document.getElementById('btn-new-game')?.addEventListener('click', () => {
  createGameAndDeal();
});

// ---- Play Button ----

renderer.handView.onPlayAction(async (cards: Card[]) => {
  if (!currentState || cards.length === 0) {
    Scoreboard.log('你必须出牌');
    return;
  }
  const cardIds = cards.map(c => c.id);
  try {
    const r = await apiCall('/play', { playerIndex: HUMAN_PLAYER_INDEX, cardIds });
    handleResponse(r);
  } catch (err) {
    Scoreboard.log('出牌失败，请选择合法的牌型');
    renderer.handView.clearSelection();
    console.error(err);
  }
});

// ---- API Layer ----

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

/** Raw POST to non-game endpoints. */
async function apiCallRaw(path: string, body: unknown): Promise<void> {
  await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

interface GameStateResponse {
  gameId: string;
  state: GameState;
  awaitingAction: string | null;
  legalActions?: { type: string; cards: string[] }[] | null;
  validBidLevels?: string[] | null;
}

// ---- Response Handler ----

function handleResponse(resp: GameStateResponse): void {
  gameId = resp.gameId;
  currentState = resp.state;
  renderer.render(currentState);

  if (!resp.awaitingAction) {
    // Round complete or no pending action
    return;
  }

  switch (resp.awaitingAction) {
    case 'bid':
      handleBid(resp);
      break;
    case 'set_trump':
      handleSetTrump(resp);
      break;
    case 'stir':
      handleStir(resp);
      break;
    case 'discard':
      handleDiscard(resp);
      break;
    case 'play':
      handlePlay(resp);
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
    default:
      Scoreboard.log(`未知操作: ${resp.awaitingAction}`);
  }
}

// ---- Action Handlers ----

async function handleBid(resp: GameStateResponse): Promise<void> {
  if (!currentState) return;

  // AI turn: auto-bid (pass 60% of the time, otherwise bid highest)
  if (currentState.currentPlayerIndex !== HUMAN_PLAYER_INDEX) {
    const playerIndex = currentState.currentPlayerIndex;
    renderer.showThinking(playerIndex);
    await new Promise(resolve => setTimeout(resolve, AI_ACTION_DELAY));

    const validLevels = resp.validBidLevels ?? [];
    const pass = validLevels.length === 0 || Math.random() < 0.6;
    const level = pass ? null : validLevels[validLevels.length - 1];

    try {
      const r = await apiCall('/bid', { playerIndex, level, pass });
      renderer.hideThinking(playerIndex);
      handleResponse(r);
    } catch (err) {
      renderer.hideThinking(playerIndex);
      console.error('AI bid error:', err);
    }
    return;
  }

  // Human turn
  const validLevels = resp.validBidLevels ?? [];
  const canPass = true;

  renderer.showBidding(validLevels as any, canPass, async (level: any, pass: boolean) => {
    try {
      const r = await apiCall('/bid', {
        playerIndex: HUMAN_PLAYER_INDEX,
        level: level ?? null,
        pass,
      });
      handleResponse(r);
    } catch (err) {
      Scoreboard.log('叫牌失败');
      console.error(err);
    }
  });
}

async function handleSetTrump(_resp: GameStateResponse): Promise<void> {
  if (!currentState) return;

  // AI turn: pick suit with most cards in hand
  if (currentState.currentPlayerIndex !== HUMAN_PLAYER_INDEX) {
    const playerIndex = currentState.currentPlayerIndex;
    renderer.showThinking(playerIndex);
    await new Promise(resolve => setTimeout(resolve, AI_ACTION_DELAY));

    const player = currentState.players[playerIndex];
    const suitCounts: Record<string, number> = {};
    for (const c of player.hand) {
      if (!c.isJoker && c.rank !== currentState.trumpRank) {
        suitCounts[c.suit] = (suitCounts[c.suit] ?? 0) + 1;
      }
    }
    let bestSuit = 'hearts';
    let bestCount = 0;
    for (const [suit, count] of Object.entries(suitCounts)) {
      if (count > bestCount) {
        bestCount = count;
        bestSuit = suit;
      }
    }

    try {
      const r = await apiCall('/set-trump', { playerIndex, trumpSuit: bestSuit });
      renderer.hideThinking(playerIndex);
      handleResponse(r);
    } catch (err) {
      renderer.hideThinking(playerIndex);
      console.error('AI set-trump error:', err);
    }
    return;
  }

  // Human turn
  renderer.showTrumpSelection(async (trumpSuit: any) => {
    try {
      const r = await apiCall('/set-trump', {
        playerIndex: HUMAN_PLAYER_INDEX,
        trumpSuit,
      });
      handleResponse(r);
    } catch (err) {
      Scoreboard.log('设置主牌失败');
      console.error(err);
    }
  });
}

async function handleStir(_resp: GameStateResponse): Promise<void> {
  if (!currentState) return;

  // AI turn: pass 70% of the time
  if (currentState.currentPlayerIndex !== HUMAN_PLAYER_INDEX) {
    const playerIndex = currentState.currentPlayerIndex;
    renderer.showThinking(playerIndex);
    await new Promise(resolve => setTimeout(resolve, AI_ACTION_DELAY));

    const pass = Math.random() < 0.7;
    try {
      let r;
      if (pass) {
        r = await apiCall('/stir', { playerIndex, pass: true });
      } else {
        const suits = ['hearts', 'spades', 'diamonds', 'clubs'];
        const otherSuits = suits.filter(s => s !== currentState!.trumpSuit);
        const newSuit = otherSuits[Math.floor(Math.random() * otherSuits.length)];
        r = await apiCall('/stir', {
          playerIndex,
          pass: false,
          newTrumpSuit: newSuit,
          level: currentState!.trumpRank,
        });
      }
      renderer.hideThinking(playerIndex);
      handleResponse(r);
    } catch (err) {
      renderer.hideThinking(playerIndex);
      console.error('AI stir error:', err);
    }
    return;
  }

  // Human turn
  renderer.showStirring(true, async (trumpSuit: any) => {
    try {
      const r = trumpSuit === null
        ? await apiCall('/stir', { playerIndex: HUMAN_PLAYER_INDEX, pass: true })
        : await apiCall('/stir', {
            playerIndex: HUMAN_PLAYER_INDEX,
            pass: false,
            newTrumpSuit: trumpSuit,
            level: currentState!.trumpRank,
          });
      handleResponse(r);
    } catch (err) {
      Scoreboard.log('炒地皮失败');
      console.error(err);
    }
  });
}

async function handleDiscard(_resp: GameStateResponse): Promise<void> {
  if (!currentState) return;
  const playerIndex = currentState.currentPlayerIndex;
  const player = currentState.players[playerIndex];
  const discardCount = currentState.settings.bottomCardCount;

  if (player.hand.length < discardCount) return;

  // Auto-discard the lowest-value cards (for both human and AI)
  if (playerIndex !== HUMAN_PLAYER_INDEX) {
    renderer.showThinking(playerIndex);
    await new Promise(resolve => setTimeout(resolve, AI_ACTION_DELAY));
  }

  const sorted = [...player.hand].sort((a, b) => b.points - a.points);
  const toDiscard = sorted.slice(-discardCount);
  const cardIds = toDiscard.map(c => c.id);

  try {
    const r = await apiCall('/discard', { playerIndex, cardIds });
    if (playerIndex !== HUMAN_PLAYER_INDEX) {
      renderer.hideThinking(playerIndex);
    }
    handleResponse(r);
  } catch (err) {
    if (playerIndex !== HUMAN_PLAYER_INDEX) {
      renderer.hideThinking(playerIndex);
    }
    Scoreboard.log('弃牌失败');
    console.error(err);
  }
}

async function handlePlay(resp: GameStateResponse): Promise<void> {
  if (!currentState) return;

  // If it's not the human's turn, this is an AI turn -- auto-play
  if (currentState.currentPlayerIndex !== HUMAN_PLAYER_INDEX) {
    await playAITurn(currentState.currentPlayerIndex, resp);
    return;
  }

  // Human turn -- enable card selection via play button
  Scoreboard.log('请选择要出的牌');
}

/** Play an AI turn: wait, then play a random legal action. */
async function playAITurn(playerIndex: number, resp: GameStateResponse): Promise<void> {
  renderer.showThinking(playerIndex);

  await new Promise(resolve => setTimeout(resolve, AI_ACTION_DELAY));

  try {
    // Use legal actions from the server response
    const legalActions = resp.legalActions ?? [];

    if (legalActions.length > 0) {
      // Pick a random legal action
      const action = legalActions[Math.floor(Math.random() * legalActions.length)];
      const r = await apiCall('/play', { playerIndex, cardIds: action.cards });
      renderer.hideThinking(playerIndex);
      handleResponse(r);
    } else {
      // No legal actions returned -- try first card in hand as fallback
      const aiPlayer = currentState?.players[playerIndex];
      if (aiPlayer && aiPlayer.hand.length > 0) {
        const card = aiPlayer.hand[0];
        const r = await apiCall('/play', { playerIndex, cardIds: [card.id] });
        renderer.hideThinking(playerIndex);
        handleResponse(r);
      } else {
        renderer.hideThinking(playerIndex);
      }
    }
  } catch (err) {
    renderer.hideThinking(playerIndex);
    console.error('AI play error:', err);
  }
}

function handleClearTrick(): void {
  setTimeout(() => {
    apiCall('/clear-trick')
      .then(handleResponse)
      .catch(err => {
        console.error('Clear trick error:', err);
      });
  }, TRICK_CLEAR_DELAY);
}

function handleNextRound(_resp: GameStateResponse): void {
  if (!currentState) return;

  // Calculate scoring display
  const defenderPts = currentState.defenderPoints;
  let message: string;
  let levelChange: number;

  if (defenderPts < 40) {
    message = '庄家方升级！';
    levelChange = 2;
  } else if (defenderPts < 80) {
    message = '庄家方升级！';
    levelChange = 1;
  } else if (defenderPts < 120) {
    message = '换庄！';
    levelChange = 0;
  } else if (defenderPts < 160) {
    message = '防守方升级！';
    levelChange = -1;
  } else if (defenderPts < 200) {
    message = '防守方升级！';
    levelChange = -2;
  } else {
    message = '防守方大升级！';
    levelChange = -3;
  }

  const details = `
    防守方得分: ${defenderPts}<br>
    庄家方变化: ${levelChange > 0 ? '+' + levelChange : levelChange} 级
  `;

  renderer.showScoring(message, details, async () => {
    try {
      const r = await apiCall('/next-round');
      handleResponse(r);
    } catch (err) {
      console.error('Next round error:', err);
    }
  });
}

function handleGameOver(_resp: GameStateResponse): void {
  if (!currentState) return;
  const winningTeam = currentState.teams[0].currentLevel > currentState.teams[1].currentLevel ? 0 : 1;
  renderer.showGameOver(winningTeam);
}

// ---- Create Game and Deal ----

async function createGameAndDeal(): Promise<void> {
  try {
    // Create game
    const createResp = await fetch('/api/game', { method: 'POST' });
    if (!createResp.ok) throw new Error('Failed to create game');
    const created: GameStateResponse = await createResp.json();
    gameId = created.gameId;

    // Deal cards
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
