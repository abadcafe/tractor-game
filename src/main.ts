/**
 * Main entry point — Game controller that orchestrates
 * the engine, UI, and AI client.
 */

import { Phase, Suit, type PlayAction, type GameState, type CompletedTrick } from './core/types';
import type { Card } from './core/card';
import { Game } from './engine/game';
import { Renderer } from './ui/renderer';
import { AIClient } from './ai/client';
import { SettingsView, type SettingsData } from './ui/settings';
import { getLegalPlays } from './rules/validator';
import { getWinningBid, isBiddingOver } from './rules/bidding';
import { HUMAN_PLAYER_INDEX, PLAYER_COUNT } from './core/constants';
import { Scoreboard } from './ui/scoreboard';

/** How long to display a completed trick before clearing (ms). */
const TRICK_DISPLAY_PAUSE = 3000;

/** Track whether we're in the post-trick pause so we don't re-trigger. */
let trickPauseInProgress = false;

// ---- Bootstrap ----

const settings = SettingsView.load();

const game = new Game({
  apiKey: settings.apiKey,
  model: settings.model,
  baseUrl: settings.baseUrl,
});

const renderer = new Renderer();

const aiClient = new AIClient({
  baseUrl: '',  // Same origin — Python serves both frontend and API
  apiKey: settings.apiKey,
  model: settings.model,
});

// ---- Settings ----

SettingsView.init((data: SettingsData) => {
  game.updateSettings({
    apiKey: data.apiKey,
    model: data.model,
    baseUrl: data.baseUrl,
  });
  aiClient.updateConfig({
    apiKey: data.apiKey,
    model: data.model,
    baseUrl: data.baseUrl,
  });
});

// ---- New Game ----

document.getElementById('btn-new-game')?.addEventListener('click', () => {
  game.startNewGame();
  game.startRound();
  renderer.render(game.state);
  runGameLoop();
});

// ---- Game State Change Handler ----

game.onChange((state: GameState) => {
  // Check if a trick just completed — we need to pause before continuing.
  // A trick is "just completed" when lastCompletedTrick is set and the
  // currentTrick still has cards (hasn't been cleared yet).
  const trickJustCompleted = state.lastCompletedTrick !== null
    && state.currentTrick.some(s => s.cards !== null);

  if (trickJustCompleted && !trickPauseInProgress) {
    trickPauseInProgress = true;

    // Render the completed trick (cards still on table)
    renderer.render(state);
    renderer.showTrickResult(state.lastCompletedTrick!);

    // Pause so the player can see the result, then clear and continue
    setTimeout(() => {
      trickPauseInProgress = false;
      game.clearTrick();  // This triggers another onChange → continueGame()
    }, TRICK_DISPLAY_PAUSE);

    return;  // Don't continue the game yet — wait for the pause
  }

  renderer.render(state);
  continueGame(state);
});

/** Continue game logic after render — trigger the next handler. */
function continueGame(state: GameState): void {
  if (state.phase === Phase.BIDDING && state.currentPlayerIndex === HUMAN_PLAYER_INDEX) {
    handleHumanBidding();
  } else if (state.phase === Phase.STIRRING && state.currentPlayerIndex === HUMAN_PLAYER_INDEX) {
    handleHumanStirring();
  } else if (state.phase === Phase.EXCHANGE && state.currentPlayerIndex === HUMAN_PLAYER_INDEX) {
    handleHumanExchange();
  } else if (state.phase === Phase.PLAYING && state.currentPlayerIndex === HUMAN_PLAYER_INDEX) {
    handleHumanPlay();
  } else if (state.phase === Phase.SCORING) {
    handleScoring();
  } else if (state.phase === Phase.GAME_OVER) {
    const winningTeam = state.teams[0].currentLevel > state.teams[1].currentLevel ? 0 : 1;
    renderer.showGameOver(winningTeam);
  } else if (state.currentPlayerIndex !== HUMAN_PLAYER_INDEX) {
    // AI turn
    handleAITurn();
  }
}

// ---- Game Loop ----

async function runGameLoop(): Promise<void> {
  // The loop is event-driven: onChange callbacks handle each step.
  // Here we just bootstrap the first phase.

  if (game.state.phase === Phase.DEALING) {
    game.startRound();
  }

  // If first player is AI, trigger AI turn
  if (game.state.currentPlayerIndex !== HUMAN_PLAYER_INDEX) {
    await handleAITurn();
  }
}

// ---- Human Interaction Handlers ----

function handleHumanBidding(): void {
  // Check if bidding is already over — if so, don't show panel
  if (isBiddingOver(game.state.biddingHistory, PLAYER_COUNT)) {
    // Bidding over — handle winner
    const winner = getWinningBid(game.state.biddingHistory);
    if (winner && winner.level) {
      if (winner.playerIndex === HUMAN_PLAYER_INDEX) {
        // Human won — show trump selection
        renderer.showTrumpSelection((trumpSuit) => {
          game.setTrumpAfterBid(HUMAN_PLAYER_INDEX, trumpSuit);
        });
      }
      // AI winner is handled in handleAITurn
    } else {
      // No one bid — this shouldn't happen often, redeal
      game.startRound();
    }
    return;
  }

  const validLevels = game.getValidBids();
  const canPass = true; // Can always pass

  renderer.showBidding(validLevels, canPass, (level, pass) => {
    const ok = game.submitBid(HUMAN_PLAYER_INDEX, level, pass);
    if (!ok) {
      Scoreboard.log('无效的叫牌');
    }
  });
}

function handleHumanStirring(): void {
  renderer.showStirring(true, (trumpSuit) => {
    if (trumpSuit === null) {
      // Pass on stirring
      game.submitStir(HUMAN_PLAYER_INDEX, null);
    } else {
      // Create a stir action
      const stir = {
        playerIndex: HUMAN_PLAYER_INDEX,
        newTrumpSuit: trumpSuit,
        level: game.state.trumpRank,
      };
      game.submitStir(HUMAN_PLAYER_INDEX, stir);
    }
  });
}

function handleHumanExchange(): void {
  // Human declarer needs to discard bottom cards
  // For now, auto-discard the lowest cards
  const humanPlayer = game.state.players[HUMAN_PLAYER_INDEX];
  const discardCount = game.state.settings.bottomCardCount;

  if (humanPlayer.hand.length >= discardCount) {
    // Auto-discard the last `discardCount` cards (lowest value)
    const sorted = [...humanPlayer.hand].sort((a, b) => b.points - a.points);
    const toDiscard = sorted.slice(-discardCount);
    game.submitDiscard(HUMAN_PLAYER_INDEX, toDiscard);
  }
}

function handleHumanPlay(): void {
  // Listen for play action
  renderer.handView.onPlayAction((cards: Card[]) => {
    if (cards.length === 0) {
      // Pass (shouldn't normally happen if leading)
      Scoreboard.log('你必须出牌');
      return;
    }

    const ok = game.submitPlay(HUMAN_PLAYER_INDEX, cards);
    if (!ok) {
      Scoreboard.log('无效的出牌，请选择合法的牌型');
      renderer.handView.clearSelection();
    }
  });
}

// ---- AI Turn Handler ----

async function handleAITurn(): Promise<void> {
  const playerIndex = game.state.currentPlayerIndex;
  if (playerIndex === HUMAN_PLAYER_INDEX) return;

  renderer.showThinking(playerIndex);

  try {
    switch (game.state.phase) {
      case Phase.BIDDING:
        await handleAIBidding(playerIndex);
        break;
      case Phase.STIRRING:
        await handleAIStirring(playerIndex);
        break;
      case Phase.EXCHANGE:
        await handleAIExchange(playerIndex);
        break;
      case Phase.PLAYING:
        await handleAIPlay(playerIndex);
        break;
    }
  } finally {
    renderer.hideThinking(playerIndex);
  }
}

async function handleAIBidding(playerIndex: number): Promise<void> {
  // Check if bidding is already over
  if (isBiddingOver(game.state.biddingHistory, PLAYER_COUNT)) {
    const winner = getWinningBid(game.state.biddingHistory);
    if (winner && winner.level && winner.playerIndex !== HUMAN_PLAYER_INDEX) {
      // AI won the bid — auto-select trump
      autoSelectTrumpForAI(winner.playerIndex);
    }
    return;
  }

  const validLevels = game.getValidBids();

  if (validLevels.length === 0) {
    // No higher bid possible — must pass
    game.submitBid(playerIndex, null, true);
    return;
  }

  // Simple AI: pass 60% of the time, otherwise bid highest
  if (Math.random() < 0.6) {
    game.submitBid(playerIndex, null, true);
  } else {
    const level = validLevels[validLevels.length - 1];
    game.submitBid(playerIndex, level, false);
  }

  // After bidding, check if it's over and handle trump selection
  if (game.state.phase === Phase.BIDDING && isBiddingOver(game.state.biddingHistory, PLAYER_COUNT)) {
    const winner = getWinningBid(game.state.biddingHistory);
    if (winner && winner.level) {
      if (winner.playerIndex === HUMAN_PLAYER_INDEX) {
        renderer.showTrumpSelection((trumpSuit) => {
          game.setTrumpAfterBid(HUMAN_PLAYER_INDEX, trumpSuit);
        });
      } else {
        autoSelectTrumpForAI(winner.playerIndex);
      }
    } else {
      // No one bid — redeal
      game.startRound();
    }
  }
}

function autoSelectTrumpForAI(playerIndex: number): void {
  const player = game.state.players[playerIndex];
  const suitCounts = new Map<Suit, number>();
  for (const c of player.hand) {
    if (!c.isJoker && c.rank !== game.state.currentLevel) {
      suitCounts.set(c.suit, (suitCounts.get(c.suit) ?? 0) + 1);
    }
  }
  let bestSuit = Suit.HEARTS;
  let bestCount = 0;
  for (const [suit, count] of suitCounts) {
    if (count > bestCount) {
      bestCount = count;
      bestSuit = suit;
    }
  }
  game.setTrumpAfterBid(playerIndex, bestSuit);
}

async function handleAIStirring(playerIndex: number): Promise<void> {
  // Simple AI stirring: pass 70% of the time
  if (Math.random() < 0.7) {
    game.submitStir(playerIndex, null);
  } else {
    // Stir with a random different suit
    const suits = [Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS];
    const otherSuits = suits.filter(s => s !== game.state.trumpSuit);
    const newSuit = otherSuits[Math.floor(Math.random() * otherSuits.length)];
    game.submitStir(playerIndex, {
      playerIndex,
      newTrumpSuit: newSuit,
      level: game.state.trumpRank,
    });
  }
}

async function handleAIExchange(playerIndex: number): Promise<void> {
  // AI declarer discards lowest-point cards
  const player = game.state.players[playerIndex];
  const discardCount = game.state.settings.bottomCardCount;

  // Sort: keep high-point cards, discard low-point non-trump cards first
  const sorted = [...player.hand].sort((a, b) => {
    // Trump cards keep
    const aIsTrump = a.isJoker || a.rank === game.state.trumpRank || a.suit === game.state.trumpSuit;
    const bIsTrump = b.isJoker || b.rank === game.state.trumpRank || b.suit === game.state.trumpSuit;
    if (aIsTrump && !bIsTrump) return 1;
    if (!aIsTrump && bIsTrump) return -1;
    return b.points - a.points;
  });

  const toDiscard = sorted.slice(-discardCount);
  game.submitDiscard(playerIndex, toDiscard);
}

async function handleAIPlay(playerIndex: number): Promise<void> {
  const player = game.state.players[playerIndex];

  // Get legal plays
  const isLeading = game.state.currentTrick.every(s => s.cards === null);
  let leadAction: PlayAction | null = null;
  if (!isLeading) {
    const leadSlot = game.state.currentTrick.find(s => s.cards !== null);
    if (leadSlot && leadSlot.cards) {
      leadAction = {
        type: game.state.leadPlayType ?? 'single' as any,
        cards: leadSlot.cards,
      };
    }
  }

  const legalPlays = getLegalPlays(
    player.hand,
    game.state.currentTrick,
    game.state.trumpSuit!,
    game.state.trumpRank,
    isLeading,
    leadAction,
  );

  if (legalPlays.length === 0) {
    // No legal plays — should not happen, but fallback
    game.submitPlay(playerIndex, [player.hand[0]]);
    return;
  }

  try {
    const legalDescriptions = legalPlays.map((p, i) =>
      `${i}: ${describePlaySimple(p)}`
    );

    const result = await aiClient.decide(
      playerIndex,
      'playing',
      game.state,
      legalDescriptions,
      player.hand,
    );

    // Try to match the AI's response to a legal play
    if (result.cardIds.length > 0) {
      const chosenCards = player.hand.filter(c => result.cardIds.includes(c.id));
      if (chosenCards.length > 0) {
        // Find which legal play matches these cards
        for (const play of legalPlays) {
          const playIds = new Set(play.cards.map(c => c.id));
          const chosenIds = new Set(chosenCards.map(c => c.id));
          if (playIds.size === chosenIds.size &&
              [...playIds].every(id => chosenIds.has(id))) {
            game.submitPlay(playerIndex, play.cards);
            return;
          }
        }
      }
    }

    // Fallback: play a random legal action
    const randomPlay = legalPlays[Math.floor(Math.random() * legalPlays.length)];
    game.submitPlay(playerIndex, randomPlay.cards);
  } catch {
    // Fallback on error
    const randomPlay = legalPlays[Math.floor(Math.random() * legalPlays.length)];
    game.submitPlay(playerIndex, randomPlay.cards);
  }
}

// ---- Scoring ----

function handleScoring(): void {
  try {
    const result = game.calculateRoundScore();
    const details = `
      防守方得分: ${result.defenderPoints}<br>
      扣底加分: ${result.bottomCardBonus}<br>
      总分: ${result.totalDefenderPoints}<br>
      庄家方变化: ${result.declarerLevelChange > 0 ? '+' + result.declarerLevelChange : result.declarerLevelChange} 级<br>
      下一局庄家: ${result.nextDeclarerTeam === 0 ? '我方' : '对方'}
    `;

    const message = result.declarerLevelChange > 0
      ? '庄家方升级！'
      : result.declarerLevelChange < 0
        ? '防守方升级！'
        : '换庄！';

    renderer.showScoring(message, details, () => {
      game.nextRound();
      // If next round, deal new cards
      if (game.state.phase === Phase.DEALING) {
        game.startRound();
      }
      runGameLoop();
    });
  } catch (err) {
    console.error('Scoring error:', err);
    Scoreboard.log('计分出错');
  }
}

// ---- Helpers ----

function describePlaySimple(play: PlayAction): string {
  const cards = play.cards.map(c => {
    if (c.isJoker) return c.isBigJoker ? '大王' : '小王';
    const s: Record<string, string> = { hearts: '♥', spades: '♠', diamonds: '♦', clubs: '♣' };
    return `${s[c.suit] ?? c.suit}${c.rank}`;
  }).join(' ');
  return `${play.type}: ${cards}`;
}

// ---- Start ----

// Auto-start or restore game
window.addEventListener('DOMContentLoaded', () => {
  // If there's saved state, restore it; otherwise start fresh
  if (game.state.phase !== Phase.DEALING || game.state.players[0].hand.length > 0) {
    // Restored state — just render and resume
    renderer.render(game.state);
    // Re-trigger the appropriate handler for the current phase
    if (game.state.phase === Phase.BIDDING || game.state.phase === Phase.STIRRING) {
      if (game.state.currentPlayerIndex === HUMAN_PLAYER_INDEX) {
        if (game.state.phase === Phase.BIDDING) handleHumanBidding();
        else handleHumanStirring();
      } else {
        handleAITurn();
      }
    } else if (game.state.phase === Phase.EXCHANGE && game.state.currentPlayerIndex === HUMAN_PLAYER_INDEX) {
      handleHumanExchange();
    } else if (game.state.phase === Phase.PLAYING) {
      if (game.state.currentPlayerIndex === HUMAN_PLAYER_INDEX) {
        handleHumanPlay();
      } else {
        handleAITurn();
      }
    } else if (game.state.phase === Phase.SCORING) {
      handleScoring();
    }
  } else {
    // Fresh game
    game.startRound();
    renderer.render(game.state);
    runGameLoop();
  }
});
