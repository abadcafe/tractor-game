/**
 * Game state management — immutable state updates.
 */

import { Card, Suit, Rank, createDecks } from '../core/card';
import { Phase, PlayType, type PlayAction, type GameState, type PlayerState, type TeamState, type TrickSlot, type CompletedTrick, type BidAction, type StirAction } from '../core/types';
import { START_LEVEL, PLAYER_COUNT, BOTTOM_CARD_COUNT, DEFAULT_SETTINGS } from '../core/constants';
import { getTeamIndex, nextPlayer } from '../core/constants';
import type { GameSettings } from '../core/types';

/**
 * Create initial game state for a new round.
 */
export function createInitialState(settings?: Partial<GameSettings>): GameState {
  const mergedSettings: GameSettings = {
    ...DEFAULT_SETTINGS,
    ...settings,
    apiKey: settings?.apiKey ?? '',
    model: settings?.model ?? DEFAULT_SETTINGS.model,
    baseUrl: settings?.baseUrl ?? DEFAULT_SETTINGS.baseUrl,
    targetLevel: settings?.targetLevel ?? DEFAULT_SETTINGS.targetLevel,
    bottomCardCount: settings?.bottomCardCount ?? BOTTOM_CARD_COUNT,
  };

  const players: PlayerState[] = [
    { index: 0, name: '同伴 (AI)', hand: [], teamIndex: 0, isHuman: false, isDeclarer: false },
    { index: 1, name: '对手A (AI)', hand: [], teamIndex: 1, isHuman: false, isDeclarer: false },
    { index: 2, name: '对手B (AI)', hand: [], teamIndex: 1, isHuman: false, isDeclarer: false },
    { index: 3, name: '你', hand: [], teamIndex: 0, isHuman: true, isDeclarer: false },
  ];

  const teams: [TeamState, TeamState] = [
    { index: 0, tricks: [], currentLevel: START_LEVEL },
    { index: 1, tricks: [], currentLevel: START_LEVEL },
  ];

  const emptyTrick: TrickSlot[] = [
    { playerIndex: 0, cards: null },
    { playerIndex: 1, cards: null },
    { playerIndex: 2, cards: null },
    { playerIndex: 3, cards: null },
  ];

  return {
    phase: Phase.DEALING,
    currentLevel: START_LEVEL,
    players,
    teams,
    currentPlayerIndex: 0,
    trumpSuit: null,
    trumpRank: START_LEVEL,
    declarerTeamIndex: 0,
    currentTrick: emptyTrick,
    leadPlayerIndex: 0,
    leadPlayType: null,
    bottomCards: [],
    trickHistory: [],
    lastCompletedTrick: null,
    biddingHistory: [],
    stirHistory: [],
    defenderPoints: 0,
    settings: mergedSettings,
  };
}

/**
 * Deal cards to all players and set bottom cards.
 */
export function dealCards(state: GameState): GameState {
  const deck = createDecks();
  const shuffled = shuffle(deck);

  const bottomCards = shuffled.slice(0, state.settings.bottomCardCount);
  const playerCards = shuffled.slice(state.settings.bottomCardCount);

  const cardsPerPlayer = playerCards.length / PLAYER_COUNT;
  const newPlayers = state.players.map((p, i) => ({
    ...p,
    hand: playerCards.slice(i * cardsPerPlayer, (i + 1) * cardsPerPlayer),
  }));

  return {
    ...state,
    players: newPlayers,
    bottomCards,
    phase: Phase.BIDDING,
    currentPlayerIndex: 0, // Start bidding from player 0
    biddingHistory: [],
    stirHistory: [],
    trickHistory: [],
    lastCompletedTrick: null,
    defenderPoints: 0,
    trumpSuit: null,
    leadPlayType: null,
    currentTrick: state.currentTrick.map(s => ({ ...s, cards: null })),
  };
}

/**
 * Record a bid.
 */
export function recordBid(state: GameState, bid: BidAction): GameState {
  return {
    ...state,
    biddingHistory: [...state.biddingHistory, bid],
    currentPlayerIndex: nextPlayer(bid.playerIndex),
  };
}

/**
 * Set the winning bidder as declarer, set trump suit.
 */
export function setDeclarer(state: GameState, playerIndex: number, trumpSuit: Suit, trumpRank: Rank): GameState {
  const teamIndex = getTeamIndex(playerIndex);
  const newPlayers = state.players.map(p => ({
    ...p,
    isDeclarer: p.teamIndex === teamIndex,
  }));

  return {
    ...state,
    trumpSuit,
    trumpRank,
    declarerTeamIndex: teamIndex,
    players: newPlayers,
    phase: Phase.STIRRING,
    // After initial bidding, move to stirring phase
    // Stirring starts from the next player clockwise after the bid winner
    currentPlayerIndex: nextPlayer(playerIndex),
  };
}

/**
 * Record a stir action.
 */
export function recordStir(state: GameState, stir: StirAction): GameState {
  return {
    ...state,
    trumpSuit: stir.newTrumpSuit,
    trumpRank: stir.level,
    declarerTeamIndex: getTeamIndex(stir.playerIndex),
    stirHistory: [...state.stirHistory, stir],
    players: state.players.map(p => ({
      ...p,
      isDeclarer: p.teamIndex === getTeamIndex(stir.playerIndex),
    })),
    currentPlayerIndex: nextPlayer(stir.playerIndex),
  };
}

/**
 * Start the exchange phase: declarer picks up bottom cards.
 */
export function pickupBottomCards(state: GameState): GameState {
  const declarerPlayer = state.players.find(p =>
    p.teamIndex === state.declarerTeamIndex && p.isDeclarer
  );
  if (!declarerPlayer) return state;

  const newPlayers = state.players.map(p => {
    if (p.index === declarerPlayer.index) {
      return {
        ...p,
        hand: [...p.hand, ...state.bottomCards],
      };
    }
    return p;
  });

  return {
    ...state,
    players: newPlayers,
    phase: Phase.EXCHANGE,
    currentPlayerIndex: declarerPlayer.index,
  };
}

/**
 * Declarer discards cards (equal to bottom card count).
 */
export function discardCards(state: GameState, playerIndex: number, cards: Card[]): GameState {
  const newPlayers = state.players.map(p => {
    if (p.index !== playerIndex) return p;
    const discardIds = new Set(cards.map(c => c.id));
    return {
      ...p,
      hand: p.hand.filter(c => !discardIds.has(c.id)),
    };
  });

  // Start playing phase: declarer leads first trick
  return {
    ...state,
    players: newPlayers,
    bottomCards: cards, // Discarded cards become new bottom cards
    phase: Phase.PLAYING,
    currentPlayerIndex: playerIndex,
    leadPlayerIndex: playerIndex,
    leadPlayType: null,
    currentTrick: state.currentTrick.map(s => ({ ...s, cards: null })),
  };
}

/**
 * Play cards to the current trick.
 */
export function playCards(state: GameState, playerIndex: number, action: PlayAction): GameState {
  const newTrick = state.currentTrick.map(s => {
    if (s.playerIndex === playerIndex) {
      return { ...s, cards: action.cards };
    }
    return s;
  });

  const isLead = state.currentTrick.every(s => s.cards === null);

  let newState: GameState = {
    ...state,
    currentTrick: newTrick,
    leadPlayType: isLead ? action.type : state.leadPlayType,
    leadPlayerIndex: isLead ? playerIndex : state.leadPlayerIndex,
  };

  // Remove played cards from hand
  const playedIds = new Set(action.cards.map(c => c.id));
  newState = {
    ...newState,
    players: newState.players.map(p => {
      if (p.index !== playerIndex) return p;
      return { ...p, hand: p.hand.filter(c => !playedIds.has(c.id)) };
    }),
  };

  // Check if trick is complete (all 4 played)
  const allPlayed = newState.currentTrick.every(s => s.cards !== null);
  if (allPlayed) {
    newState = resolveTrick(newState);
  } else {
    // Next player
    newState = {
      ...newState,
      currentPlayerIndex: nextPlayer(playerIndex),
    };
  }

  return newState;
}

/**
 * Resolve a completed trick: determine winner, collect points, start next trick.
 */
export function resolveTrick(state: GameState): GameState {
  // TODO: Use comparator to determine winner
  // For now, simplified: lead player wins if no trump played
  // Full implementation will use comparePlays from comparator.ts

  const trick = state.currentTrick as { playerIndex: number; cards: Card[] }[];

  // Simplified winner determination — to be enhanced with full comparison
  let winnerIndex = state.leadPlayerIndex;

  // Count points in this trick
  let trickPoints = 0;
  for (const slot of trick) {
    for (const card of slot.cards) {
      trickPoints += card.points;
    }
  }

  // Determine which team won
  const winnerTeamIndex = getTeamIndex(winnerIndex);
  const defenderTeamIndex = state.declarerTeamIndex === 0 ? 1 : 0;

  // If defender wins, they collect the points
  const newDefenderPoints = winnerTeamIndex === defenderTeamIndex
    ? state.defenderPoints + trickPoints
    : state.defenderPoints;

  const completedTrick: CompletedTrick = {
    leadPlayerIndex: state.leadPlayerIndex,
    leadType: state.leadPlayType ?? PlayType.SINGLE,
    slots: trick,
    winnerIndex,
    points: trickPoints,
  };

  // Add trick to winner's team
  const newTeams = state.teams.map((t) => {
    if (t.index === winnerTeamIndex) {
      return { ...t, tricks: [...t.tricks, completedTrick] };
    }
    return t;
  });

  // Assert 2-element array
  if (newTeams.length !== 2) throw new Error('Expected 2 teams');
  const teamsTuple = [newTeams[0], newTeams[1]] as [TeamState, TeamState];

  // Check if all cards are played (round over)
  const allCardsPlayed = state.players.every(p => p.hand.length === 0);
  const trickComplete = state.currentTrick.every(s => s.cards !== null);
  const lastTrickBeforeEnd = allCardsPlayed && trickComplete;

  if (lastTrickBeforeEnd) {
    return {
      ...state,
      trickHistory: [...state.trickHistory, completedTrick],
      lastCompletedTrick: completedTrick,
      teams: teamsTuple,
      defenderPoints: newDefenderPoints,
      phase: Phase.SCORING,
    };
  }

  // Start next trick — keep lastCompletedTrick so the UI can show
  // the finished trick during the pause, then clearTrick() will
  // reset currentTrick after the delay.
  return {
    ...state,
    trickHistory: [...state.trickHistory, completedTrick],
    lastCompletedTrick: completedTrick,
    teams: teamsTuple,
    defenderPoints: newDefenderPoints,
    currentPlayerIndex: winnerIndex,
    leadPlayerIndex: winnerIndex,
    leadPlayType: null,
    // Do NOT clear currentTrick yet — the UI needs to display the
    // completed trick during the 3-second pause. Game.clearTrick()
    // will be called after the pause to reset for the next trick.
  };
}

/**
 * Advance to the next round after scoring.
 */
export function advanceRound(state: GameState, newLevel: Rank, newDeclarerTeam: number): GameState {
  const newTeams: [TeamState, TeamState] = [
    { ...state.teams[0], currentLevel: newLevel, tricks: [] },
    { ...state.teams[1], currentLevel: newLevel, tricks: [] },
  ];

  const newPlayers = state.players.map(p => ({
    ...p,
    hand: [],
    isDeclarer: p.teamIndex === newDeclarerTeam,
  }));

  return {
    ...state,
    teams: newTeams,
    players: newPlayers,
    declarerTeamIndex: newDeclarerTeam,
    currentLevel: newLevel,
    trumpRank: newLevel,
    phase: Phase.DEALING,
    trumpSuit: null,
    bottomCards: [],
    trickHistory: [],
    lastCompletedTrick: null,
    biddingHistory: [],
    stirHistory: [],
    defenderPoints: 0,
    leadPlayType: null,
    currentTrick: state.currentTrick.map(s => ({ ...s, cards: null })),
  };
}

// ---- Helpers ----

function shuffle<T>(array: T[]): T[] {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * Serialize game state to a JSON-safe object for AI prompts.
 */
export function serializeForAI(state: GameState): Record<string, unknown> {
  return {
    phase: state.phase,
    currentLevel: state.currentLevel,
    trumpSuit: state.trumpSuit,
    trumpRank: state.trumpRank,
    declarerTeamIndex: state.declarerTeamIndex,
    defenderPoints: state.defenderPoints,
    currentPlayerIndex: state.currentPlayerIndex,
    currentTrick: state.currentTrick.map(s => ({
      playerIndex: s.playerIndex,
      cards: s.cards?.map(c => ({
        suit: c.suit,
        rank: c.rank,
        isJoker: c.isJoker,
        isBigJoker: c.isBigJoker,
        points: c.points,
        id: c.id,
      })),
    })),
    leadPlayerIndex: state.leadPlayerIndex,
    leadPlayType: state.leadPlayType,
    trickHistory: state.trickHistory.map(t => ({
      leadPlayerIndex: t.leadPlayerIndex,
      leadType: t.leadType,
      winnerIndex: t.winnerIndex,
      points: t.points,
      slots: t.slots.map(s => ({
        playerIndex: s.playerIndex,
        count: s.cards.length,
      })),
    })),
    teams: state.teams.map(t => ({
      index: t.index,
      currentLevel: t.currentLevel,
      tricksWon: t.tricks.length,
    })),
  };
}
