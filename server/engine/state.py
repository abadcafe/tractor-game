"""Game state transitions for 升级 (Shengji/Tractor).

Immutable state machine: every function takes a GameState and returns a new one.
All updates use Pydantic model_copy for immutability.

Bug fixes vs. original TypeScript:
  #1: resolve_trick uses compare_plays to determine actual winner.
  #2: is_declarer is set on the specific winning bidder, not the whole team.
  #3: advance_round accepts independent levels per team.
"""

from __future__ import annotations

import random

from server.engine.card import Card, Rank, Suit, create_decks
from server.engine.constants import (
    PLAYER_COUNT,
    START_LEVEL,
)
from server.engine.game_state import (
    CompletedTrick,
    CompletedTrickSlot,
    GameState,
    GameSettings,
    PlayerState,
    TeamState,
    TrickSlot,
)
from server.engine.player_utils import get_team_index, next_player
from server.engine.types import BidAction, Phase, PlayAction, PlayType, StirAction
from server.rules.comparator import compare_plays


# ---- Public API ----


def create_initial_state(settings: GameSettings | None = None) -> GameState:
    """Create a fresh game state in DEALING phase."""
    if settings is None:
        settings = GameSettings()

    players = [
        PlayerState(
            index=0, name="同伴 (AI)", hand=[], team_index=0,
            is_human=False, is_declarer=False,
        ),
        PlayerState(
            index=1, name="对手A (AI)", hand=[], team_index=1,
            is_human=False, is_declarer=False,
        ),
        PlayerState(
            index=2, name="对手B (AI)", hand=[], team_index=1,
            is_human=False, is_declarer=False,
        ),
        PlayerState(
            index=3, name="你", hand=[], team_index=0,
            is_human=True, is_declarer=False,
        ),
    ]

    teams = [
        TeamState(index=0, tricks=[], current_level=START_LEVEL),
        TeamState(index=1, tricks=[], current_level=START_LEVEL),
    ]

    current_trick = [
        TrickSlot(player_index=i) for i in range(PLAYER_COUNT)
    ]

    return GameState(
        phase=Phase.DEALING,
        current_level=START_LEVEL,
        players=players,
        teams=teams,
        current_player_index=0,
        trump_suit=None,
        trump_rank=START_LEVEL,
        declarer_team_index=0,
        current_trick=current_trick,
        lead_player_index=0,
        lead_play_type=None,
        bottom_cards=[],
        trick_history=[],
        last_completed_trick=None,
        bidding_history=[],
        stir_history=[],
        defender_points=0,
        settings=settings,
    )


def deal_cards(state: GameState) -> GameState:
    """Shuffle, deal cards to players, and transition to BIDDING phase."""
    deck = create_decks()
    shuffled = deck[:]  # copy
    random.shuffle(shuffled)

    bottom_count = state.settings.bottom_card_count
    bottom_cards = shuffled[:bottom_count]
    player_cards = shuffled[bottom_count:]
    per_player = len(player_cards) // PLAYER_COUNT

    new_players = []
    for i, p in enumerate(state.players):
        start = i * per_player
        hand = player_cards[start : start + per_player]
        new_players.append(p.model_copy(update={"hand": hand}))

    empty_trick = [TrickSlot(player_index=i) for i in range(PLAYER_COUNT)]

    return state.model_copy(update={
        "players": new_players,
        "bottom_cards": bottom_cards,
        "phase": Phase.BIDDING,
        "current_player_index": 0,
        "bidding_history": [],
        "stir_history": [],
        "trick_history": [],
        "last_completed_trick": None,
        "defender_points": 0,
        "trump_suit": None,
        "lead_play_type": None,
        "current_trick": empty_trick,
    })


def record_bid(state: GameState, bid: BidAction) -> GameState:
    """Record a bid action and advance to the next player.

    Raises:
        ValueError: if bid.player_index does not match current_player_index.
    """
    if bid.player_index != state.current_player_index:
        raise ValueError(
            f"bid.player_index={bid.player_index} does not match "
            f"current_player_index={state.current_player_index}"
        )
    return state.model_copy(update={
        "bidding_history": [*state.bidding_history, bid],
        "current_player_index": next_player(bid.player_index),
    })


def set_declarer(
    state: GameState,
    player_index: int,
    trump_suit: Suit,
    trump_rank: Rank,
) -> GameState:
    """Set the winning bidder as declarer.

    Bug #2 fix: only the specific player gets is_declarer=True,
    not the entire team.  The team is tracked via declarer_team_index.
    """
    team_index = get_team_index(player_index)

    new_players = [
        p.model_copy(update={"is_declarer": p.index == player_index})
        for p in state.players
    ]

    return state.model_copy(update={
        "trump_suit": trump_suit,
        "trump_rank": trump_rank,
        "declarer_team_index": team_index,
        "players": new_players,
        "phase": Phase.STIRRING,
        "current_player_index": next_player(player_index),
    })


def record_stir(state: GameState, stir: StirAction) -> GameState:
    """Record a stir (trump suit change) action.

    Raises:
        ValueError: if stir.player_index does not match current_player_index.
    """
    if stir.player_index != state.current_player_index:
        raise ValueError(
            f"stir.player_index={stir.player_index} does not match "
            f"current_player_index={state.current_player_index}"
        )
    team_index = get_team_index(stir.player_index)

    new_players = [
        p.model_copy(update={
            "is_declarer": p.index == stir.player_index,
        })
        for p in state.players
    ]

    return state.model_copy(update={
        "trump_suit": stir.new_trump_suit,
        "trump_rank": stir.level,
        "declarer_team_index": team_index,
        "stir_history": [*state.stir_history, stir],
        "players": new_players,
        "current_player_index": next_player(stir.player_index),
    })


def pickup_bottom_cards(state: GameState) -> GameState:
    """Give bottom cards to the declarer and enter EXCHANGE phase.

    Bug #2 fix: finds the specific player with is_declarer=True
    (not the first player on the declarer team).
    """
    declarer = next(
        (p for p in state.players if p.is_declarer), None
    )
    if declarer is None:
        raise ValueError("no declarer set; call set_declarer first")

    new_players = [
        p.model_copy(update={
            "hand": [*p.hand, *state.bottom_cards],
        })
        if p.index == declarer.index
        else p
        for p in state.players
    ]

    return state.model_copy(update={
        "players": new_players,
        "phase": Phase.EXCHANGE,
        "current_player_index": declarer.index,
    })


def discard_cards(
    state: GameState,
    player_index: int,
    cards: list[Card],
) -> GameState:
    """Declarer discards cards; transition to PLAYING phase."""
    discard_ids = {c.id for c in cards}

    new_players = [
        p.model_copy(update={
            "hand": [c for c in p.hand if c.id not in discard_ids],
        })
        if p.index == player_index
        else p
        for p in state.players
    ]

    empty_trick = [TrickSlot(player_index=i) for i in range(PLAYER_COUNT)]

    return state.model_copy(update={
        "players": new_players,
        "bottom_cards": cards,
        "phase": Phase.PLAYING,
        "current_player_index": player_index,
        "lead_player_index": player_index,
        "lead_play_type": None,
        "current_trick": empty_trick,
    })


def play_cards(
    state: GameState,
    player_index: int,
    action: PlayAction,
) -> GameState:
    """Play cards to the current trick.

    After playing, if all 4 players have played, resolve the trick.
    Otherwise, advance to the next player.

    Raises:
        ValueError: if player_index does not match current_player_index.
    """
    if player_index != state.current_player_index:
        raise ValueError(
            f"player_index={player_index} does not match "
            f"current_player_index={state.current_player_index}"
        )

    # Verify played cards exist in player's hand
    hand_ids = {c.id for c in state.players[player_index].hand}
    played_ids = {c.id for c in action.cards}
    missing = played_ids - hand_ids
    if missing:
        raise ValueError(
            f"cards {missing} not in player {player_index}'s hand"
        )

    # Place cards in trick slot
    new_trick = [
        slot.model_copy(update={"cards": action.cards})
        if slot.player_index == player_index
        else slot
        for slot in state.current_trick
    ]

    # Is this the lead play?
    is_lead = all(s.cards is None for s in state.current_trick)

    lead_play_type = action.type if is_lead else state.lead_play_type
    lead_player_index = player_index if is_lead else state.lead_player_index

    # Remove played cards from hand
    played_ids = {c.id for c in action.cards}
    new_players = [
        p.model_copy(update={
            "hand": [c for c in p.hand if c.id not in played_ids],
        })
        if p.index == player_index
        else p
        for p in state.players
    ]

    new_state = state.model_copy(update={
        "current_trick": new_trick,
        "lead_play_type": lead_play_type,
        "lead_player_index": lead_player_index,
        "players": new_players,
    })

    # Check if trick is complete (all 4 played)
    all_played = all(s.cards is not None for s in new_state.current_trick)
    if all_played:
        return resolve_trick(new_state)

    # Advance to next player
    return new_state.model_copy(update={
        "current_player_index": next_player(player_index),
    })


def resolve_trick(state: GameState) -> GameState:
    """Resolve a completed trick: determine winner, collect points.

    Bug #1 fix: uses compare_plays from the comparator module
    to determine the actual winner, rather than defaulting to lead.
    """
    trick = state.current_trick

    # Extract cards per slot (all non-None after full trick)
    lead_cards: list[Card] = []
    all_cards: list[Card] = []
    for slot in trick:
        if slot.cards:
            all_cards.extend(slot.cards)
            if slot.player_index == state.lead_player_index:
                lead_cards = slot.cards

    lead_suit = lead_cards[0].suit if lead_cards and not lead_cards[0].is_joker else None

    # Determine winner using compare_plays
    winner_index = state.lead_player_index
    best_play = lead_cards
    for slot in trick:
        if slot.cards is None:
            continue
        if slot.player_index == state.lead_player_index:
            continue
        result = compare_plays(
            slot.cards, best_play,
            state.trump_suit, state.trump_rank, lead_suit,
        )
        if result > 0:
            best_play = slot.cards
            winner_index = slot.player_index

    # Count points
    trick_points = sum(c.points for c in all_cards)

    winner_team = get_team_index(winner_index)
    defender_team = 1 if state.declarer_team_index == 0 else 0

    new_defender_points = (
        state.defender_points + trick_points
        if winner_team == defender_team
        else state.defender_points
    )

    completed = CompletedTrick(
        lead_player_index=state.lead_player_index,
        lead_type=state.lead_play_type or PlayType.SINGLE,
        slots=[
            CompletedTrickSlot(
                player_index=s.player_index,
                cards=s.cards or [],
            )
            for s in trick
        ],
        winner_index=winner_index,
        points=trick_points,
    )

    # Add trick to winner's team
    new_teams = [
        t.model_copy(update={"tricks": [*t.tricks, completed]})
        if t.index == winner_team
        else t
        for t in state.teams
    ]

    # Check if all cards are played (round over)
    all_cards_played = all(len(p.hand) == 0 for p in state.players)
    empty_trick = [TrickSlot(player_index=i) for i in range(PLAYER_COUNT)]

    if all_cards_played:
        return state.model_copy(update={
            "trick_history": [*state.trick_history, completed],
            "last_completed_trick": completed,
            "teams": new_teams,
            "defender_points": new_defender_points,
            "phase": Phase.SCORING,
            "current_trick": empty_trick,
        })

    # Start next trick
    return state.model_copy(update={
        "trick_history": [*state.trick_history, completed],
        "last_completed_trick": completed,
        "teams": new_teams,
        "defender_points": new_defender_points,
        "current_player_index": winner_index,
        "lead_player_index": winner_index,
        "lead_play_type": None,
        "current_trick": empty_trick,
    })


def advance_round(
    state: GameState,
    team0_new_level: Rank,
    team1_new_level: Rank,
    new_declarer_team: int,
) -> GameState:
    """Advance to the next round after scoring.

    Bug #3 fix: teams advance independently to their own levels.
    """
    new_teams = [
        state.teams[0].model_copy(update={
            "current_level": team0_new_level,
            "tricks": [],
        }),
        state.teams[1].model_copy(update={
            "current_level": team1_new_level,
            "tricks": [],
        }),
    ]

    # trump_rank for the new round = declarer team's new level
    declarer_level = team0_new_level if new_declarer_team == 0 else team1_new_level

    new_players = [
        p.model_copy(update={
            "hand": [],
            "is_declarer": False,
        })
        for p in state.players
    ]

    empty_trick = [TrickSlot(player_index=i) for i in range(PLAYER_COUNT)]

    return state.model_copy(update={
        "teams": new_teams,
        "players": new_players,
        "declarer_team_index": new_declarer_team,
        "current_level": declarer_level,
        "trump_rank": declarer_level,
        "phase": Phase.DEALING,
        "trump_suit": None,
        "bottom_cards": [],
        "trick_history": [],
        "last_completed_trick": None,
        "bidding_history": [],
        "stir_history": [],
        "defender_points": 0,
        "lead_play_type": None,
        "current_trick": empty_trick,
    })


def clear_trick(state: GameState) -> GameState:
    """Reset current_trick slots to empty for the next trick."""
    empty_trick = [TrickSlot(player_index=i) for i in range(PLAYER_COUNT)]
    return state.model_copy(update={
        "current_trick": empty_trick,
    })
