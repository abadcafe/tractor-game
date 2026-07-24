"""Project domain snapshots into the stable browser wire model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from server.game import Seat, snapshots
from server.game.rules import bidding, play
from server.game.rules.cards import Card, Rank, Suit
from server.game.snapshots.events import (
    BidEventSnapshot,
    BottomExchangeSnapshot,
    StirDeclarationEventSnapshot,
    StirringStateSnapshot,
)
from server.game.snapshots.review import ScoringSnapshot
from server.game.snapshots.tricks import (
    CompletedTrickSnapshot,
    TrickSnapshot,
)

_MAX_BID_HINTS = 10


class StateWire(BaseModel):
    """Existing browser state contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: str
    player_hand: tuple[Card, ...]
    player_hand_counts: tuple[int, int, int, int]
    bottom_cards: tuple[Card, ...]
    trump_suit: Suit | None
    trump_rank: Rank
    declarer_team: int | None
    declarer_player: int | None
    defender_points: int
    trick: TrickSnapshot | None
    last_completed_trick: CompletedTrickSnapshot | None
    defender_point_cards: tuple[Card, ...]
    action_hints: tuple[tuple[Card, ...], ...]
    awaiting_action: str | None
    scoring: ScoringSnapshot | None
    winning_team: int | None
    team0_level: Rank
    team1_level: Rank
    bid_events: tuple[BidEventSnapshot, ...]
    bid_winner: BidEventSnapshot | None
    own_initial_bottom_exchange: BottomExchangeSnapshot | None
    stir_events: tuple[StirDeclarationEventSnapshot, ...]
    stirring_state: StirringStateSnapshot | None
    next_round_confirmed: tuple[int, ...]


class StateMessage(BaseModel):
    """Server-to-browser state envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["state"] = "state"
    seq: int
    state: StateWire
    error: str | None


def encode_state(
    viewer: Seat,
    seq: int,
    snapshot: snapshots.PlayerSnapshot,
    error: str | None,
) -> StateMessage:
    """Encode one domain delivery without leaking domain internals."""
    return StateMessage(
        seq=seq,
        state=StateWire(
            phase=snapshot.phase,
            player_hand=snapshot.player_hand,
            player_hand_counts=snapshot.player_hand_counts,
            bottom_cards=snapshot.bottom_cards,
            trump_suit=snapshot.trump_suit,
            trump_rank=snapshot.trump_rank,
            declarer_team=snapshot.declarer_team,
            declarer_player=(
                None
                if snapshot.declarer_player is None
                else int(snapshot.declarer_player)
            ),
            defender_points=snapshot.defender_points,
            trick=snapshot.trick,
            last_completed_trick=snapshot.last_completed_trick,
            defender_point_cards=snapshot.defender_point_cards,
            action_hints=_action_hints(viewer, snapshot),
            awaiting_action=snapshot.awaiting_action,
            scoring=snapshot.scoring,
            winning_team=snapshot.winning_team,
            team0_level=snapshot.team0_level,
            team1_level=snapshot.team1_level,
            bid_events=snapshot.bid_events,
            bid_winner=snapshot.bid_winner,
            own_initial_bottom_exchange=(
                snapshot.own_initial_bottom_exchange
            ),
            stir_events=snapshot.stir_events,
            stirring_state=snapshot.stirring_state,
            next_round_confirmed=tuple(
                sorted(
                    int(seat) for seat in snapshot.next_round_confirmed
                )
            ),
        ),
        error=error,
    )


def _action_hints(
    viewer: Seat,
    snapshot: snapshots.PlayerSnapshot,
) -> tuple[tuple[Card, ...], ...]:
    action = snapshot.awaiting_action
    if action == "bid":
        if (
            snapshot.bid_winner is not None
            and snapshot.bid_winner.player == viewer
        ):
            return ()
        reveals = bidding.legal_reveals(
            snapshot.player_hand,
            snapshot.trump_rank,
            _declaration(snapshot),
        )
        if len(reveals) > _MAX_BID_HINTS:
            return ()
        return tuple(reveal.cards for reveal in reveals)
    if action == "stir":
        reveals = tuple(
            reveal
            for reveal in bidding.legal_reveals(
                snapshot.player_hand,
                snapshot.trump_rank,
                _declaration(snapshot),
            )
            if len(reveal.cards) == 2
        )
        if len(reveals) > _MAX_BID_HINTS:
            return ()
        return tuple(reveal.cards for reveal in reveals)
    if action == "play":
        lead = None
        if snapshot.trick is not None and snapshot.trick.slots:
            lead = snapshot.trick.slots[0].cards
        return play.closed_hints(
            snapshot.player_hand,
            lead,
            snapshot.trump_suit,
            snapshot.trump_rank,
        )
    return ()


def _declaration(
    snapshot: snapshots.PlayerSnapshot,
) -> bidding.Declaration | None:
    if snapshot.bid_winner is None:
        return None
    return bidding.Declaration(cards=snapshot.bid_winner.cards)
