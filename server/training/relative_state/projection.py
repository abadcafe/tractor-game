"""Project player-facing snapshots into absolute-position-free state."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game.protocol import (
    BottomExchangeSnapshot,
    CompletedTrickSnapshot,
    FailedThrowSnapshot,
    StateSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.game.rules.card_faces import (
    FaceCount,
    canonical_face_counts,
)
from server.game.rules.cards import Rank, Suit
from server.game.rules.required_progress import (
    MANDATORY_LEVELS,
    distance_to_target,
    stage_target,
)
from server.game.state_machine.constants import PLAYER_COUNT
from server.training.observation_memory import ObservationMemoryView
from server.training.relative_state.actions import (
    RelativeBidAction,
    RelativeExchangeAction,
    RelativePlayAction,
    RelativeRoundAction,
    RelativeStirAction,
)
from server.training.relative_state.contexts import (
    DecisionQuery,
    GlobalContext,
    RelativeObservation,
    RelativeTrick,
    RoundContext,
)
from server.training.relative_state.relations import (
    TrickPosition,
    TrumpMode,
    TrumpState,
    relative_actor,
    trick_position,
)


class RelativeProjectionRejected(Rejected):
    """A snapshot cannot form a valid relative policy state."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


def project_relative_observation(
    *,
    viewer: int,
    snapshot: StateSnapshot,
    memory: ObservationMemoryView,
) -> Ok[RelativeObservation] | Rejected:
    """Build one complete viewer-relative observation."""
    if viewer < 0 or viewer >= PLAYER_COUNT:
        return RelativeProjectionRejected(
            "viewer is outside player topology"
        )
    if len(snapshot.player_hand_counts) != PLAYER_COUNT:
        return RelativeProjectionRejected(
            "player hand counts do not match player topology"
        )
    own_level, opponent_level = _relative_levels(viewer, snapshot)
    own_target = stage_target(own_level)
    opponent_target = stage_target(opponent_level)
    round_context = RoundContext(
        declarer_actor=None
        if snapshot.declarer_player is None
        else relative_actor(viewer, snapshot.declarer_player),
        own_level=own_level,
        opponent_level=opponent_level,
        own_target=own_target,
        opponent_target=opponent_target,
        own_distance_to_target=distance_to_target(
            own_level, own_target
        ),
        opponent_distance_to_target=distance_to_target(
            opponent_level, opponent_target
        ),
        trump=_trump_state(snapshot),
        level_rank=snapshot.trump_rank,
        defender_points=snapshot.defender_points,
        partner_remaining=snapshot.player_hand_counts[(viewer + 2) % 4],
        left_enemy_remaining=snapshot.player_hand_counts[
            (viewer + 1) % 4
        ],
        right_enemy_remaining=snapshot.player_hand_counts[
            (viewer + 3) % 4
        ],
    )
    return Ok(
        value=RelativeObservation(
            global_context=GlobalContext(
                mandatory_levels=MANDATORY_LEVELS
            ),
            round_context=round_context,
            round_actions=_round_actions(viewer, snapshot, memory),
            tricks=_tricks(viewer, snapshot, memory),
            hand=canonical_face_counts(snapshot.player_hand),
            visible_bottom=canonical_face_counts(snapshot.bottom_cards),
            query=_query(snapshot),
        )
    )


def _relative_levels(
    viewer: int, snapshot: StateSnapshot
) -> tuple[Rank, Rank]:
    if viewer % 2 == 0:
        return (snapshot.team0_level, snapshot.team1_level)
    return (snapshot.team1_level, snapshot.team0_level)


def _trump_state(snapshot: StateSnapshot) -> TrumpState:
    suit = snapshot.trump_suit
    if snapshot.phase == "DEAL_BID":
        return TrumpState(mode=TrumpMode.UNSET, suit=None)
    if suit is None:
        return TrumpState(mode=TrumpMode.NO_TRUMP, suit=None)
    assert suit != Suit.JOKER
    return TrumpState(mode=TrumpMode.SUITED, suit=suit)


def _round_actions(
    viewer: int,
    snapshot: StateSnapshot,
    memory: ObservationMemoryView,
) -> tuple[RelativeRoundAction, ...]:
    actions: list[RelativeRoundAction] = [
        RelativeBidAction(
            actor=relative_actor(viewer, action.actor),
            disposition=action.disposition,
            revealed=canonical_face_counts(action.revealed_cards),
            event_time=action.deal_ordinal,
        )
        for action in memory.bid_actions
    ]
    event_time = 101
    if snapshot.own_initial_bottom_exchange is not None:
        actions.append(
            _exchange_action(
                snapshot.own_initial_bottom_exchange,
                event_time=event_time,
            )
        )
        event_time += 1
    for event in snapshot.stir_events:
        actions.append(
            RelativeStirAction(
                actor=relative_actor(viewer, event.player),
                disposition="pass"
                if event.kind == "pass"
                else "reveal",
                revealed=canonical_face_counts(event.cards),
                event_time=event_time,
            )
        )
        event_time += 1
        if event.own_bottom_exchange is not None:
            actions.append(
                _exchange_action(
                    event.own_bottom_exchange,
                    event_time=event_time,
                )
            )
            event_time += 1
    return tuple(actions)


def _exchange_action(
    exchange: BottomExchangeSnapshot,
    *,
    event_time: int,
) -> RelativeExchangeAction:
    return RelativeExchangeAction(
        picked_up=canonical_face_counts(
            exchange.picked_up_bottom_cards
        ),
        discarded=canonical_face_counts(
            exchange.discarded_bottom_cards
        ),
        event_time=event_time,
    )


def _tricks(
    viewer: int,
    snapshot: StateSnapshot,
    memory: ObservationMemoryView,
) -> tuple[RelativeTrick, ...]:
    result: list[RelativeTrick] = []
    total = len(memory.completed_tricks)
    for index, completed in enumerate(memory.completed_tricks):
        result.append(
            _completed_trick(
                viewer,
                completed,
                trick_time=total - index,
            )
        )
    if snapshot.trick is not None:
        result.append(_open_trick(viewer, snapshot.trick))
    return tuple(result)


def _completed_trick(
    viewer: int,
    trick: CompletedTrickSnapshot,
    *,
    trick_time: int,
) -> RelativeTrick:
    return RelativeTrick(
        status="completed",
        trick_time=trick_time,
        actions=_play_actions(
            viewer,
            lead_player=trick.lead_player,
            slots=tuple(trick.slots),
            failed_throw=trick.failed_throw,
        ),
        winner=relative_actor(viewer, trick.winner),
        points=trick.points,
    )


def _open_trick(viewer: int, trick: TrickSnapshot) -> RelativeTrick:
    return RelativeTrick(
        status="open",
        trick_time=0,
        actions=_play_actions(
            viewer,
            lead_player=trick.lead_player,
            slots=tuple(trick.slots),
            failed_throw=trick.failed_throw,
        ),
        winner=None,
        points=None,
    )


def _play_actions(
    viewer: int,
    *,
    lead_player: int,
    slots: tuple[TrickSlotSnapshot, ...],
    failed_throw: FailedThrowSnapshot | None,
) -> tuple[RelativePlayAction, ...]:
    populated = [slot for slot in slots if slot.cards]
    populated.sort(
        key=lambda slot: _position_index(
            lead_player=lead_player, actor=slot.player
        )
    )
    actions: list[RelativePlayAction] = []
    for slot in populated:
        extra: tuple[FaceCount, ...] = ()
        if (
            failed_throw is not None
            and failed_throw.player == slot.player
        ):
            extra = _revealed_extra(failed_throw)
        actions.append(
            RelativePlayAction(
                actor=relative_actor(viewer, slot.player),
                trick_position=trick_position(
                    lead_player=lead_player, actor=slot.player
                ),
                played=canonical_face_counts(slot.cards),
                revealed_extra=extra,
            )
        )
    return tuple(actions)


def _position_index(*, lead_player: int, actor: int) -> int:
    return (actor - lead_player) % PLAYER_COUNT


def _revealed_extra(
    failed_throw: FailedThrowSnapshot,
) -> tuple[FaceCount, ...]:
    attempted = canonical_face_counts(failed_throw.attempted_cards)
    forced = {
        item.face: item.count
        for item in canonical_face_counts(failed_throw.forced_cards)
    }
    result: list[FaceCount] = []
    for item in attempted:
        remaining = item.count - forced.get(item.face, 0)
        if remaining > 0:
            result.append(FaceCount(face=item.face, count=remaining))
    return tuple(result)


def _query(snapshot: StateSnapshot) -> DecisionQuery | None:
    awaiting = snapshot.awaiting_action
    if awaiting == "bid":
        return DecisionQuery(
            kind="bid",
            event_time=sum(snapshot.player_hand_counts),
            trick_position=None,
        )
    if awaiting == "stir":
        return DecisionQuery(
            kind="stir",
            event_time=101 + len(snapshot.stir_events),
            trick_position=None,
        )
    if awaiting == "discard":
        return DecisionQuery(
            kind="bottom_exchange",
            event_time=101 + len(snapshot.stir_events),
            trick_position=None,
        )
    if awaiting == "play":
        trick = snapshot.trick
        if trick is None:
            return DecisionQuery(
                kind="play",
                event_time=None,
                trick_position=TrickPosition.LEAD,
            )
        return DecisionQuery(
            kind="play",
            event_time=None,
            trick_position=trick_position(
                lead_player=trick.lead_player,
                actor=trick.current_player,
            ),
        )
    return None


__all__ = ("RelativeProjectionRejected", "project_relative_observation")
