"""Project player-facing snapshots into absolute-position-free state."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game import (
    Seat,
    next_seat,
    partner_seat,
    partnership_of,
    previous_seat,
)
from server.game.rules.cards import Rank
from server.game.rules.cards.faces import (
    FaceCount,
    canonical_face_counts,
)
from server.game.rules.progression import (
    distance_to_target,
    stage_target,
)
from server.game.snapshots import PlayerSnapshot
from server.game.snapshots.contract import (
    NoTrump,
    PendingTrump,
    SuitedTrump,
)
from server.game.snapshots.events import BottomExchangeSnapshot
from server.game.snapshots.tricks import (
    CompletedTrickSnapshot,
    FailedThrowSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.policy_model._schema.positions import (
    TrickPosition,
    TrumpMode,
    TrumpState,
    relative_actor,
    trick_position,
)
from server.policy_model.observation.history import (
    ObservationMemoryView,
)
from server.policy_model.observation.relative_state.actions import (
    RelativeBidAction,
    RelativeExchangeAction,
    RelativePlayAction,
    RelativeRoundAction,
    RelativeStirAction,
)
from server.policy_model.observation.relative_state.contexts import (
    DecisionQuery,
    GlobalContext,
    RelativeObservation,
    RelativeTrick,
    RoundContext,
)
from server.policy_model.observation.structure import (
    RoundEventOrdinal,
    TrickRecency,
)


class RelativeProjectionRejected(Rejected):
    """A snapshot cannot form a valid relative policy state."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


_DEALT_CARD_COUNT = 100


def project_relative_observation(
    *,
    viewer: Seat,
    snapshot: PlayerSnapshot,
    memory: ObservationMemoryView,
) -> Ok[RelativeObservation] | Rejected:
    """Build one complete viewer-relative observation."""
    own_level, opponent_level = _relative_levels(viewer, snapshot)
    own_target = stage_target(own_level, snapshot.mandatory_levels)
    opponent_target = stage_target(
        opponent_level,
        snapshot.mandatory_levels,
    )
    round_context = RoundContext(
        declarer_actor=None
        if snapshot.declarer is None
        else relative_actor(viewer, snapshot.declarer),
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
        partner_remaining=snapshot.remaining_cards.at(
            partner_seat(viewer)
        ),
        next_opponent_remaining=snapshot.remaining_cards.at(
            next_seat(viewer)
        ),
        previous_opponent_remaining=snapshot.remaining_cards.at(
            previous_seat(viewer)
        ),
    )
    timeline = _round_timeline(viewer, snapshot, memory)
    return Ok(
        value=RelativeObservation(
            global_context=GlobalContext(
                mandatory_levels=snapshot.mandatory_levels
            ),
            round_context=round_context,
            round_actions=timeline.actions,
            tricks=_tricks(viewer, snapshot, memory),
            hand=canonical_face_counts(snapshot.hand),
            visible_bottom=canonical_face_counts(snapshot.bottom_cards),
            query=_query(snapshot, timeline.next_ordinal),
        )
    )


def _relative_levels(
    viewer: Seat, snapshot: PlayerSnapshot
) -> tuple[Rank, Rank]:
    own = partnership_of(viewer)
    opponent = partnership_of(next_seat(viewer))
    return (
        snapshot.partnership_levels.at(own),
        snapshot.partnership_levels.at(opponent),
    )


def _trump_state(snapshot: PlayerSnapshot) -> TrumpState:
    if isinstance(snapshot.trump, PendingTrump):
        return TrumpState(mode=TrumpMode.UNSET, suit=None)
    if isinstance(snapshot.trump, NoTrump):
        return TrumpState(mode=TrumpMode.NO_TRUMP, suit=None)
    assert isinstance(snapshot.trump, SuitedTrump)
    return TrumpState(
        mode=TrumpMode.SUITED,
        suit=snapshot.trump.suit,
    )


@dataclass(frozen=True, slots=True)
class _RoundTimeline:
    actions: tuple[RelativeRoundAction, ...]
    next_ordinal: RoundEventOrdinal


def _round_timeline(
    viewer: Seat,
    snapshot: PlayerSnapshot,
    memory: ObservationMemoryView,
) -> _RoundTimeline:
    actions: list[RelativeRoundAction] = []
    next_value = 1
    deal_event_limit = _DEALT_CARD_COUNT
    for action in memory.bid_actions:
        ordinal = action.deal_ordinal
        assert ordinal.value <= deal_event_limit
        assert ordinal.value >= next_value
        actions.append(
            RelativeBidAction(
                actor=relative_actor(viewer, action.actor),
                disposition=action.disposition,
                revealed=canonical_face_counts(action.revealed_cards),
                event_ordinal=ordinal,
            )
        )
        next_value = ordinal.value + 1
    if snapshot.phase != "DEAL_BID":
        next_value = max(next_value, deal_event_limit + 1)
    if snapshot.own_initial_bottom_exchange is not None:
        actions.append(
            _exchange_action(
                snapshot.own_initial_bottom_exchange,
                event_ordinal=RoundEventOrdinal(next_value),
            )
        )
        next_value += 1
    for event in snapshot.stir_events:
        actions.append(
            RelativeStirAction(
                actor=relative_actor(viewer, event.actor),
                disposition="pass"
                if event.kind == "pass"
                else "reveal",
                revealed=canonical_face_counts(event.cards),
                event_ordinal=RoundEventOrdinal(next_value),
            )
        )
        next_value += 1
        if event.own_bottom_exchange is not None:
            actions.append(
                _exchange_action(
                    event.own_bottom_exchange,
                    event_ordinal=RoundEventOrdinal(next_value),
                )
            )
            next_value += 1
    return _RoundTimeline(
        actions=tuple(actions),
        next_ordinal=RoundEventOrdinal(next_value),
    )


def _exchange_action(
    exchange: BottomExchangeSnapshot,
    *,
    event_ordinal: RoundEventOrdinal,
) -> RelativeExchangeAction:
    return RelativeExchangeAction(
        picked_up=canonical_face_counts(
            exchange.picked_up_bottom_cards
        ),
        discarded=canonical_face_counts(
            exchange.discarded_bottom_cards
        ),
        event_ordinal=event_ordinal,
    )


def _tricks(
    viewer: Seat,
    snapshot: PlayerSnapshot,
    memory: ObservationMemoryView,
) -> tuple[RelativeTrick, ...]:
    result: list[RelativeTrick] = []
    total = len(memory.completed_tricks)
    for index, completed in enumerate(memory.completed_tricks):
        result.append(
            _completed_trick(
                viewer,
                completed,
                recency=TrickRecency(total - index),
            )
        )
    if snapshot.trick is not None:
        result.append(_open_trick(viewer, snapshot.trick))
    return tuple(result)


def _completed_trick(
    viewer: Seat,
    trick: CompletedTrickSnapshot,
    *,
    recency: TrickRecency,
) -> RelativeTrick:
    return RelativeTrick(
        status="completed",
        recency=recency,
        actions=_play_actions(
            viewer,
            lead_actor=trick.lead_actor,
            slots=tuple(trick.slots),
            failed_throw=trick.failed_throw,
        ),
        winner=relative_actor(viewer, trick.winner),
        points=trick.points,
    )


def _open_trick(viewer: Seat, trick: TrickSnapshot) -> RelativeTrick:
    return RelativeTrick(
        status="open",
        recency=TrickRecency(0),
        actions=_play_actions(
            viewer,
            lead_actor=trick.lead_actor,
            slots=tuple(trick.slots),
            failed_throw=trick.failed_throw,
        ),
        winner=None,
        points=None,
    )


def _play_actions(
    viewer: Seat,
    *,
    lead_actor: Seat,
    slots: tuple[TrickSlotSnapshot, ...],
    failed_throw: FailedThrowSnapshot | None,
) -> tuple[RelativePlayAction, ...]:
    populated = [slot for slot in slots if slot.cards]
    populated.sort(
        key=lambda slot: _position_index(
            lead_actor=lead_actor, actor=slot.actor
        )
    )
    actions: list[RelativePlayAction] = []
    for slot in populated:
        extra: tuple[FaceCount, ...] = ()
        if (
            failed_throw is not None
            and failed_throw.actor == slot.actor
        ):
            extra = _revealed_extra(failed_throw)
        actions.append(
            RelativePlayAction(
                actor=relative_actor(viewer, slot.actor),
                trick_position=trick_position(
                    lead_actor=lead_actor, actor=slot.actor
                ),
                played=canonical_face_counts(slot.cards),
                revealed_extra=extra,
            )
        )
    return tuple(actions)


def _position_index(*, lead_actor: Seat, actor: Seat) -> int:
    position = trick_position(
        lead_actor=lead_actor,
        actor=actor,
    )
    return {
        TrickPosition.LEAD: 0,
        TrickPosition.FOLLOW_1: 1,
        TrickPosition.FOLLOW_2: 2,
        TrickPosition.FOLLOW_3: 3,
    }[position]


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


def _query(
    snapshot: PlayerSnapshot,
    next_round_event: RoundEventOrdinal,
) -> DecisionQuery | None:
    awaiting = snapshot.awaiting_action
    if awaiting == "bid":
        return DecisionQuery(
            kind="bid",
            round_event=next_round_event,
            trick_position=None,
        )
    if awaiting == "stir":
        return DecisionQuery(
            kind="stir",
            round_event=next_round_event,
            trick_position=None,
        )
    if awaiting == "discard":
        return DecisionQuery(
            kind="bottom_exchange",
            round_event=next_round_event,
            trick_position=None,
        )
    if awaiting == "play":
        trick = snapshot.trick
        if trick is None:
            return DecisionQuery(
                kind="play",
                round_event=None,
                trick_position=TrickPosition.LEAD,
            )
        return DecisionQuery(
            kind="play",
            round_event=None,
            trick_position=trick_position(
                lead_actor=trick.lead_actor,
                actor=trick.current_actor,
            ),
        )
    return None


__all__ = ("RelativeProjectionRejected", "project_relative_observation")
