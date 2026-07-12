"""Build training observations from player-facing snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from server.game.protocol import (
    BidEventSnapshot,
    BottomExchangeSnapshot,
    CompletedTrickSnapshot,
    FailedThrowSnapshot,
    StateSnapshot,
    StirDeclarationEventSnapshot,
    TrickSlotSnapshot,
)
from server.game.rules.card_faces import (
    FaceCount,
    canonical_face_counts,
    face_count_signature,
)
from server.game.rules.cards import Card
from server.game.rules.required_progress import (
    MANDATORY_LEVELS,
    distance_to_target,
    progress_target_value,
    stage_target,
)
from server.game.state_machine.constants import (
    BOTTOM_CARD_COUNT,
    PLAYER_COUNT,
)
from server.training.semantic_actions.query import (
    ActionQuery,
    build_action_query,
)
from server.training.token_context import (
    RelativeRole,
    TrickRecordState,
    relative_role,
)
from server.training.tokens import (
    ActionQueryFieldToken,
    FaceCountToken,
    GlobalFieldToken,
    ObservationToken,
    RoundEventFieldToken,
    RoundFieldToken,
    TrickResultFieldToken,
    face_count_token,
)


@dataclass(frozen=True, slots=True)
class HistoryTrick:
    """One completed public trick kept by the training observer."""

    lead_player: int
    slots: tuple[TrickSlotSnapshot, ...]
    winner: int
    points: int
    failed_throw: FailedThrowSnapshot | None


@dataclass(frozen=True, slots=True)
class Observation:
    """Model-facing observation and its action-pointer query."""

    player_index: int
    tokens: tuple[ObservationToken, ...]
    hand_faces: tuple[FaceCount, ...]
    action_query: ActionQuery


def _signature_set() -> set[tuple[object, ...]]:
    return set()


def _history_list() -> list[HistoryTrick]:
    return []


@dataclass(slots=True)
class PublicHistoryRecorder:
    """Accumulate public completed tricks from repeated snapshots."""

    _signatures: set[tuple[object, ...]] = field(
        default_factory=_signature_set
    )
    _tricks: list[HistoryTrick] = field(default_factory=_history_list)
    _last_completed_signature: tuple[object, ...] | None = None
    _saw_open_play_since_last_completed: bool = False

    def update(self, snapshot: StateSnapshot) -> None:
        """Add a completed trick if this snapshot reveals a new one."""
        has_open_play = _snapshot_has_open_play(snapshot)
        completed = snapshot.last_completed_trick
        if completed is not None:
            signature = _completed_trick_signature(completed)
            if self._should_append_completed(
                signature,
                has_open_play=has_open_play,
            ):
                self._append_completed(completed, signature)
        if has_open_play:
            self._saw_open_play_since_last_completed = True

    def _should_append_completed(
        self,
        signature: tuple[object, ...],
        *,
        has_open_play: bool,
    ) -> bool:
        if self._last_completed_signature is None:
            return True
        if signature != self._last_completed_signature:
            return True
        return self._saw_open_play_since_last_completed and (
            not has_open_play
        )

    def _append_completed(
        self,
        completed: CompletedTrickSnapshot,
        signature: tuple[object, ...],
    ) -> None:
        if signature in self._signatures and (
            not self._saw_open_play_since_last_completed
        ):
            return
        self._signatures.add(signature)
        self._tricks.append(
            HistoryTrick(
                lead_player=completed.lead_player,
                slots=tuple(completed.slots),
                winner=completed.winner,
                points=completed.points,
                failed_throw=completed.failed_throw,
            )
        )
        self._last_completed_signature = signature
        self._saw_open_play_since_last_completed = False

    def tricks(self) -> tuple[HistoryTrick, ...]:
        """Return completed public tricks in observed order."""
        return tuple(self._tricks)

    def clear(self) -> None:
        """Clear per-round public trick history."""
        self._signatures.clear()
        self._tricks.clear()
        self._last_completed_signature = None
        self._saw_open_play_since_last_completed = False


def build_observation(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    history: tuple[HistoryTrick, ...],
) -> Observation:
    """Build one training observation from player-visible data only."""
    action_query = build_action_query(
        player_index=player_index,
        snapshot=snapshot,
    )
    tokens: list[ObservationToken] = []
    tokens.extend(_global_tokens())
    tokens.extend(
        _round_tokens(
            player_index=player_index,
            snapshot=snapshot,
        )
    )
    tokens.extend(_bid_event_tokens(player_index, snapshot.bid_events))
    if snapshot.own_initial_bottom_exchange is not None:
        tokens.extend(
            _own_bottom_exchange_tokens(
                snapshot.own_initial_bottom_exchange,
                event_age=len(snapshot.stir_events) + 1,
                trigger="initial",
            )
        )
    tokens.extend(
        _stir_event_tokens(player_index, snapshot.stir_events)
    )
    tokens.extend(_visible_bottom_tokens(snapshot))
    tokens.extend(_hand_tokens(snapshot))
    tokens.extend(
        _play_record_tokens(
            player_index=player_index,
            snapshot=snapshot,
            history=history,
        )
    )
    tokens.extend(_action_query_tokens(action_query))
    return Observation(
        player_index=player_index,
        tokens=tuple(tokens),
        hand_faces=canonical_face_counts(snapshot.player_hand),
        action_query=action_query,
    )


def face_count_tokens(
    observation: Observation,
) -> tuple[FaceCountToken, ...]:
    """Return all face-count tokens in sequence order."""
    return tuple(
        token
        for token in observation.tokens
        if isinstance(token, FaceCountToken)
    )


def _global_tokens() -> tuple[GlobalFieldToken, ...]:
    tokens: list[GlobalFieldToken] = [
        GlobalFieldToken("team_layout", "fixed_partner_opposite"),
        GlobalFieldToken("left_player_role", "left_enemy"),
        GlobalFieldToken("right_player_role", "right_enemy"),
        GlobalFieldToken("partner_role", "partner"),
        GlobalFieldToken("deck_count", 2),
        GlobalFieldToken("player_count", PLAYER_COUNT),
        GlobalFieldToken("bottom_card_count", BOTTOM_CARD_COUNT),
        GlobalFieldToken("rules_version", "rules-required-progress"),
    ]
    for level in MANDATORY_LEVELS:
        tokens.append(GlobalFieldToken("required_level", level.value))
    tokens.append(GlobalFieldToken("final_target", "WIN"))
    return tuple(tokens)


def _round_tokens(
    *,
    player_index: int,
    snapshot: StateSnapshot,
) -> tuple[RoundFieldToken, ...]:
    self_team = player_index % 2
    enemy_team = 1 - self_team
    self_level = (
        snapshot.team0_level if self_team == 0 else snapshot.team1_level
    )
    enemy_level = (
        snapshot.team1_level if self_team == 0 else snapshot.team0_level
    )
    self_target = stage_target(self_level)
    enemy_target = stage_target(enemy_level)
    dealer_role = (
        None
        if snapshot.declarer_player is None
        else relative_role(player_index, snapshot.declarer_player)
    )
    revealer_role = _level_card_revealer_role(player_index, snapshot)
    return (
        RoundFieldToken("phase", snapshot.phase),
        RoundFieldToken("awaiting_action", snapshot.awaiting_action),
        RoundFieldToken("dealer_role", dealer_role),
        RoundFieldToken("dealer_team", snapshot.declarer_team),
        RoundFieldToken(
            "self_team_is_declarer",
            snapshot.declarer_team == self_team,
        ),
        RoundFieldToken(
            "enemy_team_is_declarer",
            snapshot.declarer_team == enemy_team,
        ),
        RoundFieldToken("self_team_level", self_level.value),
        RoundFieldToken("enemy_team_level", enemy_level.value),
        RoundFieldToken(
            "self_team_required_level",
            progress_target_value(self_target),
        ),
        RoundFieldToken(
            "enemy_team_required_level",
            progress_target_value(enemy_target),
        ),
        RoundFieldToken(
            "self_team_distance_to_required_level",
            distance_to_target(self_level, self_target),
        ),
        RoundFieldToken(
            "enemy_team_distance_to_required_level",
            distance_to_target(enemy_level, enemy_target),
        ),
        RoundFieldToken(
            "trump_suit",
            None
            if snapshot.trump_suit is None
            else snapshot.trump_suit.value,
        ),
        RoundFieldToken("level_rank", snapshot.trump_rank.value),
        RoundFieldToken("level_card_revealer_role", revealer_role),
        RoundFieldToken("current_score", snapshot.defender_points),
        RoundFieldToken(
            "remaining_cards_self",
            _hand_count(player_index, snapshot),
        ),
        RoundFieldToken(
            "remaining_cards_partner",
            _hand_count((player_index + 2) % PLAYER_COUNT, snapshot),
        ),
        RoundFieldToken(
            "remaining_cards_left_enemy",
            _hand_count((player_index + 1) % PLAYER_COUNT, snapshot),
        ),
        RoundFieldToken(
            "remaining_cards_right_enemy",
            _hand_count((player_index + 3) % PLAYER_COUNT, snapshot),
        ),
        RoundFieldToken("winning_team", snapshot.winning_team),
    )


def _bid_event_tokens(
    player_index: int,
    bid_events: list[BidEventSnapshot],
) -> tuple[ObservationToken, ...]:
    tokens: list[ObservationToken] = []
    total = len(bid_events)
    for index, event in enumerate(bid_events):
        event_age = total - index
        actor = relative_role(player_index, event.player)
        tokens.append(
            RoundEventFieldToken("event_kind", "bid", event_age)
        )
        tokens.append(RoundEventFieldToken("actor", actor, event_age))
        tokens.append(
            RoundEventFieldToken("bid_kind", event.kind, event_age)
        )
        tokens.append(
            RoundEventFieldToken(
                "suit",
                None if event.suit is None else event.suit.value,
                event_age,
            )
        )
        tokens.append(
            RoundEventFieldToken(
                "joker_type", event.joker_type, event_age
            )
        )
        tokens.append(
            RoundEventFieldToken("count", event.count, event_age)
        )
        for face_count in canonical_face_counts(event.cards):
            tokens.append(
                face_count_token(
                    face_count,
                    segment="round_event",
                    role=actor,
                    event_age=event_age,
                )
            )
    return tuple(tokens)


def _stir_event_tokens(
    player_index: int,
    stir_events: list[StirDeclarationEventSnapshot],
) -> tuple[ObservationToken, ...]:
    tokens: list[ObservationToken] = []
    total = len(stir_events)
    for index, event in enumerate(stir_events):
        event_age = total - index
        actor = relative_role(player_index, event.player)
        tokens.append(
            RoundEventFieldToken("event_kind", "stir", event_age)
        )
        tokens.append(RoundEventFieldToken("actor", actor, event_age))
        tokens.append(
            RoundEventFieldToken("stir_kind", event.kind, event_age)
        )
        tokens.append(
            RoundEventFieldToken(
                "suit",
                None
                if event.new_suit is None
                else event.new_suit.value,
                event_age,
            )
        )
        tokens.append(
            RoundEventFieldToken("priority", event.priority, event_age)
        )
        for face_count in canonical_face_counts(event.cards):
            tokens.append(
                face_count_token(
                    face_count,
                    segment="stir_event",
                    role=actor,
                    event_age=event_age,
                )
            )
        if event.own_bottom_exchange is not None:
            tokens.extend(
                _own_bottom_exchange_tokens(
                    event.own_bottom_exchange,
                    event_age=event_age,
                    trigger="stir",
                )
            )
    return tuple(tokens)


def _own_bottom_exchange_tokens(
    event: BottomExchangeSnapshot,
    *,
    event_age: int,
    trigger: Literal["initial", "stir"],
) -> tuple[ObservationToken, ...]:
    tokens: list[ObservationToken] = []
    tokens.append(
        RoundEventFieldToken("event_kind", "own_exchange", event_age)
    )
    tokens.append(RoundEventFieldToken("actor", "self", event_age))
    tokens.append(RoundEventFieldToken("trigger", trigger, event_age))
    tokens.extend(
        _exchange_face_count_tokens(
            event.picked_up_bottom_cards,
            segment="own_exchange_pickup",
            event_age=event_age,
        )
    )
    tokens.extend(
        _exchange_face_count_tokens(
            event.discarded_bottom_cards,
            segment="own_exchange_discard",
            event_age=event_age,
        )
    )
    return tuple(tokens)


def _exchange_face_count_tokens(
    cards: list[Card],
    *,
    segment: Literal[
        "own_exchange_pickup",
        "own_exchange_discard",
    ],
    event_age: int,
) -> tuple[FaceCountToken, ...]:
    return tuple(
        face_count_token(
            face_count,
            segment=segment,
            role="self",
            event_age=event_age,
        )
        for face_count in canonical_face_counts(cards)
    )


def _hand_tokens(
    snapshot: StateSnapshot,
) -> tuple[ObservationToken, ...]:
    return tuple(
        face_count_token(
            face_count,
            segment="self_hand",
            role="self",
        )
        for face_count in canonical_face_counts(snapshot.player_hand)
    )


def _visible_bottom_tokens(
    snapshot: StateSnapshot,
) -> tuple[ObservationToken, ...]:
    return tuple(
        face_count_token(
            face_count,
            segment="visible_bottom",
        )
        for face_count in canonical_face_counts(snapshot.bottom_cards)
    )


def _play_record_tokens(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    history: tuple[HistoryTrick, ...],
) -> tuple[ObservationToken, ...]:
    tokens: list[ObservationToken] = []
    total_completed = len(history)
    for index, trick in enumerate(history):
        trick_age = total_completed - index
        _append_play_record_tokens(
            tokens=tokens,
            player_index=player_index,
            lead_player=trick.lead_player,
            slots=trick.slots,
            trick_age=trick_age,
            trick_state="completed",
        )
        tokens.append(
            TrickResultFieldToken(
                "winner",
                relative_role(player_index, trick.winner),
                trick_age,
            )
        )
        tokens.append(
            TrickResultFieldToken("points", trick.points, trick_age)
        )
        _append_failed_throw_tokens(
            tokens=tokens,
            player_index=player_index,
            failed_throw=trick.failed_throw,
            trick_age=trick_age,
            trick_state="completed",
        )
    if snapshot.trick is not None:
        _append_play_record_tokens(
            tokens=tokens,
            player_index=player_index,
            lead_player=snapshot.trick.lead_player,
            slots=tuple(snapshot.trick.slots),
            trick_age=0,
            trick_state="open",
        )
        _append_failed_throw_tokens(
            tokens=tokens,
            player_index=player_index,
            failed_throw=snapshot.trick.failed_throw,
            trick_age=0,
            trick_state="open",
        )
    return tuple(tokens)


def _append_play_record_tokens(
    *,
    tokens: list[ObservationToken],
    player_index: int,
    lead_player: int,
    slots: tuple[TrickSlotSnapshot, ...],
    trick_age: int,
    trick_state: TrickRecordState,
) -> None:
    sorted_slots = sorted(
        (slot for slot in slots if slot.cards),
        key=lambda slot: _play_order(
            lead_player=lead_player,
            player=slot.player,
        ),
    )
    for slot in sorted_slots:
        play_order = _play_order(
            lead_player=lead_player,
            player=slot.player,
        )
        actor = relative_role(player_index, slot.player)
        play_width = len(slot.cards)
        for face_count in canonical_face_counts(slot.cards):
            tokens.append(
                face_count_token(
                    face_count,
                    segment="play_record",
                    role=actor,
                    trick_age=trick_age,
                    trick_state=trick_state,
                    play_order=play_order,
                    play_width=play_width,
                )
            )


def _append_failed_throw_tokens(
    *,
    tokens: list[ObservationToken],
    player_index: int,
    failed_throw: FailedThrowSnapshot | None,
    trick_age: int,
    trick_state: TrickRecordState,
) -> None:
    if failed_throw is None:
        return
    actor = relative_role(player_index, failed_throw.player)
    for face_count in canonical_face_counts(
        failed_throw.attempted_cards
    ):
        tokens.append(
            face_count_token(
                face_count,
                segment="failed_throw_attempted",
                role=actor,
                trick_age=trick_age,
                trick_state=trick_state,
            )
        )
    for face_count in canonical_face_counts(failed_throw.forced_cards):
        tokens.append(
            face_count_token(
                face_count,
                segment="failed_throw_forced",
                role=actor,
                trick_age=trick_age,
                trick_state=trick_state,
            )
        )


def _action_query_tokens(
    query: ActionQuery,
) -> tuple[ActionQueryFieldToken, ...]:
    return (
        ActionQueryFieldToken("kind", query.kind),
        ActionQueryFieldToken("pass_allowed", query.pass_allowed),
        ActionQueryFieldToken("min_select", query.min_select),
        ActionQueryFieldToken("max_select", query.max_select),
        ActionQueryFieldToken("exact_select", query.exact_select),
        ActionQueryFieldToken(
            "action_play_order", query.action_play_order
        ),
        ActionQueryFieldToken(
            "current_trick_width", query.current_trick_width
        ),
        ActionQueryFieldToken("lead_actor", query.lead_actor),
        ActionQueryFieldToken("discard_count", query.discard_count),
        ActionQueryFieldToken(
            "trump_suit",
            None
            if query.trump_suit is None
            else query.trump_suit.value,
        ),
        ActionQueryFieldToken("level_rank", query.level_rank.value),
        ActionQueryFieldToken(
            "current_best_bid_role", query.current_best_bid_role
        ),
    )


def _hand_count(player: int, snapshot: StateSnapshot) -> int:
    if player >= len(snapshot.player_hand_counts):
        return 0
    return snapshot.player_hand_counts[player]


def _snapshot_has_open_play(snapshot: StateSnapshot) -> bool:
    trick = snapshot.trick
    if trick is None:
        return False
    return any(slot.cards for slot in trick.slots)


def _level_card_revealer_role(
    player_index: int, snapshot: StateSnapshot
) -> RelativeRole | None:
    winner = snapshot.bid_winner
    if winner is None:
        return None
    return relative_role(player_index, winner.player)


def _play_order(*, lead_player: int, player: int) -> int:
    if player >= lead_player:
        return player - lead_player
    return PLAYER_COUNT - lead_player + player


def _completed_trick_signature(
    completed: CompletedTrickSnapshot,
) -> tuple[object, ...]:
    slot_parts: list[object] = []
    for slot in completed.slots:
        slot_parts.append(slot.player)
        slot_parts.append(
            face_count_signature(canonical_face_counts(slot.cards))
        )
    failed_throw_signature: tuple[object, ...] | None = None
    if completed.failed_throw is not None:
        failed_throw_signature = (
            completed.failed_throw.player,
            face_count_signature(
                canonical_face_counts(
                    completed.failed_throw.attempted_cards
                )
            ),
            face_count_signature(
                canonical_face_counts(
                    completed.failed_throw.forced_cards
                )
            ),
        )
    return (
        completed.lead_player,
        completed.winner,
        completed.points,
        tuple(slot_parts),
        failed_throw_signature,
    )
