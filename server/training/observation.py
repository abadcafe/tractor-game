"""Build training observations from player-facing snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.protocol import (
    BidEventSnapshot,
    CompletedTrickSnapshot,
    StateSnapshot,
    TrickSlotSnapshot,
)
from server.sm.constants import BOTTOM_CARD_COUNT, PLAYER_COUNT
from server.training.action_tokens import (
    ActionQuery,
    build_action_query,
)
from server.training.progress import (
    DEFAULT_PROGRESS_CONFIG,
    ProgressConfig,
    distance_to_target,
    stage_target,
)
from server.training.tokens import (
    ActionQueryFieldToken,
    CardToken,
    GlobalFieldToken,
    ObservationToken,
    RelativeRole,
    RoundEventFieldToken,
    RoundFieldToken,
    TrickRecordState,
    TrickResultFieldToken,
    card_token,
    relative_role,
)


@dataclass(frozen=True, slots=True)
class HistoryTrick:
    """One completed public trick kept by the training observer."""

    lead_player: int
    slots: tuple[TrickSlotSnapshot, ...]
    winner: int
    points: int


@dataclass(frozen=True, slots=True)
class Observation:
    """Model-facing observation and its action-pointer query."""

    player_index: int
    tokens: tuple[ObservationToken, ...]
    hand_card_ids: tuple[str, ...]
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

    def update(self, snapshot: StateSnapshot) -> None:
        """Add a completed trick if this snapshot reveals a new one."""
        completed = snapshot.last_completed_trick
        if completed is None:
            return
        signature = _completed_trick_signature(completed)
        if signature in self._signatures:
            return
        self._signatures.add(signature)
        self._tricks.append(
            HistoryTrick(
                lead_player=completed.lead_player,
                slots=tuple(completed.slots),
                winner=completed.winner,
                points=completed.points,
            )
        )

    def tricks(self) -> tuple[HistoryTrick, ...]:
        """Return completed public tricks in observed order."""
        return tuple(self._tricks)

    def clear(self) -> None:
        """Clear per-round public trick history."""
        self._signatures.clear()
        self._tricks.clear()


def build_observation(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    history: tuple[HistoryTrick, ...],
    progress_config: ProgressConfig = DEFAULT_PROGRESS_CONFIG,
) -> Observation:
    """Build one training observation from player-visible data only."""
    action_query = build_action_query(
        player_index=player_index,
        snapshot=snapshot,
    )
    tokens: list[ObservationToken] = []
    tokens.extend(_global_tokens(progress_config))
    tokens.extend(
        _round_tokens(
            player_index=player_index,
            snapshot=snapshot,
            progress_config=progress_config,
        )
    )
    tokens.extend(
        _round_event_tokens(player_index, snapshot.bid_events)
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
        hand_card_ids=tuple(card.id for card in snapshot.player_hand),
        action_query=action_query,
    )


def card_tokens(observation: Observation) -> tuple[CardToken, ...]:
    """Return all card tokens in sequence order."""
    return tuple(
        token
        for token in observation.tokens
        if isinstance(token, CardToken)
    )


def _global_tokens(
    progress_config: ProgressConfig,
) -> tuple[GlobalFieldToken, ...]:
    tokens: list[GlobalFieldToken] = [
        GlobalFieldToken("team_layout", "fixed_partner_opposite"),
        GlobalFieldToken("left_player_role", "left_enemy"),
        GlobalFieldToken("right_player_role", "right_enemy"),
        GlobalFieldToken("partner_role", "partner"),
        GlobalFieldToken("deck_count", 2),
        GlobalFieldToken("player_count", PLAYER_COUNT),
        GlobalFieldToken("bottom_card_count", BOTTOM_CARD_COUNT),
        GlobalFieldToken("rules_version", "base-A"),
    ]
    for level in progress_config.required_levels:
        tokens.append(GlobalFieldToken("required_level", level.value))
    tokens.append(GlobalFieldToken("final_target", "WIN"))
    return tuple(tokens)


def _round_tokens(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    progress_config: ProgressConfig,
) -> tuple[RoundFieldToken, ...]:
    self_team = player_index % 2
    self_level = (
        snapshot.team0_level if self_team == 0 else snapshot.team1_level
    )
    enemy_level = (
        snapshot.team1_level if self_team == 0 else snapshot.team0_level
    )
    self_target = stage_target(self_level, progress_config)
    enemy_target = stage_target(enemy_level, progress_config)
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
        RoundFieldToken("self_team_level", self_level.value),
        RoundFieldToken("enemy_team_level", enemy_level.value),
        RoundFieldToken(
            "self_team_required_level",
            self_target if self_target == "WIN" else self_target.value,
        ),
        RoundFieldToken(
            "enemy_team_required_level",
            enemy_target
            if enemy_target == "WIN"
            else enemy_target.value,
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


def _round_event_tokens(
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
        for card_order, card in enumerate(event.cards):
            tokens.append(
                card_token(
                    card,
                    segment="round_event",
                    role=actor,
                    card_order=card_order,
                    event_age=event_age,
                )
            )
    return tuple(tokens)


def _hand_tokens(
    snapshot: StateSnapshot,
) -> tuple[ObservationToken, ...]:
    return tuple(
        card_token(
            card,
            segment="self_hand",
            role="self",
            card_order=index,
        )
        for index, card in enumerate(snapshot.player_hand)
    )


def _visible_bottom_tokens(
    snapshot: StateSnapshot,
) -> tuple[ObservationToken, ...]:
    return tuple(
        card_token(
            card,
            segment="visible_bottom",
            card_order=index,
        )
        for index, card in enumerate(snapshot.bottom_cards)
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
    if snapshot.trick is not None:
        _append_play_record_tokens(
            tokens=tokens,
            player_index=player_index,
            lead_player=snapshot.trick.lead_player,
            slots=tuple(snapshot.trick.slots),
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
        for card_order, card in enumerate(slot.cards):
            tokens.append(
                card_token(
                    card,
                    segment="play_record",
                    role=actor,
                    trick_age=trick_age,
                    trick_state=trick_state,
                    play_order=play_order,
                    card_order=card_order,
                    play_width=play_width,
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
            "selection_source", query.selection_source
        ),
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
        slot_parts.extend(card.id for card in slot.cards)
    return (
        completed.lead_player,
        completed.winner,
        completed.points,
        tuple(slot_parts),
    )
