"""Selection-based action grammar for model-generated player actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.protocol import StateSnapshot
from server.result import Ok, Rejected
from server.rules.cards import Rank, Suit
from server.training.tokens import RelativeRole, relative_role

type PlayerActionKind = Literal["bid", "stir", "discard", "play"]
type DecisionKind = Literal[
    "bid", "stir", "discard", "lead_play", "follow_play"
]
type SelectionChoiceKind = Literal["pass", "stop", "select_card"]
type SelectionSource = Literal["hand"]

MAX_HAND_CARD_SLOTS: int = 33
MAX_SELECTION_CHOICES: int = MAX_HAND_CARD_SLOTS + 1


@dataclass(frozen=True, slots=True)
class ActionQuery:
    """Player-visible decision shape for observation and decoding."""

    kind: DecisionKind | None
    hand_card_ids: tuple[str, ...]
    pass_allowed: bool
    min_select: int
    max_select: int
    exact_select: int | None
    selection_source: SelectionSource | None
    action_play_order: int | None
    current_trick_width: int | None
    lead_actor: RelativeRole | None
    discard_count: int | None
    trump_suit: Suit | None
    level_rank: Rank
    current_best_bid_role: RelativeRole | None


@dataclass(frozen=True, slots=True)
class SelectionState:
    """Current unordered set of selected hand slots."""

    selected_slots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SelectionChoice:
    """One incremental choice made by a policy head."""

    kind: SelectionChoiceKind
    slot: int | None = None


@dataclass(frozen=True, slots=True)
class SelectionTrace:
    """Full choice trace for one generated action."""

    choices: tuple[SelectionChoice, ...]


@dataclass(frozen=True, slots=True)
class GeneratedAction:
    """One model-generated action ready for Game.receive()."""

    raw: dict[str, object]
    selection_trace: SelectionTrace
    action_kind: PlayerActionKind | Literal["pass"]
    card_ids: tuple[str, ...]
    selected_slots: tuple[int, ...]


class InvalidSelectionRejected(Rejected):
    """Selection sequence violated the model action grammar."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"动作选择非法：{reason}")


def build_action_query(
    *,
    player_index: int,
    snapshot: StateSnapshot,
) -> ActionQuery:
    """Build the structured player-visible decision request."""
    kind = _decision_kind(snapshot)
    hand_size = min(len(snapshot.player_hand), MAX_HAND_CARD_SLOTS)
    pass_allowed = kind in ("bid", "stir")
    min_select, max_select, exact_select = _selection_shape(
        kind=kind,
        hand_size=hand_size,
        snapshot=snapshot,
    )
    action_play_order = _action_play_order(snapshot)
    current_trick_width = _current_trick_width(snapshot)
    return ActionQuery(
        kind=kind,
        hand_card_ids=tuple(card.id for card in snapshot.player_hand),
        pass_allowed=pass_allowed,
        min_select=min_select,
        max_select=max_select,
        exact_select=exact_select,
        selection_source="hand" if kind is not None else None,
        action_play_order=action_play_order,
        current_trick_width=current_trick_width,
        lead_actor=_lead_actor(player_index, snapshot),
        discard_count=_discard_count(snapshot)
        if kind == "discard"
        else None,
        trump_suit=snapshot.trump_suit,
        level_rank=snapshot.trump_rank,
        current_best_bid_role=_current_best_bid_role(
            player_index, snapshot
        ),
    )


def valid_selection_choices(
    query: ActionQuery,
    state: SelectionState,
) -> tuple[SelectionChoice, ...]:
    """Return syntactically valid next choices for a selection state."""
    if query.kind is None:
        return ()
    if _has_duplicate_slots(state.selected_slots):
        return ()
    selected_count = len(state.selected_slots)
    if _auto_complete(query, selected_count):
        return ()
    choices: list[SelectionChoice] = []
    if selected_count == 0 and query.pass_allowed:
        choices.append(SelectionChoice("pass"))
    if _can_stop(query, selected_count):
        choices.append(SelectionChoice("stop"))
    if selected_count < query.max_select:
        used = set(state.selected_slots)
        hand_size = min(len(query.hand_card_ids), MAX_HAND_CARD_SLOTS)
        for slot in range(hand_size):
            if slot not in used:
                choices.append(SelectionChoice("select_card", slot))
    return tuple(choices)


def selection_state_after(
    trace: SelectionTrace,
) -> Ok[SelectionState] | Rejected:
    """Return selected-slot state after applying a trace."""
    selected_slots: list[int] = []
    for choice in trace.choices:
        if choice.kind == "select_card":
            if choice.slot is None:
                return InvalidSelectionRejected(
                    "select_card 缺少手牌槽位"
                )
            selected_slots.append(choice.slot)
    return Ok(
        value=SelectionState(selected_slots=tuple(selected_slots))
    )


def decode_selection_action(
    query: ActionQuery,
    trace: SelectionTrace,
) -> Ok[GeneratedAction] | Rejected:
    """Convert a full selection trace into a raw player message."""
    validation = _validate_trace(query, trace)
    if isinstance(validation, Rejected):
        return validation
    state = validation.value
    terminal = trace.choices[-1].kind if trace.choices else None
    if terminal == "pass":
        return _decode_pass_action(query, trace)
    card_ids_result = _card_ids_from_slots(query, state.selected_slots)
    if isinstance(card_ids_result, Rejected):
        return card_ids_result
    action_kind = _player_action_kind(query)
    if action_kind is None:
        return InvalidSelectionRejected("当前没有动作请求")
    return Ok(
        value=GeneratedAction(
            raw={
                "type": action_kind,
                "cards": list(card_ids_result.value),
            },
            selection_trace=trace,
            action_kind=action_kind,
            card_ids=card_ids_result.value,
            selected_slots=state.selected_slots,
        )
    )


def selection_choice_name(choice: SelectionChoice) -> str:
    """Return a stable human-readable choice name for diagnostics."""
    if choice.kind == "select_card":
        assert choice.slot is not None
        return f"SELECT_CARD_{choice.slot}"
    return choice.kind.upper()


def _decision_kind(snapshot: StateSnapshot) -> DecisionKind | None:
    if snapshot.awaiting_action == "bid":
        return "bid"
    if snapshot.awaiting_action == "stir":
        return "stir"
    if snapshot.awaiting_action == "discard":
        return "discard"
    if snapshot.awaiting_action == "play":
        order = _action_play_order(snapshot)
        if order is None or order == 0:
            return "lead_play"
        return "follow_play"
    return None


def _player_action_kind(query: ActionQuery) -> PlayerActionKind | None:
    if query.kind in ("bid", "stir", "discard"):
        return query.kind
    if query.kind in ("lead_play", "follow_play"):
        return "play"
    return None


def _selection_shape(
    *,
    kind: DecisionKind | None,
    hand_size: int,
    snapshot: StateSnapshot,
) -> tuple[int, int, int | None]:
    if kind is None:
        return 0, 0, None
    if kind in ("bid", "stir"):
        return (1 if hand_size > 0 else 0), min(4, hand_size), None
    if kind == "discard":
        count = min(_discard_count(snapshot), hand_size)
        return count, count, count
    if kind == "follow_play":
        width = _current_trick_width(snapshot)
        assert width is not None
        exact = min(width, hand_size)
        return exact, exact, exact
    return (1 if hand_size > 0 else 0), hand_size, None


def _discard_count(snapshot: StateSnapshot) -> int:
    stir = snapshot.stirring_state
    if stir is not None and stir.exchange_count is not None:
        return stir.exchange_count
    return 8


def _current_trick_width(snapshot: StateSnapshot) -> int | None:
    trick = snapshot.trick
    if trick is None:
        return None
    for slot in trick.slots:
        if slot.player == trick.lead_player and slot.cards:
            return len(slot.cards)
    return None


def _action_play_order(snapshot: StateSnapshot) -> int | None:
    trick = snapshot.trick
    if snapshot.awaiting_action != "play" or trick is None:
        return None
    return _play_order(
        lead_player=trick.lead_player, player=trick.current_player
    )


def _lead_actor(
    player_index: int, snapshot: StateSnapshot
) -> RelativeRole | None:
    trick = snapshot.trick
    if snapshot.awaiting_action != "play" or trick is None:
        return None
    return relative_role(player_index, trick.lead_player)


def _current_best_bid_role(
    player_index: int, snapshot: StateSnapshot
) -> RelativeRole | None:
    winner = snapshot.bid_winner
    if winner is None:
        return None
    return relative_role(player_index, winner.player)


def _play_order(*, lead_player: int, player: int) -> int:
    if player >= lead_player:
        return player - lead_player
    return 4 - lead_player + player


def _auto_complete(query: ActionQuery, selected_count: int) -> bool:
    return query.kind in ("discard", "follow_play") and (
        query.exact_select is not None
        and selected_count == query.exact_select
    )


def _can_stop(query: ActionQuery, selected_count: int) -> bool:
    if query.kind not in ("bid", "stir", "lead_play"):
        return False
    if selected_count < query.min_select:
        return False
    return selected_count > 0


def _has_duplicate_slots(slots: tuple[int, ...]) -> bool:
    return len(slots) != len(set(slots))


def _validate_trace(
    query: ActionQuery,
    trace: SelectionTrace,
) -> Ok[SelectionState] | Rejected:
    state = SelectionState(selected_slots=())
    final_index = len(trace.choices) - 1
    for index, choice in enumerate(trace.choices):
        allowed = valid_selection_choices(query, state)
        if choice not in allowed:
            return InvalidSelectionRejected("选择不满足语法 mask")
        if choice.kind in ("pass", "stop"):
            if index != final_index:
                return InvalidSelectionRejected(
                    "终止选择后还有额外选择"
                )
            return Ok(value=state)
        assert choice.kind == "select_card"
        assert choice.slot is not None
        state = SelectionState(
            selected_slots=(*state.selected_slots, choice.slot)
        )
    if _auto_complete(query, len(state.selected_slots)):
        return Ok(value=state)
    return InvalidSelectionRejected("选择未完成")


def _decode_pass_action(
    query: ActionQuery,
    trace: SelectionTrace,
) -> Ok[GeneratedAction] | Rejected:
    if query.kind == "bid" and query.pass_allowed:
        return Ok(
            value=GeneratedAction(
                raw={"type": "bid", "pass": True},
                selection_trace=trace,
                action_kind="pass",
                card_ids=(),
                selected_slots=(),
            )
        )
    if query.kind == "stir" and query.pass_allowed:
        return Ok(
            value=GeneratedAction(
                raw={"type": "stir", "pass": True},
                selection_trace=trace,
                action_kind="pass",
                card_ids=(),
                selected_slots=(),
            )
        )
    return InvalidSelectionRejected("当前阶段不能 pass")


def _card_ids_from_slots(
    query: ActionQuery,
    slots: tuple[int, ...],
) -> Ok[tuple[str, ...]] | Rejected:
    card_ids: list[str] = []
    used_slots: set[int] = set()
    for slot in slots:
        if slot in used_slots:
            return InvalidSelectionRejected("同一张手牌被选择多次")
        if slot < 0 or slot >= len(query.hand_card_ids):
            return InvalidSelectionRejected("牌槽超出当前手牌")
        used_slots.add(slot)
        card_ids.append(query.hand_card_ids[slot])
    return Ok(value=tuple(card_ids))
