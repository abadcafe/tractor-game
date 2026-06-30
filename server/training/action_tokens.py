"""Action token grammar for model-generated player actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.protocol import StateSnapshot
from server.result import Ok, Rejected
from server.rules.cards import Rank, Suit
from server.training.tokens import RelativeRole, relative_role

type ModelActionKind = Literal["bid", "stir", "discard", "play"]
type SelectionSource = Literal["hand"]

PAD_TOKEN_ID: int = 0
BEGIN_TOKEN_ID: int = 1
STOP_TOKEN_ID: int = 2
ACTION_PASS_TOKEN_ID: int = 3
ACTION_BID_TOKEN_ID: int = 4
ACTION_STIR_TOKEN_ID: int = 5
ACTION_DISCARD_TOKEN_ID: int = 6
ACTION_PLAY_TOKEN_ID: int = 7
FIRST_CARD_TOKEN_ID: int = 8
MAX_HAND_CARD_SLOTS: int = 33
ACTION_TOKEN_VOCAB_SIZE: int = FIRST_CARD_TOKEN_ID + MAX_HAND_CARD_SLOTS
MAX_ACTION_TOKENS: int = MAX_HAND_CARD_SLOTS + 3

_ACTION_KIND_BY_TOKEN_ID: dict[int, ModelActionKind] = {
    ACTION_BID_TOKEN_ID: "bid",
    ACTION_STIR_TOKEN_ID: "stir",
    ACTION_DISCARD_TOKEN_ID: "discard",
    ACTION_PLAY_TOKEN_ID: "play",
}


@dataclass(frozen=True, slots=True)
class ActionQuery:
    """Player-visible decision shape for observation and decoding."""

    kind: ModelActionKind | None
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
class GeneratedAction:
    """One model-generated action ready for Game.receive()."""

    raw: dict[str, object]
    token_ids: tuple[int, ...]
    action_kind: ModelActionKind | Literal["pass"]
    card_ids: tuple[str, ...]


class InvalidActionTokensRejected(Rejected):
    """Action token sequence violated the model action grammar."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"动作 token 非法：{reason}")


def build_action_query(
    *,
    player_index: int,
    snapshot: StateSnapshot,
) -> ActionQuery:
    """Build the structured player-visible decision request."""
    kind = _model_action_kind(snapshot)
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


def valid_next_token_ids(
    query: ActionQuery,
    prefix: tuple[int, ...],
) -> tuple[int, ...]:
    """Return syntactically valid next tokens for one action prefix."""
    if not prefix:
        return (BEGIN_TOKEN_ID,)
    if prefix[0] != BEGIN_TOKEN_ID:
        return ()
    if STOP_TOKEN_ID in prefix:
        return ()
    if len(prefix) == 1:
        return _first_action_tokens(query)
    action_token = prefix[1]
    if action_token == ACTION_PASS_TOKEN_ID:
        return (STOP_TOKEN_ID,)
    if action_token not in _ACTION_KIND_BY_TOKEN_ID:
        return ()
    selected_slots = _selected_card_slots(prefix)
    selected_count = len(selected_slots)
    tokens: list[int] = []
    if _can_stop_card_action(query, selected_count):
        tokens.append(STOP_TOKEN_ID)
    if selected_count < query.max_select:
        used = set(selected_slots)
        for slot in range(
            min(len(query.hand_card_ids), MAX_HAND_CARD_SLOTS)
        ):
            if slot not in used:
                tokens.append(FIRST_CARD_TOKEN_ID + slot)
    return tuple(tokens)


def decode_action_tokens(
    query: ActionQuery,
    token_ids: tuple[int, ...],
) -> Ok[GeneratedAction] | Rejected:
    """Convert a full action token sequence into raw player message."""
    if len(token_ids) < 3:
        return InvalidActionTokensRejected("序列太短")
    if token_ids[0] != BEGIN_TOKEN_ID:
        return InvalidActionTokensRejected("缺少 BEGIN")
    if token_ids[-1] != STOP_TOKEN_ID:
        return InvalidActionTokensRejected("缺少 STOP")
    if not _prefix_is_valid(query, token_ids):
        return InvalidActionTokensRejected("序列不满足语法 mask")
    action_token = token_ids[1]
    if action_token == ACTION_PASS_TOKEN_ID:
        return _decode_pass_action(query, token_ids)
    action_kind = _ACTION_KIND_BY_TOKEN_ID.get(action_token)
    if action_kind is None:
        return InvalidActionTokensRejected("未知动作类型 token")
    if action_kind != query.kind:
        return InvalidActionTokensRejected("动作类型不匹配当前请求")
    card_ids_result = _card_ids_from_tokens(query, token_ids[2:-1])
    if isinstance(card_ids_result, Rejected):
        return card_ids_result
    return Ok(
        value=GeneratedAction(
            raw={
                "type": action_kind,
                "cards": list(card_ids_result.value),
            },
            token_ids=token_ids,
            action_kind=action_kind,
            card_ids=card_ids_result.value,
        )
    )


def token_name(token_id: int) -> str:
    """Return a stable human-readable token name for diagnostics."""
    if token_id == PAD_TOKEN_ID:
        return "PAD"
    if token_id == BEGIN_TOKEN_ID:
        return "BEGIN"
    if token_id == STOP_TOKEN_ID:
        return "STOP"
    if token_id == ACTION_PASS_TOKEN_ID:
        return "ACTION_PASS"
    if token_id == ACTION_BID_TOKEN_ID:
        return "ACTION_BID"
    if token_id == ACTION_STIR_TOKEN_ID:
        return "ACTION_STIR"
    if token_id == ACTION_DISCARD_TOKEN_ID:
        return "ACTION_DISCARD"
    if token_id == ACTION_PLAY_TOKEN_ID:
        return "ACTION_PLAY"
    if _is_card_token_id(token_id):
        return f"CARD_SLOT_{token_id - FIRST_CARD_TOKEN_ID}"
    return f"UNKNOWN_{token_id}"


def _model_action_kind(
    snapshot: StateSnapshot,
) -> ModelActionKind | None:
    if snapshot.awaiting_action == "bid":
        return "bid"
    if snapshot.awaiting_action == "stir":
        return "stir"
    if snapshot.awaiting_action == "discard":
        return "discard"
    if snapshot.awaiting_action == "play":
        return "play"
    return None


def _selection_shape(
    *,
    kind: ModelActionKind | None,
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
    width = _current_trick_width(snapshot)
    play_order = _action_play_order(snapshot)
    if width is not None and play_order is not None and play_order > 0:
        exact = min(width, hand_size)
        return exact, exact, exact
    return (1 if hand_size > 0 else 0), hand_size, None


def _first_action_tokens(query: ActionQuery) -> tuple[int, ...]:
    if query.kind == "bid":
        if query.pass_allowed:
            return (ACTION_PASS_TOKEN_ID, ACTION_BID_TOKEN_ID)
        return (ACTION_BID_TOKEN_ID,)
    if query.kind == "stir":
        if query.pass_allowed:
            return (ACTION_PASS_TOKEN_ID, ACTION_STIR_TOKEN_ID)
        return (ACTION_STIR_TOKEN_ID,)
    if query.kind == "discard":
        return (ACTION_DISCARD_TOKEN_ID,)
    if query.kind == "play":
        return (ACTION_PLAY_TOKEN_ID,)
    return ()


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


def _can_stop_card_action(
    query: ActionQuery,
    selected_count: int,
) -> bool:
    if selected_count < query.min_select:
        return False
    if query.exact_select is not None:
        return selected_count == query.exact_select
    return selected_count > 0


def _selected_card_slots(prefix: tuple[int, ...]) -> tuple[int, ...]:
    slots: list[int] = []
    for token_id in prefix[2:]:
        if _is_card_token_id(token_id):
            slots.append(token_id - FIRST_CARD_TOKEN_ID)
    return tuple(slots)


def _prefix_is_valid(
    query: ActionQuery,
    token_ids: tuple[int, ...],
) -> bool:
    prefix: tuple[int, ...] = ()
    for token_id in token_ids:
        if token_id not in valid_next_token_ids(query, prefix):
            return False
        prefix = (*prefix, token_id)
    return True


def _decode_pass_action(
    query: ActionQuery,
    token_ids: tuple[int, ...],
) -> Ok[GeneratedAction] | Rejected:
    if query.kind == "bid" and query.pass_allowed:
        return Ok(
            value=GeneratedAction(
                raw={"type": "bid", "pass": True},
                token_ids=token_ids,
                action_kind="pass",
                card_ids=(),
            )
        )
    if query.kind == "stir" and query.pass_allowed:
        return Ok(
            value=GeneratedAction(
                raw={"type": "stir", "pass": True},
                token_ids=token_ids,
                action_kind="pass",
                card_ids=(),
            )
        )
    return InvalidActionTokensRejected("当前阶段不能 pass")


def _card_ids_from_tokens(
    query: ActionQuery,
    token_ids: tuple[int, ...],
) -> Ok[tuple[str, ...]] | Rejected:
    card_ids: list[str] = []
    used_slots: set[int] = set()
    for token_id in token_ids:
        if not _is_card_token_id(token_id):
            return InvalidActionTokensRejected(
                "牌 token 位置包含非牌 token"
            )
        slot = token_id - FIRST_CARD_TOKEN_ID
        if slot in used_slots:
            return InvalidActionTokensRejected("同一张手牌被选择多次")
        if slot >= len(query.hand_card_ids):
            return InvalidActionTokensRejected("牌槽超出当前手牌")
        used_slots.add(slot)
        card_ids.append(query.hand_card_ids[slot])
    return Ok(value=tuple(card_ids))


def _is_card_token_id(token_id: int) -> bool:
    return (
        FIRST_CARD_TOKEN_ID
        <= token_id
        < FIRST_CARD_TOKEN_ID + MAX_HAND_CARD_SLOTS
    )
