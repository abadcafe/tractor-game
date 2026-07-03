"""Rule-complete semantic action spaces for training policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from server.protocol import StateSnapshot
from server.result import Ok, Rejected
from server.rules import bid as bid_rules
from server.rules.card_faces import (
    CardFace,
    FaceCount,
    canonical_face_counts,
    face_count_width,
    face_sort_key,
)
from server.rules.cards import Card, Rank, Suit
from server.rules.follow_action_space import (
    FollowActionSpace,
    build_follow_action_space,
)
from server.rules.ordering import bid_value, effective_suit
from server.training.semantic_actions import (
    ActionQuery,
    GeneratedAction,
    InvalidSemanticActionRejected,
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
    build_action_query,
    semantic_prefix_state,
)

type _CanComplete = Callable[[tuple[FaceCount, ...]], bool]


class LegalActionIndex:
    """Rule-complete next-argument mask for one player decision."""

    @property
    def query(self) -> ActionQuery:
        """Return the action query this legal index answers."""
        raise NotImplementedError

    def allowed_next(
        self, prefix: SemanticArgumentPrefix
    ) -> tuple[SemanticArgument, ...]:
        """Return legal next semantic arguments after the prefix."""
        raise NotImplementedError

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        """Decode a complete legal trace into a generated action."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class EmptyLegalActionIndex(LegalActionIndex):
    """No action is legal because the snapshot awaits nothing."""

    _query: ActionQuery

    @property
    def query(self) -> ActionQuery:
        return self._query

    def allowed_next(
        self, prefix: SemanticArgumentPrefix
    ) -> tuple[SemanticArgument, ...]:
        return ()

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        return InvalidSemanticActionRejected("当前没有动作请求")


@dataclass(frozen=True, slots=True)
class CompleteTraceLegalActionIndex(LegalActionIndex):
    """Legal index backed by a closed set of complete action traces."""

    _query: ActionQuery
    _actions: tuple[GeneratedAction, ...]

    @property
    def query(self) -> ActionQuery:
        return self._query

    def allowed_next(
        self, prefix: SemanticArgumentPrefix
    ) -> tuple[SemanticArgument, ...]:
        result: list[SemanticArgument] = []
        for action in self._actions:
            trace_args = action.semantic_trace.arguments
            prefix_args = prefix.arguments
            if len(prefix_args) >= len(trace_args):
                continue
            if trace_args[: len(prefix_args)] != prefix_args:
                continue
            argument = trace_args[len(prefix_args)]
            if argument not in result:
                result.append(argument)
        return tuple(result)

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        for action in self._actions:
            if action.semantic_trace == trace:
                return Ok(value=action)
        return InvalidSemanticActionRejected(
            "动作不在当前规则合法集合内"
        )


@dataclass(slots=True)
class DiscardLegalActionIndex(LegalActionIndex):
    """Exact-card-count discard action space."""

    _query: ActionQuery

    @property
    def query(self) -> ActionQuery:
        return self._query

    def allowed_next(
        self, prefix: SemanticArgumentPrefix
    ) -> tuple[SemanticArgument, ...]:
        selected_result = semantic_prefix_state(prefix)
        if isinstance(selected_result, Rejected):
            return ()
        selected = selected_result.value
        selected_count = face_count_width(selected)
        if selected_count >= self._required_count():
            return ()
        return _select_arguments(
            query=self._query,
            selected=selected,
            can_complete=self._discard_can_complete,
        )

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        if not _trace_is_selection_only(trace):
            return InvalidSemanticActionRejected(
                "exact-count 动作不能包含终止参数"
            )
        selected_result = semantic_prefix_state(
            SemanticArgumentPrefix(arguments=trace.arguments)
        )
        if isinstance(selected_result, Rejected):
            return selected_result
        selected = selected_result.value
        if face_count_width(selected) != self._required_count():
            return InvalidSemanticActionRejected("埋牌数量不满足规则")
        return Ok(
            value=GeneratedAction(
                action_kind="discard",
                message_type="discard",
                face_counts=selected,
                semantic_trace=trace,
                is_pass=False,
            )
        )

    def _discard_can_complete(
        self, selected: tuple[FaceCount, ...]
    ) -> bool:
        selected_count = face_count_width(selected)
        if selected_count > self._required_count():
            return False
        if selected_count == self._required_count():
            return True
        return (
            _remaining_count_after_selected(
                hand_faces=self._query.hand_faces,
                selected=selected,
            )
            >= self._required_count() - selected_count
        )

    def _required_count(self) -> int:
        assert self._query.exact_select is not None
        return self._query.exact_select


@dataclass(slots=True)
class LeadPlayLegalActionIndex(LegalActionIndex):
    """Leading action space: submit any non-empty one-suit throw."""

    _query: ActionQuery
    _hand_cards: tuple[Card, ...]

    @property
    def query(self) -> ActionQuery:
        return self._query

    def allowed_next(
        self, prefix: SemanticArgumentPrefix
    ) -> tuple[SemanticArgument, ...]:
        selected_result = semantic_prefix_state(prefix)
        if isinstance(selected_result, Rejected):
            return ()
        selected = selected_result.value
        selected_count = face_count_width(selected)
        choices: list[SemanticArgument] = []
        if selected_count > 0 and _one_effective_suit(
            selected,
            hand_cards=self._hand_cards,
            trump_suit=self._query.trump_suit,
            trump_rank=self._query.level_rank,
        ):
            choices.append(SemanticArgument("stop"))
        if selected_count >= self._query.max_select:
            return tuple(choices)
        choices.extend(
            _select_arguments(
                query=self._query,
                selected=selected,
                can_complete=self._lead_can_complete,
            )
        )
        return tuple(choices)

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        if not trace.arguments or trace.arguments[-1].kind != "stop":
            return InvalidSemanticActionRejected("领牌动作必须 stop")
        selected_result = semantic_prefix_state(
            SemanticArgumentPrefix(arguments=trace.arguments[:-1])
        )
        if isinstance(selected_result, Rejected):
            return selected_result
        selected = selected_result.value
        if face_count_width(selected) == 0:
            return InvalidSemanticActionRejected("领牌不能为空")
        if not _one_effective_suit(
            selected,
            hand_cards=self._hand_cards,
            trump_suit=self._query.trump_suit,
            trump_rank=self._query.level_rank,
        ):
            return InvalidSemanticActionRejected("领牌必须同一有效花色")
        return Ok(
            value=GeneratedAction(
                action_kind="play",
                message_type="play",
                face_counts=selected,
                semantic_trace=trace,
                is_pass=False,
            )
        )

    def _lead_can_complete(
        self, selected: tuple[FaceCount, ...]
    ) -> bool:
        selected_count = face_count_width(selected)
        if selected_count == 0:
            return True
        if selected_count > self._query.max_select:
            return False
        return _one_effective_suit(
            selected,
            hand_cards=self._hand_cards,
            trump_suit=self._query.trump_suit,
            trump_rank=self._query.level_rank,
        )


@dataclass(slots=True)
class FollowPlayLegalActionIndex(LegalActionIndex):
    """Following action space using the full follow-rule validator."""

    _query: ActionQuery
    _space: FollowActionSpace

    @property
    def query(self) -> ActionQuery:
        return self._query

    def allowed_next(
        self, prefix: SemanticArgumentPrefix
    ) -> tuple[SemanticArgument, ...]:
        selected_result = semantic_prefix_state(prefix)
        if isinstance(selected_result, Rejected):
            return ()
        selected = selected_result.value
        if (
            face_count_width(selected)
            >= self._space.analysis.lead_count
        ):
            return ()
        return tuple(
            SemanticArgument("select_face_count", face_count)
            for face_count in self._space.allowed_next(selected)
        )

    def decode(
        self, trace: SemanticArgumentTrace
    ) -> Ok[GeneratedAction] | Rejected:
        if not _trace_is_selection_only(trace):
            return InvalidSemanticActionRejected(
                "exact-count 动作不能包含终止参数"
            )
        selected_result = semantic_prefix_state(
            SemanticArgumentPrefix(arguments=trace.arguments)
        )
        if isinstance(selected_result, Rejected):
            return selected_result
        selected = selected_result.value
        decoded = self._space.decode(selected)
        if isinstance(decoded, Rejected):
            return decoded
        return Ok(
            value=GeneratedAction(
                action_kind="play",
                message_type="play",
                face_counts=selected,
                semantic_trace=trace,
                is_pass=False,
            )
        )


def build_legal_action_index(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    query: ActionQuery | None = None,
) -> LegalActionIndex:
    """Build the rule-complete action index for a snapshot."""
    action_query = (
        build_action_query(player_index=player_index, snapshot=snapshot)
        if query is None
        else query
    )
    if action_query.kind is None:
        return EmptyLegalActionIndex(action_query)
    if action_query.kind == "bid":
        return _bid_index(
            player_index=player_index,
            snapshot=snapshot,
            query=action_query,
        )
    if action_query.kind == "stir":
        return _stir_index(
            player_index=player_index,
            snapshot=snapshot,
            query=action_query,
        )
    if action_query.kind == "discard":
        return DiscardLegalActionIndex(action_query)
    if action_query.kind == "lead_play":
        return LeadPlayLegalActionIndex(
            action_query, tuple(snapshot.player_hand)
        )
    if action_query.kind == "follow_play":
        lead_cards = _lead_cards(snapshot)
        assert lead_cards
        space_result = build_follow_action_space(
            hand=snapshot.player_hand,
            lead_cards=lead_cards,
            trump_suit=action_query.trump_suit,
            trump_rank=action_query.level_rank,
        )
        assert isinstance(space_result, Ok)
        return FollowPlayLegalActionIndex(
            action_query,
            space_result.value,
        )
    assert False


def _bid_index(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    query: ActionQuery,
) -> CompleteTraceLegalActionIndex:
    actions: list[GeneratedAction] = [_pass_action("bid")]
    if snapshot.bid_winner is None or (
        snapshot.bid_winner.player != player_index
    ):
        current_cards = (
            None
            if snapshot.bid_winner is None
            else snapshot.bid_winner.cards
        )
        for cards in bid_rules.legal_bid_hints(
            snapshot.player_hand,
            snapshot.trump_rank,
            current_cards,
        ):
            actions.append(_selection_action("bid", cards))
    return CompleteTraceLegalActionIndex(query, tuple(actions))


def _stir_index(
    *,
    player_index: int,
    snapshot: StateSnapshot,
    query: ActionQuery,
) -> CompleteTraceLegalActionIndex:
    actions: list[GeneratedAction] = [_pass_action("stir")]
    if _last_stir_player(snapshot) != player_index:
        current_priority = _current_stir_priority(snapshot)
        for candidate in bid_rules.bid_card_candidates(
            snapshot.player_hand,
            snapshot.trump_rank,
        ):
            if len(candidate) != 2:
                continue
            if bid_value(candidate, snapshot.trump_rank) <= (
                current_priority
            ):
                continue
            actions.append(_selection_action("stir", candidate))
    return CompleteTraceLegalActionIndex(query, tuple(actions))


def _pass_action(message_type: str) -> GeneratedAction:
    assert message_type in ("bid", "stir")
    trace = SemanticArgumentTrace(arguments=(SemanticArgument("pass"),))
    if message_type == "bid":
        return GeneratedAction(
            action_kind="pass",
            message_type="bid",
            face_counts=(),
            semantic_trace=trace,
            is_pass=True,
        )
    return GeneratedAction(
        action_kind="pass",
        message_type="stir",
        face_counts=(),
        semantic_trace=trace,
        is_pass=True,
    )


def _selection_action(
    message_type: str, cards: Sequence[Card]
) -> GeneratedAction:
    assert message_type in ("bid", "stir")
    face_counts = canonical_face_counts(tuple(cards))
    trace = _trace_for_selection(face_counts, include_stop=True)
    if message_type == "bid":
        return GeneratedAction(
            action_kind="bid",
            message_type="bid",
            face_counts=face_counts,
            semantic_trace=trace,
            is_pass=False,
        )
    return GeneratedAction(
        action_kind="stir",
        message_type="stir",
        face_counts=face_counts,
        semantic_trace=trace,
        is_pass=False,
    )


def _trace_for_selection(
    face_counts: tuple[FaceCount, ...], *, include_stop: bool
) -> SemanticArgumentTrace:
    arguments = [
        SemanticArgument("select_face_count", face_count)
        for face_count in face_counts
    ]
    if include_stop:
        arguments.append(SemanticArgument("stop"))
    return SemanticArgumentTrace(arguments=tuple(arguments))


def _select_arguments(
    *,
    query: ActionQuery,
    selected: tuple[FaceCount, ...],
    can_complete: _CanComplete,
) -> tuple[SemanticArgument, ...]:
    return tuple(
        argument
        for argument in _raw_select_arguments(
            query=query, selected=selected
        )
        if can_complete((*selected, _required_face_count(argument)))
    )


def _raw_select_arguments(
    *,
    query: ActionQuery,
    selected: tuple[FaceCount, ...],
) -> tuple[SemanticArgument, ...]:
    selected_count = face_count_width(selected)
    if selected_count >= query.max_select:
        return ()
    last_face = selected[-1].face if selected else None
    result: list[SemanticArgument] = []
    for available in query.hand_faces:
        if _face_already_selected(selected, available.face):
            continue
        if last_face is not None and face_sort_key(
            available.face
        ) <= face_sort_key(last_face):
            continue
        for count in range(1, available.count + 1):
            if selected_count + count > query.max_select:
                continue
            result.append(
                SemanticArgument(
                    "select_face_count",
                    FaceCount(face=available.face, count=count),
                )
            )
    return tuple(result)


def _required_face_count(argument: SemanticArgument) -> FaceCount:
    assert argument.kind == "select_face_count"
    assert argument.face_count is not None
    return argument.face_count


def _cards_for_face_counts(
    face_counts: tuple[FaceCount, ...],
    hand_cards: Sequence[Card],
) -> Ok[list[Card]] | Rejected:
    result: list[Card] = []
    used_ids: set[str] = set()
    for requested in face_counts:
        matching = [
            card
            for card in hand_cards
            if card.id not in used_ids
            and card.suit == requested.face.suit
            and card.rank == requested.face.rank
        ]
        if len(matching) < requested.count:
            return InvalidSemanticActionRejected(
                "当前手牌没有足够的指定牌面"
            )
        selected = matching[: requested.count]
        result.extend(selected)
        used_ids.update(card.id for card in selected)
    return Ok(value=result)


def _one_effective_suit(
    selected: tuple[FaceCount, ...],
    *,
    hand_cards: Sequence[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    cards_result = _cards_for_face_counts(selected, hand_cards)
    if isinstance(cards_result, Rejected):
        return False
    suits = {
        effective_suit(card, trump_suit, trump_rank)
        for card in cards_result.value
    }
    return len(suits) == 1


def _remaining_count_after_selected(
    *,
    hand_faces: tuple[FaceCount, ...],
    selected: tuple[FaceCount, ...],
) -> int:
    remaining = 0
    last_face = selected[-1].face if selected else None
    for available in hand_faces:
        if last_face is not None and face_sort_key(
            available.face
        ) <= face_sort_key(last_face):
            continue
        selected_count = 0
        for item in selected:
            if item.face == available.face:
                selected_count = item.count
                break
        remaining += max(available.count - selected_count, 0)
    return remaining


def _lead_cards(snapshot: StateSnapshot) -> list[Card]:
    trick = snapshot.trick
    if trick is None:
        return []
    for slot in trick.slots:
        if slot.player == trick.lead_player:
            return list(slot.cards)
    return []


def _last_stir_player(snapshot: StateSnapshot) -> int | None:
    for event in reversed(snapshot.stir_events):
        if event.kind == "stir":
            return event.player
    return None


def _current_stir_priority(snapshot: StateSnapshot) -> int:
    if snapshot.bid_winner is None:
        return 0
    return bid_value(snapshot.bid_winner.cards, snapshot.trump_rank)


def _face_already_selected(
    selected: tuple[FaceCount, ...],
    face: CardFace,
) -> bool:
    return any(item.face == face for item in selected)


def _trace_is_selection_only(trace: SemanticArgumentTrace) -> bool:
    return all(
        argument.kind == "select_face_count"
        for argument in trace.arguments
    )
