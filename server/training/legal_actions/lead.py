"""Lead-play legal action space."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from server.result import Ok, Rejected
from server.rules.card_faces import FaceCount, face_count_width
from server.rules.cards import Card, Rank, Suit
from server.rules.ordering import effective_suit
from server.training.legal_actions.contract import LegalActionIndex
from server.training.legal_actions.selection import (
    cards_for_face_counts,
    select_arguments,
)
from server.training.semantic_actions.arguments import (
    InvalidSemanticActionRejected,
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
    semantic_prefix_state,
)
from server.training.semantic_actions.query import ActionQuery
from server.training.semantic_actions.values import GeneratedAction


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
            select_arguments(
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


def _one_effective_suit(
    selected: tuple[FaceCount, ...],
    *,
    hand_cards: Sequence[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> bool:
    cards_result = cards_for_face_counts(selected, hand_cards)
    if isinstance(cards_result, Rejected):
        return False
    suits = {
        effective_suit(card, trump_suit, trump_rank)
        for card in cards_result.value
    }
    return len(suits) == 1
