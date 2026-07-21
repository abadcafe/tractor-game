"""Lead-play legal action space."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game.rules.card_faces import FaceCount, face_count_width
from server.game.rules.cards import Card, Rank, Suit
from server.game.rules.ordering import effective_suit
from server.training.legal_actions.contract import LegalActionIndex
from server.training.legal_actions.selection import (
    cards_for_face_counts,
)
from server.training.semantic_actions.choices import (
    ActionPrefix,
    ActionTrace,
    InvalidActionRejected,
    action_prefix_cards,
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

    def decode(
        self, trace: ActionTrace
    ) -> Ok[GeneratedAction] | Rejected:
        if not trace.choices or trace.choices[-1].kind != "finish":
            return InvalidActionRejected("领牌动作必须以 finish 结束")
        selected_result = action_prefix_cards(
            ActionPrefix(choices=trace.choices[:-1])
        )
        if isinstance(selected_result, Rejected):
            return selected_result
        selected = selected_result.value
        if face_count_width(selected) == 0:
            return InvalidActionRejected("领牌不能为空")
        if not _one_effective_suit(
            selected,
            hand_cards=self._hand_cards,
            trump_suit=self._query.trump_suit,
            trump_rank=self._query.level_rank,
        ):
            return InvalidActionRejected("领牌必须同一有效花色")
        return Ok(
            value=GeneratedAction(
                action_kind="play",
                message_type="play",
                face_counts=selected,
                trace=trace,
                is_pass=False,
            )
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
