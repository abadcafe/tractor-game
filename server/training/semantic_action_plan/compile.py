"""Compile legal action indexes into queue-safe action specs."""

from __future__ import annotations

from server.game.rules.card_faces import (
    CardFace,
    FaceCount,
    canonical_face_counts,
)
from server.game.rules.cards import Rank, Suit
from server.training.legal_actions.complete_trace import (
    CompleteTraceLegalActionIndex,
)
from server.training.legal_actions.contract import (
    EmptyLegalActionIndex,
    LegalActionIndex,
)
from server.training.legal_actions.discard import (
    DiscardLegalActionIndex,
)
from server.training.legal_actions.follow import (
    FollowPlayLegalActionIndex,
)
from server.training.legal_actions.lead import LeadPlayLegalActionIndex
from server.training.semantic_action_plan.spec import (
    ACTION_FACE_COUNT,
    CompiledActionSpec,
    CompiledActionTraceSet,
    CompiledSelectionConstraints,
    FacePlan,
    PairPlanConstraints,
)
from server.training.semantic_action_plan.trace import (
    action_trace_choice_ids,
)
from server.training.semantic_actions.choices import (
    face_index as action_face_index,
)


def compile_legal_action_spec(
    legal_action: LegalActionIndex,
) -> CompiledActionSpec:
    """Compile one legal action index into an internal data spec."""
    if isinstance(legal_action, EmptyLegalActionIndex):
        return CompiledActionSpec(
            kind="empty", trace_set=None, selection=None
        )
    if isinstance(legal_action, CompleteTraceLegalActionIndex):
        return _compile_trace_set(legal_action)
    if isinstance(legal_action, DiscardLegalActionIndex):
        query = legal_action.query
        assert query.exact_select is not None
        return CompiledActionSpec(
            kind="discard",
            trace_set=None,
            selection=CompiledSelectionConstraints(
                min_select=query.exact_select,
                max_select=query.exact_select,
                exact_select=query.exact_select,
                required_same_suit_count=0,
                lead_effective_suit=-1,
                face_plan=_face_plan_from_hand(
                    query.hand_faces,
                    trump_suit=query.trump_suit,
                    trump_rank=query.level_rank,
                    lead_effective_suit=-1,
                    pair_faces=(),
                ),
                pair_plan=_empty_pair_plan(),
            ),
        )
    if isinstance(legal_action, LeadPlayLegalActionIndex):
        query = legal_action.query
        return CompiledActionSpec(
            kind="lead_play",
            trace_set=None,
            selection=CompiledSelectionConstraints(
                min_select=query.min_select,
                max_select=query.max_select,
                exact_select=None,
                required_same_suit_count=0,
                lead_effective_suit=-1,
                face_plan=_face_plan_from_hand(
                    query.hand_faces,
                    trump_suit=query.trump_suit,
                    trump_rank=query.level_rank,
                    lead_effective_suit=-1,
                    pair_faces=(),
                ),
                pair_plan=_empty_pair_plan(),
            ),
        )
    if isinstance(legal_action, FollowPlayLegalActionIndex):
        query = legal_action.query
        analysis = legal_action.space.analysis
        pair_planner = analysis.pair_planner
        lead_effective_suit = _effective_suit_code_from_value(
            analysis.lead_effective_suit
        )
        return CompiledActionSpec(
            kind="follow_play",
            trace_set=None,
            selection=CompiledSelectionConstraints(
                min_select=query.min_select,
                max_select=analysis.lead_count,
                exact_select=analysis.lead_count,
                required_same_suit_count=analysis.required_same_suit_count,
                lead_effective_suit=lead_effective_suit,
                face_plan=_face_plan_from_hand(
                    canonical_face_counts(analysis.hand_cards),
                    trump_suit=analysis.trump_suit,
                    trump_rank=analysis.trump_rank,
                    lead_effective_suit=lead_effective_suit,
                    pair_faces=tuple(pair_planner.pair_faces),
                ),
                pair_plan=PairPlanConstraints(
                    pair_plan_masks=tuple(
                        _face_mask(tuple(plan))
                        for plan in pair_planner.pair_plans
                    ),
                    has_tractor=pair_planner.has_tractor_segment(),
                    pair_floor=pair_planner.pair_floor,
                ),
            ),
        )
    raise AssertionError(
        f"unknown legal action index: {legal_action!r}"
    )


def _compile_trace_set(
    legal_action: CompleteTraceLegalActionIndex,
) -> CompiledActionSpec:
    return CompiledActionSpec(
        kind="trace_set",
        trace_set=CompiledActionTraceSet(
            traces=tuple(
                action_trace_choice_ids(action.trace)
                for action in legal_action.actions
            )
        ),
        selection=None,
    )


def _face_plan_from_hand(
    hand_faces: tuple[FaceCount, ...],
    *,
    trump_suit: Suit | None,
    trump_rank: Rank,
    lead_effective_suit: int,
    pair_faces: tuple[CardFace, ...],
) -> FacePlan:
    available = [0 for _ in range(ACTION_FACE_COUNT)]
    effective_suits = [-1 for _ in range(ACTION_FACE_COUNT)]
    same_suit = [False for _ in range(ACTION_FACE_COUNT)]
    off_suit = [False for _ in range(ACTION_FACE_COUNT)]
    for face_count in hand_faces:
        index = face_index(face_count.face)
        effective_suit = _face_effective_suit_code(
            face_count.face,
            trump_suit=trump_suit,
            trump_rank=trump_rank,
        )
        available[index] = face_count.count
        effective_suits[index] = effective_suit
        same_suit[index] = (
            lead_effective_suit >= 0
            and effective_suit == lead_effective_suit
        )
        off_suit[index] = (
            lead_effective_suit >= 0
            and effective_suit != lead_effective_suit
        )
    return FacePlan(
        available_counts=tuple(available),
        effective_suits=tuple(effective_suits),
        same_suit_mask=tuple(same_suit),
        off_suit_mask=tuple(off_suit),
        pair_face_mask=_face_mask(pair_faces),
    )


def _face_mask(faces: tuple[CardFace, ...]) -> tuple[bool, ...]:
    mask = [False for _ in range(ACTION_FACE_COUNT)]
    for face in faces:
        mask[face_index(face)] = True
    return tuple(mask)


def _empty_pair_plan() -> PairPlanConstraints:
    return PairPlanConstraints(
        pair_plan_masks=(),
        has_tractor=False,
        pair_floor=0,
    )


def face_index(face: CardFace) -> int:
    """Return the semantic model face index for a card face."""
    return action_face_index(face)


def _face_effective_suit_code(
    face: CardFace,
    *,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    if face.suit == Suit.JOKER:
        return _effective_suit_code_from_value("trump")
    if face.rank == trump_rank:
        return _effective_suit_code_from_value("trump")
    if trump_suit is not None and face.suit == trump_suit:
        return _effective_suit_code_from_value("trump")
    return _effective_suit_code_from_value(face.suit)


def _effective_suit_code_from_value(value: Suit | str) -> int:
    if value == Suit.HEARTS:
        return 0
    if value == Suit.SPADES:
        return 1
    if value == Suit.DIAMONDS:
        return 2
    if value == Suit.CLUBS:
        return 3
    if value == "trump" or value == Suit.JOKER:
        return 4
    raise AssertionError(f"unknown effective suit: {value!r}")
