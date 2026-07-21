"""Compact queue-safe typed observation rows."""

from __future__ import annotations

from dataclasses import dataclass

from server.game.rules.card_faces import RANK_ORDER, SUIT_ORDER
from server.game.rules.cards import Rank, Suit
from server.game.rules.required_progress import TerminalProgress
from server.game.state_machine.constants import (
    BOTTOM_CARD_COUNT,
    PLAYER_COUNT,
    TOTAL_CARDS,
)
from server.training.observation import Observation
from server.training.relative_state import (
    RelativeActor,
    TrickPosition,
    TrumpMode,
    TrumpState,
)
from server.training.rule_features import card_rule_features
from server.training.semantic_actions.choices import (
    CARD_CHOICE_COUNT,
    CARD_FACES,
)
from server.training.tokenization import (
    ActionToken,
    CardToken,
    GlobalToken,
    RoundToken,
    TokenAddress,
    TokenNode,
    TrickToken,
)
from server.training.tokenization.encoding_schema import (
    ACTION_KIND_INDEX,
    ACTOR_INDEX,
    CATEGORY_COUNT,
    DISPOSITION_INDEX,
    EFFECTIVE_SUIT_INDEX,
    FAMILY_INDEX,
    PAYLOAD_ROLE_INDEX,
    RANK_INDEX,
    STATE_INDEX,
    SUIT_INDEX,
    TRICK_POSITION_INDEX,
    VARIANT_INDEX,
    SemanticState,
    TokenVariant,
)
from server.training.tokenization.payloads import (
    RoundField,
    TokenActionKind,
)
from server.training.tokenization.structure import PayloadRole

type CategoryRow = tuple[int, ...]
type CardRuleRow = tuple[float, float]
type CandidateCategoryRow = tuple[int, int, int]
type CoordinateRow = tuple[int, int, int]
type CoordinateMaskRow = tuple[bool, bool, bool]

_MAX_PLAYABLE_HAND: int = (
    TOTAL_CARDS - BOTTOM_CARD_COUNT
) // PLAYER_COUNT
_MAX_TRICKS: int = _MAX_PLAYABLE_HAND
_MAX_PLAY_ACTIONS: int = _MAX_TRICKS * PLAYER_COUNT
_MAX_FAILED_EXTRA_TOKENS: int = (
    PLAYER_COUNT * _MAX_PLAYABLE_HAND * (_MAX_PLAYABLE_HAND - 1) // 2
)
_MAX_BID_NODES: int = 2 * (TOTAL_CARDS - BOTTOM_CARD_COUNT)
_MAX_STIR_DECLARATIONS: int = 6
_MAX_STIR_ACTIONS: int = (
    _MAX_STIR_DECLARATIONS + 1
) * PLAYER_COUNT + _MAX_STIR_DECLARATIONS
_MAX_EXCHANGES: int = _MAX_STIR_DECLARATIONS + 1
_MAX_EXCHANGE_NODES: int = _MAX_EXCHANGES * (1 + 2 * BOTTOM_CARD_COUNT)
_CONTEXT_NODES: int = 1 + 13
_PLAY_NODES: int = (
    _MAX_TRICKS
    + _MAX_PLAY_ACTIONS
    + (TOTAL_CARDS - BOTTOM_CARD_COUNT)
    + _MAX_FAILED_EXTRA_TOKENS
)
MAX_LOSSLESS_OBSERVATION_TOKENS: int = (
    _CONTEXT_NODES
    + _MAX_BID_NODES
    + _MAX_STIR_ACTIONS
    + _MAX_EXCHANGE_NODES
    + _PLAY_NODES
    + _MAX_PLAYABLE_HAND
    + BOTTOM_CARD_COUNT
    + 1
)


@dataclass(frozen=True, slots=True)
class PackedObservation:
    """Torch-free typed rows for one relative observation."""

    category_rows: tuple[CategoryRow, ...]
    scalar_values: tuple[float, ...]
    card_rule_rows: tuple[CardRuleRow, ...]
    coordinate_rows: tuple[CoordinateRow, ...]
    coordinate_mask_rows: tuple[CoordinateMaskRow, ...]
    candidate_category_rows: tuple[CandidateCategoryRow, ...]
    candidate_counts: tuple[float, ...]
    candidate_card_rule_rows: tuple[CardRuleRow, ...]
    query_index: int

    def __post_init__(self) -> None:
        count = len(self.category_rows)
        assert 0 < count <= MAX_LOSSLESS_OBSERVATION_TOKENS
        assert len(self.scalar_values) == count
        assert len(self.card_rule_rows) == count
        assert len(self.coordinate_rows) == count
        assert len(self.coordinate_mask_rows) == count
        assert all(
            len(row) == CATEGORY_COUNT for row in self.category_rows
        )
        assert len(self.candidate_category_rows) == CARD_CHOICE_COUNT
        assert len(self.candidate_counts) == CARD_CHOICE_COUNT
        assert len(self.candidate_card_rule_rows) == CARD_CHOICE_COUNT
        assert 0 <= self.query_index < count

    def token_count(self) -> int:
        """Return the active semantic token count."""
        return len(self.category_rows)


def pack_observation(observation: Observation) -> PackedObservation:
    """Pack one typed observation without learned absent values."""
    categories: list[CategoryRow] = []
    scalars: list[float] = []
    card_rules: list[CardRuleRow] = []
    coordinates: list[CoordinateRow] = []
    coordinate_masks: list[CoordinateMaskRow] = []
    for node in observation.tokens:
        category, scalar, card_rule = _pack_node(node, observation)
        coordinate, coordinate_mask = _pack_address(node.address)
        categories.append(category)
        scalars.append(scalar)
        card_rules.append(card_rule)
        coordinates.append(coordinate)
        coordinate_masks.append(coordinate_mask)
    candidate_categories: list[CandidateCategoryRow] = []
    candidate_counts: list[float] = []
    candidate_rules: list[CardRuleRow] = []
    for face in CARD_FACES:
        rule = card_rule_features(face, observation.round_context)
        for count in (1, 2):
            candidate_categories.append(
                (
                    _suit_id(face.suit),
                    _rank_id(face.rank),
                    rule.effective_suit_id,
                )
            )
            candidate_counts.append(float(count))
            candidate_rules.append(
                (rule.point_value, rule.relative_strength)
            )
    return PackedObservation(
        category_rows=tuple(categories),
        scalar_values=tuple(scalars),
        card_rule_rows=tuple(card_rules),
        coordinate_rows=tuple(coordinates),
        coordinate_mask_rows=tuple(coordinate_masks),
        candidate_category_rows=tuple(candidate_categories),
        candidate_counts=tuple(candidate_counts),
        candidate_card_rule_rows=tuple(candidate_rules),
        query_index=observation.token_sequence.query_index,
    )


def padded_packed_observation(
    packed: PackedObservation, *, token_count: int
) -> PackedObservation:
    """Pad one observation to a batch-local lossless token count."""
    assert packed.token_count() <= token_count
    assert token_count <= MAX_LOSSLESS_OBSERVATION_TOKENS
    padding = token_count - packed.token_count()
    return PackedObservation(
        category_rows=(
            *packed.category_rows,
            *((0,) * CATEGORY_COUNT for _ in range(padding)),
        ),
        scalar_values=(
            *packed.scalar_values,
            *(0.0 for _ in range(padding)),
        ),
        card_rule_rows=(
            *packed.card_rule_rows,
            *((0.0, 0.0) for _ in range(padding)),
        ),
        coordinate_rows=(
            *packed.coordinate_rows,
            *((0, 0, 0) for _ in range(padding)),
        ),
        coordinate_mask_rows=(
            *packed.coordinate_mask_rows,
            *((False, False, False) for _ in range(padding)),
        ),
        candidate_category_rows=packed.candidate_category_rows,
        candidate_counts=packed.candidate_counts,
        candidate_card_rule_rows=packed.candidate_card_rule_rows,
        query_index=packed.query_index,
    )


def _pack_node(
    node: TokenNode, observation: Observation
) -> tuple[CategoryRow, float, CardRuleRow]:
    values = [0 for _ in range(CATEGORY_COUNT)]
    values[FAMILY_INDEX] = int(node.family)
    scalar = 0.0
    card_rule: CardRuleRow = (0.0, 0.0)
    payload = node.payload
    if isinstance(payload, GlobalToken):
        values[VARIANT_INDEX] = int(TokenVariant.MANDATORY_LEVEL)
        values[RANK_INDEX] = _rank_id(payload.rank)
        scalar = float(payload.progress_position)
    elif isinstance(payload, RoundToken):
        scalar = _pack_round(payload, values)
    elif isinstance(payload, TrickToken):
        values[VARIANT_INDEX] = int(TokenVariant.TRICK)
        values[STATE_INDEX] = int(
            SemanticState.OPEN_TRICK
            if payload.status == "open"
            else SemanticState.COMPLETED_TRICK
        )
        if payload.winner is not None:
            values[ACTOR_INDEX] = _actor_id(payload.winner)
        if payload.points is not None:
            scalar = float(payload.points)
    elif isinstance(payload, ActionToken):
        values[VARIANT_INDEX] = int(TokenVariant.ACTION)
        values[ACTION_KIND_INDEX] = _action_kind_id(payload.kind)
        values[STATE_INDEX] = int(
            SemanticState.ACTION_FACT
            if payload.occurrence == "fact"
            else SemanticState.ACTION_QUERY
        )
        if payload.actor is not None:
            values[ACTOR_INDEX] = _actor_id(payload.actor)
        if payload.disposition is not None:
            values[DISPOSITION_INDEX] = (
                1 if payload.disposition == "pass" else 2
            )
        if payload.trick_position is not None:
            values[TRICK_POSITION_INDEX] = _trick_position_id(
                payload.trick_position
            )
    else:
        assert isinstance(payload, CardToken)
        values[VARIANT_INDEX] = int(TokenVariant.CARD)
        values[SUIT_INDEX] = _suit_id(payload.face.suit)
        values[RANK_INDEX] = _rank_id(payload.face.rank)
        scalar = float(payload.count)
        rule = card_rule_features(
            payload.face, observation.round_context
        )
        values[EFFECTIVE_SUIT_INDEX] = rule.effective_suit_id
        card_rule = (rule.point_value, rule.relative_strength)
    if node.address.payload_role is not None:
        values[PAYLOAD_ROLE_INDEX] = _payload_role_id(
            node.address.payload_role
        )
    return (tuple(values), scalar, card_rule)


def _pack_round(token: RoundToken, values: list[int]) -> float:
    field = token.field
    values[VARIANT_INDEX] = int(_round_variant(field))
    if token.actor is not None:
        values[ACTOR_INDEX] = _actor_id(token.actor)
    if field == RoundField.DECLARER_ACTOR:
        if token.value is None:
            values[STATE_INDEX] = int(SemanticState.UNSET)
        else:
            assert isinstance(token.value, RelativeActor)
            values[ACTOR_INDEX] = _actor_id(token.value)
        return 0.0
    if field in (
        RoundField.OWN_LEVEL,
        RoundField.OPPONENT_LEVEL,
        RoundField.LEVEL_RANK,
    ):
        assert isinstance(token.value, Rank)
        values[RANK_INDEX] = _rank_id(token.value)
        return 0.0
    if field in (RoundField.OWN_TARGET, RoundField.OPPONENT_TARGET):
        if token.value == TerminalProgress.WIN:
            values[STATE_INDEX] = int(SemanticState.WIN_TARGET)
        else:
            assert isinstance(token.value, Rank)
            values[RANK_INDEX] = _rank_id(token.value)
        return 0.0
    if field == RoundField.TRUMP_STATE:
        assert isinstance(token.value, TrumpState)
        if token.value.mode == TrumpMode.UNSET:
            values[STATE_INDEX] = int(SemanticState.UNSET)
        elif token.value.mode == TrumpMode.NO_TRUMP:
            values[STATE_INDEX] = int(SemanticState.NO_TRUMP)
        else:
            values[STATE_INDEX] = int(SemanticState.SUITED_TRUMP)
            assert token.value.suit is not None
            values[SUIT_INDEX] = _suit_id(token.value.suit)
        return 0.0
    assert isinstance(token.value, int)
    return float(token.value)


def _pack_address(
    address: TokenAddress,
) -> tuple[CoordinateRow, CoordinateMaskRow]:
    values = (
        0
        if address.round_event_time is None
        else address.round_event_time,
        0 if address.trick_time is None else address.trick_time,
        0
        if address.action_position is None
        else address.action_position,
    )
    masks = (
        address.round_event_time is not None,
        address.trick_time is not None,
        address.action_position is not None,
    )
    return (values, masks)


def _round_variant(field: RoundField) -> TokenVariant:
    return {
        RoundField.DECLARER_ACTOR: TokenVariant.DECLARER_ACTOR,
        RoundField.OWN_LEVEL: TokenVariant.OWN_LEVEL,
        RoundField.OPPONENT_LEVEL: TokenVariant.OPPONENT_LEVEL,
        RoundField.OWN_TARGET: TokenVariant.OWN_TARGET,
        RoundField.OPPONENT_TARGET: TokenVariant.OPPONENT_TARGET,
        RoundField.OWN_DISTANCE: TokenVariant.OWN_DISTANCE,
        RoundField.OPPONENT_DISTANCE: TokenVariant.OPPONENT_DISTANCE,
        RoundField.TRUMP_STATE: TokenVariant.TRUMP_STATE,
        RoundField.LEVEL_RANK: TokenVariant.LEVEL_RANK,
        RoundField.DEFENDER_POINTS: TokenVariant.DEFENDER_POINTS,
        RoundField.REMAINING_CARDS: TokenVariant.REMAINING_CARDS,
    }[field]


def _actor_id(actor: RelativeActor) -> int:
    return (
        RelativeActor.SELF,
        RelativeActor.PARTNER,
        RelativeActor.LEFT_ENEMY,
        RelativeActor.RIGHT_ENEMY,
    ).index(actor) + 1


def _rank_id(rank: Rank) -> int:
    return RANK_ORDER.index(rank) + 1


def _suit_id(suit: Suit) -> int:
    return SUIT_ORDER.index(suit) + 1


def _action_kind_id(kind: TokenActionKind) -> int:
    return ("bid", "stir", "bottom_exchange", "play").index(kind) + 1


def _trick_position_id(position: TrickPosition) -> int:
    return (
        TrickPosition.LEAD,
        TrickPosition.FOLLOW_1,
        TrickPosition.FOLLOW_2,
        TrickPosition.FOLLOW_3,
    ).index(position) + 1


def _payload_role_id(role: PayloadRole) -> int:
    return (
        "hand",
        "visible_bottom",
        "bid_reveal",
        "stir_reveal",
        "exchange_pickup",
        "exchange_discard",
        "played",
        "revealed_extra",
    ).index(role) + 1


__all__ = (
    "MAX_LOSSLESS_OBSERVATION_TOKENS",
    "PackedObservation",
    "pack_observation",
    "padded_packed_observation",
)
