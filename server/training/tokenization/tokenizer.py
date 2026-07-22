"""Pure flattening of relative observations into typed token nodes."""

from __future__ import annotations

from server.game.rules.card_faces import FaceCount
from server.training.observation_structure import TrickRecency
from server.training.relative_state import (
    RelativeActor,
    RelativeObservation,
)
from server.training.relative_state.actions import (
    RelativeBidAction,
    RelativeExchangeAction,
    RelativePlayAction,
    RelativeStirAction,
)
from server.training.tokenization.payloads import (
    ActionToken,
    CardToken,
    GlobalToken,
    RoundField,
    RoundToken,
    TrickToken,
)
from server.training.tokenization.structure import (
    PayloadRole,
    TokenAddress,
    TokenNode,
    TokenSequence,
)


def tokenize(observation: RelativeObservation) -> TokenSequence:
    """Return a lossless flat token sequence for one policy decision."""
    query = observation.query
    assert query is not None
    nodes: list[TokenNode] = []
    for position, level in enumerate(
        observation.global_context.mandatory_levels
    ):
        nodes.append(
            _node(GlobalToken(rank=level, progress_position=position))
        )
    nodes.extend(_round_nodes(observation))
    for action in observation.round_actions:
        _append_round_action(nodes, action)
    _append_cards(
        nodes,
        observation.visible_bottom,
        address=TokenAddress(payload_role="visible_bottom"),
    )
    _append_cards(
        nodes,
        observation.hand,
        address=TokenAddress(payload_role="hand"),
    )
    for trick in observation.tricks:
        nodes.append(
            _node(
                TrickToken(
                    status=trick.status,
                    winner=trick.winner,
                    points=trick.points,
                ),
                TokenAddress(trick=trick.recency),
            )
        )
        for action in trick.actions:
            _append_play_action(nodes, action, recency=trick.recency)
    query_index = len(nodes)
    nodes.append(
        _node(
            ActionToken(
                occurrence="query",
                kind=query.kind,
                actor=None,
                disposition=None,
                trick_position=query.trick_position,
            ),
            TokenAddress(
                round_event=query.round_event,
                trick=TrickRecency(0) if query.kind == "play" else None,
                play_position=query.trick_position,
            ),
        )
    )
    return TokenSequence(nodes=tuple(nodes), query_index=query_index)


def _round_nodes(observation: RelativeObservation) -> list[TokenNode]:
    context = observation.round_context
    return [
        _node(
            RoundToken(
                RoundField.DECLARER_ACTOR, context.declarer_actor
            )
        ),
        _node(RoundToken(RoundField.OWN_LEVEL, context.own_level)),
        _node(
            RoundToken(
                RoundField.OPPONENT_LEVEL, context.opponent_level
            )
        ),
        _node(RoundToken(RoundField.OWN_TARGET, context.own_target)),
        _node(
            RoundToken(
                RoundField.OPPONENT_TARGET,
                context.opponent_target,
            )
        ),
        _node(
            RoundToken(
                RoundField.OWN_DISTANCE,
                context.own_distance_to_target,
            )
        ),
        _node(
            RoundToken(
                RoundField.OPPONENT_DISTANCE,
                context.opponent_distance_to_target,
            )
        ),
        _node(RoundToken(RoundField.TRUMP_STATE, context.trump)),
        _node(RoundToken(RoundField.LEVEL_RANK, context.level_rank)),
        _node(
            RoundToken(
                RoundField.DEFENDER_POINTS,
                context.defender_points,
            )
        ),
        _node(
            RoundToken(
                RoundField.REMAINING_CARDS,
                context.partner_remaining,
                actor=RelativeActor.PARTNER,
            )
        ),
        _node(
            RoundToken(
                RoundField.REMAINING_CARDS,
                context.left_enemy_remaining,
                actor=RelativeActor.LEFT_ENEMY,
            )
        ),
        _node(
            RoundToken(
                RoundField.REMAINING_CARDS,
                context.right_enemy_remaining,
                actor=RelativeActor.RIGHT_ENEMY,
            )
        ),
    ]


def _append_round_action(
    nodes: list[TokenNode],
    action: RelativeBidAction
    | RelativeStirAction
    | RelativeExchangeAction,
) -> None:
    address = TokenAddress(round_event=action.event_ordinal)
    if isinstance(action, RelativeBidAction):
        nodes.append(
            _node(
                ActionToken(
                    "fact",
                    "bid",
                    action.actor,
                    action.disposition,
                    None,
                ),
                address,
            )
        )
        _append_cards(
            nodes,
            action.revealed,
            address=_with_payload(address, "bid_reveal"),
        )
        return
    if isinstance(action, RelativeStirAction):
        nodes.append(
            _node(
                ActionToken(
                    "fact",
                    "stir",
                    action.actor,
                    action.disposition,
                    None,
                ),
                address,
            )
        )
        _append_cards(
            nodes,
            action.revealed,
            address=_with_payload(address, "stir_reveal"),
        )
        return
    nodes.append(
        _node(
            ActionToken(
                "fact",
                "bottom_exchange",
                RelativeActor.SELF,
                None,
                None,
            ),
            address,
        )
    )
    _append_cards(
        nodes,
        action.picked_up,
        address=_with_payload(address, "exchange_pickup"),
    )
    _append_cards(
        nodes,
        action.discarded,
        address=_with_payload(address, "exchange_discard"),
    )


def _append_play_action(
    nodes: list[TokenNode],
    action: RelativePlayAction,
    *,
    recency: TrickRecency,
) -> None:
    address = TokenAddress(
        trick=recency,
        play_position=action.trick_position,
    )
    nodes.append(
        _node(
            ActionToken(
                "fact",
                "play",
                action.actor,
                None,
                action.trick_position,
            ),
            address,
        )
    )
    _append_cards(
        nodes,
        action.played,
        address=_with_payload(address, "played"),
    )
    _append_cards(
        nodes,
        action.revealed_extra,
        address=_with_payload(address, "revealed_extra"),
    )


def _append_cards(
    nodes: list[TokenNode],
    cards: tuple[FaceCount, ...],
    *,
    address: TokenAddress,
) -> None:
    nodes.extend(
        _node(
            CardToken(face=item.face, count=item.count),
            address,
        )
        for item in cards
    )


def _with_payload(
    address: TokenAddress, payload_role: PayloadRole
) -> TokenAddress:
    return TokenAddress(
        round_event=address.round_event,
        trick=address.trick,
        play_position=address.play_position,
        payload_role=payload_role,
    )


def _node(
    payload: GlobalToken
    | RoundToken
    | TrickToken
    | ActionToken
    | CardToken,
    address: TokenAddress | None = None,
) -> TokenNode:
    return TokenNode(
        payload=payload,
        address=address if address is not None else TokenAddress(),
    )


__all__ = ("tokenize",)
