"""Field-name vocabularies for training observation tokens."""

from __future__ import annotations

from typing import Literal

type GlobalFieldName = Literal[
    "team_layout",
    "left_player_role",
    "right_player_role",
    "partner_role",
    "deck_count",
    "player_count",
    "bottom_card_count",
    "required_level",
    "final_target",
    "rules_version",
]
type RoundFieldName = Literal[
    "phase",
    "awaiting_action",
    "dealer_role",
    "dealer_team",
    "self_team_is_declarer",
    "enemy_team_is_declarer",
    "self_team_level",
    "enemy_team_level",
    "self_team_required_level",
    "enemy_team_required_level",
    "self_team_distance_to_required_level",
    "enemy_team_distance_to_required_level",
    "trump_suit",
    "level_rank",
    "level_card_revealer_role",
    "current_score",
    "remaining_cards_self",
    "remaining_cards_partner",
    "remaining_cards_left_enemy",
    "remaining_cards_right_enemy",
    "winning_team",
]
type RoundEventFieldName = Literal[
    "event_kind",
    "actor",
    "bid_kind",
    "stir_kind",
    "suit",
    "joker_type",
    "count",
    "priority",
    "trigger",
]
type TrickResultFieldName = Literal["winner", "points"]
type ActionQueryFieldName = Literal[
    "kind",
    "pass_allowed",
    "min_select",
    "max_select",
    "exact_select",
    "action_play_order",
    "current_trick_width",
    "lead_actor",
    "discard_count",
    "trump_suit",
    "level_rank",
    "current_best_bid_role",
]
