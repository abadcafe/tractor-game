"""Scoring module: pure calculation from inputs to RoundResult.

Computes bottom card points, ambush multiplier, total defender points,
level changes, and declarer rotation per spec section 9.

This is NOT a state machine -- it is a pure function.
"""

from collections import Counter

from pydantic import BaseModel, ConfigDict

from server.sm.card_model import Card, Rank
from server.sm.constants import (
    SCORE_THRESHOLDS,
    advance_level,
    get_partner_index,
    get_team_index,
    next_player_ccw,
)
from server.sm.types import CompletedTrick, PlayType


class RoundResult(BaseModel):
    """Result of scoring a round."""

    model_config = ConfigDict(frozen=True)

    team0_new_level: Rank
    team1_new_level: Rank
    next_declarer_team: int
    next_declarer_player: int
    total_defender_points: int
    declarer_level_change: int
    switch_declarer: bool
    bottom_card_bonus: int


def _compute_ambush_multiplier(last_trick: CompletedTrick) -> int:
    """Compute the ambush multiplier based on the last trick's lead type.

    For SINGLE, PAIR, TRACTOR: use the standard multiplier.
    For THROW: analyze the actual cards to find the best sub-pattern.
    """
    lead_type = last_trick.lead_type

    if lead_type == PlayType.SINGLE:
        return 2
    if lead_type == PlayType.PAIR:
        return 4
    if lead_type == PlayType.TRACTOR:
        # Find the lead player's slot to count cards; fall back to any slot
        total_cards = _find_lead_card_count(last_trick)
        return 2 ** total_cards if total_cards > 0 else 2

    # THROW: analyze sub-patterns from the lead player's cards
    if lead_type == PlayType.THROW:
        lead_cards = _find_lead_cards(last_trick)
        return _throw_multiplier(lead_cards)

    return 2  # fallback


def _find_lead_card_count(last_trick: CompletedTrick) -> int:
    """Find the card count of the lead player's contribution to the trick."""
    for slot in last_trick.slots:
        if slot.player == last_trick.lead_player:
            return len(slot.cards)
    # Fallback: if lead_player's slot not found, use any non-empty slot
    for slot in last_trick.slots:
        if slot.cards:
            return len(slot.cards)
    return 0


def _find_lead_cards(last_trick: CompletedTrick) -> list[Card]:
    """Find the lead player's cards from the trick slots."""
    for slot in last_trick.slots:
        if slot.player == last_trick.lead_player:
            return slot.cards
    # Fallback: if lead_player's slot not found, use any non-empty slot
    for slot in last_trick.slots:
        if slot.cards:
            return slot.cards
    return []


def _throw_multiplier(cards: list[Card]) -> int:
    """Determine the multiplier for a THROW play by analyzing sub-patterns.

    Per spec: check for tractors (consecutive pairs in same effective suit),
    then pairs (same rank in same effective suit), then all singles.
    Tractor -> 2^(tractor_card_count), pair -> x4, all singles -> x2.
    """
    if not cards:
        return 2

    # Group cards by suit (ignoring deck difference)
    suit_groups: dict[str, list[Card]] = {}
    for card in cards:
        suit_groups.setdefault(card.suit.value, []).append(card)

    best_multiplier = 2  # default: all singles

    for suit_cards in suit_groups.values():
        if len(suit_cards) < 2:
            continue

        # Count ranks in this suit
        rank_counts = Counter(c.rank for c in suit_cards)

        # Find pairs (ranks with count >= 2)
        pair_ranks = sorted(
            [r for r, cnt in rank_counts.items() if cnt >= 2],
            key=lambda r: _rank_sort_key(r),
        )

        if len(pair_ranks) >= 2:
            # Check for longest consecutive pair run (tractor)
            tractor_len = _longest_consecutive_pairs(pair_ranks)
            if tractor_len >= 2:
                # tractor with tractor_len pairs = 2*tractor_len cards
                tractor_cards = tractor_len * 2
                best_multiplier = max(best_multiplier, 2 ** tractor_cards)
            elif len(pair_ranks) >= 1:
                # Has pairs but no consecutive pairs -> pair multiplier
                best_multiplier = max(best_multiplier, 4)
        elif len(pair_ranks) == 1:
            best_multiplier = max(best_multiplier, 4)

    return best_multiplier


def _rank_sort_key(rank: Rank) -> int:
    """Return a numeric sort key for rank ordering (for consecutive pair detection)."""
    order = {
        Rank.TWO: 0, Rank.THREE: 1, Rank.FOUR: 2, Rank.FIVE: 3,
        Rank.SIX: 4, Rank.SEVEN: 5, Rank.EIGHT: 6, Rank.NINE: 7,
        Rank.TEN: 8, Rank.JACK: 9, Rank.QUEEN: 10, Rank.KING: 11, Rank.ACE: 12,
    }
    return order.get(rank, -1)


def _longest_consecutive_pairs(pair_ranks: list[Rank]) -> int:
    """Find the longest run of consecutive pair ranks."""
    if not pair_ranks:
        return 0

    sorted_ranks = sorted(pair_ranks, key=_rank_sort_key)
    keys = [_rank_sort_key(r) for r in sorted_ranks]

    best = 1
    current = 1
    for i in range(1, len(keys)):
        if keys[i] == keys[i - 1] + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1

    return best


def _determine_level_change(total_points: int) -> tuple[int, bool]:
    """Return (declarer_level_change, switch_declarer) from total defender points."""
    for threshold in SCORE_THRESHOLDS:
        if total_points <= threshold.max_points:
            return (threshold.declarer_change, threshold.switch_declarer)
    # Should not reach here given the thresholds cover 0-200+
    return (-3, True)


def calculate_score(
    defender_points: int,
    bottom_cards: list[Card],
    last_trick: CompletedTrick,
    declarer_team: int,
    declarer_player: int,
    team0_level: Rank,
    team1_level: Rank,
) -> RoundResult:
    """Calculate the score result for a round.

    Pure function: takes inputs, returns RoundResult.
    """
    defender_team = 1 - declarer_team

    # Compute bottom card base points
    bottom_base = sum(card.points for card in bottom_cards)

    # Determine if defender won the last trick (ambush)
    last_trick_winner_team = get_team_index(last_trick.winner)
    is_ambush = last_trick_winner_team == defender_team

    # Compute ambush bonus
    if is_ambush:
        multiplier = _compute_ambush_multiplier(last_trick)
        bottom_card_bonus = bottom_base * multiplier
    else:
        bottom_card_bonus = 0

    total_defender_points = defender_points + bottom_card_bonus

    # Determine level change from scoring table
    declarer_change, switch = _determine_level_change(total_defender_points)

    # Compute new levels
    defender_change = -declarer_change if declarer_change < 0 else 0

    if declarer_team == 0:
        team0_new_level = advance_level(team0_level, declarer_change)
        team1_new_level = advance_level(team1_level, defender_change)
    else:
        team0_new_level = advance_level(team0_level, defender_change)
        team1_new_level = advance_level(team1_level, declarer_change)

    # Determine next declarer
    if switch:
        next_team = defender_team
        next_player = next_player_ccw(declarer_player)
    else:
        next_team = declarer_team
        next_player = get_partner_index(declarer_player)

    return RoundResult(
        team0_new_level=team0_new_level,
        team1_new_level=team1_new_level,
        next_declarer_team=next_team,
        next_declarer_player=next_player,
        total_defender_points=total_defender_points,
        declarer_level_change=declarer_change,
        switch_declarer=switch,
        bottom_card_bonus=bottom_card_bonus,
    )
