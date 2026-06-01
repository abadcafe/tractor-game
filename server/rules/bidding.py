"""Bidding rules for 升级, including 炒地皮 (Chaodipi / Stir-fry).

Phase 1: Initial Bidding (叫牌)
  - Players bid in turn, starting from the first player after the dealer.
  - First bid must be at least the current level.
  - Subsequent bids must be higher than the current highest bid.
  - A player may pass.
  - Bidding ends when 3 consecutive players pass, or everyone passes.

Phase 2: Stirring (炒地皮)
  - After initial bidding winner picks up bottom cards,
    other players may "stir" (炒) to steal declarer rights.
  - Stir at same level: must change trump suit.
  - Stir at higher level: free choice of trump.
  - Original winner can counter-stir (反炒).
  - Continues until all players pass in sequence.
  - Bug #5 fix: the same player cannot stir consecutively.
"""

from server.engine.card import Rank, Suit
from server.engine.constants import LEVELS
from server.engine.player_utils import next_player
from server.engine.types import BidAction, StirAction


# ---- Initial Bidding ----


def is_valid_bid(
    bid_level: Rank | None,
    pass_: bool,
    current_highest_bid: Rank | None,
    current_level: Rank,
) -> bool:
    """Check if a bid is valid given the current bidding state."""
    if pass_:
        return True  # Can always pass

    if bid_level is None:
        return False

    # Must bid at least the current level
    bid_index = LEVELS.index(bid_level)
    current_index = LEVELS.index(current_level)
    if bid_index < current_index:
        return False

    # If there's already a bid, must bid higher
    if current_highest_bid is not None:
        highest_index = LEVELS.index(current_highest_bid)
        if bid_index <= highest_index:
            return False

    return True


def get_valid_bid_levels(
    current_highest_bid: Rank | None,
    current_level: Rank,
) -> list[Rank]:
    """Get the valid bid levels for a player given the current state."""
    current_index = LEVELS.index(current_level)
    start_index = (
        LEVELS.index(current_highest_bid) + 1
        if current_highest_bid is not None
        else current_index
    )

    return LEVELS[start_index:]


def is_bidding_over(
    bids: list[BidAction],
    player_count: int,
) -> bool:
    """Check if the bidding round is over.

    Ends when 3 consecutive passes after the first bid, or everyone passed.
    """
    if len(bids) == 0:
        return False

    # All players passed with no bid
    any_bid = any(not b.pass_ for b in bids)
    if not any_bid and len(bids) >= player_count:
        return True

    # Three consecutive passes after a bid was made
    if any_bid:
        consecutive_passes = 0
        for bid in reversed(bids):
            if bid.pass_:
                consecutive_passes += 1
                if consecutive_passes >= 3:
                    return True
            else:
                break

    return False


def get_winning_bid(bids: list[BidAction]) -> BidAction | None:
    """Get the winning bid from bidding history."""
    winner: BidAction | None = None
    highest_index = -1

    for bid in bids:
        if bid.pass_ or bid.level is None:
            continue
        idx = LEVELS.index(bid.level)
        if idx > highest_index:
            highest_index = idx
            winner = bid

    return winner


# ---- 炒地皮 (Stirring) ----


def is_valid_stir(
    stir: StirAction,
    current_trump_suit: Suit,
    current_bid_level: Rank,
    stirring_history: list[StirAction],
    player_index: int,
) -> bool:
    """Check if a stir action is valid.

    Args:
        stir: The proposed stir action.
        current_trump_suit: The current trump suit.
        current_bid_level: The current bid level.
        stirring_history: All previous stir actions in this round.
        player_index: The player attempting to stir.
    """
    stir_level = stir.level
    if stir_level is None:
        return False

    # Must always specify a trump suit
    if stir.new_trump_suit is None:
        return False

    stir_level_index = LEVELS.index(stir_level)
    bid_level_index = LEVELS.index(current_bid_level)

    # Must be at or above current bid level
    if stir_level_index < bid_level_index:
        return False

    # Same level: must change trump suit
    if stir_level_index == bid_level_index:
        if stir.new_trump_suit == current_trump_suit:
            return False

    # Bug #5 fix: same player cannot stir consecutively
    # Use stir.player_index as the authoritative source
    if stirring_history and stirring_history[-1].player_index == stir.player_index:
        return False

    return True


def get_valid_stir_options(
    current_trump_suit: Suit,
    current_bid_level: Rank,
    player_index: int,
    stirring_history: list[StirAction],
) -> list[StirAction]:
    """Get valid stir options for a player."""
    # Bug #5: same player cannot stir consecutively
    if stirring_history and stirring_history[-1].player_index == player_index:
        return []

    options: list[StirAction] = []

    non_joker_suits = [Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS]

    # Stir at same level: change trump
    for suit in non_joker_suits:
        if suit == current_trump_suit:
            continue
        options.append(StirAction(
            player_index=player_index,
            new_trump_suit=suit,
            level=current_bid_level,
        ))

    # Stir at higher levels
    current_index = LEVELS.index(current_bid_level)
    for i in range(current_index + 1, len(LEVELS)):
        for suit in non_joker_suits:
            options.append(StirAction(
                player_index=player_index,
                new_trump_suit=suit,
                level=LEVELS[i],
            ))

    return options


def is_stirring_over(
    stir_passes: int,
    player_count: int,
) -> bool:
    """Check if the stirring round is over.

    Ends when all remaining players pass in sequence (one full round of passes).
    """
    return stir_passes >= player_count


def get_next_bidder(current_player: int) -> int:
    """Get the next player to act in bidding/stirring."""
    return next_player(current_player)
