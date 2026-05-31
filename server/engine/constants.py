"""Game structure constants and level progression for 升级 (Shengji/Tractor).

Contains only pure configuration constants: deck/player counts, player/team
mapping, next-player lookup, and the level progression order.

Scoring thresholds and scoring logic are in the scoring module.
Utility functions are in the player_utils module.
Card-related constants (POINTS_MAP, TOTAL_POINTS) are in the card_utils module.
"""

from server.engine.card import Rank

# ---- Deck Configuration ----

DECK_COUNT: int = 2
TOTAL_CARDS: int = 108
PLAYER_COUNT: int = 4
BOTTOM_CARD_COUNT: int = 8
CARDS_PER_PLAYER: int = (TOTAL_CARDS - BOTTOM_CARD_COUNT) // PLAYER_COUNT  # 25

# ---- Player Positioning ----
#
# Player indices and seating:
#   0 = North (human's partner, AI)
#   1 = West  (opponent, AI)
#   2 = East  (opponent, AI)
#   3 = South (human)
#
# Teams: {0,3} vs {1,2}
#   Team 0: North (AI) + South (Human)
#   Team 1: West (AI) + East (AI)

HUMAN_PLAYER_INDEX: int = 3

TEAM_0: list[int] = [0, 3]  # North + South
TEAM_1: list[int] = [1, 2]  # West + East

# Clockwise next-player lookup:
#        North(0)
#   West(1)    East(2)
#        South(3)
# Clockwise: N(0)->E(2)->S(3)->W(1)->N(0)
NEXT_PLAYER: dict[int, int] = {
    0: 2,  # North -> East
    1: 0,  # West  -> North
    2: 3,  # East  -> South
    3: 1,  # South -> West
}

# ---- Level Progression ----

LEVELS: list[Rank] = [
    Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE,
    Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE,
    Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
]

START_LEVEL: Rank = Rank.TWO
