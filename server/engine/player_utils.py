"""Player positioning utility functions for 升级 (Shengji/Tractor).

Provides player turn order and team relationship utilities.

This module fixes a bug from the original TypeScript ``getPartnerIndex``:
the ``(playerIndex + 2) % 4`` formula returns a player from the opposite
team instead of the same-team partner.  With TEAM_0=[0,3] and TEAM_1=[1,2],
player 0's partner is player 3 (not player 2).
"""

from server.engine.constants import NEXT_PLAYER, TEAM_0, TEAM_1


def next_player(current: int) -> int:
    """Advance to the next player clockwise."""
    return NEXT_PLAYER[current]


def clockwise_distance(from_idx: int, to_idx: int) -> int:
    """Number of clockwise steps from *from_idx* to *to_idx*."""
    steps = 0
    cur = from_idx
    while cur != to_idx and steps < 4:
        cur = NEXT_PLAYER[cur]
        steps += 1
    return steps


def get_team_index(player_index: int) -> int:
    """Return the team index (0 or 1) for *player_index*."""
    if player_index in TEAM_0:
        return 0
    return 1


def get_partner_index(player_index: int) -> int:
    """Return the index of the same-team partner.

    Partners sit opposite each other:
      N(0) <-> S(3)  (TEAM_0)
      W(1) <-> E(2)  (TEAM_1)

    Note: the original TS formula ``(playerIndex + 2) % 4`` is WRONG here
    because it returns the opposite-team member.
    """
    if player_index in TEAM_0:
        return TEAM_0[1] if player_index == TEAM_0[0] else TEAM_0[0]
    return TEAM_1[1] if player_index == TEAM_1[0] else TEAM_1[0]
