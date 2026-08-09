"""Black-box tests for deterministic balanced rollout identities."""

from __future__ import annotations

from server.game import Partnership, Seat
from server.training.rollout import GameIdentity


def test_identity_is_deterministic_and_separates_random_streams() -> (
    None
):
    identity = GameIdentity(
        base_seed=17,
        worker_index=2,
        game_env_index=3,
        game_ordinal=4,
    )
    same = GameIdentity(
        base_seed=17,
        worker_index=2,
        game_env_index=3,
        game_ordinal=4,
    )

    assert identity.game_seed() == same.game_seed()
    assert identity.auto_seed(Seat.A) == same.auto_seed(Seat.A)
    assert identity.auto_seed(Seat.A) != identity.auto_seed(Seat.C)
    assert identity.game_seed() != identity.auto_seed(Seat.A)


def test_adjacent_games_swap_model_and_auto_partnerships() -> None:
    first = GameIdentity(
        base_seed=0,
        worker_index=0,
        game_env_index=0,
        game_ordinal=0,
    )
    second = first.next_game()

    assert first.model_partnership() == Partnership.FIRST
    assert second.model_partnership() == Partnership.SECOND
    assert second.next_game().model_partnership() == Partnership.FIRST
    assert first.game_seed() != second.game_seed()
