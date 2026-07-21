"""Black-box tests for viewer-relative policy state."""

from __future__ import annotations

from server.foundation.result import Ok
from server.game.players.test_helpers import card, make_snapshot
from server.game.protocol import (
    FailedThrowSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.training.observation_memory import ObservationMemoryView
from server.training.relative_state import (
    RelativeActor,
    TrumpMode,
    project_relative_observation,
)


def test_project_removes_viewer_and_absolute_team_identity() -> None:
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer_team=1,
        declarer_player=3,
        team0_level="10",
        team1_level="K",
        player_hand=[card("spades", "A")],
        player_hand_counts=[1, 4, 2, 3],
        trick=_open_trick(lead_player=1, current_player=0),
    )

    result = project_relative_observation(
        viewer=0,
        snapshot=snapshot,
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )

    assert isinstance(result, Ok)
    observation = result.value
    assert not hasattr(observation, "player_index")
    assert not hasattr(observation.round_context, "declarer_team")
    assert observation.round_context.declarer_actor == (
        RelativeActor.RIGHT_ENEMY
    )
    assert observation.round_context.own_level.value == "10"
    assert observation.round_context.opponent_level.value == "K"


def test_project_rotated_seats_produce_equal_relative_state() -> None:
    original = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer_team=0,
        declarer_player=2,
        team0_level="J",
        team1_level="A",
        player_hand=[card("clubs", "5")],
        player_hand_counts=[1, 3, 2, 4],
        trick=_open_trick(lead_player=1, current_player=0),
    )
    rotated = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        declarer_team=1,
        declarer_player=3,
        team0_level="A",
        team1_level="J",
        player_hand=[card("clubs", "5")],
        player_hand_counts=[4, 1, 3, 2],
        trick=_open_trick(lead_player=2, current_player=1),
    )
    empty = ObservationMemoryView(bid_actions=(), completed_tricks=())

    first = project_relative_observation(
        viewer=0, snapshot=original, memory=empty
    )
    second = project_relative_observation(
        viewer=1, snapshot=rotated, memory=empty
    )

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value == second.value


def test_project_distinguishes_unset_from_no_trump() -> None:
    empty = ObservationMemoryView(bid_actions=(), completed_tricks=())
    unset = project_relative_observation(
        viewer=0,
        snapshot=make_snapshot(
            phase="DEAL_BID",
            awaiting_action="bid",
            trump_suit=None,
            player_hand=[card("clubs", "2")],
            player_hand_counts=[1, 0, 0, 0],
        ),
        memory=empty,
    )
    no_trump = project_relative_observation(
        viewer=0,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action="play",
            trump_suit=None,
            player_hand=[card("clubs", "2")],
            player_hand_counts=[1, 0, 0, 0],
            trick=_open_trick(lead_player=0, current_player=0),
        ),
        memory=empty,
    )

    assert isinstance(unset, Ok)
    assert isinstance(no_trump, Ok)
    assert unset.value.round_context.trump.mode == TrumpMode.UNSET
    assert no_trump.value.round_context.trump.mode == TrumpMode.NO_TRUMP


def test_project_failed_throw_keeps_only_revealed_extra() -> None:
    forced = card("hearts", "Q")
    extra = card("hearts", "K")
    trick = TrickSnapshot(
        lead_player=1,
        current_player=2,
        slots=[
            TrickSlotSnapshot(player=0, cards=[]),
            TrickSlotSnapshot(player=1, cards=[forced]),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
        failed_throw=FailedThrowSnapshot(
            player=1,
            attempted_cards=[forced, extra],
            forced_cards=[forced],
        ),
    )

    result = project_relative_observation(
        viewer=0,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action=None,
            player_hand_counts=[0, 0, 0, 0],
            trick=trick,
        ),
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )

    assert isinstance(result, Ok)
    action = result.value.tricks[-1].actions[0]
    assert action.actor == RelativeActor.LEFT_ENEMY
    assert [
        (item.face.rank.value, item.count) for item in action.played
    ] == [("Q", 1)]
    assert [
        (item.face.rank.value, item.count)
        for item in action.revealed_extra
    ] == [("K", 1)]


def _open_trick(
    *, lead_player: int, current_player: int
) -> TrickSnapshot:
    lead_card = card("hearts", "2")
    slots = [
        TrickSlotSnapshot(player=player, cards=[])
        for player in range(4)
    ]
    if lead_player != current_player:
        slots[lead_player] = TrickSlotSnapshot(
            player=lead_player, cards=[lead_card]
        )
    return TrickSnapshot(
        lead_player=lead_player,
        current_player=current_player,
        slots=slots,
    )
