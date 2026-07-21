"""Black-box tests for training observation memory."""

from __future__ import annotations

from typing import Literal

from server.foundation.result import Ok, Rejected
from server.game.players.test_helpers import card, make_snapshot
from server.game.protocol import (
    BidEventSnapshot,
    CompletedTrickSnapshot,
    StateMessage,
    StateSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.training.observation_memory import ObservationMemory


def test_observe_records_bid_pass_and_reveal_at_decision_ordinal() -> (
    None
):
    memory = ObservationMemory()
    first = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand_counts=[1, 0, 0, 0],
        player_hand=[card("hearts", "2")],
    )
    second = make_snapshot(
        phase="DEAL_BID",
        awaiting_action=None,
        player_hand_counts=[1, 1, 0, 0],
    )
    revealed = card("spades", "2")
    bid = BidEventSnapshot(
        player=1,
        cards=[revealed],
        kind="trump_rank",
        suit=revealed.suit,
        joker_type=None,
        count=1,
    )
    third = make_snapshot(
        phase="DEAL_BID",
        awaiting_action=None,
        player_hand_counts=[1, 1, 1, 0],
        bid_events=[bid],
        bid_winner=bid,
    )

    assert isinstance(memory.observe(_message(1, first)), Ok)
    assert isinstance(memory.observe(_message(2, second)), Ok)
    result = memory.observe(_message(3, third))

    assert isinstance(result, Ok)
    actions = result.value.bid_actions
    assert len(actions) == 2
    assert actions[0].actor == 0
    assert actions[0].deal_ordinal == 1
    assert actions[0].revealed_cards == ()
    assert actions[1].actor == 1
    assert actions[1].deal_ordinal == 2
    assert actions[1].revealed_cards == (revealed,)


def test_observe_ignores_error_message_at_current_sequence() -> None:
    memory = ObservationMemory()
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand_counts=[1, 0, 0, 0],
    )
    first = memory.observe(_message(1, snapshot))
    error = memory.observe(_message(1, snapshot, error="invalid bid"))

    assert isinstance(first, Ok)
    assert isinstance(error, Ok)
    assert error.value == first.value


def test_observe_accepts_duplicate_state_sync_idempotently() -> None:
    memory = ObservationMemory()
    snapshot = make_snapshot(
        phase="WAITING", awaiting_action="next_round"
    )
    first = memory.observe(_message(7, snapshot))
    duplicate = memory.observe(_message(7, snapshot))

    assert isinstance(first, Ok)
    assert isinstance(duplicate, Ok)
    assert duplicate.value == first.value


def test_observe_rejects_sequence_gap() -> None:
    memory = ObservationMemory()
    first = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand_counts=[1, 0, 0, 0],
    )
    third = make_snapshot(
        phase="DEAL_BID",
        awaiting_action=None,
        player_hand_counts=[1, 1, 0, 0],
    )

    assert isinstance(memory.observe(_message(1, first)), Ok)
    result = memory.observe(_message(3, third))

    assert isinstance(result, Rejected)
    assert result.reason == "observation memory missed state sequence 2"


def test_observe_rejects_mid_round_initial_snapshot() -> None:
    memory = ObservationMemory()
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
    )

    result = memory.observe(_message(9, snapshot))

    assert isinstance(result, Rejected)
    assert (
        result.reason
        == "observation memory did not observe round start"
    )


def test_observe_keeps_consecutive_equal_face_tricks() -> None:
    memory = ObservationMemory()
    deal = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand_counts=[1, 0, 0, 0],
    )
    playing = make_snapshot(
        phase="PLAYING",
        awaiting_action=None,
        player_hand_counts=[25, 25, 25, 25],
    )
    first = _completed_trick(deck=1)
    second = _completed_trick(deck=2)
    open_second = TrickSnapshot(
        lead_player=0,
        current_player=1,
        slots=[
            TrickSlotSnapshot(player=0, cards=[card("hearts", "A", 2)]),
            TrickSlotSnapshot(player=1, cards=[]),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
    )

    assert isinstance(memory.observe(_message(1, deal)), Ok)
    assert isinstance(memory.observe(_message(2, playing)), Ok)
    assert isinstance(
        memory.observe(
            _message(
                3,
                playing.model_copy(
                    update={"last_completed_trick": first}
                ),
            )
        ),
        Ok,
    )
    assert isinstance(
        memory.observe(
            _message(
                4,
                playing.model_copy(update={"trick": open_second}),
            )
        ),
        Ok,
    )
    result = memory.observe(
        _message(
            5,
            playing.model_copy(update={"last_completed_trick": second}),
        )
    )

    assert isinstance(result, Ok)
    assert result.value.completed_tricks == (first, second)


def _message(
    seq: int,
    snapshot: StateSnapshot,
    *,
    error: str | None = None,
) -> StateMessage:
    return StateMessage(seq=seq, state=snapshot, error=error)


def _completed_trick(*, deck: Literal[1, 2]) -> CompletedTrickSnapshot:
    played = card("hearts", "A", deck)
    return CompletedTrickSnapshot(
        lead_player=0,
        slots=[
            TrickSlotSnapshot(player=0, cards=[played]),
            TrickSlotSnapshot(player=1, cards=[]),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
        winner=0,
        points=0,
    )
