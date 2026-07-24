"""Black-box tests for the browser game-wire facade."""

from __future__ import annotations

from server.foundation.result import Ok
from server.game import CommandRejected, commands
from server.game.rules.cards import CardId
from server.web.game_wire import read_frame


def test_frame_reads_sequence_before_decoding_action() -> None:
    frame = read_frame(
        {
            "seq": 7,
            "type": "play",
            "cards": ["D1-hearts-2"],
        }
    )

    assert frame.seq == 7
    decoded = frame.decoder.decode()
    assert isinstance(decoded, Ok)
    assert decoded.value == commands.Play((CardId("D1-hearts-2"),))


def test_non_object_becomes_sequence_zero_state_request() -> None:
    frame = read_frame(["invalid"])

    assert frame.seq == 0
    assert isinstance(frame.decoder.decode(), CommandRejected)


def test_invalid_sequence_becomes_zero_without_decoding_action() -> (
    None
):
    frame = read_frame({"seq": True, "type": "unknown"})

    assert frame.seq == 0
    assert isinstance(frame.decoder.decode(), CommandRejected)
