"""Black-box tests for exact browser command decoding."""

from server.foundation.result import Ok
from server.game import CommandRejected, commands
from server.game.rules.cards import CardId
from server.web.game_wire import read_frame


def test_read_frame_decodes_every_command_variant() -> None:
    cases: tuple[tuple[dict[str, object], commands.Command], ...] = (
        (
            {"seq": 1, "type": "next_round"},
            commands.ConfirmRound(),
        ),
        (
            {"seq": 2, "type": "bid", "pass": True},
            commands.PassBid(),
        ),
        (
            {"seq": 3, "type": "bid", "cards": ["a"]},
            commands.RevealBid((CardId("a"),)),
        ),
        (
            {"seq": 4, "type": "stir", "pass": True},
            commands.PassStir(),
        ),
        (
            {
                "seq": 5,
                "type": "stir",
                "cards": ["a", "b"],
            },
            commands.Stir((CardId("a"), CardId("b"))),
        ),
        (
            {"seq": 6, "type": "discard", "cards": ["a"]},
            commands.Bury((CardId("a"),)),
        ),
        (
            {"seq": 7, "type": "play", "cards": ["a"]},
            commands.Play((CardId("a"),)),
        ),
    )

    for raw, expected in cases:
        frame = read_frame(raw)
        decoded = frame.decoder.decode()
        assert frame.seq == raw["seq"]
        assert isinstance(decoded, Ok)
        assert decoded.value == expected


def test_read_frame_keeps_action_lazy_for_zero_and_stale_gate() -> None:
    frame = read_frame(
        {
            "seq": 0,
            "type": "unknown",
            "unexpected": object(),
        }
    )

    assert frame.seq == 0


def test_read_frame_rejects_non_object_and_invalid_sequence() -> None:
    non_object = read_frame(["invalid"])
    invalid_seq = read_frame({"seq": True, "type": "play"})

    assert non_object.seq == 0
    assert isinstance(non_object.decoder.decode(), CommandRejected)
    assert invalid_seq.seq == 0


def test_decode_rejects_missing_or_unknown_action_type() -> None:
    missing = read_frame({"seq": 1})
    unknown = read_frame({"seq": 1, "type": "unknown"})

    assert isinstance(missing.decoder.decode(), CommandRejected)
    assert isinstance(unknown.decoder.decode(), CommandRejected)


def test_decode_requires_string_card_id_array() -> None:
    missing = read_frame({"seq": 1, "type": "play"})
    object_card = read_frame(
        {
            "seq": 1,
            "type": "play",
            "cards": [{"id": "a"}],
        }
    )
    mixed = read_frame({"seq": 1, "type": "play", "cards": ["a", 2]})

    assert isinstance(missing.decoder.decode(), CommandRejected)
    assert isinstance(object_card.decoder.decode(), CommandRejected)
    assert isinstance(mixed.decoder.decode(), CommandRejected)


def test_decode_rejects_unknown_fields_and_conflicting_pass() -> None:
    extra = read_frame(
        {
            "seq": 1,
            "type": "play",
            "cards": ["a"],
            "reason": "legacy",
        }
    )
    pass_with_cards = read_frame(
        {
            "seq": 1,
            "type": "bid",
            "pass": True,
            "cards": ["a"],
        }
    )
    invalid_stir_pass = read_frame(
        {
            "seq": 1,
            "type": "stir",
            "pass": "false",
            "cards": ["a", "b"],
        }
    )

    assert isinstance(extra.decoder.decode(), CommandRejected)
    assert isinstance(pass_with_cards.decoder.decode(), CommandRejected)
    assert isinstance(
        invalid_stir_pass.decoder.decode(), CommandRejected
    )
