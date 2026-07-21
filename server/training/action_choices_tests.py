"""Black-box contract tests for the closed action vocabulary."""

from server.foundation.result import Ok, Rejected
from server.game.rules.card_faces import FaceCount
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    CARD_FACES,
    FINISH_CHOICE_ID,
    PASS_CHOICE_ID,
    ActionChoice,
    ActionPrefix,
    action_choice_from_id,
    action_choice_id,
    action_choice_name,
    action_prefix_cards,
)


def test_action_vocabulary_is_exactly_two_controls_plus_108_cards() -> (
    None
):
    assert ACTION_CHOICE_COUNT == 110
    assert action_choice_id(ActionChoice("pass")) == PASS_CHOICE_ID
    assert action_choice_id(ActionChoice("finish")) == FINISH_CHOICE_ID
    assert len(CARD_FACES) == 54
    assert len(set(CARD_FACES)) == 54


def test_every_action_choice_round_trips() -> None:
    for choice_id in range(ACTION_CHOICE_COUNT):
        decoded = action_choice_from_id(choice_id)
        assert isinstance(decoded, Ok)
        assert action_choice_id(decoded.value) == choice_id


def test_card_names_express_payload_instead_of_decoder_operation() -> (
    None
):
    choice = ActionChoice(
        "card", FaceCount(face=CARD_FACES[0], count=2)
    )
    assert action_choice_name(choice).startswith("CARD_")


def test_prefix_rejects_duplicate_card_face() -> None:
    card = ActionChoice("card", FaceCount(face=CARD_FACES[0], count=1))
    result = action_prefix_cards(ActionPrefix((card, card)))
    assert isinstance(result, Rejected)


def test_vocabulary_has_no_reserved_or_start_choice() -> None:
    assert isinstance(action_choice_from_id(-1), Rejected)
    assert isinstance(
        action_choice_from_id(ACTION_CHOICE_COUNT), Rejected
    )
