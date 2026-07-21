"""Tests for rule-complete semantic legal action indexes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from server.foundation.result import Ok, Rejected
from server.game.players.test_helpers import card, make_snapshot
from server.game.protocol import (
    BidEventSnapshot,
    StirDeclarationEventSnapshot,
    StirringStateSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.game.rules.card_faces import CardFace, FaceCount
from server.game.rules.cards import Card
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.semantic_action_plan import (
    ActionChoiceLogitDecoder,
    ActionSampler,
    action_plan_generation_step_count,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.training.semantic_actions import (
    ActionChoice,
    ActionTrace,
)
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    action_choice_id,
)


@dataclass(slots=True)
class _PreferredChoiceDecoder:
    target_choice_id: int
    batch_size: int
    device: torch.device
    step_index: int = 0

    def next_choice_logits(self) -> torch.Tensor:
        logits = torch.zeros(
            (self.batch_size, ACTION_CHOICE_COUNT),
            dtype=torch.float32,
            device=self.device,
        )
        if self.step_index == 0:
            logits[:, self.target_choice_id] = 100.0
        return logits

    def advance(self, selected_choice_ids: torch.Tensor) -> None:
        assert selected_choice_ids.shape == (self.batch_size,)
        self.step_index += 1


def test_build_legal_action_index_ignores_action_hints_for_follow() -> (
    None
):
    lead = card("hearts", "A", 1)
    heart = card("hearts", "3", 1)
    spade = card("spades", "K", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[heart, spade],
        action_hints=[],
        trick=_trick(
            lead_player=1,
            current_player=2,
            lead_cards=[lead],
        ),
    )

    legal_actions = build_legal_action_index(
        player_index=2,
        snapshot=snapshot,
    )

    decoded_heart = legal_actions.decode(
        ActionTrace(choices=(_card_choice(heart, 1),))
    )
    rejected_spade = legal_actions.decode(
        ActionTrace(choices=(_card_choice(spade, 1),))
    )

    assert isinstance(decoded_heart, Ok)
    assert isinstance(rejected_spade, Rejected)


def test_follow_decode_accepts_only_full_rule_legal_play() -> None:
    lead = card("hearts", "A", 1)
    heart = card("hearts", "3", 1)
    spade = card("spades", "K", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[heart, spade],
        trick=_trick(
            lead_player=1,
            current_player=2,
            lead_cards=[lead],
        ),
    )
    legal_actions = build_legal_action_index(
        player_index=2,
        snapshot=snapshot,
    )

    decoded = legal_actions.decode(
        ActionTrace(choices=(_card_choice(heart, 1),))
    )
    rejected = legal_actions.decode(
        ActionTrace(choices=(_card_choice(spade, 1),))
    )

    assert isinstance(decoded, Ok)
    assert isinstance(rejected, Rejected)


def test_lead_mask_keeps_selected_cards_in_one_effective_suit() -> None:
    heart = card("hearts", "3", 1)
    spade = card("spades", "K", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[heart, spade],
    )
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
    )

    decoded_stop = legal_actions.decode(
        ActionTrace(
            choices=(_card_choice(heart, 1), ActionChoice("finish"))
        )
    )
    rejected_mixed = legal_actions.decode(
        ActionTrace(
            choices=(
                _card_choice(heart, 1),
                _card_choice(spade, 1),
                ActionChoice("finish"),
            )
        )
    )

    assert isinstance(decoded_stop, Ok)
    assert isinstance(rejected_mixed, Rejected)


def test_discard_auto_completes_at_exact_count_without_stop() -> None:
    first = card("hearts", "3", 1)
    second = card("spades", "K", 1)
    snapshot = make_snapshot(
        phase="STIRRING",
        awaiting_action="discard",
        player_hand=[first, second],
        stirring_state=StirringStateSnapshot(
            phase="EXCHANGING",
            trump_suit=None,
            current_player=0,
            declarer_player=0,
            exchanging_player=0,
            exchange_count=2,
        ),
    )
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
    )

    trace = ActionTrace(
        choices=(_card_choice(first, 1), _card_choice(second, 1))
    )

    rejected_extra_stop = legal_actions.decode(
        ActionTrace(choices=(*trace.choices, ActionChoice("finish")))
    )
    assert isinstance(legal_actions.decode(trace), Ok)
    assert isinstance(rejected_extra_stop, Rejected)


def test_bid_current_winner_can_only_pass() -> None:
    first = card("hearts", "2", 1)
    second = card("hearts", "2", 2)
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        trump_rank="2",
        player_hand=[first, second],
        bid_winner=BidEventSnapshot(
            player=0,
            cards=[first],
            kind="trump_rank",
            suit=first.suit,
            joker_type=None,
            count=1,
        ),
    )

    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
    )

    pass_result = legal_actions.decode(
        ActionTrace(choices=(ActionChoice("pass"),))
    )
    select_result = legal_actions.decode(
        ActionTrace(choices=(_card_choice(first, 1),))
    )

    assert isinstance(pass_result, Ok)
    assert isinstance(select_result, Rejected)


def test_stir_mask_uses_current_priority() -> None:
    heart_first = card("hearts", "2", 1)
    heart_second = card("hearts", "2", 2)
    spade_first = card("spades", "2", 1)
    spade_second = card("spades", "2", 2)
    diamond_first = card("diamonds", "2", 1)
    diamond_second = card("diamonds", "2", 2)
    snapshot = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        trump_rank="2",
        trump_suit="hearts",
        player_hand=[
            spade_first,
            spade_second,
            diamond_first,
            diamond_second,
        ],
        bid_winner=BidEventSnapshot(
            player=1,
            cards=[heart_first, heart_second],
            kind="trump_rank",
            suit=heart_first.suit,
            joker_type=None,
            count=2,
        ),
        stirring_state=StirringStateSnapshot(
            phase="WAITING",
            trump_suit=heart_first.suit,
            current_player=0,
            declarer_player=1,
            exchanging_player=None,
            exchange_count=None,
        ),
    )

    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
    )
    pass_result = legal_actions.decode(
        ActionTrace(choices=(ActionChoice("pass"),))
    )
    spade_choice = action_choice_id(_card_choice(spade_first, 2))
    diamond_choice = action_choice_id(_card_choice(diamond_first, 2))

    assert isinstance(pass_result, Ok)
    assert (
        _sampled_first_choice_id(
            legal_actions=legal_actions,
            target_choice_id=spade_choice,
        )
        == spade_choice
    )
    assert (
        _sampled_first_choice_id(
            legal_actions=legal_actions,
            target_choice_id=diamond_choice,
        )
        != diamond_choice
    )


def test_stir_mask_uses_stir_event_priority_over_bid_winner() -> None:
    diamond_first = card("diamonds", "2", 1)
    diamond_second = card("diamonds", "2", 2)
    spade_first = card("spades", "2", 1)
    spade_second = card("spades", "2", 2)
    heart_first = card("hearts", "2", 1)
    heart_second = card("hearts", "2", 2)
    small_joker_first = card("joker", "SJ", 1)
    small_joker_second = card("joker", "SJ", 2)
    snapshot = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        trump_rank="2",
        trump_suit="spades",
        player_hand=[
            heart_first,
            heart_second,
            small_joker_first,
            small_joker_second,
        ],
        bid_winner=BidEventSnapshot(
            player=0,
            cards=[diamond_first, diamond_second],
            kind="trump_rank",
            suit=diamond_first.suit,
            joker_type=None,
            count=2,
        ),
        stir_events=[
            StirDeclarationEventSnapshot(
                player=1,
                kind="stir",
                cards=[spade_first, spade_second],
                new_suit=spade_first.suit,
                priority=203,
                own_bottom_exchange=None,
            )
        ],
        stirring_state=StirringStateSnapshot(
            phase="WAITING",
            trump_suit=spade_first.suit,
            current_player=2,
            declarer_player=0,
            exchanging_player=None,
            exchange_count=None,
        ),
    )

    legal_actions = build_legal_action_index(
        player_index=2,
        snapshot=snapshot,
    )
    pass_result = legal_actions.decode(
        ActionTrace(choices=(ActionChoice("pass"),))
    )
    heart_choice = action_choice_id(_card_choice(heart_first, 2))
    joker_choice = action_choice_id(_card_choice(small_joker_first, 2))

    assert isinstance(pass_result, Ok)
    assert (
        _sampled_first_choice_id(
            legal_actions=legal_actions,
            target_choice_id=heart_choice,
        )
        != heart_choice
    )
    assert (
        _sampled_first_choice_id(
            legal_actions=legal_actions,
            target_choice_id=joker_choice,
        )
        == joker_choice
    )


def _trick(
    *,
    lead_player: int,
    current_player: int,
    lead_cards: list[Card],
) -> TrickSnapshot:
    return TrickSnapshot(
        lead_player=lead_player,
        current_player=current_player,
        slots=[
            TrickSlotSnapshot(player=0, cards=[]),
            TrickSlotSnapshot(
                player=lead_player,
                cards=list(lead_cards),
            ),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
    )


def _card_choice(card_value: Card, count: int) -> ActionChoice:
    return ActionChoice(
        "card",
        FaceCount(
            CardFace(card_value.suit, card_value.rank),
            count,
        ),
    )


def _sampled_first_choice_id(
    *, legal_actions: LegalActionIndex, target_choice_id: int
) -> int:
    device = torch.device("cpu")
    action_plan = compile_legal_action_frame(legal_actions)
    generation_steps = action_plan_generation_step_count(action_plan)
    action_batch = plan_batch_to_device((action_plan,), device=device)

    logit_decoder: ActionChoiceLogitDecoder = _PreferredChoiceDecoder(
        target_choice_id=target_choice_id,
        batch_size=1,
        device=device,
    )
    sampler = ActionSampler.create(batch_capacity=1, device=device)
    sample_result = sampler.sample(
        action_batch=action_batch,
        generation_step_counts=torch.tensor(
            (generation_steps,), dtype=torch.long, device=device
        ),
        sampling_thresholds=torch.full(
            (1, generation_steps),
            0.5,
            dtype=torch.float64,
            device=device,
        ),
        padded_generation_steps=generation_steps,
        logit_decoder=logit_decoder,
    )
    assert isinstance(sample_result, Ok)
    first_choice = sample_result.value.choice_ids_padded[0, 0]
    return int(first_choice.item())
