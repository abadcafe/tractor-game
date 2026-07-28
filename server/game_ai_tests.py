"""Black-box tests for the trained-model AI player boundary."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor

from server.foundation.result import Ok, Rejected
from server.game import (
    GameConfig,
    GameSeed,
    GameState,
    Seat,
    apply,
    commands,
    create,
    observe,
    seats,
)
from server.game.rules.cards.faces import FaceCount
from server.game_ai.controller import AIController
from server.game_ai.search import SearchConfig
from server.game_ai.simulation import SimulationBranch
from server.policy_model._schema.actions import (
    ACTION_CHOICE_COUNT,
)
from server.policy_model.actions import GeneratedAction
from server.policy_model.actions.decoding import (
    ActionChoiceLogitDecoder,
    ActionSampler,
    action_plan_generation_step_count,
    action_trace_from_choice_ids,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.policy_model.actions.legality.complete_trace import (
    trace_for_selection,
)
from server.policy_model.actions.semantic.binding import (
    bind_generated_action,
)
from server.policy_model.inference import (
    ActionProposalRequest,
    ActionProposals,
    ActionSamples,
    PolicyQuery,
    PolicySampleRequest,
    PolicyScoreRequest,
    ProposedAction,
    SampledAction,
)
from server.policy_model.observation import (
    Observation,
    build_observation,
)
from server.policy_model.observation.history import ObservationMemory


@dataclass(slots=True)
class _ZeroDecoder(ActionChoiceLogitDecoder):
    batch_size: int

    def next_choice_logits(
        self,
        active_rows: Tensor,
        scored_rows: Tensor,
    ) -> Tensor:
        assert active_rows.shape == (self.batch_size,)
        assert scored_rows.shape == active_rows.shape
        return torch.zeros(
            (self.batch_size, ACTION_CHOICE_COUNT),
            dtype=torch.float32,
        )

    def advance(
        self,
        selected_choice_ids: Tensor,
        active_rows: Tensor,
    ) -> None:
        assert selected_choice_ids.shape == (self.batch_size,)
        assert active_rows.shape == (self.batch_size,)


def _actions() -> list[GeneratedAction]:
    return []


def _seats() -> list[Seat]:
    return []


@dataclass(slots=True)
class _PolicyImprovementModel:
    root_actions: list[GeneratedAction] = field(
        default_factory=_actions
    )
    scored_seats: list[Seat] = field(default_factory=_seats)
    value_call_count: int = 0
    policy_row_count: int = 0

    async def propose(
        self,
        *,
        requests: tuple[ActionProposalRequest, ...],
    ) -> Ok[tuple[ActionProposals, ...]] | Rejected:
        results: list[ActionProposals] = []
        for request in requests:
            if (
                request.query.observation.action_query.kind
                == "lead_play"
            ):
                first = _single_card_lead(
                    request.query,
                    first=True,
                )
                if isinstance(first, Rejected):
                    return first
                second = _single_card_lead(
                    request.query,
                    first=False,
                )
                if isinstance(second, Rejected):
                    return second
                self.root_actions = [first.value, second.value]
                results.append(
                    ActionProposals(
                        candidates=(
                            ProposedAction(
                                action=first.value,
                                log_probability=0.0,
                                perturbed_root_score=1.0,
                            ),
                            ProposedAction(
                                action=second.value,
                                log_probability=-1.0,
                                perturbed_root_score=0.0,
                            ),
                        )
                    )
                )
                continue
            action = _first_legal_action(request.query)
            if isinstance(action, Rejected):
                return action
            results.append(
                ActionProposals(
                    candidates=(
                        ProposedAction(
                            action=action.value,
                            log_probability=0.0,
                            perturbed_root_score=0.0,
                        ),
                    )
                )
            )
        return Ok(tuple(results))

    async def sample(
        self,
        *,
        requests: tuple[PolicySampleRequest, ...],
    ) -> Ok[tuple[ActionSamples, ...]] | Rejected:
        self.policy_row_count += len(requests)
        groups: list[ActionSamples] = []
        for request in requests:
            action = _first_legal_action(request.query)
            if isinstance(action, Rejected):
                return action
            groups.append(
                ActionSamples(
                    samples=tuple(
                        SampledAction(
                            action=action.value,
                            log_probability=0.0,
                        )
                        for _draw in request.draws
                    )
                )
            )
        return Ok(tuple(groups))

    async def score(
        self,
        *,
        requests: tuple[PolicyScoreRequest, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        self.scored_seats.extend(
            request.query.seat for request in requests
        )
        return Ok(tuple(-0.25 for _request in requests))

    async def values(
        self,
        *,
        observations: tuple[Observation, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        assert len(observations) == 2
        self.value_call_count += 1
        return Ok((-1.0, 1.0))


async def test_ai_improves_policy_without_scoring_self() -> None:
    state = _round_start_with_actor(Seat.A)
    model = _PolicyImprovementModel()
    controller = AIController(
        seat=Seat.A,
        model=model,
        particle_count=1,
        search_config=SearchConfig(
            root_candidate_count=2,
            macro_simulation_budget=2,
        ),
        random_source=random.Random(7),
    )
    seq = 0
    current = observe(state, Seat.A)
    observed = controller.observe(
        seq=seq,
        snapshot=current,
        error=None,
    )
    assert isinstance(observed, Ok)

    while current.awaiting_action != "play":
        actor = _active_actor(state)
        actor_view = observe(state, actor)
        command: commands.Command
        if actor == Seat.A:
            decided = await controller.decide(
                seq=seq,
                snapshot=current,
            )
            assert isinstance(decided, Ok)
            command = decided.value
        elif actor_view.awaiting_action == "bid":
            command = commands.PassBid()
        elif actor_view.awaiting_action == "discard":
            assert False
        else:
            assert actor_view.awaiting_action == "stir"
            command = commands.PassStir()
        advanced = apply(state, actor, command)
        assert isinstance(advanced, Ok)
        state = advanced.value
        seq += 1
        current = observe(state, Seat.A)
        observed = controller.observe(
            seq=seq,
            snapshot=current,
            error=None,
        )
        assert isinstance(observed, Ok)

    decided = await controller.decide(seq=seq, snapshot=current)

    assert isinstance(decided, Ok)
    assert isinstance(decided.value, commands.Play)
    assert len(model.root_actions) == 2
    expected = bind_generated_action(
        model.root_actions[1],
        current.hand,
    )
    assert isinstance(expected, Ok)
    assert decided.value == expected.value
    assert model.scored_seats
    assert Seat.A not in model.scored_seats
    assert model.value_call_count == 1
    assert model.policy_row_count >= 6


def test_simulation_branch_matches_all_player_observations() -> None:
    state = _round_start_with_actor(Seat.A)
    started = SimulationBranch.start(state)
    assert isinstance(started, Ok)
    branch = started.value
    memories = {seat: ObservationMemory() for seat in seats()}
    for seat in seats():
        remembered = memories[seat].observe(
            seq=0,
            snapshot=observe(state, seat),
        )
        assert isinstance(remembered, Ok)
    seq = 0
    compared: set[Seat] = set()
    for _index in range(12):
        actor = _active_actor(state)
        actor_view = observe(state, actor)
        expected = branch.model_input(actor)[0]
        actual = build_observation(
            viewer=actor,
            snapshot=actor_view,
            memory=memories[actor].view(),
        )
        assert expected == actual
        compared.add(actor)
        assert actor_view.awaiting_action == "bid"
        command = commands.PassBid()
        advanced = apply(state, actor, command)
        assert isinstance(advanced, Ok)
        state = advanced.value
        branched = branch.advance(actor=actor, command=command)
        assert isinstance(branched, Ok)
        branch = branched.value
        seq += 1
        for seat in seats():
            remembered = memories[seat].observe(
                seq=seq,
                snapshot=observe(state, seat),
            )
            assert isinstance(remembered, Ok)
    assert compared == set(seats())


def _round_start_with_actor(actor: Seat) -> GameState:
    for seed_value in range(100):
        state = create(GameConfig(), GameSeed(seed_value))
        for seat in seats():
            confirmed = apply(state, seat, commands.ConfirmRound())
            assert isinstance(confirmed, Ok)
            state = confirmed.value
        if observe(state, actor).awaiting_action == "bid":
            return state
    assert False


def _active_actor(state: GameState) -> Seat:
    active = [
        seat
        for seat in seats()
        if observe(state, seat).awaiting_action
        in ("bid", "discard", "stir", "play")
    ]
    assert len(active) == 1
    return active[0]


def _single_card_lead(
    query: PolicyQuery,
    *,
    first: bool,
) -> Ok[GeneratedAction] | Rejected:
    faces = query.observation.hand_faces
    assert faces
    face = faces[0] if first else faces[-1]
    trace = trace_for_selection(
        (FaceCount(face.face, 1),),
        include_finish=True,
    )
    return query.legal_actions.decode(trace)


def _first_legal_action(
    query: PolicyQuery,
) -> Ok[GeneratedAction] | Rejected:
    frame = compile_legal_action_frame(query.legal_actions)
    steps = action_plan_generation_step_count(frame)
    device = torch.device("cpu")
    sampled = ActionSampler.create(
        batch_capacity=1,
        device=device,
    ).sample(
        action_batch=plan_batch_to_device((frame,), device=device),
        generation_step_counts=torch.tensor(
            (steps,),
            dtype=torch.long,
            device=device,
        ),
        sampling_thresholds=torch.zeros(
            (1, steps),
            dtype=torch.float64,
            device=device,
        ),
        padded_generation_steps=steps,
        logit_decoder=_ZeroDecoder(batch_size=1),
    )
    if isinstance(sampled, Rejected):
        return sampled
    count = int(sampled.value.step_counts[0].item())
    choice_ids = tuple(
        int(sampled.value.choice_ids_padded[0, index].item())
        for index in range(count)
    )
    trace = action_trace_from_choice_ids(choice_ids)
    if isinstance(trace, Rejected):
        return trace
    return query.legal_actions.decode(trace.value)
