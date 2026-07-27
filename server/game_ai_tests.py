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
from server.game_ai import (
    ActionSampleRequest,
    ActionSamples,
    ActionScoreRequest,
    ModelQuery,
    SampledAction,
)
from server.game_ai.controller import AIController
from server.game_ai.search import SearchConfig
from server.training.legal_actions.complete_trace import (
    trace_for_selection,
)
from server.training.observation import Observation
from server.training.semantic_action_plan import (
    ActionChoiceLogitDecoder,
    ActionSampler,
    action_plan_generation_step_count,
    action_trace_from_choice_ids,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.training.semantic_actions import GeneratedAction
from server.training.semantic_actions.binding import (
    bind_generated_action,
)
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
)


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


def _sample_request_counts() -> list[tuple[int, ...]]:
    return []


def _score_batch_sizes() -> list[int]:
    return []


@dataclass(slots=True)
class _SearchBiasedModel:
    root_actions: list[GeneratedAction] = field(
        default_factory=_actions
    )
    public_score_count: int = 0
    sample_request_counts: list[tuple[int, ...]] = field(
        default_factory=_sample_request_counts
    )
    score_batch_sizes: list[int] = field(
        default_factory=_score_batch_sizes
    )

    async def sample(
        self,
        *,
        requests: tuple[ActionSampleRequest, ...],
    ) -> Ok[tuple[ActionSamples, ...]] | Rejected:
        assert requests
        self.sample_request_counts.append(
            tuple(len(request.draws) for request in requests)
        )
        queries = tuple(
            request.query
            for request in requests
            for _draw in request.draws
        )
        if (
            len(queries) == 2
            and queries[0].observation.action_query.kind == "lead_play"
            and not self.root_actions
        ):
            actions = (
                _single_card_lead(queries[0], first=True),
                _single_card_lead(queries[1], first=False),
            )
            if isinstance(actions[0], Rejected):
                return actions[0]
            if isinstance(actions[1], Rejected):
                return actions[1]
            self.root_actions.extend(
                (actions[0].value, actions[1].value)
            )
            return Ok(
                (
                    ActionSamples(
                        samples=(
                            SampledAction(
                                action=actions[0].value,
                                log_probability=10.0,
                            ),
                            SampledAction(
                                action=actions[1].value,
                                log_probability=0.0,
                            ),
                        )
                    ),
                )
            )
        groups: list[ActionSamples] = []
        for request in requests:
            samples: list[SampledAction] = []
            for _draw in request.draws:
                action = _first_legal_action(request.query)
                if isinstance(action, Rejected):
                    return action
                samples.append(
                    SampledAction(
                        action=action.value,
                        log_probability=0.0,
                    )
                )
            groups.append(ActionSamples(samples=tuple(samples)))
        return Ok(tuple(groups))

    async def score(
        self,
        *,
        requests: tuple[ActionScoreRequest, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        self.public_score_count += len(requests)
        self.score_batch_sizes.append(len(requests))
        return Ok(tuple(-0.25 for _request in requests))

    async def values(
        self,
        *,
        observations: tuple[Observation, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        assert len(observations) == 2
        return Ok((-1.0, 1.0))


async def test_ai_uses_rollout_value_instead_of_argmax() -> None:
    state = _round_start_with_actor(Seat.A)
    model = _SearchBiasedModel()
    controller = AIController(
        seat=Seat.A,
        model=model,
        particle_count=1,
        direct_samples=1,
        search_config=SearchConfig(
            candidate_samples=2,
            rollouts_per_particle=2,
            rollout_depth=1,
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
        if actor_view.awaiting_action == "bid":
            command = commands.PassBid()
        elif actor_view.awaiting_action == "discard":
            assert actor == Seat.A
            decided = await controller.decide(
                seq=seq,
                snapshot=current,
            )
            assert isinstance(decided, Ok)
            command = decided.value
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
    assert model.public_score_count >= 100
    assert max(model.score_batch_sizes) > 1
    assert (2, 2) in model.sample_request_counts


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
    query: ModelQuery,
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
    query: ModelQuery,
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
