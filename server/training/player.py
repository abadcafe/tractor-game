"""Training player implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game.players.base import GameView, Player
from server.game.protocol import PlayerMessage, StateMessage
from server.game.rules.cards import Card
from server.training.legal_actions import build_legal_action_index
from server.training.observation import (
    PublicHistoryRecorder,
    build_observation,
)
from server.training.policy import TrainingPolicy
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_actions.binding import (
    bind_generated_action,
)
from server.training.semantic_actions.values import GeneratedAction
from server.training.trajectory import DecisionStep, TrajectoryRecorder


@dataclass(frozen=True, slots=True)
class TrainingPlayerStats:
    """Per-round model action statistics."""

    generated_action_count: int = 0
    accepted_action_count: int = 0
    action_choice_count: int = 0


@dataclass(frozen=True, slots=True)
class PendingDecision:
    """Model decision submitted to Game and awaiting settlement."""

    seq: int
    step: DecisionStep


class TrainingPlayer(Player):
    """
    Policy-driven player that records accepted model decisions.

    It participates through the same Game.receive() boundary as human,
    auto, and AI players.  It owns no optimizer and writes no
    checkpoint; trainer modules consume its accepted trajectory records.
    """

    def __init__(
        self,
        index: int,
        *,
        policy: TrainingPolicy,
        recorder: TrajectoryRecorder | None = None,
        history: PublicHistoryRecorder | None = None,
    ) -> None:
        super().__init__(index)
        self._policy = policy
        self._recorder = recorder or TrajectoryRecorder()
        self._history = history or PublicHistoryRecorder()
        self._pending: PendingDecision | None = None
        self._held_scoring_message: StateMessage | None = None
        self._action_tasks: set[asyncio.Task[None]] = set()
        self._policy_rejection: Rejected | None = None
        self._stats = TrainingPlayerStats()
        self._base_seed = 0
        self._policy_version = 0
        self._episode_id = 0
        self._decision_index = 0

    async def run(self, game: GameView) -> None:
        """Request the current state and start the player loop."""
        await game.receive(self.index, PlayerMessage(seq=0, raw={}))

    async def on_state(
        self, game: GameView, message: StateMessage
    ) -> None:
        if message.state.scoring is None:
            self._history.update(message.state)
        if self._settle_pending(message):
            return
        if message.state.awaiting_action is None:
            if message.state.scoring is not None:
                self._held_scoring_message = None
            return
        if message.state.awaiting_action == "next_round":
            self._handle_next_round_state(game, message)
            return
        await self._submit_model_action(game, message)

    def recorder(self) -> TrajectoryRecorder:
        return self._recorder

    def stats(self) -> TrainingPlayerStats:
        """Return per-round action stats."""
        return self._stats

    def raise_background_errors(self) -> Ok[None] | Rejected:
        """Raise any programming error from submitted action tasks."""
        if self._policy_rejection is not None:
            return self._policy_rejection
        for task in tuple(self._action_tasks):
            if not task.done():
                continue
            self._action_tasks.discard(task)
            task.result()
        return Ok(value=None)

    def reset_round_tracking(
        self,
        *,
        base_seed: int,
        policy_version: int,
        episode_id: int,
    ) -> Ok[None] | Rejected:
        """Clear per-round history and action stats."""
        assert base_seed >= 0
        assert policy_version >= 0
        assert episode_id >= 0
        background_result = self.raise_background_errors()
        if isinstance(background_result, Rejected):
            return background_result
        self._history.clear()
        self._pending = None
        self._policy_rejection = None
        self._stats = TrainingPlayerStats()
        self._base_seed = base_seed
        self._policy_version = policy_version
        self._episode_id = episode_id
        self._decision_index = 0
        return Ok(value=None)

    async def cancel_background_tasks(self) -> None:
        """Cancel in-flight actions before discarding a game."""
        tasks = tuple(self._action_tasks)
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._action_tasks.clear()
        self._pending = None

    async def confirm_held_scoring_next_round(
        self, game: GameView
    ) -> bool:
        """Submit the held next-round confirmation after training."""
        message = self._held_scoring_message
        if message is None:
            return False
        self._held_scoring_message = None
        await game.receive(
            self.index,
            PlayerMessage(seq=message.seq, raw={"type": "next_round"}),
        )
        return True

    def _settle_pending(self, message: StateMessage) -> bool:
        pending = self._pending
        if pending is None:
            return False
        if message.error is not None and message.seq == pending.seq:
            self._pending = None
            raise AssertionError(
                "training legal action was rejected: "
                f"player={self.index}, seq={message.seq}, "
                f"error={message.error}, action={pending.step.action!r}"
            )
        if message.error is None and message.seq != pending.seq:
            self._recorder.append(pending.step)
            self._stats = _add_accepted(self._stats)
            self._pending = None
            return False
        return True

    def _handle_next_round_state(
        self,
        game: GameView,
        message: StateMessage,
    ) -> None:
        if message.state.scoring is not None:
            self._held_scoring_message = message
            return
        self._held_scoring_message = None
        self._submit_raw_action(
            game,
            message.seq,
            {"type": "next_round"},
        )

    async def _submit_model_action(
        self,
        game: GameView,
        message: StateMessage,
    ) -> None:
        observation = build_observation(
            player_index=self.index,
            snapshot=message.state,
            history=self._history.tricks(),
        )
        legal_actions = build_legal_action_index(
            player_index=self.index,
            snapshot=message.state,
            query=observation.action_query,
        )
        decision_result = await self._policy.decide(
            observation,
            legal_actions,
            PolicyDecisionKey(
                base_seed=self._base_seed,
                policy_version=self._policy_version,
                episode_id=self._episode_id,
                player_index=self.index,
                decision_index=self._decision_index,
            ),
        )
        if isinstance(decision_result, Rejected):
            self._policy_rejection = decision_result
            return
        decision = decision_result.value
        self._decision_index += 1
        step = DecisionStep(
            player_index=self.index,
            seq=message.seq,
            action=decision.action,
            decision_handle=decision.decision_handle,
            choice_count=decision.choice_count,
        )
        self._pending = PendingDecision(seq=message.seq, step=step)
        self._stats = _add_generated(
            self._stats,
            choice_count=decision.choice_count,
        )
        self._submit_generated_action(
            game,
            message.seq,
            decision.action,
            message.state.player_hand,
        )

    def _submit_generated_action(
        self,
        game: GameView,
        seq: int,
        action: GeneratedAction,
        hand_cards: list[Card],
    ) -> None:
        bound = bind_generated_action(action, hand_cards)
        assert isinstance(bound, Ok)
        self._submit_raw_action(game, seq, bound.value.raw)

    def _submit_raw_action(
        self,
        game: GameView,
        seq: int,
        raw: dict[str, object],
    ) -> None:
        task = asyncio.create_task(
            game.receive(self.index, PlayerMessage(seq=seq, raw=raw))
        )
        self._action_tasks.add(task)


def _add_generated(
    stats: TrainingPlayerStats,
    *,
    choice_count: int,
) -> TrainingPlayerStats:
    return TrainingPlayerStats(
        generated_action_count=stats.generated_action_count + 1,
        accepted_action_count=stats.accepted_action_count,
        action_choice_count=stats.action_choice_count + choice_count,
    )


def _add_accepted(stats: TrainingPlayerStats) -> TrainingPlayerStats:
    return TrainingPlayerStats(
        generated_action_count=stats.generated_action_count,
        accepted_action_count=stats.accepted_action_count + 1,
        action_choice_count=stats.action_choice_count,
    )
