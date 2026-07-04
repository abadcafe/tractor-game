"""Tests for training trajectory data invariants."""

from __future__ import annotations

import subprocess
import sys

import torch

from server.training.choice_trace import (
    SemanticChoiceTrace,
    semantic_choice_step_from_argument,
)
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.values import GeneratedAction
from server.training.tensorize import ObservationTensorBatch
from server.training.trajectory import (
    DecisionStep,
    DecisionTransition,
    TeamTrajectory,
)


def test_team_trajectory_accepts_same_team_players() -> None:
    trajectory = TeamTrajectory(
        team_index=0,
        transitions=(
            _transition(player_index=0),
            _transition(player_index=2),
        ),
        terminal_reward=1.0,
    )

    assert trajectory.team_index == 0


def test_team_trajectory_rejects_cross_team_player() -> None:
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch\n"
                "from server.training.choice_trace import (\n"
                "    SemanticChoiceTrace,\n"
                "    semantic_choice_step_from_argument,\n"
                ")\n"
                "from server.training.semantic_actions import (\n"
                "    SemanticArgument,\n"
                "    SemanticArgumentTrace,\n"
                ")\n"
                "from server.training.semantic_actions.values import "
                "GeneratedAction\n"
                "from server.training.tensorize import "
                "ObservationTensorBatch\n"
                "from server.training.trajectory import (\n"
                "    DecisionStep,\n"
                "    DecisionTransition,\n"
                "    TeamTrajectory,\n"
                ")\n"
                "def transition(player_index):\n"
                "    argument = SemanticArgument('pass')\n"
                "    trace = SemanticArgumentTrace(\n"
                "        arguments=(argument,)\n"
                "    )\n"
                "    decision = DecisionStep(\n"
                "        player_index=player_index,\n"
                "        seq=1,\n"
                "        observation_batch=ObservationTensorBatch(\n"
                "            component_ids=torch.zeros(\n"
                "                (1, 1, 1), dtype=torch.long\n"
                "            ),\n"
                "            numeric_values=torch.zeros(\n"
                "                (1, 1, 1), dtype=torch.float32\n"
                "            ),\n"
                "            numeric_masks=torch.zeros(\n"
                "                (1, 1, 1), dtype=torch.float32\n"
                "            ),\n"
                "        ),\n"
                "        choice_trace=SemanticChoiceTrace(\n"
                "            steps=(\n"
                "                semantic_choice_step_from_argument(\n"
                "                    allowed=(argument,),\n"
                "                    selected_argument=argument,\n"
                "                ),\n"
                "            )\n"
                "        ),\n"
                "        action=GeneratedAction(\n"
                "            action_kind='pass',\n"
                "            message_type='bid',\n"
                "            face_counts=(),\n"
                "            semantic_trace=trace,\n"
                "            is_pass=True,\n"
                "        ),\n"
                "        log_probability=0.0,\n"
                "        value_estimate=0.0,\n"
                "        entropy=0.0,\n"
                "        choice_count=1,\n"
                "    )\n"
                "    return DecisionTransition(\n"
                "        decision=decision, reward_after_step=0.0\n"
                "    )\n"
                "TeamTrajectory(\n"
                "    team_index=0,\n"
                "    transitions=(\n"
                "        transition(0),\n"
                "        transition(1),\n"
                "    ),\n"
                "    terminal_reward=1.0,\n"
                ")\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "AssertionError" in completed.stderr


def _transition(*, player_index: int) -> DecisionTransition:
    return DecisionTransition(
        decision=_decision_step(player_index=player_index),
        reward_after_step=0.0,
    )


def _decision_step(*, player_index: int) -> DecisionStep:
    argument = SemanticArgument("pass")
    trace = SemanticArgumentTrace(arguments=(argument,))
    return DecisionStep(
        player_index=player_index,
        seq=1,
        observation_batch=ObservationTensorBatch(
            component_ids=torch.zeros((1, 1, 1), dtype=torch.long),
            numeric_values=torch.zeros((1, 1, 1), dtype=torch.float32),
            numeric_masks=torch.zeros((1, 1, 1), dtype=torch.float32),
        ),
        choice_trace=SemanticChoiceTrace(
            steps=(
                semantic_choice_step_from_argument(
                    allowed=(argument,),
                    selected_argument=argument,
                ),
            )
        ),
        action=GeneratedAction(
            action_kind="pass",
            message_type="bid",
            face_counts=(),
            semantic_trace=trace,
            is_pass=True,
        ),
        log_probability=0.0,
        value_estimate=0.0,
        entropy=0.0,
        choice_count=1,
    )
