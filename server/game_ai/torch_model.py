"""Current-checkpoint PyTorch implementation of model evaluation."""

from __future__ import annotations

import math
import threading
from pathlib import Path

import torch

from server.foundation.result import Ok, Rejected
from server.training.model import ModelConfig, TractorPolicyModel
from server.training.observation import Observation
from server.training.policy_inference_batch import (
    PolicyRequestCompiler,
    PolicyRequestInput,
    PolicyRequestRoute,
    materialize_borrowed_policy_request_batch,
)
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_action_plan import (
    ActionSampler,
    action_plan_generation_step_count,
    action_trace_choice_ids,
    action_trace_from_choice_ids,
    compile_legal_action_frame,
    plan_batch_to_device,
)
from server.training.tensorize import tensorize_observations
from server.training.torch_checkpoints.load import (
    load_inference_checkpoint,
)
from server.training.torch_sampler import sample_policy_batch

from .model import (
    ActionEvaluation,
    ActionScoreRequest,
    ModelQuery,
)


class TorchModelEvaluator:
    """Own one application-scoped read-only policy/value model."""

    def __init__(
        self,
        *,
        model: TractorPolicyModel,
        model_config: ModelConfig,
        device: torch.device,
    ) -> None:
        self._model = model
        self._model_config = model_config
        self._device = device
        self._lock = threading.Lock()

    @classmethod
    def load_for_device_name(
        cls,
        *,
        checkpoint_path: Path,
        device_name: str,
    ) -> Ok[TorchModelEvaluator] | Rejected:
        """Resolve a configured device and load the current model."""
        device = _resolve_device(device_name)
        if isinstance(device, Rejected):
            return device
        return cls.load(
            checkpoint_path=checkpoint_path,
            device=device.value,
        )

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: Path,
        device: torch.device,
    ) -> Ok[TorchModelEvaluator] | Rejected:
        """Load exactly the model declared by the current checkpoint."""
        loaded = load_inference_checkpoint(
            path=checkpoint_path,
            device=device,
        )
        if isinstance(loaded, Rejected):
            return loaded
        return Ok(
            cls(
                model=loaded.value.model,
                model_config=loaded.value.model_config,
                device=device,
            )
        )

    def sample(
        self,
        *,
        queries: tuple[ModelQuery, ...],
        decision_seed: int,
    ) -> Ok[tuple[ActionEvaluation, ...]] | Rejected:
        """Sample one legal action for each heterogeneous query."""
        assert queries
        assert decision_seed >= 0
        requests = tuple(
            PolicyRequestInput(
                route=PolicyRequestRoute(
                    worker_index=0,
                    request_id=index,
                ),
                observation=query.observation,
                legal_actions=query.legal_actions,
                decision_key=PolicyDecisionKey(
                    base_seed=decision_seed,
                    policy_version=0,
                    rollout_id="live-inference",
                    episode_id=0,
                    seat=query.seat,
                    decision_index=index,
                ),
            )
            for index, query in enumerate(queries)
        )
        compiler = PolicyRequestCompiler(batch_capacity=len(requests))
        compiled = compiler.compile_batch(requests)
        if isinstance(compiled, Rejected):
            return compiled
        materialized = materialize_borrowed_policy_request_batch(
            batch=compiled.value,
            device=self._device,
        )
        if isinstance(materialized, Rejected):
            return materialized
        sampler = ActionSampler.create(
            batch_capacity=len(requests),
            device=self._device,
        )
        with self._lock, torch.inference_mode():
            sampled = sample_policy_batch(
                model=self._model,
                config=self._model_config,
                device=self._device,
                requests=materialized.value,
                sampler=sampler,
            )
        if isinstance(sampled, Rejected):
            return sampled
        rows = sampled.value
        step_counts = _int_tuple(rows.step_counts)
        choice_ids = rows.choice_ids_padded.detach().cpu()
        log_probabilities = _float_tuple(rows.old_log_probabilities)
        values = _float_tuple(rows.old_values)
        evaluations: list[ActionEvaluation] = []
        for row_index, step_count in enumerate(step_counts):
            trace_ids = tuple(
                int(choice_ids[row_index, step].item())
                for step in range(step_count)
            )
            trace = action_trace_from_choice_ids(trace_ids)
            if isinstance(trace, Rejected):
                return trace
            decoded = queries[row_index].legal_actions.decode(
                trace.value
            )
            if isinstance(decoded, Rejected):
                return decoded
            evaluations.append(
                ActionEvaluation(
                    action=decoded.value,
                    log_probability=log_probabilities[row_index],
                    value=values[row_index],
                )
            )
        return Ok(tuple(evaluations))

    def score(
        self,
        *,
        requests: tuple[ActionScoreRequest, ...],
    ) -> Ok[tuple[ActionEvaluation, ...]] | Rejected:
        """Score actions with the policy's exact legal masks."""
        assert requests
        plans = tuple(
            compile_legal_action_frame(request.query.legal_actions)
            for request in requests
        )
        generation_steps = tuple(
            action_plan_generation_step_count(plan) for plan in plans
        )
        padded_steps = max(generation_steps)
        trace_ids = tuple(
            action_trace_choice_ids(request.action.trace)
            for request in requests
        )
        for request, ids, capacity in zip(
            requests,
            trace_ids,
            generation_steps,
            strict=True,
        ):
            decoded = request.query.legal_actions.decode(
                request.action.trace
            )
            if isinstance(decoded, Rejected):
                return decoded
            if decoded.value != request.action:
                return Rejected(
                    reason=(
                        "model action does not match legal action index"
                    )
                )
            if len(ids) > capacity:
                return Rejected(reason="model action trace is too long")
        padded_ids = tuple(
            ids + (0,) * (padded_steps - len(ids)) for ids in trace_ids
        )
        observation_batch = tensorize_observations(
            observations=tuple(
                request.query.observation for request in requests
            ),
            device=self._device,
        )
        action_batch = plan_batch_to_device(
            plans,
            device=self._device,
        )
        selected = torch.tensor(
            padded_ids,
            dtype=torch.long,
            device=self._device,
        )
        step_counts = torch.tensor(
            tuple(len(ids) for ids in trace_ids),
            dtype=torch.long,
            device=self._device,
        )
        sampler = ActionSampler.create(
            batch_capacity=len(requests),
            device=self._device,
        )
        with self._lock, torch.inference_mode():
            encoding = self._model.encode_observations(
                observation_batch
            )
            values = self._model.value_estimates(encoding)
            decoder = self._model.begin_action_decode_session(
                encoding,
                max_steps=padded_steps,
            )
            scored = sampler.score(
                action_batch=action_batch,
                selected_choice_ids=selected,
                step_counts=step_counts,
                padded_generation_steps=padded_steps,
                logit_decoder=decoder,
            )
        if isinstance(scored, Rejected):
            return scored
        probabilities = _float_tuple(scored.value.log_probabilities)
        value_rows = _float_tuple(values)
        if not all(
            math.isfinite(value)
            for value in (*probabilities, *value_rows)
        ):
            return Rejected(reason="model evaluation must be finite")
        return Ok(
            tuple(
                ActionEvaluation(
                    action=request.action,
                    log_probability=probabilities[index],
                    value=value_rows[index],
                )
                for index, request in enumerate(requests)
            )
        )

    def values(
        self,
        *,
        observations: tuple[Observation, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        """Estimate a heterogeneous observation batch."""
        assert observations
        batch = tensorize_observations(
            observations=observations,
            device=self._device,
        )
        with self._lock, torch.inference_mode():
            encoding = self._model.encode_observations(batch)
            values = self._model.value_estimates(encoding)
        result = _float_tuple(values)
        if not all(math.isfinite(value) for value in result):
            return Rejected(
                reason="model value estimate must be finite"
            )
        return Ok(result)


def _int_tuple(values: torch.Tensor) -> tuple[int, ...]:
    assert values.ndim == 1
    return tuple(
        int(values[index].detach().cpu().item())
        for index in range(int(values.shape[0]))
    )


def _float_tuple(values: torch.Tensor) -> tuple[float, ...]:
    assert values.ndim == 1
    return tuple(
        float(values[index].detach().cpu().item())
        for index in range(int(values.shape[0]))
    )


def _resolve_device(name: str) -> Ok[torch.device] | Rejected:
    if name == "cpu":
        return Ok(torch.device("cpu"))
    if name == "cuda":
        if not torch.cuda.is_available():
            return Rejected(reason="AI CUDA device is unavailable")
        return Ok(torch.device("cuda"))
    if name == "mps":
        if not torch.backends.mps.is_available():
            return Rejected(reason="AI MPS device is unavailable")
        return Ok(torch.device("mps"))
    return Rejected(reason="AI device name is invalid")


__all__ = ("TorchModelEvaluator",)
