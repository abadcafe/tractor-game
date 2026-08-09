"""Autoregressive action decoding over shared semantic choices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, final

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from server.policy_model._schema.actions import (
    ACTION_CHOICE_COUNT,
    MAX_ACTION_STEPS,
)

from .._module_call import call_attention, call_tensor
from ..observation_backbone import (
    ActionDecoderInputs,
    EncodedObservation,
)


@dataclass(frozen=True, slots=True)
class ActionTraceScores:
    """Fixed-vocabulary logits for every teacher-forced action step."""

    choice_logits: Tensor


class ActionDecodeSession(Protocol):
    """Incremental action-decoding session contract."""

    def next_choice_logits(
        self,
        active_rows: Tensor,
        scored_rows: Tensor,
    ) -> Tensor: ...

    def advance(
        self,
        selected_choice_ids: Tensor,
        active_rows: Tensor,
    ) -> None: ...


@dataclass(slots=True)
class _ActionDecodeSession:
    """Fixed-lane decoder state without host synchronization."""

    _decoder: ActionDecoder
    _cache: _ActionDecodeCache
    _observation_context: Tensor
    _choice_embeddings: Tensor
    _choice_keys: Tensor
    _lane_count: int
    _step_index: int = 0

    def next_choice_logits(
        self,
        active_rows: Tensor,
        scored_rows: Tensor,
    ) -> Tensor:
        """Evaluate all fixed lanes and mask rows that need no score."""
        assert active_rows.shape == (self._lane_count,)
        assert scored_rows.shape == (self._lane_count,)
        assert active_rows.dtype == torch.bool
        assert scored_rows.dtype == torch.bool
        assert active_rows.device == self._choice_keys.device
        assert scored_rows.device == self._choice_keys.device
        prefix_context = self._decoder.decode_live_action_step(
            cache=self._cache
        )
        logits = self._decoder.choice_logits_from_prefix_context(
            observation_context=self._observation_context,
            prefix_context=prefix_context,
            choice_keys=self._choice_keys,
        )
        return torch.where(
            scored_rows.unsqueeze(1),
            logits,
            torch.zeros_like(logits),
        )

    def advance(
        self,
        selected_choice_ids: Tensor,
        active_rows: Tensor,
    ) -> None:
        """Append one selected choice for every fixed lane."""
        assert selected_choice_ids.shape == (self._lane_count,)
        assert active_rows.shape == (self._lane_count,)
        assert selected_choice_ids.dtype == torch.long
        assert active_rows.dtype == torch.bool
        assert selected_choice_ids.device == self._choice_keys.device
        assert active_rows.device == self._choice_keys.device
        next_index = self._step_index + 1
        assert next_index < self._cache.max_steps
        embeddings = self._decoder.embed_selected_choices(
            choice_embeddings=self._choice_embeddings,
            choice_ids=selected_choice_ids,
            position=next_index,
        )
        self._decoder.append_live_action_choice(
            cache=self._cache,
            choice_embeddings=embeddings,
        )
        self._step_index = next_index


@dataclass(slots=True)
class _ActionDecodeCache:
    """Projected key/value cache for one live action batch."""

    self_keys: Tensor
    self_values: Tensor
    memory_keys: Tensor
    memory_values: Tensor
    memory_padding_mask: Tensor
    current_embedding: Tensor
    current_query: Tensor
    current_length: int
    max_steps: int

    def __post_init__(self) -> None:
        assert self.self_keys.ndim == 4
        assert self.self_values.shape == self.self_keys.shape
        assert self.memory_keys.ndim == 4
        assert self.memory_values.shape == self.memory_keys.shape
        assert self.memory_padding_mask.shape == (
            self.batch_size,
            int(self.memory_keys.shape[2]),
        )
        model_width = int(
            self.self_keys.shape[1] * self.self_keys.shape[3]
        )
        assert self.current_embedding.shape == (
            self.batch_size,
            model_width,
        )
        assert self.current_query.shape == (
            self.batch_size,
            int(self.self_keys.shape[1]),
            int(self.self_keys.shape[3]),
        )
        assert 0 < self.current_length <= self.max_steps

    @property
    def batch_size(self) -> int:
        return int(self.self_keys.shape[0])


@final
class ActionDecoder(nn.Module):
    """Own all parameters and state transitions for action decoding."""

    def __init__(self, *, d_model: int, heads: int) -> None:
        super().__init__()
        self._control_choice_embeddings = nn.Parameter(
            torch.empty(2, d_model)
        )
        _ = nn.init.normal_(
            self._control_choice_embeddings,
            mean=0.0,
            std=1.0 / math.sqrt(float(d_model)),
        )
        self._action_position_embedding = nn.Embedding(
            MAX_ACTION_STEPS, d_model
        )
        self._selected_choice_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self._query_seed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self._transformer = _ActionDecoderLayer(
            d_model=d_model,
            heads=heads,
        )
        self._decision_projection = nn.Linear(d_model * 2, d_model)
        self._choice_query = nn.Linear(d_model, d_model, bias=False)
        self._choice_key = nn.Linear(d_model, d_model, bias=False)
        self._choice_logit_bias = nn.Parameter(
            torch.zeros(ACTION_CHOICE_COUNT)
        )

    def score_action_traces(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: Tensor,
        choice_ids_padded: Tensor,
        step_counts: Tensor,
    ) -> ActionTraceScores:
        """Score complete traces with causal teacher forcing."""
        assert choice_ids_padded.ndim == 2
        assert step_counts.shape == (int(choice_ids_padded.shape[0]),)
        assert source_rows.shape == (int(choice_ids_padded.shape[0]),)
        assert source_rows.dtype == torch.long
        assert source_rows.device == encoding.device
        base_inputs = encoding.action_decoder_inputs()
        inputs = ActionDecoderInputs(
            memory=base_inputs.memory.index_select(0, source_rows),
            memory_padding_mask=(
                base_inputs.memory_padding_mask.index_select(
                    0, source_rows
                )
            ),
            observation_context=(
                base_inputs.observation_context.index_select(
                    0, source_rows
                )
            ),
            card_choice_embeddings=(
                base_inputs.card_choice_embeddings.index_select(
                    0, source_rows
                )
            ),
        )
        prefix_embeddings = self._teacher_forced_prefix_embeddings(
            inputs=inputs,
            choice_ids_padded=choice_ids_padded,
        )
        max_steps = int(choice_ids_padded.shape[1])
        positions = torch.arange(
            max_steps,
            dtype=torch.long,
            device=choice_ids_padded.device,
        )
        prefix_padding_mask = positions.unsqueeze(
            0
        ) >= step_counts.unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones(
                (max_steps, max_steps),
                dtype=torch.bool,
                device=choice_ids_padded.device,
            ),
            diagonal=1,
        )
        decoded = self._transformer.forward_prefix(
            prefix_embeddings=prefix_embeddings,
            memory=inputs.memory,
            prefix_padding_mask=prefix_padding_mask,
            memory_padding_mask=inputs.memory_padding_mask,
            self_attention_mask=causal_mask,
        )
        observation_context = inputs.observation_context.unsqueeze(
            1
        ).expand(-1, max_steps, -1)
        decision_context = torch.tanh(
            call_tensor(
                self._decision_projection,
                torch.cat((observation_context, decoded), dim=-1),
            )
        )
        return ActionTraceScores(
            choice_logits=self._score_choices(
                decision_context=decision_context,
                choice_embeddings=self._choice_embeddings(
                    batch_size=encoding.batch_size,
                    inputs=inputs,
                ),
            )
        )

    def begin_decode_session(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: Tensor,
        max_steps: int,
    ) -> ActionDecodeSession:
        """Begin fixed-lane decoding without device-to-host reads."""
        assert 0 < max_steps <= MAX_ACTION_STEPS
        assert source_rows.ndim == 1
        assert source_rows.dtype == torch.long
        assert source_rows.device == encoding.device
        assert int(source_rows.shape[0]) > 0
        inputs = encoding.action_decoder_inputs()
        observation_context = inputs.observation_context.index_select(
            0, source_rows
        )
        seed = call_tensor(self._query_seed, observation_context)
        seed = seed + call_tensor(
            self._action_position_embedding,
            torch.zeros(
                (int(source_rows.shape[0]),),
                dtype=torch.long,
                device=seed.device,
            ),
        )
        cache = self._transformer.begin_decode_cache(
            first_embeddings=seed,
            memory=inputs.memory.index_select(0, source_rows),
            memory_padding_mask=(
                inputs.memory_padding_mask.index_select(0, source_rows)
            ),
            max_steps=max_steps,
        )
        choice_embeddings = self._choice_embeddings(
            batch_size=encoding.batch_size,
            inputs=inputs,
        ).index_select(0, source_rows)
        return _ActionDecodeSession(
            _decoder=self,
            _cache=cache,
            _observation_context=observation_context,
            _choice_embeddings=choice_embeddings,
            _choice_keys=call_tensor(
                self._choice_key, choice_embeddings
            ),
            _lane_count=int(source_rows.shape[0]),
        )

    def decode_live_action_step(
        self, *, cache: _ActionDecodeCache
    ) -> Tensor:
        return self._transformer.decode_cached_step(cache=cache)

    def append_live_action_choice(
        self,
        *,
        cache: _ActionDecodeCache,
        choice_embeddings: Tensor,
    ) -> None:
        self._transformer.append_cached_choice(
            cache=cache,
            choice_embeddings=choice_embeddings,
        )

    def choice_logits_from_prefix_context(
        self,
        *,
        observation_context: Tensor,
        prefix_context: Tensor,
        choice_keys: Tensor,
    ) -> Tensor:
        decision_context = torch.tanh(
            call_tensor(
                self._decision_projection,
                torch.cat(
                    (observation_context, prefix_context),
                    dim=-1,
                ),
            )
        )
        return self._score_projected_choices(
            decision_context=decision_context,
            choice_keys=choice_keys,
        )

    def embed_selected_choices(
        self,
        *,
        choice_embeddings: Tensor,
        choice_ids: Tensor,
        position: int,
    ) -> Tensor:
        assert choice_ids.ndim == 1
        assert 0 < position < MAX_ACTION_STEPS
        batch_indices = torch.arange(
            int(choice_ids.shape[0]),
            dtype=torch.long,
            device=choice_ids.device,
        )
        selected = choice_embeddings[batch_indices, choice_ids]
        positions = torch.full_like(choice_ids, position)
        return call_tensor(
            self._selected_choice_adapter, selected
        ) + call_tensor(self._action_position_embedding, positions)

    def _teacher_forced_prefix_embeddings(
        self,
        *,
        inputs: ActionDecoderInputs,
        choice_ids_padded: Tensor,
    ) -> Tensor:
        max_steps = int(choice_ids_padded.shape[1])
        assert 0 < max_steps <= MAX_ACTION_STEPS
        seed = call_tensor(
            self._query_seed, inputs.observation_context
        ).unsqueeze(1)
        if max_steps == 1:
            prefix = seed
        else:
            batch_size = int(choice_ids_padded.shape[0])
            batch_indices = torch.arange(
                batch_size,
                dtype=torch.long,
                device=choice_ids_padded.device,
            ).unsqueeze(1)
            selected = self._choice_embeddings(
                batch_size=batch_size,
                inputs=inputs,
            )[
                batch_indices,
                choice_ids_padded[:, : max_steps - 1],
            ]
            prefix = torch.cat(
                (
                    seed,
                    call_tensor(
                        self._selected_choice_adapter, selected
                    ),
                ),
                dim=1,
            )
        positions = torch.arange(
            max_steps,
            dtype=torch.long,
            device=choice_ids_padded.device,
        ).unsqueeze(0)
        return prefix + call_tensor(
            self._action_position_embedding, positions
        )

    def _choice_embeddings(
        self,
        *,
        batch_size: int,
        inputs: ActionDecoderInputs,
    ) -> Tensor:
        controls = self._control_choice_embeddings.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        choice_embeddings = torch.cat(
            (controls, inputs.card_choice_embeddings),
            dim=1,
        )
        assert int(choice_embeddings.shape[1]) == ACTION_CHOICE_COUNT
        return choice_embeddings

    def _score_choices(
        self,
        *,
        decision_context: Tensor,
        choice_embeddings: Tensor,
    ) -> Tensor:
        return self._score_projected_choices(
            decision_context=decision_context,
            choice_keys=call_tensor(
                self._choice_key, choice_embeddings
            ),
        )

    def _score_projected_choices(
        self,
        *,
        decision_context: Tensor,
        choice_keys: Tensor,
    ) -> Tensor:
        queries = call_tensor(self._choice_query, decision_context)
        if queries.ndim == 2:
            logits = torch.einsum("bd,bcd->bc", queries, choice_keys)
        else:
            assert queries.ndim == 3
            logits = torch.einsum("bsd,bcd->bsc", queries, choice_keys)
        return logits / math.sqrt(float(choice_keys.shape[-1])) + (
            self._choice_logit_bias
        )


@final
class _ActionDecoderLayer(nn.Module):
    """Single-layer decoder with teacher-forced and cached paths."""

    def __init__(self, *, d_model: int, heads: int) -> None:
        super().__init__()
        self._heads = heads
        self._self_attn = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self._cross_attn = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self._linear1 = nn.Linear(d_model, d_model * 4)
        self._linear2 = nn.Linear(d_model * 4, d_model)
        self._norm1 = nn.LayerNorm(d_model)
        self._norm2 = nn.LayerNorm(d_model)
        self._norm3 = nn.LayerNorm(d_model)

    def forward_prefix(
        self,
        *,
        prefix_embeddings: Tensor,
        memory: Tensor,
        prefix_padding_mask: Tensor,
        memory_padding_mask: Tensor,
        self_attention_mask: Tensor,
    ) -> Tensor:
        self_attention_result = call_attention(
            self._self_attn,
            prefix_embeddings,
            prefix_embeddings,
            prefix_embeddings,
            key_padding_mask=prefix_padding_mask,
            attn_mask=self_attention_mask,
            need_weights=False,
        )
        self_attended, _weights = self_attention_result
        target = call_tensor(
            self._norm1, prefix_embeddings + self_attended
        )
        cross_attention_result = call_attention(
            self._cross_attn,
            target,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        cross_attended, _weights = cross_attention_result
        target = call_tensor(self._norm2, target + cross_attended)
        return call_tensor(
            self._norm3, target + self._feed_forward(target)
        )

    def begin_decode_cache(
        self,
        *,
        first_embeddings: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
        max_steps: int,
    ) -> _ActionDecodeCache:
        assert first_embeddings.ndim == 2
        assert memory.ndim == 3
        assert max_steps > 0
        batch_size = int(first_embeddings.shape[0])
        self_key, self_value = self._project_self_key_value(
            first_embeddings.unsqueeze(1)
        )
        memory_key, memory_value = self._project_cross_key_value(memory)
        keys = torch.empty(
            (
                batch_size,
                int(self_key.shape[1]),
                max_steps,
                int(self_key.shape[3]),
            ),
            dtype=self_key.dtype,
            device=self_key.device,
        )
        values = torch.empty_like(keys)
        _ = keys[:, :, 0:1].copy_(self_key)
        _ = values[:, :, 0:1].copy_(self_value)
        return _ActionDecodeCache(
            self_keys=keys,
            self_values=values,
            memory_keys=memory_key,
            memory_values=memory_value,
            memory_padding_mask=memory_padding_mask,
            current_embedding=first_embeddings,
            current_query=self._project_self_query(first_embeddings),
            current_length=1,
            max_steps=max_steps,
        )

    def append_cached_choice(
        self,
        *,
        cache: _ActionDecodeCache,
        choice_embeddings: Tensor,
    ) -> None:
        assert choice_embeddings.shape == cache.current_embedding.shape
        assert cache.current_length < cache.max_steps
        key, value = self._project_self_key_value(
            choice_embeddings.unsqueeze(1)
        )
        position = cache.current_length
        _ = cache.self_keys[:, :, position : position + 1].copy_(key)
        _ = cache.self_values[:, :, position : position + 1].copy_(
            value
        )
        cache.current_embedding = choice_embeddings
        cache.current_query = self._project_self_query(
            choice_embeddings
        )
        cache.current_length += 1

    def decode_cached_step(
        self, *, cache: _ActionDecodeCache
    ) -> Tensor:
        self_attended = self._scaled_attention(
            query=cache.current_query,
            key=cache.self_keys[:, :, : cache.current_length],
            value=cache.self_values[:, :, : cache.current_length],
            key_padding_mask=None,
            out_projection=self._self_attn.out_proj,
        )
        target = call_tensor(
            self._norm1, cache.current_embedding + self_attended
        )
        cross_attended = self._scaled_attention(
            query=self._project_cross_query(target),
            key=cache.memory_keys,
            value=cache.memory_values,
            key_padding_mask=cache.memory_padding_mask,
            out_projection=self._cross_attn.out_proj,
        )
        target = call_tensor(self._norm2, target + cross_attended)
        return call_tensor(
            self._norm3, target + self._feed_forward(target)
        )

    def _feed_forward(self, target: Tensor) -> Tensor:
        hidden = call_tensor(self._linear1, target)
        return call_tensor(self._linear2, F.gelu(hidden))

    def _project_self_query(self, embeddings: Tensor) -> Tensor:
        return self._project_query(
            embeddings,
            attention=self._self_attn,
        )

    def _project_cross_query(self, embeddings: Tensor) -> Tensor:
        return self._project_query(
            embeddings,
            attention=self._cross_attn,
        )

    def _project_query(
        self,
        embeddings: Tensor,
        *,
        attention: nn.MultiheadAttention,
    ) -> Tensor:
        weight = attention.in_proj_weight
        bias = attention.in_proj_bias
        embed_dim = attention.embed_dim
        projected = F.linear(
            embeddings,
            weight[:embed_dim],
            bias[:embed_dim],
        )
        return _split_heads(
            projected.unsqueeze(1),
            heads=self._heads,
        ).squeeze(2)

    def _project_self_key_value(
        self, embeddings: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self._project_key_value(
            embeddings,
            attention=self._self_attn,
        )

    def _project_cross_key_value(
        self, embeddings: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self._project_key_value(
            embeddings,
            attention=self._cross_attn,
        )

    def _project_key_value(
        self,
        embeddings: Tensor,
        *,
        attention: nn.MultiheadAttention,
    ) -> tuple[Tensor, Tensor]:
        weight = attention.in_proj_weight
        bias = attention.in_proj_bias
        embed_dim = attention.embed_dim
        key = F.linear(
            embeddings,
            weight[embed_dim : embed_dim * 2],
            bias[embed_dim : embed_dim * 2],
        )
        value = F.linear(
            embeddings,
            weight[embed_dim * 2 :],
            bias[embed_dim * 2 :],
        )
        return (
            _split_heads(key, heads=self._heads),
            _split_heads(value, heads=self._heads),
        )

    def _scaled_attention(
        self,
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None,
        out_projection: nn.Linear,
    ) -> Tensor:
        scores = torch.matmul(
            query.unsqueeze(2),
            key.transpose(-2, -1),
        ).squeeze(2) / math.sqrt(float(query.shape[-1]))
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -torch.inf,
            )
        context = torch.matmul(
            torch.softmax(scores, dim=-1).unsqueeze(2),
            value,
        ).squeeze(2)
        return call_tensor(out_projection, _merge_heads(context))


def _split_heads(values: Tensor, *, heads: int) -> Tensor:
    assert values.ndim == 3
    batch_size, sequence_length, embed_dim = values.shape
    assert int(embed_dim) % heads == 0
    return values.view(
        int(batch_size),
        int(sequence_length),
        heads,
        int(embed_dim) // heads,
    ).transpose(1, 2)


def _merge_heads(values: Tensor) -> Tensor:
    assert values.ndim == 3
    return values.reshape(
        int(values.shape[0]),
        int(values.shape[1] * values.shape[2]),
    )


__all__ = ("ActionDecodeSession", "ActionDecoder", "ActionTraceScores")
