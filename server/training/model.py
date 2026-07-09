"""Torch Transformer policy/value model for Tractor self-play."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import (
    OBSERVATION_COMPONENT_COUNT,
    ObservationTensorBatch,
    observation_component_tensors,
)
from server.training.vocab_schema import VOCAB_SCHEMA


@dataclass(frozen=True, slots=True)
class ArgumentTraceScores:
    """Per-step argument logits for one teacher-forced trace batch."""

    argument_logits: Tensor


@dataclass(frozen=True, slots=True)
class ObservationEncoding:
    """Reusable encoded observation memory for policy/value heads."""

    memory: Tensor
    memory_padding_mask: Tensor
    observation_context: Tensor


@dataclass(slots=True)
class ArgumentDecodeSession:
    """Incremental semantic argument decoder for live inference."""

    _model: TractorPolicyModel
    _encoding: ObservationEncoding
    _cache: ArgumentDecodeCache
    _step_index: int = 0

    def next_logits(self) -> Tensor:
        """Return logits for the current live decoding prefix."""
        prefix_context = self._model.decode_live_argument_step(
            cache=self._cache,
        )
        return self._model.argument_logits_from_prefix_context(
            encoding=self._encoding,
            prefix_context=prefix_context,
        )

    def advance(self, selected_token_ids: Tensor) -> None:
        """Append sampled argument tokens to the live prefix cache."""
        next_index = self._step_index + 1
        assert next_index < self._cache.max_steps
        assert selected_token_ids.shape == (self._cache.batch_size,)
        positions = torch.full(
            selected_token_ids.shape,
            next_index,
            dtype=torch.long,
            device=selected_token_ids.device,
        )
        token_embeddings = self._model.embed_argument_tokens(
            token_ids=selected_token_ids,
            positions=positions,
        )
        self._model.append_live_argument_token(
            cache=self._cache,
            token_embeddings=token_embeddings,
        )
        self._step_index = next_index


@dataclass(slots=True)
class ArgumentDecodeCache:
    """Projected K/V cache for one live semantic argument batch."""

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
        assert self.current_embedding.shape == (
            self.batch_size,
            int(self.self_keys.shape[1] * self.self_keys.shape[3]),
        )
        assert self.current_query.shape == (
            self.batch_size,
            int(self.self_keys.shape[1]),
            int(self.self_keys.shape[3]),
        )
        assert self.current_length > 0
        assert self.max_steps >= self.current_length

    @property
    def batch_size(self) -> int:
        """Return cached batch row count."""
        return int(self.self_keys.shape[0])


class TractorPolicyModel(nn.Module):
    """Shared observation encoder with semantic argument decoder."""

    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self._token_type_embedding = _embedding(
            VOCAB_SCHEMA.token_type_vocab_size, d_model
        )
        self._segment_embedding = _embedding(
            VOCAB_SCHEMA.segment_vocab_size, d_model
        )
        self._field_embedding = _embedding(
            VOCAB_SCHEMA.field_vocab_size, d_model
        )
        self._value_embedding = _embedding(
            VOCAB_SCHEMA.value_vocab_size, d_model
        )
        self._suit_embedding = _embedding(
            VOCAB_SCHEMA.suit_vocab_size, d_model
        )
        self._rank_embedding = _embedding(
            VOCAB_SCHEMA.rank_vocab_size, d_model
        )
        self._points_embedding = _embedding(
            VOCAB_SCHEMA.points_vocab_size, d_model
        )
        self._color_embedding = _embedding(
            VOCAB_SCHEMA.color_vocab_size, d_model
        )
        self._role_embedding = _embedding(
            VOCAB_SCHEMA.role_vocab_size, d_model
        )
        self._trick_age_embedding = _embedding(
            VOCAB_SCHEMA.trick_age_vocab_size, d_model
        )
        self._trick_state_embedding = _embedding(
            VOCAB_SCHEMA.trick_state_vocab_size, d_model
        )
        self._play_order_embedding = _embedding(
            VOCAB_SCHEMA.play_order_vocab_size, d_model
        )
        self._count_embedding = _embedding(
            VOCAB_SCHEMA.count_vocab_size, d_model
        )
        self._play_width_embedding = _embedding(
            VOCAB_SCHEMA.play_width_vocab_size, d_model
        )
        self._event_age_embedding = _embedding(
            VOCAB_SCHEMA.event_age_vocab_size, d_model
        )
        self._categorical_projection = nn.Linear(
            OBSERVATION_COMPONENT_COUNT * d_model,
            d_model,
            bias=False,
        )
        self._numeric_projection = nn.Linear(
            NUMERIC_FEATURE_COUNT * 2,
            d_model,
            bias=False,
        )
        observation_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self._observation_encoder = nn.TransformerEncoder(
            observation_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self._argument_embedding = _embedding(
            SEMANTIC_CODEC.argument_vocab_size, d_model
        )
        self._argument_position_embedding = nn.Embedding(
            SEMANTIC_CODEC.max_argument_tokens, d_model
        )
        self._argument_decoder = _ArgumentDecoderLayer(
            d_model=d_model, heads=heads
        )
        self._decision_projection = nn.Linear(d_model * 2, d_model)
        self._argument_head = nn.Linear(
            d_model, SEMANTIC_CODEC.argument_vocab_size
        )
        self._value_head = nn.Linear(d_model, 1)

    def encode_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> ObservationEncoding:
        """Encode observations once for value and argument decoding."""
        components = observation_component_tensors(observation)
        memory_padding_mask = components.token_type_ids.eq(
            VOCAB_SCHEMA.obs_pad_id
        )
        memory = self._encode_observation(
            observation,
            memory_padding_mask=memory_padding_mask,
        )
        query_mask = components.segment_ids.eq(
            VOCAB_SCHEMA.segment_action_query_id
        ) & (~memory_padding_mask)
        observation_context = _query_or_all_mean(
            memory,
            padding_mask=memory_padding_mask,
            query_mask=query_mask,
        )
        return ObservationEncoding(
            memory=memory,
            memory_padding_mask=memory_padding_mask,
            observation_context=observation_context,
        )

    def select_observation_encoding(
        self,
        encoding: ObservationEncoding,
        *,
        active_indices: Tensor,
    ) -> ObservationEncoding:
        """Select rows from a reusable observation encoding."""
        assert active_indices.ndim == 1
        assert int(active_indices.shape[0]) > 0
        index = active_indices.to(
            dtype=torch.long, device=encoding.memory.device
        )
        return ObservationEncoding(
            memory=encoding.memory.index_select(0, index),
            memory_padding_mask=encoding.memory_padding_mask.index_select(
                0, index
            ),
            observation_context=encoding.observation_context.index_select(
                0, index
            ),
        )

    def value_estimates(self, encoding: ObservationEncoding) -> Tensor:
        """Return value estimates from an encoded observation batch."""
        return self._value_head(encoding.observation_context).squeeze(
            -1
        )

    def begin_argument_decode_session(
        self,
        encoding: ObservationEncoding,
        *,
        max_steps: int,
    ) -> ArgumentDecodeSession:
        """Create an incremental decoder session for live sampling."""
        assert max_steps > 0
        assert max_steps <= SEMANTIC_CODEC.max_argument_tokens
        batch_size = int(encoding.memory.shape[0])
        bos_ids = torch.full(
            (batch_size,),
            SEMANTIC_CODEC.argument_bos_id,
            dtype=torch.long,
            device=encoding.memory.device,
        )
        positions = torch.zeros(
            (batch_size,),
            dtype=torch.long,
            device=encoding.memory.device,
        )
        bos_embeddings = self.embed_argument_tokens(
            token_ids=bos_ids, positions=positions
        )
        cache = self._argument_decoder.begin_decode_cache(
            first_token_embeddings=bos_embeddings,
            memory=encoding.memory,
            memory_padding_mask=encoding.memory_padding_mask,
            max_steps=max_steps,
        )
        return ArgumentDecodeSession(
            _model=self,
            _encoding=encoding,
            _cache=cache,
        )

    def decode_live_argument_step(
        self,
        *,
        cache: ArgumentDecodeCache,
    ) -> Tensor:
        """Decode the current live token using cached projected K/V."""
        return self._argument_decoder.decode_cached_step(cache=cache)

    def append_live_argument_token(
        self, *, cache: ArgumentDecodeCache, token_embeddings: Tensor
    ) -> None:
        """Append one selected token embedding to a live cache."""
        self._argument_decoder.append_cached_token(
            cache=cache,
            token_embeddings=token_embeddings,
        )

    def score_argument_traces(
        self,
        encoding: ObservationEncoding,
        *,
        selected_token_ids_padded: Tensor,
        step_counts: Tensor,
    ) -> ArgumentTraceScores:
        """Return per-step logits for recorded traces in one pass."""
        assert selected_token_ids_padded.ndim == 2
        assert step_counts.shape == (
            int(selected_token_ids_padded.shape[0]),
        )
        prefix_embeddings = self._embed_teacher_forced_trace_prefixes(
            selected_token_ids_padded=selected_token_ids_padded,
        )
        max_step_count = int(selected_token_ids_padded.shape[1])
        positions = torch.arange(
            max_step_count,
            dtype=torch.long,
            device=selected_token_ids_padded.device,
        )
        prefix_padding_mask = positions.unsqueeze(0) >= (
            step_counts.unsqueeze(1)
        )
        causal_mask = torch.triu(
            torch.ones(
                (max_step_count, max_step_count),
                dtype=torch.bool,
                device=selected_token_ids_padded.device,
            ),
            diagonal=1,
        )
        decoded = self._argument_decoder.forward_prefix(
            prefix_embeddings=prefix_embeddings,
            memory=encoding.memory,
            prefix_padding_mask=prefix_padding_mask,
            memory_padding_mask=encoding.memory_padding_mask,
            self_attention_mask=causal_mask,
        )
        observation_context = encoding.observation_context.unsqueeze(
            1
        ).expand(-1, max_step_count, -1)
        decision_context = torch.tanh(
            self._decision_projection(
                torch.cat((observation_context, decoded), dim=-1)
            )
        )
        return ArgumentTraceScores(
            argument_logits=self._argument_head(decision_context)
        )

    def _encode_observation(
        self,
        observation: ObservationTensorBatch,
        *,
        memory_padding_mask: Tensor,
    ) -> Tensor:
        obs_embedded = self._embed_observation(observation)
        return self._observation_encoder(
            obs_embedded,
            src_key_padding_mask=memory_padding_mask,
        )

    def _embed_observation(
        self,
        observation: ObservationTensorBatch,
    ) -> Tensor:
        components = observation_component_tensors(observation)
        categorical_input = torch.cat(
            (
                self._token_type_embedding(components.token_type_ids),
                self._segment_embedding(components.segment_ids),
                self._field_embedding(components.field_ids),
                self._value_embedding(components.value_ids),
                self._suit_embedding(components.suit_ids),
                self._rank_embedding(components.rank_ids),
                self._points_embedding(components.points_ids),
                self._color_embedding(components.color_ids),
                self._role_embedding(components.role_ids),
                self._trick_age_embedding(components.trick_age_ids),
                self._trick_state_embedding(components.trick_state_ids),
                self._play_order_embedding(components.play_order_ids),
                self._count_embedding(components.count_ids),
                self._play_width_embedding(components.play_width_ids),
                self._event_age_embedding(components.event_age_ids),
            ),
            dim=-1,
        )
        numeric_values = observation.numeric_values * (
            observation.numeric_masks
        )
        numeric_input = torch.cat(
            (numeric_values, observation.numeric_masks),
            dim=-1,
        )
        return self._categorical_projection(
            categorical_input
        ) + self._numeric_projection(numeric_input)

    def argument_logits_from_prefix_context(
        self,
        *,
        encoding: ObservationEncoding,
        prefix_context: Tensor,
    ) -> Tensor:
        decision_context = torch.tanh(
            self._decision_projection(
                torch.cat(
                    (encoding.observation_context, prefix_context),
                    dim=-1,
                )
            )
        )
        return self._argument_head(decision_context)

    def _embed_teacher_forced_trace_prefixes(
        self, *, selected_token_ids_padded: Tensor
    ) -> Tensor:
        assert selected_token_ids_padded.ndim == 2
        max_step_count = int(selected_token_ids_padded.shape[1])
        assert max_step_count > 0
        prefix_ids = torch.empty_like(selected_token_ids_padded)
        prefix_ids[:, 0].fill_(SEMANTIC_CODEC.argument_bos_id)
        if max_step_count > 1:
            prefix_ids[:, 1:].copy_(
                selected_token_ids_padded[:, : max_step_count - 1]
            )
        positions = torch.arange(
            max_step_count,
            dtype=torch.long,
            device=selected_token_ids_padded.device,
        ).unsqueeze(0)
        return self._argument_embedding(
            prefix_ids
        ) + self._argument_position_embedding(positions)

    def embed_argument_tokens(
        self, *, token_ids: Tensor, positions: Tensor
    ) -> Tensor:
        assert token_ids.shape == positions.shape
        assert token_ids.ndim == 1
        return self._argument_embedding(
            token_ids
        ) + self._argument_position_embedding(positions)


class _ArgumentDecoderLayer(nn.Module):
    """Single-layer decoder with full-prefix and live-last paths."""

    def __init__(self, *, d_model: int, heads: int) -> None:
        super().__init__()
        self._self_attn = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        self._cross_attn = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        self._linear1 = nn.Linear(d_model, d_model * 4)
        self._linear2 = nn.Linear(d_model * 4, d_model)
        self._norm1 = nn.LayerNorm(d_model)
        self._norm2 = nn.LayerNorm(d_model)
        self._norm3 = nn.LayerNorm(d_model)
        self._dropout = nn.Dropout(0.0)

    def forward_prefix(
        self,
        *,
        prefix_embeddings: Tensor,
        memory: Tensor,
        prefix_padding_mask: Tensor,
        memory_padding_mask: Tensor,
        self_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Decode every prefix position for PPO replay evaluation."""
        self_attended = self._self_attention(
            query=prefix_embeddings,
            key_value=prefix_embeddings,
            key_padding_mask=prefix_padding_mask,
            attention_mask=self_attention_mask,
        )
        target = self._norm1(prefix_embeddings + self_attended)
        cross_attended = self._cross_attention(
            query=target,
            memory=memory,
            memory_padding_mask=memory_padding_mask,
        )
        target = self._norm2(target + cross_attended)
        return self._norm3(target + self._feed_forward(target))

    def begin_decode_cache(
        self,
        *,
        first_token_embeddings: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
        max_steps: int,
    ) -> ArgumentDecodeCache:
        """Create a projected K/V cache seeded with BOS embeddings."""
        assert first_token_embeddings.ndim == 2
        assert memory.ndim == 3
        assert max_steps > 0
        batch_size = int(first_token_embeddings.shape[0])
        assert batch_size > 0
        self_key, self_value = self._project_self_key_value(
            first_token_embeddings.unsqueeze(1)
        )
        memory_key, memory_value = self._project_cross_key_value(memory)
        heads = int(self_key.shape[1])
        head_dim = int(self_key.shape[3])
        key_cache = torch.empty(
            (batch_size, heads, max_steps, head_dim),
            dtype=self_key.dtype,
            device=self_key.device,
        )
        value_cache = torch.empty_like(key_cache)
        key_cache[:, :, 0:1, :].copy_(self_key)
        value_cache[:, :, 0:1, :].copy_(self_value)
        return ArgumentDecodeCache(
            self_keys=key_cache,
            self_values=value_cache,
            memory_keys=memory_key,
            memory_values=memory_value,
            memory_padding_mask=memory_padding_mask,
            current_embedding=first_token_embeddings,
            current_query=self._project_self_query(
                first_token_embeddings
            ),
            current_length=1,
            max_steps=max_steps,
        )

    def append_cached_token(
        self, *, cache: ArgumentDecodeCache, token_embeddings: Tensor
    ) -> None:
        """Append one projected token to a live K/V cache."""
        assert token_embeddings.ndim == 2
        assert token_embeddings.shape[0] == cache.batch_size
        assert cache.current_length < cache.max_steps
        key, value = self._project_self_key_value(
            token_embeddings.unsqueeze(1)
        )
        cache.self_keys[
            :, :, cache.current_length : cache.current_length + 1, :
        ].copy_(key)
        cache.self_values[
            :, :, cache.current_length : cache.current_length + 1, :
        ].copy_(value)
        cache.current_embedding = token_embeddings
        cache.current_query = self._project_self_query(token_embeddings)
        cache.current_length += 1

    def decode_cached_step(
        self, *, cache: ArgumentDecodeCache
    ) -> Tensor:
        """Decode the current live prefix token from cached K/V."""
        query = cache.current_query
        self_attended = self._scaled_attention(
            query=query,
            key=cache.self_keys[:, :, : cache.current_length, :],
            value=cache.self_values[:, :, : cache.current_length, :],
            key_padding_mask=None,
            out_projection=self._self_attn.out_proj,
        )
        target = self._norm1(cache.current_embedding + self_attended)
        cross_attended = self._scaled_attention(
            query=self._project_cross_query(target),
            key=cache.memory_keys,
            value=cache.memory_values,
            key_padding_mask=cache.memory_padding_mask,
            out_projection=self._cross_attn.out_proj,
        )
        target = self._norm2(target + cross_attended)
        return self._norm3(target + self._feed_forward(target))

    def _self_attention(
        self,
        *,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        output, _weights = self._self_attn(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        return output

    def _cross_attention(
        self,
        *,
        query: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
    ) -> Tensor:
        output, _weights = self._cross_attn(
            query,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        return output

    def _feed_forward(self, target: Tensor) -> Tensor:
        hidden = torch.nn.functional.gelu(self._linear1(target))
        return self._linear2(self._dropout(hidden))

    def _project_self_query(self, embeddings: Tensor) -> Tensor:
        assert embeddings.ndim == 2
        weight = self._self_attn.in_proj_weight
        bias = self._self_attn.in_proj_bias
        embed_dim = self._self_attn.embed_dim
        projected = F.linear(
            embeddings,
            weight[:embed_dim],
            bias[:embed_dim],
        )
        return _split_heads(
            projected.unsqueeze(1), heads=self._self_attn.num_heads
        ).squeeze(2)

    def _project_cross_query(self, embeddings: Tensor) -> Tensor:
        assert embeddings.ndim == 2
        weight = self._cross_attn.in_proj_weight
        bias = self._cross_attn.in_proj_bias
        embed_dim = self._cross_attn.embed_dim
        projected = F.linear(
            embeddings,
            weight[:embed_dim],
            bias[:embed_dim],
        )
        return _split_heads(
            projected.unsqueeze(1), heads=self._cross_attn.num_heads
        ).squeeze(2)

    def _project_self_key_value(
        self, embeddings: Tensor
    ) -> tuple[Tensor, Tensor]:
        assert embeddings.ndim == 3
        weight = self._self_attn.in_proj_weight
        bias = self._self_attn.in_proj_bias
        embed_dim = self._self_attn.embed_dim
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
            _split_heads(key, heads=self._self_attn.num_heads),
            _split_heads(value, heads=self._self_attn.num_heads),
        )

    def _project_cross_key_value(
        self, memory: Tensor
    ) -> tuple[Tensor, Tensor]:
        assert memory.ndim == 3
        weight = self._cross_attn.in_proj_weight
        bias = self._cross_attn.in_proj_bias
        embed_dim = self._cross_attn.embed_dim
        key = F.linear(
            memory,
            weight[embed_dim : embed_dim * 2],
            bias[embed_dim : embed_dim * 2],
        )
        value = F.linear(
            memory,
            weight[embed_dim * 2 :],
            bias[embed_dim * 2 :],
        )
        return (
            _split_heads(key, heads=self._cross_attn.num_heads),
            _split_heads(value, heads=self._cross_attn.num_heads),
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
        assert query.ndim == 3
        scores = torch.matmul(
            query.unsqueeze(2), key.transpose(-2, -1)
        ).squeeze(2)
        scores = scores / math.sqrt(float(query.shape[-1]))
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1), -torch.inf
            )
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.matmul(
            probabilities.unsqueeze(2), value
        ).squeeze(2)
        return out_projection(_merge_heads(context))


def _embedding(vocab_size: int, d_model: int) -> nn.Embedding:
    return nn.Embedding(
        vocab_size, d_model, padding_idx=VOCAB_SCHEMA.obs_pad_id
    )


def _split_heads(values: Tensor, *, heads: int) -> Tensor:
    assert values.ndim == 3
    batch_size = int(values.shape[0])
    sequence_length = int(values.shape[1])
    embed_dim = int(values.shape[2])
    assert embed_dim % heads == 0
    head_dim = embed_dim // heads
    return values.view(
        batch_size, sequence_length, heads, head_dim
    ).transpose(1, 2)


def _merge_heads(values: Tensor) -> Tensor:
    assert values.ndim == 3
    batch_size = int(values.shape[0])
    heads = int(values.shape[1])
    head_dim = int(values.shape[2])
    return values.reshape(batch_size, heads * head_dim)


def _masked_mean(values: Tensor, padding_mask: Tensor) -> Tensor:
    keep_mask = (~padding_mask).unsqueeze(-1).to(dtype=values.dtype)
    summed = (values * keep_mask).sum(dim=-2)
    counts = keep_mask.sum(dim=-2).clamp_min(1.0)
    return summed / counts


def _query_or_all_mean(
    values: Tensor,
    *,
    padding_mask: Tensor,
    query_mask: Tensor,
) -> Tensor:
    query_keep = query_mask.unsqueeze(-1).to(dtype=values.dtype)
    query_summed = (values * query_keep).sum(dim=-2)
    query_counts = query_keep.sum(dim=-2)
    query_mean = query_summed / query_counts.clamp_min(1.0)
    all_mean = _masked_mean(values, padding_mask)
    return torch.where(query_counts.gt(0), query_mean, all_mean)
