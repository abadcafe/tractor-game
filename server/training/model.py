"""Position-independent Transformer policy/value model for Tractor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from server.training.network import (
    StructuredObservationEncoder,
    TypedTokenEncoder,
)
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    CARD_CHOICE_COUNT,
    MAX_ACTION_STEPS,
)
from server.training.tensorize import ObservationTensorBatch


@dataclass(frozen=True, slots=True)
class ActionTraceScores:
    """Fixed-vocabulary logits for every teacher-forced action step."""

    choice_logits: Tensor


@dataclass(frozen=True, slots=True)
class ObservationEncoding:
    """Reusable observation memory and observation-specific choices."""

    memory: Tensor
    memory_padding_mask: Tensor
    observation_context: Tensor
    choice_embeddings: Tensor


@dataclass(slots=True)
class ActionDecodeSession:
    """Incremental decoder seeded by the decision query itself."""

    _model: TractorPolicyModel
    _encoding: ObservationEncoding
    _cache: ActionDecodeCache
    _step_index: int = 0

    def next_choice_logits(self) -> Tensor:
        """Return logits for all 110 choices at the current prefix."""
        prefix_context = self._model.decode_live_action_step(
            cache=self._cache
        )
        return self._model.choice_logits_from_prefix_context(
            encoding=self._encoding,
            prefix_context=prefix_context,
        )

    def advance(self, selected_choice_ids: Tensor) -> None:
        """Append selected choices to the live prefix cache."""
        next_index = self._step_index + 1
        assert next_index < self._cache.max_steps
        assert selected_choice_ids.shape == (self._cache.batch_size,)
        embeddings = self._model.embed_selected_choices(
            encoding=self._encoding,
            choice_ids=selected_choice_ids,
            position=next_index,
        )
        self._model.append_live_action_choice(
            cache=self._cache, choice_embeddings=embeddings
        )
        self._step_index = next_index


@dataclass(slots=True)
class ActionDecodeCache:
    """Projected K/V cache for one live action batch."""

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


class TractorPolicyModel(nn.Module):
    """Typed observation encoder and shared-card action decoder."""

    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        assert d_model > 0
        assert d_model % heads == 0
        self._token_encoder = TypedTokenEncoder(d_model=d_model)
        self._observation_encoder = StructuredObservationEncoder(
            d_model=d_model, layers=layers, heads=heads
        )
        self._control_choice_embeddings = nn.Parameter(
            torch.empty(2, d_model)
        )
        nn.init.normal_(
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
        self._action_decoder = _ActionDecoderLayer(
            d_model=d_model, heads=heads
        )
        self._decision_projection = nn.Linear(d_model * 2, d_model)
        self._choice_query = nn.Linear(d_model, d_model, bias=False)
        self._choice_key = nn.Linear(d_model, d_model, bias=False)
        self._choice_logit_bias = nn.Parameter(
            torch.zeros(ACTION_CHOICE_COUNT)
        )
        self._value_head = nn.Linear(d_model, 1)

    def encode_observations(
        self, observation: ObservationTensorBatch
    ) -> ObservationEncoding:
        """Encode observations and rule-conditioned card choices."""
        memory_padding_mask = observation.category_ids[:, :, 0].eq(0)
        memory = self._observation_encoder(
            self._token_encoder(
                category_ids=observation.category_ids,
                scalar_values=observation.scalar_values,
                card_rule_values=observation.card_rule_values,
            ),
            padding_mask=memory_padding_mask,
            encoded_structure_coordinates=(
                observation.encoded_structure_coordinates
            ),
        )
        batch_indices = torch.arange(
            int(memory.shape[0]),
            dtype=torch.long,
            device=memory.device,
        )
        observation_context = memory[
            batch_indices, observation.query_indices
        ]
        card_categories = observation.candidate_category_ids
        card_choices = self._token_encoder.encode_card_candidates(
            suit_ids=card_categories[:, :, 0],
            rank_ids=card_categories[:, :, 1],
            effective_suit_ids=card_categories[:, :, 2],
            counts=observation.candidate_counts,
            rule_values=observation.candidate_card_rule_values,
        )
        assert int(card_choices.shape[1]) == CARD_CHOICE_COUNT
        controls = self._control_choice_embeddings.unsqueeze(0).expand(
            int(memory.shape[0]), -1, -1
        )
        return ObservationEncoding(
            memory=memory,
            memory_padding_mask=memory_padding_mask,
            observation_context=observation_context,
            choice_embeddings=torch.cat(
                (controls, card_choices), dim=1
            ),
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
            choice_embeddings=encoding.choice_embeddings.index_select(
                0, index
            ),
        )

    def value_estimates(self, encoding: ObservationEncoding) -> Tensor:
        return self._value_head(encoding.observation_context).squeeze(
            -1
        )

    def begin_action_decode_session(
        self,
        encoding: ObservationEncoding,
        *,
        max_steps: int,
    ) -> ActionDecodeSession:
        """Decode from the contextual query without a start token."""
        assert 0 < max_steps <= MAX_ACTION_STEPS
        seed = self._query_seed(encoding.observation_context)
        seed = seed + self._action_position_embedding(
            torch.zeros(
                (int(seed.shape[0]),),
                dtype=torch.long,
                device=seed.device,
            )
        )
        cache = self._action_decoder.begin_decode_cache(
            first_embeddings=seed,
            memory=encoding.memory,
            memory_padding_mask=encoding.memory_padding_mask,
            max_steps=max_steps,
        )
        return ActionDecodeSession(
            _model=self, _encoding=encoding, _cache=cache
        )

    def decode_live_action_step(
        self, *, cache: ActionDecodeCache
    ) -> Tensor:
        return self._action_decoder.decode_cached_step(cache=cache)

    def append_live_action_choice(
        self, *, cache: ActionDecodeCache, choice_embeddings: Tensor
    ) -> None:
        self._action_decoder.append_cached_choice(
            cache=cache, choice_embeddings=choice_embeddings
        )

    def score_action_traces(
        self,
        encoding: ObservationEncoding,
        *,
        choice_ids_padded: Tensor,
        step_counts: Tensor,
    ) -> ActionTraceScores:
        """Score traces without Python autoregressive replay."""
        assert choice_ids_padded.ndim == 2
        assert step_counts.shape == (int(choice_ids_padded.shape[0]),)
        prefix_embeddings = self._teacher_forced_prefix_embeddings(
            encoding=encoding, choice_ids_padded=choice_ids_padded
        )
        max_steps = int(choice_ids_padded.shape[1])
        positions = torch.arange(
            max_steps, dtype=torch.long, device=choice_ids_padded.device
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
        decoded = self._action_decoder.forward_prefix(
            prefix_embeddings=prefix_embeddings,
            memory=encoding.memory,
            prefix_padding_mask=prefix_padding_mask,
            memory_padding_mask=encoding.memory_padding_mask,
            self_attention_mask=causal_mask,
        )
        observation_context = encoding.observation_context.unsqueeze(
            1
        ).expand(-1, max_steps, -1)
        decision_context = torch.tanh(
            self._decision_projection(
                torch.cat((observation_context, decoded), dim=-1)
            )
        )
        return ActionTraceScores(
            choice_logits=self._score_choices(
                decision_context=decision_context,
                choice_embeddings=encoding.choice_embeddings,
            )
        )

    def choice_logits_from_prefix_context(
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
        return self._score_choices(
            decision_context=decision_context,
            choice_embeddings=encoding.choice_embeddings,
        )

    def embed_selected_choices(
        self,
        *,
        encoding: ObservationEncoding,
        choice_ids: Tensor,
        position: int,
    ) -> Tensor:
        """Embed outputs with their shared candidate encodings."""
        assert choice_ids.ndim == 1
        assert 0 < position < MAX_ACTION_STEPS
        batch_indices = torch.arange(
            int(choice_ids.shape[0]),
            dtype=torch.long,
            device=choice_ids.device,
        )
        selected = encoding.choice_embeddings[batch_indices, choice_ids]
        positions = torch.full_like(choice_ids, position)
        return self._selected_choice_adapter(
            selected
        ) + self._action_position_embedding(positions)

    def _teacher_forced_prefix_embeddings(
        self,
        *,
        encoding: ObservationEncoding,
        choice_ids_padded: Tensor,
    ) -> Tensor:
        max_steps = int(choice_ids_padded.shape[1])
        assert 0 < max_steps <= MAX_ACTION_STEPS
        seed = self._query_seed(encoding.observation_context).unsqueeze(
            1
        )
        if max_steps == 1:
            prefix = seed
        else:
            batch_size = int(choice_ids_padded.shape[0])
            batch_indices = torch.arange(
                batch_size,
                dtype=torch.long,
                device=choice_ids_padded.device,
            ).unsqueeze(1)
            selected = encoding.choice_embeddings[
                batch_indices,
                choice_ids_padded[:, : max_steps - 1],
            ]
            prefix = torch.cat(
                (seed, self._selected_choice_adapter(selected)), dim=1
            )
        positions = torch.arange(
            max_steps,
            dtype=torch.long,
            device=choice_ids_padded.device,
        ).unsqueeze(0)
        return prefix + self._action_position_embedding(positions)

    def _score_choices(
        self, *, decision_context: Tensor, choice_embeddings: Tensor
    ) -> Tensor:
        queries = self._choice_query(decision_context)
        keys = self._choice_key(choice_embeddings)
        if queries.ndim == 2:
            logits = torch.einsum("bd,bcd->bc", queries, keys)
        else:
            assert queries.ndim == 3
            logits = torch.einsum("bsd,bcd->bsc", queries, keys)
        return logits / math.sqrt(float(keys.shape[-1])) + (
            self._choice_logit_bias
        )


class _ActionDecoderLayer(nn.Module):
    """Single-layer decoder with teacher-forced and cached paths."""

    def __init__(self, *, d_model: int, heads: int) -> None:
        super().__init__()
        self._heads = heads
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

    def forward_prefix(
        self,
        *,
        prefix_embeddings: Tensor,
        memory: Tensor,
        prefix_padding_mask: Tensor,
        memory_padding_mask: Tensor,
        self_attention_mask: Tensor,
    ) -> Tensor:
        self_attended, _weights = self._self_attn(
            prefix_embeddings,
            prefix_embeddings,
            prefix_embeddings,
            key_padding_mask=prefix_padding_mask,
            attn_mask=self_attention_mask,
            need_weights=False,
        )
        target = self._norm1(prefix_embeddings + self_attended)
        cross_attended, _weights = self._cross_attn(
            target,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        target = self._norm2(target + cross_attended)
        return self._norm3(target + self._feed_forward(target))

    def begin_decode_cache(
        self,
        *,
        first_embeddings: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
        max_steps: int,
    ) -> ActionDecodeCache:
        """Create a cache seeded by the contextual decision query."""
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
        keys[:, :, 0:1].copy_(self_key)
        values[:, :, 0:1].copy_(self_value)
        return ActionDecodeCache(
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
        self, *, cache: ActionDecodeCache, choice_embeddings: Tensor
    ) -> None:
        assert choice_embeddings.shape == cache.current_embedding.shape
        assert cache.current_length < cache.max_steps
        key, value = self._project_self_key_value(
            choice_embeddings.unsqueeze(1)
        )
        position = cache.current_length
        cache.self_keys[:, :, position : position + 1].copy_(key)
        cache.self_values[:, :, position : position + 1].copy_(value)
        cache.current_embedding = choice_embeddings
        cache.current_query = self._project_self_query(
            choice_embeddings
        )
        cache.current_length += 1

    def decode_cached_step(self, *, cache: ActionDecodeCache) -> Tensor:
        self_attended = self._scaled_attention(
            query=cache.current_query,
            key=cache.self_keys[:, :, : cache.current_length],
            value=cache.self_values[:, :, : cache.current_length],
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

    def _feed_forward(self, target: Tensor) -> Tensor:
        return self._linear2(F.gelu(self._linear1(target)))

    def _project_self_query(self, embeddings: Tensor) -> Tensor:
        return self._project_query(
            embeddings, attention=self._self_attn
        )

    def _project_cross_query(self, embeddings: Tensor) -> Tensor:
        return self._project_query(
            embeddings, attention=self._cross_attn
        )

    def _project_query(
        self, embeddings: Tensor, *, attention: nn.MultiheadAttention
    ) -> Tensor:
        weight = attention.in_proj_weight
        bias = attention.in_proj_bias
        embed_dim = attention.embed_dim
        projected = F.linear(
            embeddings, weight[:embed_dim], bias[:embed_dim]
        )
        return _split_heads(
            projected.unsqueeze(1), heads=self._heads
        ).squeeze(2)

    def _project_self_key_value(
        self, embeddings: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self._project_key_value(
            embeddings, attention=self._self_attn
        )

    def _project_cross_key_value(
        self, embeddings: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self._project_key_value(
            embeddings, attention=self._cross_attn
        )

    def _project_key_value(
        self, embeddings: Tensor, *, attention: nn.MultiheadAttention
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
            query.unsqueeze(2), key.transpose(-2, -1)
        ).squeeze(2) / math.sqrt(float(query.shape[-1]))
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1), -torch.inf
            )
        context = torch.matmul(
            torch.softmax(scores, dim=-1).unsqueeze(2), value
        ).squeeze(2)
        return out_projection(_merge_heads(context))


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
        int(values.shape[0]), int(values.shape[1] * values.shape[2])
    )


__all__ = (
    "ActionDecodeSession",
    "ActionTraceScores",
    "ObservationEncoding",
    "TractorPolicyModel",
)
