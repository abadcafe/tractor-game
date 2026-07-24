"""LLM-backed runtime game agent."""

from __future__ import annotations

import asyncio
import json

from server.foundation.result import Rejected
from server.game import CommandRejected, Seat, commands, snapshots
from server.game.rules import bidding, play
from server.game.rules.cards import Card, CardId
from server.game_runtime.session import (
    AgentSubmission,
    Delivery,
    DisconnectReason,
)

from .._submission import TypedCommandDecoder
from .client import (
    APIInteraction,
    ClientRejected,
    Decision,
    DecisionClient,
    DecisionPrompt,
    JSONObject,
    JSONValue,
    ToolSpec,
)
from .config import LLMConfig
from .openai import OpenAIChatCompletionsClient
from .transcript import (
    LLMTranscript,
    TranscriptRecord,
    TranscriptRecordDict,
)

_BID_DECISION_COUNTS: frozenset[int] = frozenset((5, 11, 17, 23))


class LLMAgent:
    """Make strategic decisions through one provider-neutral client."""

    def __init__(
        self,
        seat: Seat,
        target: AgentSubmission,
        *,
        config: LLMConfig | None = None,
        client: DecisionClient | None = None,
    ) -> None:
        self._seat = seat
        self._target = target
        self._config = config or LLMConfig.from_env()
        self._client = client or OpenAIChatCompletionsClient(
            self._config
        )
        self._transcript = LLMTranscript()
        self._queue: asyncio.Queue[Delivery | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._attempts: dict[int, int] = {}
        self._pending_records: dict[int, TranscriptRecord] = {}

    async def start(self) -> None:
        assert self._task is None
        self._task = asyncio.create_task(self._drain())
        await self._target.submit(
            0,
            TypedCommandDecoder(commands.ConfirmRound()),
        )
        await self._ready.wait()

    async def offer(self, delivery: Delivery) -> None:
        await self._queue.put(delivery)

    async def disconnect(self, reason: DisconnectReason) -> None:
        del reason
        await self.stop()

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        self._task = None
        await self._queue.put(None)
        if task is not asyncio.current_task():
            await task

    def transcript(self) -> list[TranscriptRecordDict]:
        """Return the complete debug transcript."""
        return self._transcript.to_dict()

    def transcript_stream(self) -> list[TranscriptRecordDict]:
        """Return records ordered for an initial stream replay."""
        return self._transcript.stream_dicts()

    def subscribe_transcript(
        self,
    ) -> asyncio.Queue[TranscriptRecordDict]:
        """Subscribe to live transcript records."""
        return self._transcript.subscribe()

    def unsubscribe_transcript(
        self,
        queue: asyncio.Queue[TranscriptRecordDict],
    ) -> None:
        """Remove a live transcript subscription."""
        self._transcript.unsubscribe(queue)

    async def _drain(self) -> None:
        while True:
            delivery = await self._queue.get()
            if delivery is None:
                return
            await self._decide(delivery)
            self._ready.set()

    async def _decide(self, delivery: Delivery) -> None:
        snapshot = delivery.snapshot
        for completed_seq in tuple(self._pending_records):
            if completed_seq < delivery.seq:
                self._pending_records.pop(completed_seq)
        previous_record = self._pending_records.pop(
            delivery.seq,
            None,
        )
        if delivery.error is not None and previous_record is not None:
            self._transcript.update_tool_result(
                previous_record,
                _tool_result("rejected", delivery.error),
            )
        local = _local_command(snapshot)
        if local is not None:
            await self._submit(delivery.seq, local)
            return
        if snapshot.awaiting_action is None:
            return
        attempt = self._attempts.get(delivery.seq, 0) + 1
        maximum = self._config.decision_max_retries + 1
        if attempt > maximum:
            return
        self._attempts[delivery.seq] = attempt
        prompt = _prompt(snapshot, delivery.error)
        tools = [_tool(snapshot)]
        decision = await self._client.decide(prompt, tools)
        if isinstance(decision, Rejected):
            api = _api_from_rejection(decision)
            self._transcript.add_record(
                player_index=int(self._seat),
                seq=delivery.seq,
                attempt=attempt,
                api_request=api.request,
                api_response=api.response,
                api_error=api.error or decision.reason,
                tool_result=None,
            )
            return
        command = _command(decision.value, snapshot)
        if isinstance(command, CommandRejected):
            self._transcript.add_record(
                player_index=int(self._seat),
                seq=delivery.seq,
                attempt=attempt,
                api_request=decision.value.api.request,
                api_response=decision.value.api.response,
                api_error=decision.value.api.error,
                tool_result=_tool_result(
                    "rejected",
                    command.reason,
                ),
            )
            if attempt < maximum:
                await self._queue.put(
                    Delivery(
                        viewer=delivery.viewer,
                        seq=delivery.seq,
                        snapshot=snapshot,
                        error=command.reason,
                    )
                )
            return
        record = self._transcript.add_record(
            player_index=int(self._seat),
            seq=delivery.seq,
            attempt=attempt,
            api_request=decision.value.api.request,
            api_response=decision.value.api.response,
            api_error=decision.value.api.error,
            tool_result=_tool_result("accepted", None),
        )
        self._pending_records[delivery.seq] = record
        await self._submit(delivery.seq, command)

    async def _submit(
        self,
        seq: int,
        command: commands.Command,
    ) -> None:
        await self._target.submit(
            seq,
            TypedCommandDecoder(command),
        )


def _local_command(
    snapshot: snapshots.PlayerSnapshot,
) -> commands.Command | None:
    action = snapshot.awaiting_action
    if action == "next_round":
        return commands.ConfirmRound()
    if action == "bid":
        reveals = _legal_reveals(snapshot)
        if (
            len(snapshot.player_hand) not in _BID_DECISION_COUNTS
            or not reveals
        ):
            return commands.PassBid()
    if action == "stir" and not _legal_stirs(snapshot):
        return commands.PassStir()
    return None


def _prompt(
    snapshot: snapshots.PlayerSnapshot,
    previous_error: str | None,
) -> DecisionPrompt:
    hand = "\n".join(
        f"- {card.id}: {card.suit.value} {card.rank.value}"
        for card in snapshot.player_hand
    )
    groups = _legal_groups(snapshot)
    group_text = "\n".join(
        f"- [{', '.join(card.id for card in group)}]"
        for group in groups
    )
    if not group_text:
        group_text = "- 无完整枚举；仍可从当前手牌中依法选择"
    error = (
        ""
        if previous_error is None
        else f"\n上一次动作被拒绝：{previous_error}\n"
    )
    return DecisionPrompt(
        system=(
            "你是升级/拖拉机玩家。只调用给出的一个工具。"
            "card_ids 必须逐字来自当前手牌；若合法动作组非空，"
            "必须完整选择其中一组。"
        ),
        user=(
            f"当前阶段={snapshot.phase}，"
            f"需要动作={snapshot.awaiting_action}，"
            f"主级牌={snapshot.trump_rank.value}，"
            f"主花色={snapshot.trump_suit}，"
            f"闲家分={snapshot.defender_points}。"
            f"{error}\n当前手牌：\n{hand}\n"
            f"合法动作组：\n{group_text}"
        ),
    )


def _tool(snapshot: snapshots.PlayerSnapshot) -> ToolSpec:
    action = snapshot.awaiting_action
    assert action is not None
    name = {
        "bid": "bid_trump",
        "stir": "stir_trump",
        "discard": "discard_bottom",
        "play": "play_cards",
        "next_round": "confirm_round",
    }[action]
    ids: list[JSONValue] = [card.id for card in snapshot.player_hand]
    card_ids: JSONObject = {
        "type": "array",
        "items": {"type": "string", "enum": ids},
    }
    optional = action in ("bid", "stir")
    required: list[JSONValue] = [] if optional else ["card_ids"]
    return ToolSpec(
        name=name,
        description="选择当前动作使用的牌；空数组表示叫牌或炒牌 pass",
        parameters={
            "type": "object",
            "properties": {"card_ids": card_ids},
            "required": required,
            "additionalProperties": False,
        },
    )


def _command(
    decision: Decision,
    snapshot: snapshots.PlayerSnapshot,
) -> commands.Command | CommandRejected:
    expected_tool = {
        "bid": "bid_trump",
        "stir": "stir_trump",
        "discard": "discard_bottom",
        "play": "play_cards",
        "next_round": "confirm_round",
        None: None,
    }[snapshot.awaiting_action]
    if decision.tool_call.name != expected_tool:
        return CommandRejected("模型调用了当前动作之外的工具")
    raw = decision.tool_call.arguments.get("card_ids", [])
    if not isinstance(raw, list):
        return CommandRejected("card_ids 必须是数组")
    ids: list[CardId] = []
    hand_ids = {card.id for card in snapshot.player_hand}
    for value in raw:
        if not isinstance(value, str):
            return CommandRejected("card_ids 只能包含字符串")
        if value not in hand_ids:
            return CommandRejected(f"牌 {value} 不在当前手牌")
        card_id = CardId(value)
        if card_id in ids:
            return CommandRejected(f"不能重复选择牌 {value}")
        ids.append(card_id)
    action = snapshot.awaiting_action
    if action == "bid":
        if not ids:
            return commands.PassBid()
        if not _matches_group(ids, _legal_reveals(snapshot)):
            return CommandRejected("抢主牌不属于完整合法动作组")
        return commands.RevealBid(card_ids=tuple(ids))
    if action == "stir":
        if not ids:
            return commands.PassStir()
        if not _matches_group(ids, _legal_stirs(snapshot)):
            return CommandRejected("反主牌不属于完整合法动作组")
        return commands.Stir(card_ids=tuple(ids))
    if action == "discard":
        if len(ids) != len(snapshot.bottom_cards):
            return CommandRejected(
                f"必须正好埋 {len(snapshot.bottom_cards)} 张牌"
            )
        return commands.Bury(card_ids=tuple(ids))
    if action == "play":
        if not ids:
            return CommandRejected("出牌不能为空")
        groups = _legal_play_groups(snapshot)
        if groups and not _matches_group(ids, groups):
            return CommandRejected("出牌不属于完整合法动作组")
        return commands.Play(card_ids=tuple(ids))
    return CommandRejected("当前不需要 LLM 决策")


def _legal_groups(
    snapshot: snapshots.PlayerSnapshot,
) -> tuple[tuple[Card, ...], ...]:
    if snapshot.awaiting_action == "bid":
        return tuple(
            reveal.cards for reveal in _legal_reveals(snapshot)
        )
    if snapshot.awaiting_action == "stir":
        return tuple(reveal.cards for reveal in _legal_stirs(snapshot))
    if snapshot.awaiting_action == "play":
        return _legal_play_groups(snapshot)
    return ()


def _legal_reveals(
    snapshot: snapshots.PlayerSnapshot,
) -> tuple[bidding.Declaration, ...]:
    return bidding.legal_reveals(
        snapshot.player_hand,
        snapshot.trump_rank,
        _current_declaration(snapshot),
    )


def _legal_stirs(
    snapshot: snapshots.PlayerSnapshot,
) -> tuple[bidding.Declaration, ...]:
    return tuple(
        reveal
        for reveal in _legal_reveals(snapshot)
        if len(reveal.cards) == 2
    )


def _legal_play_groups(
    snapshot: snapshots.PlayerSnapshot,
) -> tuple[tuple[Card, ...], ...]:
    lead = None
    if snapshot.trick is not None and snapshot.trick.slots:
        lead = snapshot.trick.slots[0].cards
    return play.closed_hints(
        snapshot.player_hand,
        lead,
        snapshot.trump_suit,
        snapshot.trump_rank,
    )


def _current_declaration(
    snapshot: snapshots.PlayerSnapshot,
) -> bidding.Declaration | None:
    if snapshot.bid_winner is None:
        return None
    return bidding.Declaration(cards=snapshot.bid_winner.cards)


def _matches_group(
    ids: list[CardId],
    groups: tuple[bidding.Declaration, ...]
    | tuple[tuple[Card, ...], ...],
) -> bool:
    selected = tuple(sorted(str(card_id) for card_id in ids))
    for group in groups:
        cards = (
            group.cards
            if isinstance(group, bidding.Declaration)
            else group
        )
        if tuple(sorted(card.id for card in cards)) == selected:
            return True
    return False


def _api_from_rejection(rejected: Rejected) -> APIInteraction:
    if isinstance(rejected, ClientRejected):
        return rejected.api
    return APIInteraction(error=rejected.reason)


def _tool_result(status: str, reason: str | None) -> str:
    value: JSONObject = {"status": status}
    if reason is not None:
        value["reason"] = reason
    return json.dumps(value, ensure_ascii=False)
