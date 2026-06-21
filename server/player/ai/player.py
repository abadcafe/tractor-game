"""AI player implementation."""

from __future__ import annotations

import asyncio
import json

from server.player.ai.client import (
    AIAPIInteraction,
    AIClient,
    AIClientRejected,
    AIDecisionPrompt,
    AIToolCall,
    DisabledAIClient,
    JSONObject,
    JSONValue,
    is_json_value,
)
from server.player.ai.config import AIConfig
from server.player.ai.context import build_decision_prompt
from server.player.ai.local_actions import local_message
from server.player.ai.memory import AIMemory
from server.player.ai.openai_client import OpenAIChatCompletionsClient
from server.player.ai.rejections import (
    AIRejectionFeedback,
    feedback_from_rejected,
    rule_feedback,
)
from server.player.ai.rules import RuleBook
from server.player.ai.tools import (
    allowed_tool_specs,
    tool_call_to_message,
)
from server.player.ai.transcript import (
    AITranscript,
    TranscriptRecord,
    TranscriptRecordDict,
)
from server.player.base import GameView, Player
from server.protocol import PlayerMessage, StateMessage, StateSnapshot
from server.result import Ok, Rejected


class AIPlayer(Player):
    """
    LLM-backed player with local handling for non-strategic actions.
    """

    def __init__(
        self,
        index: int,
        *,
        config: AIConfig | None = None,
        client: AIClient | None = None,
        rules: RuleBook | None = None,
    ) -> None:
        super().__init__(index)
        self._config = config or AIConfig.from_env()
        self._client = client or _client_from_config(self._config)
        self._rules = rules or RuleBook.from_default()
        self._memory = AIMemory()
        self._transcript = AITranscript()
        self._transcript_records_by_seq: dict[
            int, TranscriptRecord
        ] = {}
        self._pending_seq: int | None = None

    async def run(self, game: GameView) -> None:
        """Start this player by requesting current state with seq=0."""
        await game.receive(self.index, PlayerMessage(seq=0, raw={}))

    async def on_state(
        self, game: GameView, message: StateMessage
    ) -> None:
        """
        Update visible memory and schedule one decision if action is
        awaited.
        """
        self._memory.update(message.state, seq=message.seq)
        if message.error is not None:
            rejected_record = self._record_server_rejection(message)
            self._clear_pending(message.seq)
            if self._should_repair_server_rejection(
                message, rejected_record
            ):
                self._pending_seq = message.seq
                asyncio.create_task(
                    self._decide_and_submit(
                        game,
                        message,
                        repair_reason=message.error,
                        start_attempt=rejected_record.attempt + 1,
                    )
                )
            return
        if message.state.awaiting_action is None:
            return
        if self._pending_seq == message.seq and message.error is None:
            return
        self._pending_seq = message.seq
        asyncio.create_task(self._decide_and_submit(game, message))

    async def _decide_and_submit(
        self,
        game: GameView,
        message: StateMessage,
        *,
        repair_reason: str | None = None,
        start_attempt: int = 1,
    ) -> None:
        snapshot = message.state
        if repair_reason is None:
            local_result = local_message(message.seq, snapshot)
            if local_result is not None:
                await self._submit_if_ok(game, local_result)
                self._clear_pending(message.seq)
                return

        tools = allowed_tool_specs(snapshot)
        if not tools:
            self._clear_pending(message.seq)
            return

        prompt = build_decision_prompt(
            player_index=self.index,
            snapshot=snapshot,
            memory=self._memory,
            rules=self._rules,
        )
        if repair_reason is not None:
            prompt = _repair_prompt(
                prompt, snapshot, rule_feedback(repair_reason)
            )
        max_attempts = max(self._config.decision_retries, 0) + 1
        for attempt in range(start_attempt, max_attempts + 1):
            decision_result_from_client = await self._client.decide(
                prompt, tools
            )
            if isinstance(decision_result_from_client, Rejected):
                record = self._transcript.add_record(
                    player_index=self.index,
                    seq=message.seq,
                    attempt=attempt,
                    api_request=_api_from_rejection(
                        decision_result_from_client
                    ).request,
                    api_response=_api_from_rejection(
                        decision_result_from_client
                    ).response,
                    api_error=_client_rejection_api_error(
                        decision_result_from_client
                    ),
                    tool_result=None,
                )
                self._transcript_records_by_seq[message.seq] = record
                self._clear_pending(message.seq)
                return

            decision = decision_result_from_client.value
            decision_result = tool_call_to_message(
                message.seq, snapshot, decision.tool_call
            )
            if isinstance(decision_result, Ok):
                record = self._transcript.add_record(
                    player_index=self.index,
                    seq=message.seq,
                    attempt=attempt,
                    api_request=decision.api.request,
                    api_response=decision.api.response,
                    api_error=decision.api.error,
                    tool_result=_accepted_tool_result(
                        decision.tool_call, decision_result.value
                    ),
                )
                self._transcript_records_by_seq[message.seq] = record
                await self._submit_if_ok(game, decision_result)
                self._clear_pending(message.seq)
                return

            feedback = feedback_from_rejected(decision_result)
            record = self._transcript.add_record(
                player_index=self.index,
                seq=message.seq,
                attempt=attempt,
                api_request=decision.api.request,
                api_response=decision.api.response,
                api_error=decision.api.error,
                tool_result=_rejected_tool_result(
                    decision.tool_call, feedback
                ),
            )
            self._transcript_records_by_seq[message.seq] = record
            if attempt < max_attempts:
                prompt = _repair_prompt(prompt, snapshot, feedback)
            else:
                await self._submit_if_ok(game, decision_result)
        self._clear_pending(message.seq)

    async def _submit_if_ok(
        self,
        game: GameView,
        decision_result: Ok[PlayerMessage] | Rejected,
    ) -> None:
        if isinstance(decision_result, Ok):
            await game.receive(self.index, decision_result.value)

    def _clear_pending(self, seq: int) -> None:
        if self._pending_seq == seq:
            self._pending_seq = None

    def transcript(self) -> list[TranscriptRecordDict]:
        return self._transcript.to_dict()

    def transcript_stream(self) -> list[TranscriptRecordDict]:
        return self._transcript.stream_dicts()

    def subscribe_transcript(
        self,
    ) -> asyncio.Queue[TranscriptRecordDict]:
        return self._transcript.subscribe()

    def unsubscribe_transcript(
        self, queue: asyncio.Queue[TranscriptRecordDict]
    ) -> None:
        self._transcript.unsubscribe(queue)

    def _record_server_rejection(
        self, message: StateMessage
    ) -> TranscriptRecord:
        record = self._transcript_records_by_seq.get(message.seq)
        assert message.error is not None
        if record is None:
            record = self._transcript.add_record(
                player_index=self.index,
                seq=message.seq,
                attempt=0,
                api_request=None,
                api_response=None,
                api_error=None,
                tool_result=_server_rejection_tool_result(
                    message.error, previous=None
                ),
            )
            self._transcript_records_by_seq[message.seq] = record
            return record
        self._transcript.update_tool_result(
            record,
            _server_rejection_tool_result(
                message.error, previous=record.tool_result
            ),
        )
        return record

    def _should_repair_server_rejection(
        self,
        message: StateMessage,
        rejected_record: TranscriptRecord,
    ) -> bool:
        if message.state.awaiting_action is None:
            return False
        if rejected_record.attempt <= 0:
            return False
        max_attempts = max(self._config.decision_retries, 0) + 1
        return rejected_record.attempt < max_attempts


def _client_from_config(config: AIConfig) -> AIClient:
    if config.provider == "openai":
        return OpenAIChatCompletionsClient(config)
    return DisabledAIClient()


def _api_from_rejection(rejection: Rejected) -> AIAPIInteraction:
    if isinstance(rejection, AIClientRejected):
        return rejection.api
    return AIAPIInteraction()


def _client_rejection_api_error(rejection: Rejected) -> str:
    api = _api_from_rejection(rejection)
    if (
        isinstance(rejection, AIClientRejected)
        and api.error is not None
    ):
        return api.error
    return _json_text(
        {
            "title": "CLIENT REJECTION",
            "reason": rejection.reason,
        }
    )


def _accepted_tool_result(
    call: AIToolCall, message: PlayerMessage
) -> str:
    return _json_text(
        {
            "status": "accepted",
            "tool_call": _tool_call_payload(call),
            "message": {
                "seq": message.seq,
                "raw": _json_or_text(message.raw),
            },
        }
    )


def _rejected_tool_result(
    call: AIToolCall, feedback: AIRejectionFeedback
) -> str:
    return _json_text(
        {
            "status": "rejected",
            "error_type": feedback.error_type,
            "reason": feedback.reason,
            "repair": feedback.repair,
            "tool_call": _tool_call_payload(call),
        }
    )


def _server_rejection_tool_result(
    reason: str, *, previous: str | None
) -> str:
    feedback = rule_feedback(reason)
    payload: JSONObject = {
        "status": "rejected",
        "error_type": feedback.error_type,
        "reason": feedback.reason,
        "repair": feedback.repair,
    }
    if previous is not None:
        payload["previous_tool_result"] = _raw_json_value(previous)
    return _json_text(payload)


def _tool_call_payload(call: AIToolCall) -> JSONObject:
    return {
        "name": call.name,
        "arguments": call.arguments,
    }


def _json_text(value: JSONValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
    )


def _json_or_text(value: object) -> JSONValue:
    if is_json_value(value):
        return value
    return str(value)


def _raw_json_value(raw: str) -> JSONValue:
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return _json_or_text(parsed)


def _repair_prompt(
    prompt: AIDecisionPrompt,
    snapshot: StateSnapshot,
    feedback: AIRejectionFeedback,
) -> AIDecisionPrompt:
    legal_ids = ", ".join(card.id for card in snapshot.player_hand)
    if not legal_ids:
        legal_ids = "无"
    hint_groups = _hint_groups_text(snapshot)
    repair_text = "\n".join(
        [
            "上一次动作被拒绝。",
            f"- error_type: {feedback.error_type}",
            f"- reason: {feedback.reason}",
            f"- repair: {feedback.repair}",
            f"- 当前手牌 card_ids: {legal_ids}",
            hint_groups,
            "请重新调用一个允许的工具。",
        ]
    )
    return AIDecisionPrompt(
        system=prompt.system,
        user=f"{prompt.user}\n\n{repair_text}",
    )


def _hint_groups_text(snapshot: StateSnapshot) -> str:
    if not snapshot.action_hints:
        return "- legal_action_hint_groups: 无"
    groups: list[str] = []
    for index, hint in enumerate(snapshot.action_hints):
        groups.append(
            f"hint {index}: [{', '.join(card.id for card in hint)}]"
        )
    return f"- legal_action_hint_groups: {'; '.join(groups)}"
