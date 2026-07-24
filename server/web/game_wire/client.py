"""Decode untrusted WebSocket frames into typed game commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeGuard

from server.foundation.result import Ok
from server.game import CommandRejected, commands
from server.game.rules.cards import CardId
from server.game_runtime.session import CommandDecoder


@dataclass(frozen=True, slots=True)
class ClientFrame:
    """A sequence number and lazily decoded command."""

    seq: int
    decoder: CommandDecoder


class WireCommandDecoder:
    """Decode action fields only after the runtime sequence gate."""

    def __init__(
        self,
        raw: dict[str, object] | None,
    ) -> None:
        self._raw = raw

    def decode(
        self,
    ) -> Ok[commands.Command] | CommandRejected:
        raw = self._raw
        if raw is None:
            return CommandRejected("消息必须是 JSON 对象")
        action = raw.get("type")
        if not isinstance(action, str):
            return CommandRejected("缺少操作类型")
        if action == "next_round":
            extra = _extra_fields(raw, ("seq", "type"))
            if extra is not None:
                return extra
            return Ok(commands.ConfirmRound())
        if action == "bid":
            if raw.get("pass") is True:
                extra = _extra_fields(
                    raw,
                    ("seq", "type", "pass"),
                )
                if extra is not None:
                    return extra
                return Ok(commands.PassBid())
            extra = _extra_fields(raw, ("seq", "type", "cards"))
            if extra is not None:
                return extra
            return self._decode_cards("bid", raw)
        if action == "stir":
            if raw.get("pass") is True:
                extra = _extra_fields(
                    raw,
                    ("seq", "type", "pass"),
                )
                if extra is not None:
                    return extra
                return Ok(commands.PassStir())
            extra = _extra_fields(
                raw,
                ("seq", "type", "cards", "pass"),
            )
            if extra is not None:
                return extra
            if "pass" in raw and raw["pass"] is not False:
                return CommandRejected("stir.pass 必须是布尔值")
            return self._decode_cards("stir", raw)
        if action == "discard":
            extra = _extra_fields(raw, ("seq", "type", "cards"))
            if extra is not None:
                return extra
            return self._decode_cards("discard", raw)
        if action == "play":
            extra = _extra_fields(raw, ("seq", "type", "cards"))
            if extra is not None:
                return extra
            return self._decode_cards("play", raw)
        return CommandRejected(f"未知操作类型: {action}")

    def _decode_cards(
        self,
        action: str,
        raw: dict[str, object],
    ) -> Ok[commands.Command] | CommandRejected:
        ids = _card_ids(raw.get("cards"))
        if isinstance(ids, CommandRejected):
            return ids
        if action == "bid":
            return Ok(commands.RevealBid(card_ids=ids))
        if action == "stir":
            return Ok(commands.Stir(card_ids=ids))
        if action == "discard":
            return Ok(commands.Bury(card_ids=ids))
        assert action == "play"
        return Ok(commands.Play(card_ids=ids))


def read_frame(value: object) -> ClientFrame:
    """Read the sequence envelope without decoding action fields."""
    raw = value if _is_str_object(value) else None
    seq_value = 0 if raw is None else raw.get("seq", 0)
    seq = (
        seq_value
        if isinstance(seq_value, int)
        and not isinstance(seq_value, bool)
        else 0
    )
    return ClientFrame(
        seq=seq,
        decoder=WireCommandDecoder(raw),
    )


def _card_ids(
    value: object,
) -> tuple[CardId, ...] | CommandRejected:
    if not _is_object_list(value):
        return CommandRejected("cards 必须是字符串 id 数组")
    result: list[CardId] = []
    for item in value:
        if isinstance(item, str):
            result.append(CardId(item))
            continue
        return CommandRejected("cards 只能包含字符串 id")
    return tuple(result)


def _extra_fields(
    raw: dict[str, object],
    allowed: tuple[str, ...],
) -> CommandRejected | None:
    extra = tuple(sorted(set(raw) - set(allowed)))
    if not extra:
        return None
    return CommandRejected(f"消息包含未知字段: {', '.join(extra)}")


def _is_str_object(
    value: object,
) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)
