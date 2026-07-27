"""Reconstruct public engine commands from sequenced observations."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands, seats
from server.game.rules.cards import CardId
from server.game.snapshots import PlayerSnapshot
from server.game.snapshots.tricks import (
    FailedThrowSnapshot,
    TrickSlotSnapshot,
)
from server.training.observation_memory import ObservationMemoryView

from .runtime import EngineBranch


@dataclass(frozen=True, slots=True)
class ObservedFrame:
    """One real player view with its complete public history."""

    seq: int
    snapshot: PlayerSnapshot
    memory: ObservationMemoryView


def awaited_actor(branch: EngineBranch) -> Ok[Seat] | Rejected:
    """Return the single strategic actor in a branch."""
    actors = [
        seat
        for seat in seats()
        if branch.snapshot(seat).awaiting_action
        in ("bid", "discard", "stir", "play")
    ]
    if len(actors) != 1:
        return Rejected(
            reason="particle does not have one active actor"
        )
    return Ok(actors[0])


def bid_command(
    previous: PlayerSnapshot,
    current: PlayerSnapshot,
) -> commands.RevealBid | commands.PassBid:
    """Reconstruct one publicly observed bid action."""
    if len(current.bid_events) == len(previous.bid_events):
        return commands.PassBid()
    assert len(current.bid_events) == len(previous.bid_events) + 1
    return commands.RevealBid(
        tuple(CardId(card.id) for card in current.bid_events[-1].cards)
    )


def stir_command(
    previous: PlayerSnapshot,
    current: PlayerSnapshot,
) -> commands.Stir | commands.PassStir:
    """Reconstruct one publicly observed stir action."""
    assert len(current.stir_events) == len(previous.stir_events) + 1
    event = current.stir_events[-1]
    if event.kind == "pass":
        return commands.PassStir()
    return commands.Stir(tuple(CardId(card.id) for card in event.cards))


def play_command(
    previous: ObservedFrame,
    current: ObservedFrame,
) -> Ok[commands.Play] | Rejected:
    """Reconstruct the play appended by one state transition."""
    before = _public_plays(previous)
    after = _public_plays(current)
    if len(after) <= len(before) or after[: len(before)] != before:
        return Rejected(reason="AI could not identify observed play")
    return Ok(after[len(before)][1])


def _public_plays(
    frame: ObservedFrame,
) -> tuple[tuple[Seat, commands.Play], ...]:
    result: list[tuple[Seat, commands.Play]] = []
    for trick in frame.memory.completed_tricks:
        result.extend(_trick_plays(trick.slots, trick.failed_throw))
    trick = frame.snapshot.trick
    if trick is not None:
        result.extend(_trick_plays(trick.slots, trick.failed_throw))
    return tuple(result)


def _trick_plays(
    slots: tuple[TrickSlotSnapshot, ...],
    failed: FailedThrowSnapshot | None,
) -> tuple[tuple[Seat, commands.Play], ...]:
    return tuple(
        (
            slot.actor,
            commands.Play(
                tuple(
                    CardId(card.id)
                    for card in (
                        failed.attempted_cards
                        if failed is not None
                        and failed.actor == slot.actor
                        else slot.cards
                    )
                )
            ),
        )
        for slot in slots
    )


__all__ = (
    "ObservedFrame",
    "awaited_actor",
    "bid_command",
    "play_command",
    "stir_command",
)
