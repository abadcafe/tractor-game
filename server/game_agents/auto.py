"""Rule-driven automatic game agent."""

from __future__ import annotations

import asyncio
import random

from server.game import commands, snapshots
from server.game.rules import bidding, play
from server.game.rules.cards import CardId
from server.game_runtime.session import (
    AgentSubmission,
    Delivery,
    DisconnectReason,
)

from ._submission import TypedCommandDecoder


class AutoAgent:
    """Consume deliveries and submit rule-legal commands."""

    def __init__(
        self,
        target: AgentSubmission,
        random_source: random.Random,
    ) -> None:
        self._target = target
        self._random = random_source
        self._queue: asyncio.Queue[Delivery | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()

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

    async def _drain(self) -> None:
        while True:
            delivery = await self._queue.get()
            if delivery is None:
                return
            command = automatic_command(
                delivery.snapshot,
                self._random,
            )
            if command is None:
                self._ready.set()
                continue
            await self._target.submit(
                delivery.seq,
                TypedCommandDecoder(command),
            )
            self._ready.set()


def automatic_command(
    snapshot: snapshots.PlayerSnapshot,
    random_source: random.Random,
) -> commands.Command | None:
    """Choose one rule-legal command from a player snapshot."""
    action = snapshot.awaiting_action
    if action is None:
        return None
    if action == "next_round":
        return commands.ConfirmRound()
    if action == "bid":
        current = (
            None
            if snapshot.bid_winner is None
            else bidding.Declaration(cards=snapshot.bid_winner.cards)
        )
        reveals = bidding.legal_reveals(
            snapshot.player_hand,
            snapshot.trump_rank,
            current,
        )
        if reveals and random_source.random() < 0.15:
            reveal = random_source.choice(reveals)
            return commands.RevealBid(
                card_ids=tuple(CardId(card.id) for card in reveal.cards)
            )
        return commands.PassBid()
    if action == "discard":
        selected = random_source.sample(
            list(snapshot.player_hand),
            len(snapshot.bottom_cards),
        )
        return commands.Bury(
            card_ids=tuple(CardId(card.id) for card in selected)
        )
    if action == "stir":
        current = (
            None
            if snapshot.bid_winner is None
            else bidding.Declaration(cards=snapshot.bid_winner.cards)
        )
        reveals = tuple(
            reveal
            for reveal in bidding.legal_reveals(
                snapshot.player_hand,
                snapshot.trump_rank,
                current,
            )
            if len(reveal.cards) == 2
        )
        if reveals and random_source.random() < 0.15:
            reveal = random_source.choice(reveals)
            return commands.Stir(
                card_ids=tuple(CardId(card.id) for card in reveal.cards)
            )
        return commands.PassStir()
    assert action == "play"
    lead = None
    if snapshot.trick is not None and snapshot.trick.slots:
        lead = snapshot.trick.slots[0].cards
    selected = play.choose_legal_play(
        snapshot.player_hand,
        lead,
        snapshot.trump_suit,
        snapshot.trump_rank,
    )
    return commands.Play(
        card_ids=tuple(CardId(card.id) for card in selected)
    )
