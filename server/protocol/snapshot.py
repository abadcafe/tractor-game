"""Player-facing state snapshot models."""

from __future__ import annotations

from typing import Literal, overload

from pydantic import BaseModel, ConfigDict

from server.rules.cards import Card, Rank, Suit

type RoundPhase = Literal[
    "DEAL_BID",
    "STIRRING",
    "PLAYING",
    "SCORING",
    "WAITING",
]
type StirringPhase = Literal["WAITING", "EXCHANGING", "COMPLETE"]
type AwaitingAction = Literal[
    "bid", "stir", "discard", "play", "next_round"
]
type BidEventKind = Literal["trump_rank", "joker"]
type JokerType = Literal["big", "small"]


class SnapshotModel(BaseModel):
    """Frozen base model for player-facing snapshots."""

    model_config = ConfigDict(frozen=True)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in type(self).model_fields


class TrickSlotSnapshot(SnapshotModel):
    """One player's contribution in a trick."""

    player: int
    cards: list[Card]

    @overload
    def __getitem__(self, key: Literal["player"]) -> int: ...

    @overload
    def __getitem__(self, key: Literal["cards"]) -> list[Card]: ...

    def __getitem__(
        self, key: Literal["player", "cards"]
    ) -> int | list[Card]:
        if key == "player":
            return self.player
        return self.cards


class TrickSnapshot(SnapshotModel):
    """Current in-progress trick."""

    lead_player: int
    slots: list[TrickSlotSnapshot]
    current_player: int

    @overload
    def __getitem__(self, key: Literal["lead_player"]) -> int: ...

    @overload
    def __getitem__(
        self, key: Literal["slots"]
    ) -> list[TrickSlotSnapshot]: ...

    @overload
    def __getitem__(self, key: Literal["current_player"]) -> int: ...

    def __getitem__(
        self,
        key: Literal["lead_player", "slots", "current_player"],
    ) -> int | list[TrickSlotSnapshot]:
        if key == "lead_player":
            return self.lead_player
        if key == "slots":
            return self.slots
        return self.current_player


class CompletedTrickSnapshot(SnapshotModel):
    """Completed trick visible to players."""

    lead_player: int
    slots: list[TrickSlotSnapshot]
    winner: int
    points: int

    @overload
    def __getitem__(self, key: Literal["lead_player"]) -> int: ...

    @overload
    def __getitem__(
        self, key: Literal["slots"]
    ) -> list[TrickSlotSnapshot]: ...

    @overload
    def __getitem__(self, key: Literal["winner"]) -> int: ...

    @overload
    def __getitem__(self, key: Literal["points"]) -> int: ...

    def __getitem__(
        self,
        key: Literal["lead_player", "slots", "winner", "points"],
    ) -> int | list[TrickSlotSnapshot]:
        if key == "lead_player":
            return self.lead_player
        if key == "slots":
            return self.slots
        if key == "winner":
            return self.winner
        return self.points


class FailedThrowSnapshot(SnapshotModel):
    """Public event emitted when a throw attempt is forced smaller."""

    player: int
    attempted_cards: list[Card]
    forced_cards: list[Card]

    @overload
    def __getitem__(self, key: Literal["player"]) -> int: ...

    @overload
    def __getitem__(
        self, key: Literal["attempted_cards"]
    ) -> list[Card]: ...

    @overload
    def __getitem__(
        self, key: Literal["forced_cards"]
    ) -> list[Card]: ...

    def __getitem__(
        self,
        key: Literal["player", "attempted_cards", "forced_cards"],
    ) -> int | list[Card]:
        if key == "player":
            return self.player
        if key == "attempted_cards":
            return self.attempted_cards
        return self.forced_cards


class ScoringSnapshot(SnapshotModel):
    """Round scoring information."""

    declarer_team: int | None
    defender_points: int
    total_defender_points: int
    bottom_card_bonus: int
    bottom_cards: list[Card]

    @overload
    def __getitem__(
        self, key: Literal["declarer_team"]
    ) -> int | None: ...

    @overload
    def __getitem__(self, key: Literal["defender_points"]) -> int: ...

    @overload
    def __getitem__(
        self, key: Literal["total_defender_points"]
    ) -> int: ...

    @overload
    def __getitem__(self, key: Literal["bottom_card_bonus"]) -> int: ...

    @overload
    def __getitem__(
        self, key: Literal["bottom_cards"]
    ) -> list[Card]: ...

    def __getitem__(
        self,
        key: Literal[
            "declarer_team",
            "defender_points",
            "total_defender_points",
            "bottom_card_bonus",
            "bottom_cards",
        ],
    ) -> int | None | list[Card]:
        if key == "declarer_team":
            return self.declarer_team
        if key == "defender_points":
            return self.defender_points
        if key == "total_defender_points":
            return self.total_defender_points
        if key == "bottom_card_bonus":
            return self.bottom_card_bonus
        return self.bottom_cards


class StirringStateSnapshot(SnapshotModel):
    """Public stirring phase state."""

    phase: StirringPhase
    trump_suit: Suit | None
    current_player: int
    declarer_player: int
    exchanging_player: int | None
    exchange_count: int | None

    @overload
    def __getitem__(self, key: Literal["phase"]) -> StirringPhase: ...

    @overload
    def __getitem__(
        self, key: Literal["trump_suit"]
    ) -> Suit | None: ...

    @overload
    def __getitem__(self, key: Literal["current_player"]) -> int: ...

    @overload
    def __getitem__(self, key: Literal["declarer_player"]) -> int: ...

    @overload
    def __getitem__(
        self, key: Literal["exchanging_player"]
    ) -> int | None: ...

    @overload
    def __getitem__(
        self, key: Literal["exchange_count"]
    ) -> int | None: ...

    def __getitem__(
        self,
        key: Literal[
            "phase",
            "trump_suit",
            "current_player",
            "declarer_player",
            "exchanging_player",
            "exchange_count",
        ],
    ) -> StirringPhase | Suit | int | None:
        if key == "phase":
            return self.phase
        if key == "trump_suit":
            return self.trump_suit
        if key == "current_player":
            return self.current_player
        if key == "declarer_player":
            return self.declarer_player
        if key == "exchanging_player":
            return self.exchanging_player
        return self.exchange_count


class BidEventSnapshot(SnapshotModel):
    """Public bid/stir trump declaration event."""

    player: int
    cards: list[Card]
    kind: BidEventKind
    suit: Suit | None
    joker_type: JokerType | None
    count: int

    @overload
    def __getitem__(self, key: Literal["player"]) -> int: ...

    @overload
    def __getitem__(self, key: Literal["cards"]) -> list[Card]: ...

    @overload
    def __getitem__(self, key: Literal["kind"]) -> BidEventKind: ...

    @overload
    def __getitem__(self, key: Literal["suit"]) -> Suit | None: ...

    @overload
    def __getitem__(
        self, key: Literal["joker_type"]
    ) -> JokerType | None: ...

    @overload
    def __getitem__(self, key: Literal["count"]) -> int: ...

    def __getitem__(
        self,
        key: Literal[
            "player", "cards", "kind", "suit", "joker_type", "count"
        ],
    ) -> int | list[Card] | BidEventKind | Suit | JokerType | None:
        if key == "player":
            return self.player
        if key == "cards":
            return self.cards
        if key == "kind":
            return self.kind
        if key == "suit":
            return self.suit
        if key == "joker_type":
            return self.joker_type
        return self.count


class StateSnapshot(SnapshotModel):
    """Full player-facing state snapshot.

    ``action_hints`` is a complete closed hint set when non-empty. An
    empty
    list means no closed hint set is provided, not that the player has
    no
    legal action. Clients must still allow user input where the action
    type
    allows free card selection; the backend remains authoritative.
    """

    phase: RoundPhase
    player_hand: list[Card]
    player_hand_counts: list[int]
    bottom_cards: list[Card]
    trump_suit: Suit | None
    trump_rank: Rank
    declarer_team: int | None
    declarer_player: int | None
    defender_points: int
    trick: TrickSnapshot | None
    last_completed_trick: CompletedTrickSnapshot | None
    defender_point_cards: list[Card]
    failed_throw: FailedThrowSnapshot | None
    action_hints: list[list[Card]]
    awaiting_action: AwaitingAction | None
    scoring: ScoringSnapshot | None
    winning_team: int | None
    team0_level: Rank
    team1_level: Rank
    bid_events: list[BidEventSnapshot]
    bid_winner: BidEventSnapshot | None
    stirring_state: StirringStateSnapshot | None
    next_round_confirmed: list[int]

    @overload
    def __getitem__(self, key: Literal["phase"]) -> RoundPhase: ...

    @overload
    def __getitem__(
        self, key: Literal["player_hand"]
    ) -> list[Card]: ...

    @overload
    def __getitem__(
        self, key: Literal["player_hand_counts"]
    ) -> list[int]: ...

    @overload
    def __getitem__(
        self, key: Literal["bottom_cards"]
    ) -> list[Card]: ...

    @overload
    def __getitem__(
        self, key: Literal["trump_suit"]
    ) -> Suit | None: ...

    @overload
    def __getitem__(self, key: Literal["trump_rank"]) -> Rank: ...

    @overload
    def __getitem__(
        self, key: Literal["declarer_team"]
    ) -> int | None: ...

    @overload
    def __getitem__(
        self, key: Literal["declarer_player"]
    ) -> int | None: ...

    @overload
    def __getitem__(self, key: Literal["defender_points"]) -> int: ...

    @overload
    def __getitem__(
        self, key: Literal["trick"]
    ) -> TrickSnapshot | None: ...

    @overload
    def __getitem__(
        self,
        key: Literal["last_completed_trick"],
    ) -> CompletedTrickSnapshot | None: ...

    @overload
    def __getitem__(
        self, key: Literal["defender_point_cards"]
    ) -> list[Card]: ...

    @overload
    def __getitem__(
        self, key: Literal["failed_throw"]
    ) -> FailedThrowSnapshot | None: ...

    @overload
    def __getitem__(
        self, key: Literal["action_hints"]
    ) -> list[list[Card]]: ...

    @overload
    def __getitem__(
        self, key: Literal["awaiting_action"]
    ) -> AwaitingAction | None: ...

    @overload
    def __getitem__(
        self, key: Literal["scoring"]
    ) -> ScoringSnapshot | None: ...

    @overload
    def __getitem__(
        self, key: Literal["winning_team"]
    ) -> int | None: ...

    @overload
    def __getitem__(self, key: Literal["team0_level"]) -> Rank: ...

    @overload
    def __getitem__(self, key: Literal["team1_level"]) -> Rank: ...

    @overload
    def __getitem__(
        self, key: Literal["bid_events"]
    ) -> list[BidEventSnapshot]: ...

    @overload
    def __getitem__(
        self, key: Literal["bid_winner"]
    ) -> BidEventSnapshot | None: ...

    @overload
    def __getitem__(
        self, key: Literal["stirring_state"]
    ) -> StirringStateSnapshot | None: ...

    @overload
    def __getitem__(
        self, key: Literal["next_round_confirmed"]
    ) -> list[int]: ...

    def __getitem__(
        self,
        key: Literal[
            "phase",
            "player_hand",
            "player_hand_counts",
            "bottom_cards",
            "trump_suit",
            "trump_rank",
            "declarer_team",
            "declarer_player",
            "defender_points",
            "trick",
            "last_completed_trick",
            "defender_point_cards",
            "failed_throw",
            "action_hints",
            "awaiting_action",
            "scoring",
            "winning_team",
            "team0_level",
            "team1_level",
            "bid_events",
            "bid_winner",
            "stirring_state",
            "next_round_confirmed",
        ],
    ) -> (
        RoundPhase
        | list[Card]
        | list[int]
        | Suit
        | Rank
        | int
        | TrickSnapshot
        | CompletedTrickSnapshot
        | FailedThrowSnapshot
        | list[list[Card]]
        | AwaitingAction
        | ScoringSnapshot
        | list[BidEventSnapshot]
        | BidEventSnapshot
        | StirringStateSnapshot
        | None
    ):
        if key == "phase":
            return self.phase
        if key == "player_hand":
            return self.player_hand
        if key == "player_hand_counts":
            return self.player_hand_counts
        if key == "bottom_cards":
            return self.bottom_cards
        if key == "trump_suit":
            return self.trump_suit
        if key == "trump_rank":
            return self.trump_rank
        if key == "declarer_team":
            return self.declarer_team
        if key == "declarer_player":
            return self.declarer_player
        if key == "defender_points":
            return self.defender_points
        if key == "trick":
            return self.trick
        if key == "last_completed_trick":
            return self.last_completed_trick
        if key == "defender_point_cards":
            return self.defender_point_cards
        if key == "failed_throw":
            return self.failed_throw
        if key == "action_hints":
            return self.action_hints
        if key == "awaiting_action":
            return self.awaiting_action
        if key == "scoring":
            return self.scoring
        if key == "winning_team":
            return self.winning_team
        if key == "team0_level":
            return self.team0_level
        if key == "team1_level":
            return self.team1_level
        if key == "bid_events":
            return self.bid_events
        if key == "bid_winner":
            return self.bid_winner
        if key == "stirring_state":
            return self.stirring_state
        return self.next_round_confirmed
