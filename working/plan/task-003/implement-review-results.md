# Implement Review Results: Task-003

## Task Review Issues

### TR-001: Player model missing frozen=True
- Status: Resolved
- Description: The task spec states "All Pydantic models use `frozen=True` for immutability." The Player model (server/sm/types.py line 62-68) does not have `model_config = ConfigDict(frozen=True)`. All other 5 Pydantic models (PlayAction, BidEvent, StirAction, CompletedTrickSlot, CompletedTrick) correctly have frozen=True. While Player's hand is intended to be mutated during gameplay, the spec explicitly requires all models to be frozen. This should either be addressed by adding frozen=True or explicitly documented as a deliberate deviation.
- Decision Reason: Player intentionally left unfrozen because `hand: list[Card]` must be mutable during gameplay (spec.md shows hand mutations in deal, exchange, and trick phases). All other models remain frozen. Added test_player_hand_mutable to document this deliberate deviation.

## Code Quality Issues

### CQ-001: BidEvent.kind and StirAction.kind use unconstrained str instead of Literal types
- Status: Resolved
- Description: The spec defines `kind: "trump_rank" | "joker"` for BidEvent (spec.md line 315) and `kind: "stir" | "pass"` for StirAction (spec.md line 488). The implementation uses plain `str` for both fields (server/sm/types.py:43, server/sm/types.py:55). This allows any arbitrary string value (e.g., `BidEvent(kind="invalid")` passes validation), defeating type safety in a state machine architecture. Using `Literal["trump_rank", "joker"]` and `Literal["stir", "pass"]` would enforce valid values at construction time and provide better pyright static analysis.
- Decision Reason: Changed to `Literal["trump_rank", "joker"]` and `Literal["stir", "pass"]`. Added test_bid_event_kind_literal and test_stir_action_kind_literal to verify invalid values are rejected.

### CQ-002: Player.team allows any int instead of constraining to 0|1 per spec
- Status: Resolved
- Description: The spec defines `team: 0 | 1` for Player (spec.md line 20). The implementation uses plain `int` (server/sm/types.py:67), allowing values like `team=2` or `team=99`. This should use `Literal[0, 1]` to match the spec and prevent invalid team assignments that would silently propagate through state machines.
- Decision Reason: Changed to `Literal[0, 1]`. Added test_player_team_literal to verify invalid team values are rejected.

### CQ-003: Tests do not verify frozen model immutability behavior
- Status: Resolved
- Description: All 13 tests in types_tests.py verify construction and field access but none test that frozen models actually prevent attribute mutation (e.g., `action.type = PlayType.PAIR` should raise). The frozen=True constraint is a key design requirement from the spec ("All Pydantic models use `frozen=True` for immutability"). Without tests verifying immutability, a future change that removes frozen=True would pass all existing tests silently.
- Decision Reason: Added frozen immutability tests for all 5 frozen models: test_play_action_frozen, test_bid_event_frozen, test_stir_action_frozen, test_completed_trick_frozen, test_completed_trick_slot_frozen. Total test count increased from 13 to 23.

### CQ-004: BidEvent.joker_type also uses unconstrained str instead of Literal type
- Status: Resolved
- Description: The spec defines `joker_type: "big" | "small" | None` for BidEvent (spec.md line 317). The implementation uses `str | None` (server/sm/types.py:45), allowing arbitrary strings like `"big"` but also `"INVALID"`. This should use `Literal["big", "small"] | None` to match the spec, consistent with the fix needed for the `kind` fields (CQ-001).
- Decision Reason: Changed to `Literal["big", "small"] | None`. Added test_bid_event_joker_type_literal to verify invalid joker_type values are rejected.
