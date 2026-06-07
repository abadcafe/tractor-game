# Implement Review Results: Task-010

## Task Review Issues

## Code Quality Issues

### CQ-001: TrickState.phase should be Literal["LEADING", "FOLLOWING", "RESOLVED"] not str
- Status: Resolved
- Description: At trick.py:48, `phase: str` uses a plain string type with comment-only annotation. Every state machine should enforce phase as a Literal type. A plain `str` allows invalid phase values like `"INVALID"` to be set without any type-checking or runtime error. This is inconsistent with the architecture pattern and allows silent corruption of state machine invariants.
- Decision Reason:

### CQ-002: play() mutates state in-place, violating immutable state machine architecture
- Status: Resolved
- Description: At trick.py:83-152, `play()` mutates the input `TrickState` (slots, hands, played, cur, lead_type, phase) and returns the same object. The docstring on line 92 explicitly acknowledges this ("mutates and returns for efficiency"). However, the project architecture specifies "serial state machines with no shared mutable state. Each machine takes input, returns output." All other state machines return new state objects. This in-place mutation means any caller holding a reference to the pre-play state will see it change unexpectedly, breaking referential transparency.
- Decision Reason:

### CQ-003: TrickState.slots uses untyped list[dict] instead of proper model
- Status: Resolved
- Description: At trick.py:51, `slots: list[dict]` stores `{player: int, cards: list[Card] | None}` as raw dicts. The `CompletedTrickSlot` model in types.py already models this exact structure. Using untyped dicts means `state.slots[player]["cards"]` has no type checking, and the multiple `# type: ignore` comments on lines 189, 202, 204 are symptoms of this design choice.
- Decision Reason:

### CQ-004: Six assert statements in production code paths
- Status: Resolved
- Description: At trick.py:111, 159, 160, 169, 176, 189, `assert` is used for invariant checks in production code. `assert` statements are stripped entirely when Python runs with `-O` (optimized mode). For example, `assert lead_cards is not None` on line 111 protects against a crash if the lead cards are somehow None. These should be `if ... is None: raise ValueError(...)` to match the defensive coding pattern already used on lines 94-99.
- Decision Reason:

### CQ-005: Missing test for play() on already-resolved trick
- Status: Resolved
- Description: The code at trick.py:98-99 implements a guard: `if state.phase == "RESOLVED": raise ValueError("Trick is already resolved")`. No test exercises this branch. The `test_play_wrong_player_rejected` test only checks wrong-player validation. A dedicated test should verify that calling `play()` after the trick is resolved raises ValueError.
- Decision Reason:

### CQ-006: test_play_resolve_defender_points_update uses conditional assertion
- Status: Resolved
- Description: At trick_tests.py:233-234, the test uses `if result.winner in (1, 2): assert ...` instead of a deterministic assertion. The test setup has player 1 (team 1, defender) playing ♥A which is the highest card, so winner should deterministically be player 1. The conditional makes this test vacuously pass if the winner is wrong, defeating the purpose of testing defender point accumulation.
- Decision Reason:

### CQ-007: Missing tests for pair follow-suit validation and empty cards edge case
- Status: Resolved
- Description: While single follow-suit is tested (test_play_follow_must_follow_suit), there is no test verifying that pair follow-suit validation works through the trick module (e.g., when lead is a pair and follower must play a pair of the same suit). Additionally, there is no test for calling play() with an empty cards list `[]`, which could cause issues in the resolution logic or infer_play_type.
- Decision Reason:
