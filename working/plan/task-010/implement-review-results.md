# Implement Review Results: Task-010

## Task Review Issues

### TR-001: Dead None-check on lead_cards in follow-suit validation (line 117)
- Status: Resolved
- Description: At trick.py:117, the code reads `if lead_cards is None or len(lead_cards) == 0:`. The `is None` check is dead code because `CompletedTrickSlot.cards` is typed as `list[Card]` (not Optional) in types.py:95, and all slots are initialized with `cards=[]` in `create_trick` (line 73) or `cards=list(cards)` in `play` (line 145). The `lead_cards` variable is `lead_slot.cards` (line 116), which is always a list, never None. This is the same class of issue reported in CQ-009 but in the `play()` function rather than `_resolve()`. The `_resolve` function was fixed (lines 188, 199, 207 now only check `len(...) == 0`), but line 117 in `play()` was missed.
- Decision Reason: Removed `is None` check to match the pattern already used in `_resolve()`. `lead_cards` is always `list[Card]`, never None.

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

### CQ-008: Unused import CompletedTrick in test file (ruff F401)
- Status: Resolved
- Description: At trick_tests.py:4, `CompletedTrick` is imported from `server.sm.types` but never used anywhere in the test file. Ruff flags this as F401. While this does not affect correctness, unused imports clutter the module namespace and violate standard clean-code practices. The ruff linter explicitly catches this.
- Decision Reason:

### CQ-009: Dead None-checks in _resolve for non-optional CompletedTrickSlot.cards
- Status: Resolved
- Description: At trick.py:188, 199, 207, `_resolve()` checks `if lead_cards is None or len(lead_cards) == 0` and similar for `p_cards` and `best_cards`. However, `CompletedTrickSlot.cards` is typed as `list[Card]` (not Optional) in types.py:95, and all slots are initialized with `cards=[]` in `create_trick` (line 73) or `cards=list(cards)` in `play` (line 145). The None checks are dead code -- they can never be True. This makes the error messages on lines 189, 200, 208 unreachable, adding confusion about actual invariants. The `len(...) == 0` check could theoretically be reached if a slot somehow ended up with an empty list, but the None part is dead.
- Decision Reason:

### CQ-010: assert statement in play() follow-suit validation (line 119) not converted to ValueError
- Status: Resolved
- Description: At trick.py:119, `assert state.lead_type is not None, "lead_type must be set in FOLLOWING phase"` is the last remaining assert in production code. CQ-004 was marked Resolved and _resolve() was cleaned up, but this assert in play() was missed. Under `python -O`, this assert is stripped entirely, meaning the code would pass a None lead_type into `PlayAction(type=None, ...)` on line 120, causing a confusing downstream error rather than a clear message. Should be `if state.lead_type is None: raise ValueError("lead_type must be set in FOLLOWING phase")`.
- Decision Reason: Replaced assert with explicit ValueError guard: `if state.lead_type is None: raise ValueError(...)`. Matches the defensive coding pattern used elsewhere in the module.

### CQ-011: # type: ignore[arg-type] on CompletedTrick construction (line 232) indicates unverified invariant
- Status: Resolved
- Description: At trick.py:232, `lead_type=state.lead_type,  # type: ignore[arg-type]` suppresses the type error because `state.lead_type` is `PlayType | None` but `CompletedTrick.lead_type` is `PlayType`. This `# type: ignore` is a workaround for the type system being unable to prove the invariant that `lead_type` is always set by the time `_resolve()` is called. This could be resolved by adding an explicit `if state.lead_type is None: raise ValueError(...)` guard before the CompletedTrick construction, which both documents the invariant AND removes the need for the type ignore comment. This is the same class of issue as CQ-010 -- both stem from the missing guard.
- Decision Reason: Added `if state.lead_type is None: raise ValueError(...)` guard at the top of `_resolve()`. This both documents the invariant and enables Pyright to narrow the type, eliminating the type: ignore comment.

### CQ-012: Pyright type error in trick_tests.py deck parameter (line 21)
- Status: Resolved
- Description: At trick_tests.py:21, the `_card()` helper passes `deck` as `int` to `Card(deck=...)`, but `Card.deck` is typed as `Literal[1, 2]`. Pyright reports: "Argument of type 'int' cannot be assigned to parameter 'deck' of type 'Literal[1, 2]'". The function signature at line 11 declares `deck: int = 1` which should be `deck: Literal[1, 2] = 1`. While all current callers pass literal 1 or 2, the broad type allows invalid values (e.g., `deck=3`) to be passed at the call site without a type error, and pyright fails on the module. This is a pyright error (exit code 1) and should be fixed.
- Decision Reason: Changed `deck: int = 1` to `deck: Literal[1, 2] = 1` and added `from typing import Literal` import. Matches Card's type constraint exactly.

### CQ-013: Missing test for tractor lead follow-suit validation through trick module
- Status: Resolved
- Description: The task test list specifies tests for single and pair follow-suit, but there is no test verifying tractor follow-suit validation through the trick module. When a tractor is led, followers must play a matching-length tractor of the same effective suit if possible. While the underlying `get_legal_plays` handles this, the trick module's integration with tractor follow-suit is untested. This leaves a gap in coverage for the trick module's delegation to play_rules for complex lead types.
- Decision Reason: Added `test_play_follow_tractor_must_follow_suit` that leads a hearts tractor (4 cards), verifies follower cannot play off-suit spades tractor, and must play on-suit hearts tractor.

### CQ-014: _resolve mutates state in-place but play() returns it -- inconsistent with stated immutable pattern
- Status: Resolved
- Description: At trick.py:175-178, `play()` creates a new state on line 159, then immediately calls `_resolve(new_state)` which mutates `new_state.phase` and `new_state.result` in-place (lines 238-244). While this works correctly (the mutation happens before the return), it means `_resolve()` is a side-effecting function that mutates its argument, contradicting the "immutable state machine" docstring on line 93. This was originally flagged as CQ-002 (which was about play() itself). The play() function was fixed to build a new state, but _resolve() still uses mutation. The mutation is technically safe here because no external reference to new_state exists yet, but it creates a fragile pattern -- if anyone ever calls _resolve() on a state they hold a reference to, that reference silently changes.
- Decision Reason: Refactored `_resolve()` to return a new `TrickState` via `state.model_copy(update=...)` instead of mutating in-place. The `play()` function now uses the return value: `return _resolve(new_state)`. This maintains referential transparency throughout.
