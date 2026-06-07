# Implement Review Results: Task-011

## Task Review Issues

### TR-001: RoundState missing `current_lead_player` field specified in task
- Status: Resolved
- Description: The task spec (Step 3, RoundState definition) explicitly requires a `current_lead_player (int|None)` field on `RoundState`. This field is absent from the implemented `RoundState` in `server/sm/round_sm.py`. The field would track the lead player of the current trick at the round level. While the lead player is available via `trick_state.lead_player`, the task specifically requested this field on `RoundState` and it was not implemented.
- Decision Reason:

### TR-002: Dead `RoundResult` class defined in round_sm.py but never used
- Status: Resolved
- Description: `server/sm/round_sm.py` defines its own `RoundResult` class (lines 43-55) with the same fields as `scoring.RoundResult`. However, the `RoundState.result` field is typed as `scoring.RoundResult | None` (line 77), and `get_round_result` returns `scoring.RoundResult | None` (line 304). The local `RoundResult` class is never referenced anywhere in the module. It is dead code that should either be removed or used as the canonical type (re-exporting from scoring, or replacing the `scoring.RoundResult` reference).
- Decision Reason:

### TR-003: `test_scoring_produces_round_result` has no meaningful assertion
- Status: Resolved
- Description: The test `test_scoring_produces_round_result` in `TestScoringPhase` (line 382) only asserts that `is_round_complete(state) is False` when in SCORING phase, then has a comment noting "SCORING is a transient state." In practice the state transitions directly to COMPLETE during `play()`, so the `if state.phase == "SCORING"` branch is never entered (the test passes vacuously). The test should assert that after playing all tricks, `is_round_complete(state)` is True and `get_round_result(state)` is not None with expected fields, similar to `test_round_full_round_flow`.
- Decision Reason:

### TR-004: Test file was modified from task spec to work around issues
- Status: Resolved
- Description: Per `test-case-changes.md`, three modifications were made to the test file: (1) `test_stir_during_stirring` was rewritten to dynamically find valid cards instead of using fabricated IDs, (2) `test_playing_all_tricks_to_scoring` assertion was relaxed from `"SCORING"` to `("SCORING", "COMPLETE")`, (3) `_play_first_legal` was changed from dict access to attribute access for `CompletedTrickSlot`. Changes (2) and (3) are justified corrections. Change (1) is reasonable but the test now uses `pytest.skip` when no trump-rank pair is found, meaning the test can silently not run under certain deck shuffles. The seed-dependent behavior means this test is non-deterministic across different deck configurations.
- Decision Reason:

### TR-005: `test_playing_trick_resolved_starts_next` does not assert trick resolution
- Status: Resolved
- Description: The test `test_playing_trick_resolved_starts_next` (line 331) plays up to 4 legal plays but never asserts that the trick actually resolved, that a new trick started, or that the next trick's lead player matches the previous trick's winner. The test name implies it verifies "trick resolved starts next" but the assertions are missing. The loop just plays until the trick resolves or 4 plays are made, with no post-condition checks.
- Decision Reason:

## Code Quality Issues

### CQ-001: Unused helper function `_shuffled_deck` in test file
- Status: Resolved
- Description: The function `_shuffled_deck` defined at line 13 of `server/sm/round_sm_tests.py` is never called anywhere in the test file. It is dead code that should be removed to reduce confusion and maintain test code cleanliness.
- Decision Reason: Removed unused function. No test depends on it.

### CQ-002: Unused import `PlayType` in test file
- Status: Resolved
- Description: `PlayType` is imported on line 5 of `server/sm/round_sm_tests.py` but never used anywhere in the file. This is an unused import that should be removed. Running `ruff check` would flag this.
- Decision Reason: Removed unused import. Also removed unused `create_decks` import.

### CQ-003: Redundant unreachable condition in `_transition_to_stirring` Case B
- Status: Resolved
- Description: In `server/sm/round_sm.py` line 319, the expression `state.start_player if declarer_team is None else state.last_declarer_player` contains a condition `declarer_team is None` that is always False at that point. This code is inside the `else` branch of `if declarer_team is None:` (line 310), so `declarer_team` is guaranteed to be not None. The ternary always evaluates to `state.last_declarer_player`. The redundant condition suggests the developer was uncertain about the control flow and could confuse future maintainers. It should be simplified to just `declarer_player = state.last_declarer_player`.
- Decision Reason: Simplified to `declarer_player = state.last_declarer_player` since `declarer_team` is guaranteed non-None in the else branch.

### CQ-004: Redundant `if declarer_team is not None` check in `_transition_to_stirring`
- Status: Resolved
- Description: In `server/sm/round_sm.py` line 327, the check `if declarer_team is not None:` is always True at that point. In Case A (line 310-313), `declarer_team` is set to `winner_team`. In Case B (line 314-322), `declarer_team` was already not None (since we entered the `else` branch of `if declarer_team is None:`). The guard is redundant and should be removed or replaced with a comment explaining why `declarer_team` is guaranteed to be set.
- Decision Reason: Removed redundant guard. `declarer_team` is always set at that point in both Case A and Case B.

### CQ-005: Inconsistent state mutation pattern in `pass_stir` and `play`
- Status: Resolved
- Description: In `server/sm/round_sm.py`, the `pass_stir` function (line 174) and `play` function (lines 271, 274) directly mutate `new_state` attributes after creating it via `model_copy`. This is inconsistent with the rest of the module which uses `model_copy(update=...)` for all state transitions. While `frozen=False` allows direct mutation, the inconsistency makes the code harder to reason about and violates the immutable state machine pattern used elsewhere.
- Decision Reason: Replaced direct attribute mutation with `model_copy(update=...)` in both pass_stir and play functions for consistency.

### CQ-006: `import random` placed inside function body instead of module top-level
- Status: Resolved
- Description: In `server/sm/round_sm.py` line 76, `import random` is placed inside the `create_round` function body instead of at the module's top-level imports. While this works, it is unconventional for a standard library module. Module-level imports are the Python convention and make dependencies more visible.
- Decision Reason: Moved `import random` to module top-level imports per Python conventions.

### CQ-007: No test coverage for Case B (subsequent round with bid winner) transition logic
- Status: Resolved
- Description: The `_transition_to_stirring` function's Case B (subsequent round with a bid winner) is never tested. Specifically, there is no test that creates a round with a pre-determined `declarer_team`, completes the deal-bid phase with a winner on that team, and verifies that `declarer_player` is set to the winner and `declarer_team` remains unchanged. Similarly, the "invalid winner" path (line 317-320, where the winner is not on the declarer team) is untested.
- Decision Reason: Added `test_round_subsequent_round_bid_winner_on_team` test that creates a round with `declarer_team=0`, uses seed 3 to ensure a team 0 player has trump rank cards, reveals one, and asserts `declarer_team` is unchanged and `declarer_player` is set.

### CQ-008: `test_deal_bid_to_stirring_with_winner` can pass vacuously
- Status: Resolved
- Description: The test `test_deal_bid_to_stirring_with_winner` (line 183) uses a conditional assertion `if state.deal_bid_state.bid_winner is not None:` on line 191. If the `_complete_deal_bid_with_reveal` helper fails to produce a winner (which is unlikely but possible with certain random deck orderings since no fixed seed is set), the entire assertion block is skipped and the test passes vacuously. The test should either set a fixed seed to guarantee a winner is produced, or use `assert` unconditionally to fail if the reveal didn't work.
- Decision Reason: Added `random.seed(42)` before `create_round` and replaced conditional assertions with unconditional ones. Also fixed `test_round_first_round_declarer_from_bid` similarly.

### CQ-009: Overly broad regex match in `test_stir_cards_not_in_hand_rejected`
- Status: Resolved
- Description: In `server/sm/round_sm_tests.py` line 271, `pytest.raises(ValueError, match="hand|not in")` uses a regex that matches either "hand" or "not in". The actual error message is `"Card {card.id} not in hand of player {cur}"` which contains both substrings. While this works, the regex is overly broad and would match unintended error messages. A more specific pattern like `"not in hand"` would be more precise.
- Decision Reason: Changed regex to `"not in hand"` to match only the specific error message.

### CQ-010: Unused import `next_player_ccw` in round_sm.py
- Status: Resolved
- Description: `server/sm/round_sm.py` imports `next_player_ccw` on line 20 from `server.sm.constants`, but this function is never used anywhere in the module body. This is an unused import that should be removed to maintain code cleanliness.
- Decision Reason: Removed unused import.

### CQ-011: Missing test coverage for Case B "invalid winner" path in `_transition_to_stirring`
- Status: Resolved
- Description: The `_transition_to_stirring` function's Case B has an "invalid winner" branch (lines 325-328 in round_sm.py) where `winner_team != declarer_team`. When this occurs, the winner is ignored, `declarer_player` falls back to `state.last_declarer_player`, and `trump_suit` is set to `None`. This branch has no test coverage. The existing test `test_round_subsequent_round_bid_winner_on_team` only tests the happy path where the winner IS on the declarer team. A test should be added that creates a round with a pre-determined `declarer_team`, ensures a player from the OTHER team wins the bid, and verifies the fallback behavior.
- Decision Reason: Added `test_round_subsequent_round_bid_winner_wrong_team` test that creates a round with `declarer_team=0`, uses seed 3 to ensure a team 1 player has trump rank cards, reveals one, and asserts `declarer_team` is unchanged, `declarer_player` falls back to `last_declarer_player`, and `trump_suit` is None.

### CQ-012: Wasted intermediate state in `pass_stir` when phase is COMPLETE
- Status: Resolved
- Description: In `server/sm/round_sm.py`, the `pass_stir` function creates `new_state` via `model_copy` on line 171, but when `new_ss.phase == "COMPLETE"` (line 173), a second `model_copy` on line 174 immediately overwrites it. The first `model_copy` is wasted work in the COMPLETE path. This could be restructured to avoid the redundant allocation by moving the `model_copy` after the phase check.
- Decision Reason: Restructured to check phase before creating new_state, returning early on COMPLETE path and returning the single model_copy in the non-complete path.

### CQ-013: Wasted intermediate state in `play` when trick resolves
- Status: Resolved
- Description: In `server/sm/round_sm.py`, the `play` function creates `new_state` via `model_copy` on lines 267-270, but when `new_trick.phase == "RESOLVED"` (line 272), a second `model_copy` on lines 277-282 immediately overwrites it with the same base fields plus additional trick history and defender points. The first `model_copy` is wasted work in the resolved path. The allocation could be deferred until after the phase check.
- Decision Reason: Restructured to check phase first, creating the complete state only in the RESOLVED path, and a simpler state in the non-resolved path.

### CQ-014: Unused variable `lead_player` flagged by ruff F841 in test
- Status: Resolved
- Description: In `server/sm/round_sm_tests.py` line 338, the variable `lead_player` is assigned from `trick.lead_player` but never referenced. Ruff reports F841. The variable appears to have been intended for assertions comparing the new trick's lead player against the first trick's winner, but the actual assertions use `state.trick_history[0].winner` directly instead. The dead assignment should be removed.
- Decision Reason: Removed unused variable assignment.
