# Implement Review Results: Task-018

## Task Compliance Issues

## Code Quality Issues

### CQ-001: getLastError() public method has no test coverage
- Status: Resolved
- Description: `GameLoop.getLastError()` (game-loop.ts:47-49) is a public method that returns the stored error message, but no test exercises it. The test `test_handleMessage_error_does_not_update_state` only verifies that state is not updated and render is not called -- it never calls `getLastError()` to confirm the error was actually stored. This means a regression (e.g., removing the `this.lastError = msg.message` assignment) would go undetected.
- Decision Reason: Added test_handleMessage_error_stores_error_message to verify getLastError() returns null before and the correct message after an error.

### CQ-002: Error message storage not verified in error handling test
- Status: Resolved
- Description: The test `test_handleMessage_error_does_not_update_state` (game-loop.test.ts:161-169) checks that the error does not update state or trigger a render, but it does not verify that `getLastError()` returns the error message "something went wrong". The implementation stores the error (game-loop.ts:33) but the test only validates the negative case (no state update). A follow-up assertion like `assertEquals(loop.getLastError(), "something went wrong")` would confirm the error is correctly captured.
- Decision Reason: Added test_handleMessage_error_stores_error_message which verifies getLastError() returns the correct error message.

### CQ-003: Default branch in computeInteractionMode switch is not tested
- Status: Resolved
- Description: The `default` case in `computeInteractionMode` (game-loop.ts:74-75) returns `null` for any unknown `awaiting` value. No test passes an `awaiting` value that is not one of "stir", "discard", "play", or "next_round" with a matching human player to exercise this branch. For example, sending `awaiting = "unknown_action"` with `current_player: 3` should yield `null`. This untested branch means a future typo or new awaiting value would silently fall through without test validation.
- Decision Reason: Added test_handleMessage_unknown_awaiting_returns_null which sends awaiting="unknown_action" with human player and verifies null interaction mode.

### CQ-004: Task module design lists start/stop as public operations but they are not implemented
- Status: Don't Fix
- Description: The task module design (task.md) declares `public operations: GameLoop.start, GameLoop.stop, GameLoop.handleMessage`, but the implementation (game-loop.ts) only has `handleMessage` and `getLastError`. The `start` and `stop` methods are entirely absent. If the module is intended to manage WebSocket subscription lifecycle (subscribe on start, unsubscribe on stop), this is a missing feature. If these were intentionally omitted from the implementation steps, the module design spec is inaccurate and should be reconciled.
- Decision Reason:
  Tried: (1) Review task.md steps - none mention start/stop implementation, only handleMessage is specified.
  Tried: (2) Check if start/stop are needed for current functionality - handleMessage is called externally by WsClient, no lifecycle management needed within GameLoop itself.
  Tried: (3) Consider adding stub implementations - would violate YAGNI and add untested code.
  Resolution: start/stop are lifecycle operations for future WebSocket subscription management. Task steps only require handleMessage. Adding unimplemented methods would be over-engineering. The module design should be updated to remove start/stop from public operations, or they can be added when WS lifecycle management is implemented in a future task.

### CQ-005: Module-level mutable test state creates fragile test isolation
- Status: Resolved
- Description: The test file (game-loop.test.ts:38-39) uses module-level mutable variables `lastRenderedSnapshot` and `lastInteractionMode` that are reset manually at the start of some tests but not others. Tests like `test_handleMessage_stirring_human` (line 67) set `lastInteractionMode = null` but not `lastRenderedSnapshot = null`, relying on the prior test's state having been set. If Deno ever runs these tests in parallel or the execution order changes, tests will read stale values from other tests. Each test should fully reset its own state, or use a setup/teardown pattern.
- Decision Reason: Updated all test functions to reset both lastRenderedSnapshot and lastInteractionMode at the start of each test for proper isolation.
