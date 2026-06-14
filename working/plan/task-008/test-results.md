# Test Results: Task-008

## Status
UNEXPECTED

## Test Results

| Test Case | Result | Expected | Blocked | Root Cause |
|-----------|--------|----------|---------|---------|
| test_create_game_returns_201 | PASS | PASS | no | - |
| test_health_check | PASS | PASS | no | - |
| test_list_games | PASS | PASS | no | - |
| test_delete_game_closes_ws | PASS | PASS | no | - |
| test_connect_nonexistent_game | PASS | PASS | no | - |
| test_connection_takeover | PASS | PASS | no | - |
| test_reconnect_resumes_game | PASS | PASS | no | - |
| test_full_game | FAIL | PASS | no | Disconnector thread + _interleave_error timing hang |

## Unfixed Blocked Tests

### test_full_game
- File: tests/test_full_game_ws.py::test_full_game
- Expected: PASS — play complete game from start to GAME_OVER through WS protocol
- Actual: FAIL — test hangs indefinitely
- Root cause: The test's disconnector thread fires every 0.5-2s, disrupting do_action() calls inside _interleave_error. Each disruption triggers WS reconnection which sends {"type":"next_round","seq":0}, producing extra state pushes. These extra pushes cause the test's receive_state() loop to re-enter _interleave_error on every bid/stir/play turn. Even with tested_interleave flags, the extra state pushes flood the WS buffer and the loop spins indefinitely waiting for a phase transition that never comes because the game never advances past the first bid turn.
- 3 attempted approaches:
  1. Added "unknown" to the unknown action assertion keyword list — fixed assertion, no effect on hang
  2. Added tested_interleave flags to prevent re-entry — reduced interleave calls but test still hangs because the extra state pushes from reconnection confuse the receive_state() loop even after the interleave is skipped
  3. Added do_bid_pass() fallback when do_bid() fails — prevents one deadlock scenario but doesn't fix the fundamental timing issue where disconnector reconnections produce state pushes that keep the loop spinning
- Resolution: Task needs redesign. The disconnector thread's reconnection strategy (sending next_round with seq=0 on every reconnect) produces state pushes that interfere with the test's state machine loop. Either remove the disconnector, change the reconnection to not produce state pushes, or redesign the test loop to drain all stale state pushes before acting.

## Summary
- EXPECTED (Result=Expected, Blocked=no): 7
- UNEXPECTED (Result≠Expected, Blocked=no): 1
- Blocked (Blocked=yes): 0
