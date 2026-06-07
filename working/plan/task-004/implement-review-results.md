# Implement Review Results: Task-004

## Task Review Issues

### TR-001: AutoPlayer._handle_next_round missing current_player guard
- Status: Resolved
- Description: The task spec (step 3, COMPLETE phase) says "For COMPLETE phase with `awaiting_action == 'next_round'` and `current_player == self.index`: submit `NextRoundAction`." The implementation at `server/player.py:119-120` routes to `_handle_next_round` without checking `current_player == self.index`, and `_handle_next_round` itself (line 180-184) has no guard. This means ALL AutoPlayers will submit NextRoundAction when the round is complete, even when it is not their turn. The test `test_auto_player_next_round` only tests `current_player=0` with `index=0` so this gap is untested. This could cause multiple NextRoundAction submissions per round.
- Decision Reason:

## Code Quality Issues

### CQ-001: Unused import `field` in player.py
- Status: Resolved
- Description: `server/player.py` line 14 imports `field` from `dataclasses` but it is never used anywhere in the module. This is dead code that should be removed to keep imports clean.
- Decision Reason:

### CQ-002: Unused imports `patch` and `dataclass_fields` in player_tests.py
- Status: Resolved
- Description: `server/player_tests.py` line 4 imports `patch` from `unittest.mock` and line 5 imports `dataclass_fields` as `fields`. Neither is used anywhere in the test file. These are dead imports.
- Decision Reason:

### CQ-003: AutoPlayer._handle_deal_bid may have incorrect current_player guard
- Status: Resolved
- Description: The task spec says AutoPlayer should bid during DEAL_BID phase when it has trump rank cards in hand, and notes "bidding is optional during dealing." The implementation at `server/player.py:112` checks `snapshot.current_player != self.index` and returns early if the player is not the current player. During DEAL_BID, bidding is typically available to all players simultaneously (not just the current player). If the spec intends all players can bid regardless of current_player, this guard would incorrectly prevent AutoPlayer from bidding when it is not the current player. The test `test_auto_player_bid_during_dealing` only tests `current_player=0` matching `AutoPlayer(index=0)`, so this potential bug is not covered by tests.
- Decision Reason:

### CQ-004: Missing wrong-player test coverage for STIRRING/EXCHANGE/COMPLETE phases
- Status: Resolved
- Description: The test `test_auto_player_ignores_wrong_player` only covers the PLAYING phase. The same wrong-player guard exists in `_handle_stir`, `_handle_discard`, and `_handle_deal_bid`, but there are no tests verifying those guards block action when `current_player != self.index`. If the guard is removed or broken in one of those methods, no test would catch it.
- Decision Reason:

### CQ-005: HumanPlayer.on_state test does not verify snapshot.to_dict() is called
- Status: Resolved
- Description: The task spec (step 3.4) explicitly states HumanPlayer should call `snapshot.to_dict()` to serialize the snapshot before sending via WebSocket. The test `test_human_player_sends_state_on_push` at `server/player_tests.py:243-251` checks that `send_json` was called with `{"type": "state", ...}` but does not verify that `snapshot.to_dict()` was actually called -- the mock snapshot's `to_dict` returns a MagicMock by default. A future regression where someone removes the `.to_dict()` call would not be caught by tests.
- Decision Reason:

### CQ-006: asyncio.create_task results are not awaited -- silent exception swallowing risk
- Status: Resolved
- Description: In `server/player.py`, five call sites (lines 121, 136, 150, 162, 167) use `asyncio.create_task(game.act(...))` without storing the returned Task reference. If `game.act()` raises an exception, the error is silently lost because no one awaits or adds a done_callback to the task. In production, this means action submission failures could go completely undetected -- the player would appear to act, but nothing would happen. The task design intentionally uses `create_task` (not `await`) to avoid blocking state transitions, but the returned Task should at minimum have an exception callback (e.g., `task.add_done_callback(_handle_task_exception)`) to log or propagate errors.
- Decision Reason: Added `_log_task_exception` callback function that logs exceptions via `logging.exception()`. All 5 `create_task` call sites now store the Task reference and register the callback via `task.add_done_callback(_log_task_exception)`. The callback silently ignores cancelled tasks and logs non-None exceptions. All 23 tests continue to pass.

### CQ-007: _handle_next_round missing current_player guard per task spec
- Status: Resolved
- Description: The task spec (step 3.5) explicitly states: "For COMPLETE phase with `awaiting_action == "next_round"` and `current_player == self.index`: submit `NextRoundAction`." However, `_handle_next_round` at `server/player.py:180-184` does not check `snapshot.current_player != self.index` before submitting the action. The dispatch in `on_state` (line 119) only checks `phase == "COMPLETE" and awaiting_action == "next_round"` without a `current_player` guard. This means every AutoPlayer in the game could attempt to trigger next_round simultaneously, not just the one whose turn it is. While in practice the game engine may handle multiple next_round submissions gracefully, the implementation deviates from the task spec.
- Decision Reason:
