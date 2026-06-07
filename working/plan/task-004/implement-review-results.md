# Implement Review Results: Task-004

## Task Review Issues

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
