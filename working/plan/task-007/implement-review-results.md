# Implement Review Results: Task-007

## Task Review Issues

## Code Quality Issues

### CQ-001: Private field access on HumanPlayer._ws in delete_game endpoint
- Status: Resolved

### CQ-002: Deprecated @app.on_event("startup") used instead of lifespan
- Status: Resolved

### CQ-003: Tests swallow exceptions silently via bare except/try-except-pass in WS action tests
- Status: Resolved

### CQ-004: on_game_over callback does not close WebSocket before registry.delete
- Status: Don't Fix

### CQ-005: No test coverage for DELETE game with active WebSocket connection
- Status: Resolved

### CQ-006: create_game does not await game.run() startup, race condition possible
- Status: Don't Fix

### CQ-007: No state push after game.act() on successful non-game-over actions
- Status: Don't Fix

### CQ-008: Game not cancelled in on_game_over callback before registry.delete
- Status: Don't Fix
- Description: At `server/server.py:70-71`, the `on_game_over` callback only calls `registry.delete(game_id)` but does not call `game.cancel()` to stop the dealing loop background task. When the game reaches GAME_OVER, the dealing loop (if still somehow running) should be stopped. Looking at the actual code flow: `Game.act()` sets `self._cancelled = True` at `server/game.py:260` before starting a new dealing loop in the non-GAME_OVER branch, and in the GAME_OVER branch (line 255-258) there is no new loop started. The dealing loop itself checks `self._cancelled` at `game.py:476` and also breaks when `phase != "DEAL_BID"` at line 479. So the dealing loop will naturally stop. However, the `on_game_over` callback should still call `game.cancel()` defensively to ensure any pending `_dealing_task` is properly cleaned up, matching the pattern used in the `delete_game` endpoint (line 96). Without this, the `_dealing_task` reference in `game._dealing_task` may remain set to a completed-but-not-awaited task, and the game object retains stale state until garbage collection.
- Decision Reason: On re-verification, the dealing loop at game.py:468-492 naturally exits when `phase != "DEAL_BID"` (line 479). By the time GAME_OVER is reached, the game has progressed through all phases and the dealing loop has completed long ago. The `_dealing_task` is a done task. Calling `game.cancel()` would be purely defensive with no functional benefit. The `delete_game` endpoint calls `game.cancel()` because it may interrupt a still-running game, but the `on_game_over` callback only fires after the game has ended.

### CQ-009: Exception handling in delete_game silently swallows errors from human.on_state
- Status: Resolved

### CQ-010: _parse_action does not handle KeyError from _extract_card_ids on malformed dict cards
- Status: Resolved

### CQ-011: Static file mount uses relative path that may resolve incorrectly
- Status: Resolved

### CQ-012: No input validation on game_id path parameter in DELETE endpoint
- Status: Don't Fix
- Description: At `server/server.py:85`, the `game_id` parameter is a plain `str` with no validation. While `registry.get()` gracefully returns `None` for invalid IDs, the endpoint accepts arbitrarily long strings, special characters, or path-traversal-like inputs (e.g., `../../../etc/passwd`). FastAPI's path parameter extraction handles URL decoding, but the downstream `registry.get(game_id)` does a dictionary lookup, so there is no actual security vulnerability. However, for production robustness, a Pydantic model or regex constraint (e.g., `Path(game_id, pattern=r"^[a-f0-9]{32}$")`) would be appropriate since `GameRegistry.create()` generates `uuid4().hex` strings.
- Decision Reason: No security vulnerability exists - `registry.get(game_id)` is a dictionary lookup that returns None for unknown keys. Adding regex validation would change the API contract and require updating tests. The test `test_delete_nonexistent_returns_200` already verifies graceful handling of arbitrary strings. This is a production hardening concern, not a task requirement. All 20 tests pass without this change.

### CQ-013: Initial state push in normal WS connect path is not wrapped in try/finally
- Status: Resolved
- Description: At `server/server.py:137-144`, the normal (non-game-over) WebSocket connect path sets the WS reference on the human player (`human_player.set_ws(websocket)` at line 137), then pushes initial state via `game.snapshot()` (line 139) and `websocket.send_json()` (line 140). If either of these raises an exception (e.g., the websocket connection was already closed by the client between accept and send, or snapshot raises), the exception propagates out of the function WITHOUT entering the `try/finally` block at lines 146-171. This means `human_player.set_ws(None)` at line 171 never runs, leaving the HumanPlayer with a stale/dead WebSocket reference. Subsequent connection attempts would be rejected by the `is_connected()` check at line 113, effectively making the game unplayable. Compare with the game-over branch (lines 118-135) which correctly wraps snapshot/send in its own `try/finally` that always calls `set_ws(None)`. The fix is to move the `human_player.set_ws(websocket)` call inside the `try` block at line 146, or wrap lines 137-144 in their own try/except that calls `set_ws(None)` and re-raises.
- Decision Reason: Fixed in commit 0871719. The implementation now wraps `websocket.accept()`, `game.snapshot()`, and `websocket.send_json()` in a try/except block (lines 138-148). On exception, `human_player.set_ws(None)` is called and the function returns. If successful, execution falls through to the main message loop whose finally block handles cleanup.

### CQ-014: websocket.accept() in game-over path is outside try/finally, leaking stale WS reference
- Status: Resolved
- Description: At `server/server.py:118-119`, in the game-over WebSocket connect path, `human_player.set_ws(websocket)` is called at line 118 and `await websocket.accept()` is called at line 119, both BEFORE the try/finally block that starts at line 120. If `websocket.accept()` fails (e.g., client disconnects between HTTP upgrade and accept), the exception propagates out of the function WITHOUT the finally block at lines 129-134 executing. This means `human_player.set_ws(None)` at line 130 never runs, leaving the HumanPlayer with a stale/dead WebSocket reference. Subsequent connection attempts would be rejected by the `is_connected()` check at line 113, effectively making the game unplayable until the game object is garbage-collected. The fix is to move `human_player.set_ws(websocket)` and `await websocket.accept()` inside the try block (i.e., change `try:` at line 120 to encompass lines 118-119 as well), matching the defensive pattern used in the normal connect path at lines 138-148 where the except block calls `set_ws(None)`.

### CQ-015: WS connect path duplicates on_state logic instead of calling human_player.on_state()
- Status: Resolved
- Description: At `server/server.py:120-126` (game-over path) and lines 139-145 (normal connect path), the code manually builds the state message by calling `game.snapshot()` and `websocket.send_json()` directly. The task specification (task.md lines 449-450) explicitly says to call `human_player.on_state(game)` which does exactly the same thing (see `server/player.py:199-208`). The `delete_game` endpoint at line 92 already correctly uses `human.on_state(game)`. This duplication means there are 3 separate implementations of the same state-push logic (player.on_state, game-over connect, normal connect) rather than one. If the state message format ever changes, all 3 must be updated in sync. The fix is to replace lines 120-126 with `await human_player.on_state(game)` (after accepting the websocket), and similarly replace lines 139-145.
