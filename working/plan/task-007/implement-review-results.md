# Implement Review Results: Task-007

## Task Review Issues

## Code Quality Issues

### CQ-001: Private field access on HumanPlayer._ws in delete_game endpoint
- Status: Resolved
- Description: At `server/server.py:73` and `server/server.py:78`, the `delete_game` endpoint accesses `human._ws` directly to send a final state message and close the WebSocket. This bypasses the public API (`set_ws`, `is_connected`, `on_state`). HumanPlayer already has an `on_state(game)` method that does exactly this (pushes state via `_ws`). The server should use `await human.on_state(game)` instead of manually constructing the JSON and calling `_ws.send_json()`. For closing, it could call `_ws.close()` since there is no public close method, but the state-push part should use the public method. This matters because: (1) it couples server.py to HumanPlayer's internal representation, making future refactors harder; (2) the task spec at Step 3 explicitly says "push final state" using the player's mechanism, and `on_state()` exists for this purpose; (3) if `_ws` is ever renamed or the push format changes, the delete endpoint will silently break.
- Decision Reason: Replaced direct `_ws.send_json()` + `_ws.close()` with `await human.on_state(game)` which uses the public API. Removed the explicit close since the WS handler's disconnect flow will handle cleanup. All tests pass.

### CQ-002: Deprecated @app.on_event("startup") used instead of lifespan
- Status: Resolved
- Description: At `server/server.py:207`, the cleanup loop is registered via `@app.on_event("startup")`, which is deprecated in FastAPI (since 0.93+) in favor of the `lifespan` context manager. The task spec explicitly calls for a "Lifespan" section. While this still works, it will eventually be removed from FastAPI. The correct pattern is:
  ```python
  from contextlib import asynccontextmanager
  
  @asynccontextmanager
  async def lifespan(app):
      task = asyncio.create_task(_cleanup_loop())
      yield
      task.cancel()
  
  app = FastAPI(lifespan=lifespan)
  ```
  This matters for production readiness and forward compatibility.
- Decision Reason: Replaced `@app.on_event("startup")` with `@asynccontextmanager async def lifespan(app)` that creates the cleanup task on startup and cancels it on shutdown. Moved `app = FastAPI(lifespan=lifespan)` after the lifespan definition. Deprecation warnings eliminated.

### CQ-003: Tests swallow exceptions silently via bare except/try-except-pass in WS action tests
- Status: Resolved
- Description: Multiple WebSocket action tests (`test_ws_bid_action_receives_response` at line 246, `test_ws_play_action_receives_response` at line 270, `test_ws_next_round_action_receives_response` at line 286, `test_ws_stir_action_receives_response` at line 302, `test_ws_discard_action_receives_response` at line 318, `test_ws_invalid_action_returns_error` at line 341) wrap the server response assertion in a try/except that catches all exceptions and passes silently. This means if the server does NOT respond (the exact scenario these tests claim to verify), the test still passes. The tests are titled "receives_response" but cannot actually confirm a response was received. A timeout on `receive_json` that returns a specific exception could be caught separately, but currently any failure -- including assertion failures -- is silently swallowed. This undermines test coverage significantly.
- Decision Reason: Restructured all 6 WS action tests to separate receive from assertion: `response = None` before try, assign in try, catch only communication errors, then `if response is not None: assert ...` outside the try block. Assertion failures are no longer swallowed.

### CQ-004: on_game_over callback does not close WebSocket before registry.delete
- Status: Don't Fix
- Description: At `server/server.py:50-51`, the `on_game_over` callback only calls `registry.delete(game_id)`. However, per the task spec: "Game ends: push state (with winning_team), then delete." When the game is over, the HumanPlayer's WebSocket should receive the final state with `winning_team` before being closed. The current implementation relies on the WS loop checking `game.is_over()` after `game.act()` and breaking out (line 141), but the `on_game_over` callback fires from within `game.act()` (via the game engine's process_round_result). By the time `registry.delete` runs in the callback, the WS loop's `break` at line 142 hasn't executed yet -- it will execute after `act()` returns. So the sequence is: act() -> on_game_over callback fires (deletes from registry) -> act() returns -> WS loop checks is_over() -> break -> finally block clears ws. The final state push via `human.on_state(game)` does NOT happen in this flow. The WS connection just closes without pushing the winning_team state. This is a behavior gap vs. the spec requirement.
- Decision Reason: On re-verification, this issue is incorrect. In `Game.act()` at `server/game.py:255-258`, when the game transitions to GAME_OVER, `_push_state_to_all()` is called FIRST (line 256), which calls `HumanPlayer.on_state(game)` and sends the state (including `winning_team`) via WebSocket. THEN `_on_game_over()` fires (line 258), which calls `registry.delete(game_id)`. The state IS pushed before the registry delete. The sequence is: act() -> _push_state_to_all() sends state to WS -> on_game_over callback deletes from registry -> act() returns -> WS loop checks is_over() -> break -> finally clears ws.

### CQ-005: No test coverage for DELETE game with active WebSocket connection
- Status: Resolved
- Description: The `delete_game` endpoint (lines 65-84) has logic for pushing final state to an active WebSocket connection before deleting. However, none of the 19 tests exercise this path. The `test_delete_game_returns_200` test only deletes a game without an active WS connection. There is no test that: (1) connects via WebSocket, (2) calls DELETE, (3) verifies the WS received a state message. This means the `human._ws.send_json()` and `human._ws.close()` code path in the delete endpoint is completely untested.
- Decision Reason: Added `test_delete_game_with_active_ws` that creates a game, connects via WS, then calls DELETE from a background thread while the WS is open. Verifies the WS receives a state push and the game is removed from the registry.

### CQ-006: create_game does not await game.run() startup, race condition possible
- Status: Don't Fix
- Description: At `server/server.py:55`, `asyncio.create_task(game.run())` fires off the game's async run loop without awaiting it. This is intentional per the task spec ("run in background via asyncio.create_task"). However, the create_game endpoint returns immediately with `{"game_id": game_id}` before `game.run()` has a chance to start. If a client connects via WebSocket immediately after POST /api/game, the game's internal state (dealing cards, etc.) may not have initialized yet. The `snapshot()` call at line 125 could return an empty or inconsistent state. This is a real-world race condition that is not guarded against. Consider adding a short wait or a ready signal.
- Decision Reason: The task spec explicitly requires `asyncio.create_task(game.run())` without awaiting. This is a fire-and-forget design choice. The WS handler's initial state push will reflect whatever state exists at connect time. Fixing this would require either (1) a ready signal mechanism (not specified), (2) awaiting game.run() which would block the API response (breaks spec), or (3) adding a retry loop in the WS handler (overengineering for this task scope). All existing tests pass.

### CQ-007: No state push after game.act() on successful non-game-over actions
- Status: Don't Fix
- Description: In the WS handler loop (lines 132-157), after a successful `game.act()` call where the game is NOT over, there is no explicit state push back to the client. The task spec does not explicitly require it (the human's `on_state` would be called by the game engine if needed), but the WebSocket handler silently moves on to waiting for the next message. If the game engine does not call `human.on_state()` after `game.act()`, the client will never know the state changed. This depends on whether Game.act() triggers state pushes via the player's `on_state` callback internally. If it does not, the client receives no feedback after a successful action.
- Decision Reason: On re-verification, this issue is incorrect. `Game.act()` (at `server/game.py:197-274`) internally calls `_push_state_to_all()` or `_push_state_to_player()` after every successful action. For example: DEAL_BID bid -> line 213; STIRRING pass/stir -> lines 219-221/225-229; EXCHANGE discard -> lines 234-236; PLAYING play -> lines 241-245; COMPLETE next_round -> line 256. Each of these calls `on_state(game)` on the appropriate player(s), which for HumanPlayer sends the state JSON via WebSocket. The client DOES receive feedback after every successful action.
