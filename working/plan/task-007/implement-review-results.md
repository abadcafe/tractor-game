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
- Description: At `server/server.py:91-94`, when pushing the final state to an active WebSocket before deletion, the code catches `Exception` and silently passes. If `on_state` fails (e.g., WebSocket already half-closed, serialization error), the human player receives no notification and the error is invisible. The `set_ws(None)` at line 95 still runs, which is correct for cleanup, but there is no logging of the failure. In production, this makes debugging connection issues difficult. At minimum, a `logger.debug` or `logger.warning` should be added inside the except block to record the failure reason.
- Decision Reason: (implementer fills for Don't Fix status)

### CQ-010: _parse_action does not handle KeyError from _extract_card_ids on malformed dict cards
- Status: Resolved
- Description: At `server/server.py:212`, `_extract_card_ids` accesses `c["id"]` for dict-format cards without a `try/except KeyError`. If a client sends `{"cards": [{"suit": "hearts"}]}` (missing the "id" key), this raises an unhandled `KeyError` which propagates up. The `_parse_action` caller catches `ValueError` (line 157) but not `KeyError`, so the exception reaches the generic `except Exception` handler at line 162 and returns an error message. This technically works but the error message will be a generic Python traceback string rather than a clean user-facing message like `"Invalid card format: missing 'id' field"`. The `ValueError` raised by the `else` branch at line 214 is the intended pattern -- the dict branch should follow the same convention.
- Decision Reason: (implementer fills for Don't Fix status)

### CQ-011: Static file mount uses relative path that may resolve incorrectly
- Status: Resolved
- Description: At `server/server.py:220`, `_static_dir` is computed as `os.path.join(os.path.dirname(__file__), "..", "static")`. This uses a relative `..` component which is resolved at import time. While this works for the standard project layout, the resulting path may contain `..` segments (e.g., `/home/lfw/works/tractor-game/server/../static`). `os.path.abspath()` should be called to normalize the path. The same issue exists at line 227 for `html_path`. This matters if any downstream code compares paths or if FastAPI/Starlette does path resolution that doesn't handle `..` consistently.
- Decision Reason: (implementer fills for Don't Fix status)

### CQ-012: No input validation on game_id path parameter in DELETE endpoint
- Status: Don't Fix
- Description: At `server/server.py:85`, the `game_id` parameter is a plain `str` with no validation. While `registry.get()` gracefully returns `None` for invalid IDs, the endpoint accepts arbitrarily long strings, special characters, or path-traversal-like inputs (e.g., `../../../etc/passwd`). FastAPI's path parameter extraction handles URL decoding, but the downstream `registry.get(game_id)` does a dictionary lookup, so there is no actual security vulnerability. However, for production robustness, a Pydantic model or regex constraint (e.g., `Path(game_id, pattern=r"^[a-f0-9]{32}$")`) would be appropriate since `GameRegistry.create()` generates `uuid4().hex` strings.
- Decision Reason: No security vulnerability exists - `registry.get(game_id)` is a dictionary lookup that returns None for unknown keys. Adding regex validation would change the API contract and require updating tests. The test `test_delete_nonexistent_returns_200` already verifies graceful handling of arbitrary strings. This is a production hardening concern, not a task requirement. All 20 tests pass without this change.
