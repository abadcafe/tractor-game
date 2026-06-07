# Implement Review Results: Task-005

## Task Review Issues

### TR-001: EXCHANGE phase does not push state back to player after discard when phase stays EXCHANGE
- Status: Resolved
- Description: In Game.act() (server/game.py lines 200-203), after a DiscardAction in EXCHANGE phase, state is only pushed to the current player when the phase transitions to PLAYING. When the exchange is not yet complete and the phase stays in EXCHANGE, no state is pushed to the declarer at all. The task spec states: "During all other phases (STIRRING, EXCHANGE, PLAYING, COMPLETE): call _push_state_to_player(current_player) where current_player is determined by the phase: exchange_state.declarer_player for EXCHANGE". For HumanPlayer, this means after discarding cards, they would not receive a WebSocket state update telling them to discard more cards. The AutoPlayer would similarly not be triggered for its next discard action. The fix is to add an else branch that pushes state to self._round_state.exchange_state.declarer_player when the phase remains EXCHANGE after a discard.
- Decision Reason: Fixed by adding elif branch that pushes state to exchange_state.declarer_player when phase stays EXCHANGE after discard.

### TR-002: snapshot().phase returns round-level phase instead of "GAME_OVER" when game is over
- Status: Resolved
- Description: In `snapshot()` at line 359, `phase=rs.phase` uses the round state's phase (which remains "COMPLETE" after the game ends) instead of checking `_game_state.phase`. When the game transitions to GAME_OVER via `act()` with NextRoundAction, `get_phase()` correctly returns "GAME_OVER" (line 391-392), but `snapshot().phase` still returns "COMPLETE" because it reads from `_round_state.phase`. This means when the game pushes final state to all players via `_push_state_to_all()` at line 232, the client receives a snapshot with `phase: "COMPLETE"` instead of `phase: "GAME_OVER"`. The `to_dict()` method at line 72 serializes `self.phase` directly, so the WebSocket JSON also reports "COMPLETE" when the game is over. This inconsistency between `get_phase()` and `snapshot().phase` can mislead clients that rely on the snapshot to determine game state. The fix is to use `self.get_phase()` instead of `rs.phase` at line 359, or check `_game_state.phase == "GAME_OVER"` and override accordingly.
- Decision Reason: Fixed by using `self.get_phase()` instead of `rs.phase` at line 359. This ensures snapshot().phase is consistent with get_phase() and correctly reports "GAME_OVER" when the game is over.

## Code Quality Issues

### CQ-001: `act()` type annotation is `Player` instead of `PlayerAction`
- Status: Resolved
- Description: At `server/game.py:178`, the `act()` method signature is `async def act(self, player_index: int, action: Player) -> None:`. The `action` parameter is annotated as `Player` (the base class for AutoPlayer/HumanPlayer), but it should be a union of the action types (BidAction, PlayAction, StirAction, SkipStirAction, DiscardAction, NextRoundAction). This is misleading -- `Player` is a player entity, not an action type. It happens to work because Python does not enforce type annotations at runtime, but it defeats type checkers and misleads readers. The task spec clearly states the interface should accept PlayerAction types. A proper annotation would be `action: BidAction | PlayAction | StirAction | SkipStirAction | DiscardAction | NextRoundAction` or a `PlayerAction` type alias.
- Decision Reason: Fixed by changing annotation to union type `BidAction | PlayAction | StirAction | SkipStirAction | DiscardAction | NextRoundAction`.

### CQ-002: EXCHANGE phase action does not push state to all players when phase transitions to PLAYING
- Status: Don't Fix
- Description: At `server/game.py:200-203`, when a `DiscardAction` causes the phase to transition from EXCHANGE to PLAYING, only the next trick player is pushed state via `_push_state_to_player(self._round_state.trick_state.cur)`. However, all players should receive updated state because: (a) the declarer player's hand changes during exchange (cards are discarded to bottom), and (b) all players need to know the phase has changed to PLAYING so they can prepare their play strategies. Only pushing to the trick's current player leaves the other 3 players with stale state from the EXCHANGE phase.
- Decision Reason: The task spec explicitly states: "During all other phases (STIRRING, EXCHANGE, PLAYING, COMPLETE): call _push_state_to_player(current_player) where current_player is determined by the phase: trick_state.cur for PLAYING". The spec only mandates _push_state_to_all() during DEAL_BID phase and when transitioning to GAME_OVER. The current code correctly follows the spec by pushing only to the current player for non-DEAL_BID phases. Pushing to all players would deviate from the designed architecture. The AutoPlayer only acts when current_player matches its index, so pushing to others has no functional effect for AI. For HumanPlayer, subsequent state pushes will occur as the game progresses through tricks.

### CQ-003: `_serialize_completed_trick` does not handle `lead_type` when `trick` is already a dict
- Status: Resolved
- Description: At `server/game.py:131-147`, `_serialize_completed_trick()` handles both dict and object cases. When `trick` is already a dict (lines 133-134), it returns the dict as-is. But if that dict contains `lead_type` as an enum object (not yet serialized to a string), it will remain non-JSON-serializable. Currently, `trick_history` is populated from `rs.trick_history` which contains sm CompletedTrick objects, so the object branch (lines 135-147) is always taken. The dict branch is a defensive fallback that could return non-JSON-serializable data if trick_history entries are ever dicts with enum values. To be safe, the dict branch should also serialize `lead_type` if present.
- Decision Reason: Fixed by adding lead_type and slots serialization to the dict branch of _serialize_completed_trick.

### CQ-004: No bounds checking on `get_player(index)`
- Status: Resolved
- Description: At `server/game.py:381-382`, `get_player()` directly indexes into `self._players[index]` without bounds checking. If an invalid index is passed (e.g., -1 or 5), it will raise an `IndexError` with an unhelpful message. Since this is a public API used by the server layer to route WebSocket connections, an explicit `ValueError` with a clear message would be more appropriate and easier to debug in production.
- Decision Reason: Fixed by adding explicit bounds check that raises ValueError with descriptive message.

### CQ-005: `_convert_bid_action` assumes `cards` list is non-empty
- Status: Resolved
- Description: At `server/game.py:436-456`, `_convert_bid_action()` accesses `cards[0]` at line 440 without checking if the list is empty. If a `BidAction` is created with an empty `cards` list, this will raise an `IndexError` before the sm layer gets a chance to validate and reject it with a meaningful error. The task spec says the method derives `kind`, `suit`, `joker_type` from the first card, implying non-empty cards. A defensive check or early validation would produce a clearer error message.
- Decision Reason: Fixed by adding empty cards check that raises ValueError with descriptive message before accessing cards[0].

### CQ-006: PLAYING phase does not push state to declarer when round transitions to COMPLETE
- Status: Resolved
- Description: At `server/game.py:218-221`, when a `PlayAction` in PLAYING phase causes the round to transition to COMPLETE (after 25 tricks), the code only pushes state if `self._round_state.phase == "PLAYING"`. When the phase is now COMPLETE, no state is pushed to any player. The declarer player (who needs to submit `NextRoundAction`) is never notified of the phase change. This means in a real game with HumanPlayer, the WebSocket client would never receive the COMPLETE-phase state, and the player would not know they need to act. The spec says: "During all other phases (STIRRING, EXCHANGE, PLAYING, COMPLETE): call _push_state_to_player(current_player) where current_player is determined by the phase: ... declarer_player for COMPLETE". The fix is to add an elif branch that pushes state to `self._round_state.declarer_player` when the phase transitions to COMPLETE after a PlayAction.
- Decision Reason: Fixed by adding elif branch at lines 222-225 that pushes state to declarer_player when phase transitions to COMPLETE.

### CQ-007: Unused import `AsyncMock` in game_tests.py
- Status: Resolved
- Description: At `server/game_tests.py:11`, `AsyncMock` is imported from `unittest.mock` but never used anywhere in the test file. All mocked async methods (like `on_state` in `test_set_on_game_over_callback_fires_on_game_over`) use `MagicMock` (via `patch`), not `AsyncMock`. Unused imports reduce code clarity and can confuse readers about the testing approach used.
- Decision Reason: Removed the unused `AsyncMock` import.

### CQ-008: `snapshot()` and `resolve_cards()` have no bounds check on `player_index` parameter
- Status: Resolved
- Description: At `server/game.py:257`, `snapshot(for_player)` accesses `rs.players_hand[for_player]` with a guard only for `for_player < len(rs.players_hand)` for the hand field, but negative indices (e.g., -1) silently pass the check and access `players_hand[-1]` (last player's hand). At `server/game.py:421`, `resolve_cards()` accesses `players_hand[player_index]` without any bounds check, which would raise an unhelpful `IndexError` for out-of-range indices. This is inconsistent with `get_player()` (line 399) which now has an explicit bounds check raising `ValueError`. Both `snapshot()` and `resolve_cards()` should validate `player_index` with explicit bounds checking consistent with the pattern established in `get_player()`.
- Decision Reason: Added explicit bounds checks raising ValueError in both snapshot() and resolve_cards(), consistent with get_player().

### CQ-009: `_dealing_loop` has no exception handling for `deal_next_card` failures
- Status: Resolved
- Description: At `server/game.py:446`, `_dealing_loop` calls `round_sm.deal_next_card(self._round_state)` without any try/except. If `deal_next_card` raises an exception (e.g., due to corrupted state, an edge case in the sm layer, or a card model validation error), the background asyncio task will terminate silently with an unhandled exception. The game would become stuck -- the dealing loop stops but the game remains in DEAL_BID phase with no way to recover. The dealing loop should catch unexpected exceptions, log them, and either retry or transition to a failure state. At minimum, the exception should be logged so operators can diagnose the issue.
- Decision Reason: Wrapped the dealing loop body in try/except Exception with logging to prevent silent failures.

### CQ-010: `test_set_on_game_over_callback_fires_on_game_over` uses broad try/except that masks failures
- Status: Resolved
- Description: At `server/game_tests.py:405-408`, the test catches `(ValueError, AttributeError, TypeError)` and passes silently. Then at lines 410-412, the assertion is guarded by `if game.is_over()`, meaning if the try/except swallowed the actual error that would have led to GAME_OVER, the test passes without actually verifying the callback was invoked. This makes the test brittle: it can pass both when the feature works correctly AND when it fails silently. The test should either (a) not catch these exceptions (let them propagate as test failures), or (b) use a more targeted approach that ensures the GAME_OVER path is exercised without needing to catch broad exception types.
- Decision Reason: Removed the broad try/except and the conditional assertion. Now the test always asserts game.is_over() and callback.assert_called_once_with(game), ensuring failures are not masked.

### CQ-011: `act()` passes potentially None `round_result` to `process_round_result` without validation
- Status: Resolved
- Description: At `server/game.py:228`, `round_result = round_sm.get_round_result(self._round_state)` returns `RoundResult | None` (see `server/sm/round_sm.py:308-310` where it returns `state.result`). At line 229, this value is passed directly to `game_sm.process_round_result(self._game_state, round_result)` which expects a non-None `RoundResult` (see `server/sm/game_sm.py:70`). If `state.result` is ever `None` while the round phase is "COMPLETE" (e.g., due to an sm layer bug or an edge case not covered by existing tests), this would pass `None` to `process_round_result`, causing a runtime error deep in the sm layer with an unhelpful traceback. A defensive check that raises a clear error before calling `process_round_result` would make failures easier to diagnose in production. In practice, `result` is always populated when the round reaches COMPLETE, but the type system explicitly allows `None`, so the code should defend against it.
- Decision Reason: Fixed by adding an explicit None check before passing round_result to process_round_result, raising ValueError with a clear diagnostic message.

### CQ-012: `_serialize_trick` accesses `.value` on `lead_type` without `hasattr` guard
- Status: Resolved
- Description: At `server/game.py:106-107`, `_serialize_trick()` accesses `result["lead_type"].value` directly without checking `hasattr(result["lead_type"], "value")`. If `lead_type` were ever a string (not an enum), this would raise an `AttributeError`. Currently, `lead_type` is always a `PlayType` enum from `trick_state`, so this is not a runtime issue. However, `_serialize_completed_trick()` at line 135 defensively checks `hasattr(result["lead_type"], "value")` before accessing `.value`, creating an inconsistency. Both functions should use the same defensive pattern. If the codebase evolves to store `lead_type` as a string (e.g., after serialization/deserialization), `_serialize_trick` would break while `_serialize_completed_trick` would not.
- Decision Reason: Added `hasattr(result["lead_type"], "value")` guard to `_serialize_trick()`, consistent with `_serialize_completed_trick()`.

### CQ-013: `resolve_cards()` "Game not started" error path is untested
- Status: Resolved
- Description: At `server/game.py:432-433`, `resolve_cards()` raises `RuntimeError("Game not started")` when `_round_state` is None (before `run()` is called). The test file has `test_snapshot_raises_before_run` for `snapshot()` but no equivalent test for `resolve_cards()`. This means the "Game not started" guard in `resolve_cards()` is untested. While the code path is straightforward, untested error paths can silently break when refactored.
- Decision Reason: Added `test_resolve_cards_raises_before_run` test that verifies RuntimeError is raised when resolve_cards() is called before run().
