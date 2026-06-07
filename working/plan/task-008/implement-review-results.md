# Implement Review Results: Task-008

## Task Compliance Issues

### TC-001: Missing `setWsHost()` method specified in task objective
- Status: Resolved
- Description: The task objective (task.md line 13) states: "The caller provides the WS host via `setWsHost()` or the connect method's optional wsHost parameter." The implementation has neither a `setWsHost()` method nor an optional `wsHost` parameter on `connect()`. The `connect()` signature is `connect(gameId: string, wsHost: string)` with `wsHost` as a required positional argument. The Steps section describes the same required-parameter signature, but the task objective explicitly specifies a setter pattern or optional parameter. The implementer followed one interpretation of an internally inconsistent spec and omitted the `setWsHost()` method entirely.
- Decision Reason:

### TC-002: No test coverage for reconnection behavior
- Status: Resolved
- Description: The task objective requires "exponential backoff reconnection (1s, 2s, 4s, max 3 retries)" as a core feature. The implementation in `_attemptReconnect()` (ws-client.ts lines 98-110) implements this logic, but none of the 5 tests verify reconnection -- that after a server-initiated disconnect, the client automatically reconnects with correct backoff delays and respects the 3-retry limit. The task lists 5 specific test names in the Module Design section, and while all 5 pass, none cover the reconnection feature which is a central requirement of the task.
- Decision Reason:

## Code Quality Issues

### CQ-001: connect() parameter order inconsistency with task specification
- Status: Resolved
- Description: Task spec line 13 states: "connect(gameId: string) takes a gameId and constructs the WebSocket URL internally using WS_PATH(gameId). The caller provides the WS host via setWsHost() or the connect method's optional wsHost parameter." However, the implementation at `frontend/net/ws-client.ts:29` defines `connect(gameId: string, wsHost: string)` where wsHost is a required positional parameter, not an optional parameter, and there is no `setWsHost()` method. The task spec implies wsHost should be optional with a setter pattern, but the implementation requires it as a mandatory argument.
- Decision Reason:

### CQ-002: No test for reconnection behavior
- Status: Resolved
- Description: The task spec explicitly requires "exponential backoff reconnection (1s, 2s, 4s, max 3 retries)" and the implementation in `_attemptReconnect()` (ws-client.ts:98-109) implements this logic. However, none of the 5 test cases verify the reconnection logic -- that after a disconnect, the client automatically reconnects with the correct exponential backoff delays and respects the 3-retry limit. This is a core feature per the task objective and should have dedicated test coverage.
- Decision Reason:

### CQ-003: Error handler resolves promise even on initial connection failure
- Status: Resolved
- Description: At `frontend/net/ws-client.ts:87-94`, the error handler resolves the promise if `readyState !== WebSocket.OPEN`, which means a failed initial connection silently resolves the connect() promise instead of rejecting it. The caller has no way to know the connection failed. This could cause downstream code to assume the connection is alive when it is not. The `reject` parameter at line 60 is declared but never called -- it is dead code. The error handler should either reject the promise on initial connect failure or provide a separate mechanism to detect connection failure.
- Decision Reason:

### CQ-004: Malformed messages silently swallowed
- Status: Resolved
- Description: At `frontend/net/ws-client.ts:73-76`, the catch block in the message handler silently ignores JSON parse errors with an empty comment. There is no logging, no error callback, and no way for the application to detect or respond to malformed messages from the server. In production, this could mask protocol bugs.
- Decision Reason:

### CQ-005: Test uses fixed timeouts for synchronization instead of deterministic waiting
- Status: Resolved
- Description: All tests use `await new Promise((r) => setTimeout(r, N))` with fixed timeouts (50ms-200ms) to wait for async operations. This is inherently fragile -- under load or slow CI environments, these timeouts may not be sufficient, causing flaky tests. Tests should use more deterministic synchronization like waiting for specific events or polling with retry instead of fixed delays.
- Decision Reason:

### CQ-006: Error handler checks `this._ws` instead of local `ws` variable
- Status: Resolved
- Description: At `frontend/net/ws-client.ts:87-94`, the error handler captures the class property `this._ws` rather than the local variable `ws` from line 62. The error handler should check `ws.readyState !== WebSocket.OPEN` (the local socket), not `this._ws?.readyState` (which may point to a different socket after reconnection). During reconnection, `this._ws` gets reassigned to a new WebSocket, so the old socket's error handler would check the new socket's state instead of its own. This is a correctness bug.
- Decision Reason:

### CQ-007: `test_onDisconnect_called` does not clean up client
- Status: Resolved
- Description: At `frontend/tests/ws-client-mock.test.ts:137-165`, the `test_onDisconnect_called` test never calls `client.disconnect()` in cleanup. Unlike all other tests which call `client.disconnect()` at the end, this test leaves the client in a state where `_attemptReconnect()` may schedule a reconnect timer after the test completes. This can cause dangling async operations and potential interference with other tests.
- Decision Reason:
