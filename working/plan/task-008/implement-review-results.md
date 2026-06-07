# Implement Review Results: Task-008

## Task Compliance Issues

## Code Quality Issues

### CQ-001: Reconnect promise rejection is unhandled
- Status: Resolved
- Description: In `_attemptReconnect()` (ws-client.ts:119-122), `_doConnect()` is called without awaiting or catching its returned Promise. If the reconnection fails (e.g., server is still down), the Promise returned by `_doConnect()` rejects, and since nothing handles that rejection, it produces an unhandled promise rejection. During reconnection, the initial `connect()` call has already resolved, so the caller has no way to know reconnection attempts are failing. This can produce unhandled promise rejection warnings in Deno/Node runtimes and in production could cause silent connection loss with no error visibility.
- Decision Reason:

### CQ-002: Malformed messages silently swallowed with no diagnostic
- Status: Resolved
- Description: At ws-client.ts:87-89, the catch block in the message handler silently ignores JSON parse errors with only a comment `// Ignore malformed messages`. In production, if the server sends a message that cannot be parsed, the client will silently drop it with no logging or error callback. This makes protocol-level debugging extremely difficult -- there is no way to detect or diagnose malformed message issues in deployed code. At minimum, `console.warn` should be used so that errors are visible in browser developer tools / server logs.
- Decision Reason:

### CQ-003: send() silently discards when socket not open
- Status: Resolved
- Description: At ws-client.ts:50-52, `send()` silently drops the action when `this._ws` is null or not in `OPEN` state. The caller (`client.send(action)`) receives no indication that the message was not sent. This could cause hard-to-diagnose issues in production where user actions (bids, plays) are silently lost without any feedback. A warning or throwing would make failures visible.
- Decision Reason:

### CQ-004: test_reconnect_respects_max_retries uses 9s fixed timeout
- Status: Resolved
- Description: At ws-client-mock.test.ts:282, the test uses `await new Promise((r) => setTimeout(r, 9000))` -- a fixed 9-second sleep. This is the only test that does not use the `waitFor()` helper. The 9-second sleep makes this test extremely slow and is also fragile: if the reconnect delays shift or the system is under load, the assertion at line 285 may fire before or after the correct window. The test should use `waitFor()` to poll for the expected connection count instead of sleeping a fixed duration.
- Decision Reason:

### CQ-005: `_reconnectAttempts` not reset on successful reconnect
- Status: Resolved
- Description: The task spec states: "On successful reconnect, reset `_reconnectAttempts` to 0." In the implementation at `frontend/net/ws-client.ts`, the `_doConnect` `open` handler (lines 76-81) resolves the promise but never resets `_reconnectAttempts`. This means if a connection succeeds after a reconnect attempt (e.g., `_reconnectAttempts` is 1), and the server later disconnects again, the client will skip retries 0 and 1 (since `_reconnectAttempts` is still >= 1), proceeding directly to retry 2. The `_reconnectAttempts` counter only resets when `connect()` is explicitly called again (line 43) or `disconnect()` is called (line 66). This is a functional correctness bug -- the exponential backoff does not fully reset after a successful reconnect, limiting future retry capacity.
- Decision Reason:
