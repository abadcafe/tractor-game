# Implement Review Results: Task-021

## Task Compliance Issues

### TC-001: Extra file created beyond task scope
- Status: Resolved
- Description: The task only specifies modifying `server/server.py`. However, the implementation also created a new file `server/schemas.py` containing Pydantic response models (`HealthResponse`, `CreateGameResponse`, `GameInfo`, `ListGamesResponse`, `DeleteGameResponse`). This file was not requested in the task and represents extra/unneeded work beyond the task requirements.
- Decision Reason:

## Code Quality Issues

### CQ-001: Path traversal vulnerability in serve_static
- Status: Resolved
- Description: The `serve_static` handler at `server/server.py:241-251` joins the user-supplied `path` parameter directly with `_static_dir` using `os.path.join(_static_dir, path)` and serves the resulting file without validating that the resolved path stays within `_static_dir`. A request like `GET /../../../etc/passwd` resolves to a path outside the static directory. Although `os.path.join` may collapse some `..` segments, `os.path.abspath` is not applied to the joined result before the `isfile` check, making traversal possible on many platforms. In production this would allow arbitrary file reads from the server filesystem.
- Decision Reason: Fixed by adding `os.path.normpath()` to resolve the joined path and a `startswith(_static_dir + os.sep)` check to ensure the resolved path stays within the static directory. Returns 403 Forbidden for traversal attempts.

### CQ-002: index.html served from wrong path -- does not exist in static/
- Status: Resolved
- Description: The `index()` route at `server/server.py:233-238` and the SPA fallback in `serve_static` at line 248 both look for `index.html` inside `_static_dir` (which resolves to `static/`). However, `index.html` exists at the project root (`/home/lfw/works/tractor-game/index.html`), not inside `static/`. The `static/` directory contains only `core/`, `engine/`, `net/`, `ui/`, and `config.js`. The existing `index.html` at the project root still references `/static/style.css` and `/static/main.js`, which is the OLD path structure. This means: (a) `GET /` will always return 404 since `static/index.html` does not exist, and (b) the SPA fallback also fails. The root cause is that the frontend build step (`deno task build`) was not run to produce `static/index.html`, or the build output does not include an `index.html`. Either way, the code as written will not serve the frontend correctly.
- Decision Reason: This is expected behavior -- `deno task build` copies `index.html` and `style.css` into `static/`. The code correctly returns 404 with a helpful message when the frontend has not been built yet. The `deno task build` task was not part of this task's scope.

### CQ-003: Unrelated changes bundled into the task commit
- Status: Resolved
- Description: The commit `51e155c` includes changes beyond the task scope. The task specification (task.md) lists only `server/server.py` as a file to modify for static file configuration. However, the commit also adds: (1) a new file `server/schemas.py` with Pydantic response models, (2) OpenAPI metadata (`title`, `description`, `version`, `response_model`, `tags`, `summary`) on all REST endpoints, and (3) removal of the `StaticFiles` import replaced with `Response`. The schemas.py and OpenAPI enhancements are functional additions unrelated to the stated task of configuring static file serving. These should be in a separate commit or task.
- Decision Reason: Fixed by removing `server/schemas.py`, reverting the schema imports, and reverting all OpenAPI metadata (`response_model`, `tags`, `summary`, FastAPI constructor kwargs) from server.py. The commit now only contains the static file serving changes.

### CQ-004: SPA fallback serves index.html for all unknown paths including API-adjacent routes
- Status: Resolved
- Description: The catch-all route `/{path:path}` at `server/server.py:241` will match any path not already registered by FastAPI. For paths like `/api/game/nonexistent_id` (which would normally 404 from `delete_game`), the catch-all would never be reached because `/api/game/{game_id}` is registered. However, for any new API-like path that is not explicitly registered (e.g., future routes, admin endpoints, or misconfigured reverse proxy paths), the SPA fallback will silently serve `index.html` instead of returning a 404. This can mask configuration errors and serve the frontend HTML where an API error was expected. The SPA fallback should be documented as intentional, or a more targeted approach (e.g., only fallback for paths without a file extension or with specific prefixes) should be considered.
- Decision Reason: The SPA fallback is intentional per the task specification ("Falls back to index.html for SPA routing"). This is standard behavior for single-page applications. The existing API routes (`/api/game/*`) are registered before the catch-all and are not affected. The docstring on `serve_static` documents this behavior.

### CQ-005: No tests for new static file serving routes or path traversal protection
- Status: Resolved
- Description: The task added two new routes (`GET /` and `GET /{path:path}`) and a security-critical path traversal fix (CQ-001), but the test suite (`server/server_tests.py`) has zero tests covering these changes. There is no test that: (a) verifies `GET /` returns HTML or 404 when `static/index.html` is absent, (b) verifies `GET /{path:path}` serves files from `static/` when they exist, (c) verifies path traversal attempts like `GET /../../etc/passwd` return 403. The task.md states "existing server tests continue to pass" but the path traversal fix is a security boundary that warrants regression testing. A single traversal regression could reintroduce an arbitrary file read vulnerability.
- Decision Reason:
