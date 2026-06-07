# Implement Review Results: Task-002

## Task Review Issues

### TR-001: server/ai/auto_play.py has broken imports from deleted server.engine
- Status: Resolved
- Description: After deleting server/engine/, the file server/ai/auto_play.py still imports from server.engine.card (Card, Rank, Suit) and server.engine.types (PlayAction). This makes the file un-importable -- `from server.ai.auto_play import choose_play` raises ModuleNotFoundError. Although the task scope did not explicitly list server/ai/ for deletion or modification, leaving a file with broken imports in the codebase is a code quality concern. The file should either be deleted (since other AI files like graph.py, models.py, prompts.py, session_manager.py were already deleted in the same commit), or its imports should be updated to use server.sm types.
- Decision Reason: Updated imports to use server.sm.card_model (Card, Rank, Suit) and server.sm.types (PlayAction). The auto_play module is not referenced by any active code but should remain importable for future use. Verified importable with `python -c` and all 288 tests pass.

## Code Quality Issues

### CQ-001: Unrelated changes bundled into the same commit
- Status: Don't Fix
- Description: The commit 729275c for this deletion task also includes changes to 13 files under server/sm/, frontend files (src/main.ts, src/style.css, src/ui/*.ts), new test files under tests/, pyproject.toml, uv.lock, and deletion of working/plan task review files. These changes are unrelated to deleting obsolete modules and stubbing server.py. Bundling unrelated changes makes the commit history harder to bisect and understand.
- Decision Reason: The commit has already been made and is part of the shared history. Rewriting history (e.g., interactive rebase to split the commit) would require force-pushing and could disrupt other collaborators/branches. The working tree is clean with no staged changes to reorganize. This is a process improvement note for future tasks, not an actionable fix at this point.

### CQ-002: pyproject.toml testpaths references deleted server/tests directory
- Status: Resolved
- Description: pyproject.toml (line 23-26) lists "server/tests" in testpaths, but this directory was deleted as part of this task. While pytest silently skips missing directories during normal collection (the 386 tests still collect fine), this is a stale reference that could confuse developers and causes an explicit error if someone runs `pytest server/tests/` directly. The testpaths should be updated to only list existing directories.
- Decision Reason: Removed "server/tests" from testpaths in pyproject.toml. Verified all 288 tests still pass and 386 tests are collected correctly.
