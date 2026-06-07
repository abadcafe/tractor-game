# Implement Review Results: Task-015

## Task Compliance Issues

## Code Quality Issues

### TC-001: Missing test for scoring: null case
- Status: Resolved
- Description: `StateSnapshot.scoring` is typed as `{ ... } | null` (types.ts line 83-87). The implementation guards with `if (snapshot.scoring)` on line 20 of scoring-overlay.ts, but there is no test verifying that `renderScoringOverlay` handles a `scoring: null` snapshot gracefully (returns an empty overlay without throwing). The `test_renderScoringOverlay_no_button_when_not_human` test overrides `declarer_player` and `interactionMode` but still uses the default `scoring` object -- it does not exercise the null-scoring code path. Edge-case correctness is untested.
- Decision Reason:

### TC-002: Missing test for onNextRound callback not provided
- Status: Resolved
- Description: When `interactionMode === "next_round"` but `onNextRound` is not provided (scoring-overlay.ts line 41: `if (onNextRound)`), the button is still rendered but has no click handler. There is no test verifying this behavior -- that the button renders but clicking it is a no-op. This is a valid edge case (e.g. spectator mode where button visibility is driven by a different check). Without a test, this behavior could regress silently.
- Decision Reason:
