# Implement Review Results: Task-008

## Task Compliance Issues

### TC-001: _extract_from_tractor only extracts from beginning, missing branching requirement from task spec
- Status: Resolved
- Description: The task spec (Step 3, point 5) explicitly requires: "when extracting K pairs from an N-pair tractor (K < N), there are (N - K + 1) possible consecutive K-pair combinations (starting at pair 1, pair 2, ..., pair N-K+1). Each combination produces a separate option branch." However, `_extract_from_tractor` (line 586) always extracts from the beginning: "For now, take the first 'count' pairs (lowest ranks in the tractor)." This means that when a tractor has 3 pairs and the lead requires 2 pairs, only one branch (the first 2 pairs) is generated instead of 2 possible branches. This causes `_following_plays_new` to return incomplete results -- some legal plays are missing. The test `test_following_tractor_priority` passes only because it asserts `len(p) == 4` without checking the full set of legal options.
- Decision Reason:

## Code Quality Issues

### CQ-001: _extract_from_tractor only extracts from beginning, missing branching requirement
- Status: Resolved
- Description: The task spec (Step 3, point 5) explicitly requires: "when extracting K pairs from an N-pair tractor (K < N), there are (N - K + 1) possible consecutive K-pair combinations (starting at pair 1, pair 2, ..., pair N-K+1). Each combination produces a separate option branch." However, `_extract_from_tractor` (line 586) always extracts from the beginning: "For now, take the first 'count' pairs (lowest ranks in the tractor)." This means that when a tractor has 3 pairs and the lead requires 2 pairs, only one branch (the first 2 pairs) is generated instead of 2 possible branches. This causes `_following_plays_new` to return incomplete results -- some legal plays are missing. The test `test_following_tractor_priority` passes only because it asserts `len(p) == 4` without checking the full set of legal options.
- Decision Reason:

### CQ-002: Redundant filtering in _enumerate_follow_branches fill logic
- Status: Resolved
- Description: At lines 475-480, `suit_singles` is filtered from `all_suit_cards` excluding `used_card_ids`, and then `fill_suit` is filtered again with the identical `c.id not in used_card_ids` check. The second filter is redundant since `suit_singles` already excludes used IDs.
- Decision Reason:

### CQ-003: No deduplication of leading plays across groups
- Status: Don't Fix
- Description: In `_leading_plays_new` (lines 369-373), decompose is called per effective-suit group, and each SubPlay's cards are emitted as an option. However, when two different effective-suit groups produce SubPlays with identical card contents (e.g., same card appearing in two decompositions), no deduplication is applied. Unlike `_following_plays_new` which has a dedup check at line 418-420, `_leading_plays_new` does not deduplicate. While unlikely to cause issues in practice (effective suits partition cards), this inconsistency could be a source of subtle bugs if decompose has edge cases.
- Decision Reason: Effective suits strictly partition cards by construction (each card belongs to exactly one effective suit group). decompose operates on a single effective suit group, so it cannot produce duplicate cards across groups. No bug possible.

### CQ-004: _generate_extractions computes pair_count incorrectly for singles
- Status: Resolved
- Description: At line 547, `total_pair_count += extracted` counts each single card extraction as a "pair" toward `pairs_must`. However, singles (pair_count=0) are not pairs -- they should not contribute to the pair count floor requirement. The comment at line 532 says "but doesn't count as a pair for pair_count" but the code still adds to `total_pair_count`. This could cause incorrect filtering: a valid branch that plays only singles may be rejected because `total_pair_count < pairs_must`, when singles should not count toward the pair requirement.
- Decision Reason:

### CQ-005: Task spec requires `other_hands` parameter in `_following_plays_new` but it is not passed
- Status: Don't Fix
- Description: `get_legal_plays_new` receives `other_hands` as a parameter (line 340) and passes it to `_leading_plays_new` for throw detection, but does not pass it to `_following_plays_new` (line 354). The `_following_plays_new` function signature (line 386) does not include `other_hands`. While this is not strictly needed for the current following logic, it means the function cannot do any verification against other players' known cards (e.g., for future validation or pruning of obviously losing plays). This is a design limitation that should be documented or addressed.
- Decision Reason: The task spec only requires passing `other_hands` to `_leading_plays_new` for throw detection. The following logic does not need `other_hands` -- it uses `is_legal_follow` which only checks suit-following rules and sub-play priority, not what other players hold. Adding it would be premature optimization (YAGNI). If future tasks need it, the signature change is trivial.
