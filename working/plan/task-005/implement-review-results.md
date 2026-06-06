# Implement Review Results: Task-005

## Task Review Issues

### TR-001: test_detect_tractors_three_pairs uses Rank.FIVE instead of task-specified Rank.SIX
- Status: Don't Fix
- Description: The task.md test code at line 100 specifies `Rank.SIX` for the third pair in `test_detect_tractors_three_pairs`, but the implementation uses `Rank.FIVE`. The test description says "Three consecutive pairs form a 6-card tractor" but the task.md data (THREE, FOUR, SIX) is NOT consecutive -- there is a gap at FIVE. The implementation correctly changed this to Rank.FIVE to make the test match its description and be a meaningful assertion. This is a correct fix of a bug in the task specification.
- Decision Reason: The task.md test specification had a bug: Rank.SIX with Rank.THREE/FOUR and trump_rank=TWO does not form consecutive pairs, so the test assertion would fail. The implementer correctly fixed this to Rank.FIVE which is consecutive with THREE and FOUR.

## Code Quality Issues

### CQ-001: _follow_tractor produces duplicate cards in play
- Status: Resolved
- Description: In `_follow_tractor` (play_rules.py:463-467), the second loop that collects "unused singles from matching suit" duplicates cards already added by the first loop (lines 456-461). The first loop adds non-pair cards to `singles_in_suit`; the second loop iterates ALL `suit_cards` and adds any card whose rank is not in `used_ranks`, but those cards are already in `singles_in_suit`. This results in the same card appearing twice in the returned play. Verified: hand with 1 King produces a 4-card play with the same King appearing twice. This violates the invariant that all cards in a play must be distinct cards from the player's hand.
- Decision Reason: Removed the duplicate loop. Also fixed the fill logic to draw from any remaining hand cards when same-suit singles are insufficient, per spec "补足" rule.

### CQ-002: Unused import `combinations` from itertools
- Status: Resolved
- Description: `from itertools import combinations` at play_rules.py:7 is imported but never used anywhere in the module. Dead code.
- Decision Reason: Removed unused import.

### CQ-003: Unused import `pytest` in test file
- Status: Resolved
- Description: `import pytest` at play_rules_tests.py:2 is imported but never used. No pytest fixtures or markers are referenced.
- Decision Reason: Removed unused import.

### CQ-004: get_legal_plays leading returns duplicate play interpretations
- Status: Don't Fix
- Description: When leading, `get_legal_plays` (play_rules.py:257-262) returns all singles + all pairs + all tractors + all throws. A pair of cards (e.g., two Aces of hearts) is returned as both a PAIR play and a THROW play with identical card sets. This means the same physical play appears twice with different type labels. While this may be acceptable (the caller picks one interpretation), it could cause confusion in the UI or game logic if both are presented as distinct options. The spec (line 659) says `lead_type = infer(cards)` which uses `infer_play_type` to determine the single correct type, so having both in the options list is redundant.
- Decision Reason: The task Step 1 test code is authoritative and includes tests verifying leading returns singles, pairs, tractors, and throws as separate lists. The caller uses `infer_play_type` to determine the actual type. Deduplication is not required by the spec or task.

### CQ-005: _find_consecutive_pairs docstring is inaccurate
- Status: Resolved
- Description: The docstring at play_rules.py:79-83 says "Also returns individual pairs as separate PlayActions" but the function only returns TRACTOR plays. Individual pairs are not emitted. The docstring is misleading. The public `detect_tractors` function is correct (it only needs tractors), and `detect_pairs` handles individual pairs separately, so this is purely a documentation issue.
- Decision Reason: Removed inaccurate sentence from docstring.

### CQ-006: Tests have weak assertions on throw detection
- Status: Don't Fix
- Description: Several throw detection tests (TestDetectThrows) only assert `isinstance(throws, list)` without checking actual behavior. For example, `test_detect_throws_all_highest_in_suit` and `test_detect_throws_partial_high_cards` don't verify that throws are correctly detected or rejected. The tests pass trivially. While the test for `test_detect_throws_no_throw_when_lower_rank_exists` is more meaningful, the overall throw test coverage is weak.
- Decision Reason: The test code is authoritative as specified in Task Step 1. The tests match the task specification exactly. Strengthening assertions is outside the scope of this task.

### CQ-007: _find_consecutive_pairs uses incorrect rank ordering for trump effective suit
- Status: Resolved
- Description: In `_find_consecutive_pairs` (play_rules.py:93), when sorting pair ranks for a "trump" effective suit group, the code falls back to `Suit.HEARTS` when `suit` is the string `"trump"`: `_rank_order_for_suit(r, suit if isinstance(suit, Suit) else Suit.HEARTS, trump_suit, trump_rank)`. This means `_rank_order_for_suit` only uses `_trump_rank_order` when `trump_suit == HEARTS`. When `trump_suit` is any other suit (e.g., SPADES), trump cards incorrectly use `_non_trump_rank_order` (which skips the trump rank from ordering), potentially giving wrong consecutive detection results. The same fallback appears in `infer_play_type` (lines 319-324, 328-339). The correct fix would be to special-case `"trump"` in `_rank_order_for_suit` or detect it in `_find_consecutive_pairs` and call `_trump_rank_order` directly. Impact: When `trump_suit != HEARTS`, the rank ordering for trump card tractors may be wrong, potentially causing valid tractors to be missed or invalid ones to be detected.
- Decision Reason: Added "trump" string check in `_rank_order_for_suit` and removed the Suit.HEARTS fallback at all call sites. Now the effective suit string "trump" is passed directly, which correctly resolves to trump rank ordering regardless of which suit is the trump suit.

### CQ-008: detect_throws with known_remaining_cards=None returns all throws instead of none
- Status: Resolved
- Description: The task Step 3 specification states: "If known_remaining_cards is None, assume all cards not in hand are potentially remaining (conservative: no THROW possible unless we can verify)." However, the implementation (play_rules.py:224-232) treats `None` as "no remaining cards exist" -- when `remaining_by_suit` is empty, `any(...)` over an empty set returns `False`, so all cards are considered throwable. This is the OPPOSITE of the specified conservative behavior. The docstring (lines 189-191) also states the conservative intent but the code does not match. The tests pass because they only check `isinstance(throws, list)` (CQ-006). In practice, `get_legal_plays` when leading calls `detect_throws` without `known_remaining_cards`, so ALL same-suit multi-card plays are offered as THROW options even when the throw cannot be verified.
- Decision Reason: Added early return of `[]` when `known_remaining_cards is None`, matching the specified conservative behavior.

### CQ-009: _follow_throw returns only one play option when player has enough suit cards
- Status: Resolved
- Description: In `_follow_throw` (play_rules.py:516-523), when the player has enough cards of the matching suit, only one play is returned: the highest-ranked cards sorted by `_non_trump_rank_order`. The spec says "有同花色足够牌→必须出" (must play same-suit cards), but the player should have the option of WHICH cards to play from the suit (e.g., they might want to hold high cards). The function hardcodes "play highest" as the only option. This limits player agency and may not match the intended game behavior where the player chooses which suit cards to play.
- Decision Reason: Changed to return all C(n,k) combinations of suit cards of the correct length using itertools.combinations, giving the player full choice of which cards to play.

### CQ-010: _follow_pair no-match fallback generates O(n^2) combinations
- Status: Resolved
- Description: In `_follow_pair` (play_rules.py:406-409), when the player has no matching pair, the code generates all C(n,2) combinations of hand cards. For a full hand of 25 cards, this produces 300 play options. While functionally correct, this is inefficient and may cause performance issues when the hand is large. Each combination is a separate PlayAction with 2 cards. The caller must process all of them. A more practical approach would be to return a smaller set of representative options or a single "play any 2 cards" sentinel.
- Decision Reason: Replaced O(n^2) enumeration with a single representative play (first 2 cards from hand), since the caller can choose which specific cards to play from a generic "play any 2" option.

### TR-002: Unused imports is_trump_card and trump_order in play_rules.py
- Status: Resolved
- Description: Line 8 of play_rules.py imports `is_trump_card` and `trump_order` from `server.sm.comparator`, but neither symbol is used anywhere in the module. This is dead code (similar to the previously resolved CQ-002 for itertools.combinations and CQ-003 for pytest). While it does not affect correctness, it adds unnecessary coupling and violates clean-import conventions.
- Decision Reason: Removed unused imports.

### CQ-011: Unused variable `tractors` in test_detect_tractors_trump_group
- Status: Resolved
- Description: In play_rules_tests.py:95, the variable `tractors` is assigned from `detect_tractors(hand, Suit.HEARTS, Rank.TWO)` but is never referenced. The test only uses `pairs` (line 98). This causes a ruff F841 lint warning. Dead code in tests reduces maintainability and signals that the test may be incomplete -- either the variable should be asserted on or removed.
- Decision Reason: Removed the unused variable assignment. The function is still called for its side-effect of not crashing, but the return value is discarded.

### CQ-012: _follow_tractor type annotation is inaccurate on rank_groups
- Status: Resolved
- Description: In `_follow_tractor` (play_rules.py:449), the type annotation is `dict[Suit | tuple, list[Card]]`. The actual keys are always `tuple[Suit, Rank]` (line 451: `key = (c.suit, c.rank)`). The `Suit` in the union is never used as a key type, making the annotation misleading. Should be `dict[tuple[Suit, Rank], list[Card]]`.
- Decision Reason: Fixed the type annotation to `dict[tuple[Suit, Rank], list[Card]]`.

### CQ-013: _follow_tractor returns play with fewer cards than required when hand is insufficient
- Status: Resolved
- Description: In `_follow_tractor` (play_rules.py:462-488), when the player has pairs in the matching suit but insufficient total cards to fill `tractor_len`, the function still returns a play with fewer cards than required. For example, with a 6-card tractor lead, a hand containing 1 pair (2 cards) + 2 singles (2 cards) + 1 card of another suit (1 card) = 5 total cards returns a 5-card play instead of 0 plays. The spec requires playing the same number of cards as the lead. The `fill_needed` calculation (line 466) becomes positive but `remaining[:max(0, fill_needed - len(fill_cards))]` draws fewer cards than needed, and `combo_cards.extend(fill_cards[:fill_needed])` silently truncates. The resulting play violates the invariant that a follow play must have exactly `tractor_len` cards.
- Decision Reason: Added a check after collecting fill_cards: if `len(fill_cards) < fill_needed`, return empty list `[]`. This ensures the invariant that a follow play must have exactly `tractor_len` cards.

### CQ-014: _follow_throw returns play with fewer cards than required when hand is insufficient
- Status: Resolved
- Description: In `_follow_throw` (play_rules.py:528-532), when the player has some cards of the matching suit but fewer than `throw_len` total cards, the function returns a play with fewer cards than the lead requires. For example, with a 2-card throw lead, a hand with 1 matching suit card and 0 other cards returns a 1-card play instead of 0 plays. The spec says "有同花色足够牌→必须出；不足→出所有+补足" -- the "fill" part assumes enough other cards exist. When `len(suit_cards) + len(other_cards) < throw_len`, the fill silently truncates. Similarly, with 1 suit card + 1 other card and a 3-card throw lead, only 2 cards are returned. The play should have exactly `throw_len` cards or not exist at all.
- Decision Reason: Added a check: if `len(other_cards) < fill_needed`, return empty list `[]`. This ensures the invariant that a follow play must have exactly `throw_len` cards.

### CQ-015: Local import of itertools.combinations inside _follow_throw
- Status: Resolved
- Description: `from itertools import combinations` at play_rules.py:522 is a local import inside `_follow_throw`. All other imports in the module are at the top of the file (lines 7-9). Local imports are inconsistent with the codebase convention and are less visible during code review. The import was originally at module level (CQ-002 era), removed as "unused", then re-added as a local import when CQ-009 added the combinations logic. It should be moved to the top-level imports alongside the other standard library and project imports.
- Decision Reason: Moved `from itertools import combinations` to the top-level imports block.

### CQ-016: detect_pairs and follow functions use actual suit for grouping instead of effective suit for trump cards
- Status: Resolved
- Description: `detect_pairs` (line 151-158), `_follow_pair` (line 399-402), `_follow_tractor` (line 451-454), and `infer_play_type` (line 318-321) all group cards by `(c.suit, c.rank)` -- the card's actual suit and rank. The spec (line 696) defines a pair as "同有效花色、同Rank的2张" (same effective suit, same rank). For trump cards, the effective suit is "trump" regardless of actual suit. This means two cards of the same trump rank but different actual suits (e.g., hearts-2 + spades-2 when trump_suit=HEARTS, trump_rank=TWO) are NOT detected as a pair, even though they should be per spec. The `detect_pairs` function should group by `(effective_suit(c), c.rank)` for trump cards. Similarly, `_follow_pair` and `_follow_tractor` should use effective suit grouping when the lead is trump. In practice, this means cross-suit trump rank pairs are silently missed -- the fallback "any N cards" path returns the correct number of cards but via the wrong mechanism, and does not provide the pair/tractor option the player is entitled to.
- Decision Reason: Fixed all four functions to group by `(effective_suit(c), c.rank)` instead of `(c.suit, c.rank)`. Added optional `trump_suit`/`trump_rank` params to `detect_pairs` for backward compatibility. Also fixed `detect_pairs` to emit pairs of 2 from groups larger than 2 (e.g., 4 cards of same trump rank now yield 2 pairs). Updated `test_detect_tractors_trump_group` to pass trump params. Fixed `infer_play_type` 2-card pair check to use effective suit comparison.
