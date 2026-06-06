# Implement Review Results: Task-007

## Task Review Issues

## Code Quality Issues

### CQ-005: Unused `pytest` import in test file
- Status: Resolved
- Description: `server/sm/deal_bid_tests.py:2` imports `pytest` but never uses it (no `pytest.raises`, fixtures, or marks). Ruff flags this as F401. While this does not affect correctness, unused imports reduce code clarity and fail lint checks.
- Decision Reason: Removed unused import.

### CQ-006: Unused `bottom` variables in test helper code
- Status: Resolved
- Description: In `server/sm/deal_bid_tests.py:428` and `server/sm/deal_bid_tests.py:464`, the variable `bottom` is assigned `rest[95:103]` but never used. These appear in `test_reveal_joker_pair_accepted` and `test_reveal_joker_pair_sets_no_trump`, where a custom deck is constructed for both tests with identical boilerplate. Ruff flags these as F841. The fix is to either remove the `bottom` variable assignment or prefix with `_` to indicate intentional non-use.
- Decision Reason: Removed by extracting shared helper `_make_joker_pair_deck()` which does not compute bottom cards.

### CQ-007: Duplicate custom deck construction in two joker tests
- Status: Resolved
- Description: `test_reveal_joker_pair_accepted` (line 415-428) and `test_reveal_joker_pair_sets_no_trump` (line 453-464) contain nearly identical custom deck construction code (same seed 77, same big joker placement at positions 0 and 4, same remaining pool setup). This duplicated setup (~15 lines each) could be extracted into a shared helper like `_make_joker_pair_deck()` to improve maintainability and reduce the risk of the two setups diverging silently in the future.
- Decision Reason: Extracted shared helper `_make_joker_pair_deck()` and replaced both duplicated blocks.
