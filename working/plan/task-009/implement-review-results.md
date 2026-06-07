# Implement Review Results: Task-009

## Task Review Issues

No issues found. The implementation correctly matches spec section 7 and the task requirements:

- All 3 models (ExchangeInput, ExchangeResult, ExchangeState) match spec definitions.
- `create_exchange` correctly combines declarer hand + bottom cards and sets count.
- `discard` correctly validates count and card membership, then returns new state with COMPLETE phase and ExchangeResult.
- All 10 required tests are present and pass.
- The implementer correctly improved `_make_hand` to accept an `offset` parameter, fixing a bug in the original task specification where hand and bottom cards would share 8 overlapping IDs.

## Code Quality Issues

### CQ-001: Duplicate card IDs allowed in discard
- Status: Resolved
- Description: The `discard` function (exchange.py:72-77) validates that each card ID exists in the hand via a membership check, but does not check for duplicate card IDs in the input `cards` list. If the same card is passed twice (e.g., the same card object repeated to fill 8 slots), each duplicate passes the "in hand" check, but the set-based filtering on line 80 (`discard_set`) only removes each ID once. This means `new_hand` would retain the "extra" copies and be larger than expected, while `new_bottom_cards` would contain duplicate entries. This is a data integrity issue: the declarer could discard fewer unique cards than intended, corrupting game state. The test at exchange_tests.py:110 uses `[fake] * 8` which tests the "not in hand" error path, but a scenario where duplicates of a valid card are passed would silently produce an incorrect result.
- Decision Reason:
