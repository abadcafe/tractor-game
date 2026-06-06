# Implement Review Results: Task-007

## Task Review Issues

## Code Quality Issues

### CQ-001: reveal() does not validate len(cards) == event.count for trump_rank bids
- Status: Resolved
- Description: In `server/sm/deal_bid.py:155-163`, the `reveal()` function validates `event.count in (1, 2)` for trump_rank bids but never checks that `len(event.cards) == event.count`. This allows a player to submit a bid with `count=2` but only 1 card (or vice versa). The `bid_value()` function in `server/sm/comparator.py:162` computes value based on `len(cards)`, not `event.count`, so the comparison logic works on the actual card count. However, `bid_winner.count` is set to the inconsistent `event.count` value. This creates a state where `bid_winner.count` says "single" but the actual bid value is "pair" level (e.g., 1 card submitted with count=2 gets stored as count=2 but bid_value=102 because len(cards)=1). A subsequent valid bid would be compared against the wrong internal representation. The joker path correctly validates both `event.count == 2` and `len(event.cards) == 2`, but the trump_rank path lacks the equivalent `len(cards) == event.count` check.
- Decision Reason: Fixed by adding `if len(event.cards) != event.count: return state` check after the `event.count in (1, 2)` validation in reveal().

### CQ-002: Missing test coverage for count/cards length mismatch
- Status: Resolved
- Description: There is no test in `server/sm/deal_bid_tests.py` that verifies `reveal()` rejects bids where `event.count` does not match `len(event.cards)` for trump_rank bids. The existing test suite does not cover this edge case, which is why CQ-001 was not caught during implementation. Tests should verify that: (1) a bid with count=2 but only 1 card is rejected, (2) a bid with count=1 but 2 cards is rejected or accepted consistently.
- Decision Reason: Added two tests: test_reveal_count_cards_mismatch_rejected (count=2 with 1 card) and test_reveal_count_one_with_two_cards_rejected (count=1 with 2 cards). Both verify bid_events count is unchanged after the rejected bid.
