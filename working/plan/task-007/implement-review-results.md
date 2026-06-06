# Implement Review Results: Task-007

## Task Review Issues

### TR-001: reveal() does not reject duplicate card IDs in event.cards
- Status: Resolved
- Description: In `server/sm/deal_bid.py:152-156`, the hand validation builds a set of card IDs from the player's hand and checks each card in event.cards is in that set. However, it does not check for duplicate card IDs within event.cards itself. A player could submit the same card ID twice in event.cards (e.g., for a pair bid with count=2), and both entries would pass the hand check since the ID exists in the hand set. This means a player could claim to reveal a pair using the same physical card twice, which is semantically invalid -- a pair requires two distinct physical cards. With two decks in play, cards of the same suit and rank have different IDs, so a legitimate pair would always have two distinct IDs. The fix is to check `len(set(c.id for c in event.cards)) == len(event.cards)` after the hand validation.
- Decision Reason: Added duplicate ID check `len(set(c.id for c in event.cards)) != len(event.cards)` after hand validation. Added test_reveal_duplicate_card_ids_rejected to verify.

### TR-002: reveal() does not validate that pair cards are distinct physical cards (same suit + rank for trump_rank)
- Status: Resolved
- Description: For trump_rank pair bids (count=2), the implementation checks that each card has rank==trump_rank and suit==event.suit, but does not verify that the two cards are actually a pair (same suit and rank from different decks). While the existing checks do ensure same suit and same rank (since each card must match event.suit and trump_rank), the deeper issue is that with two identical decks, a valid pair requires two cards with the same suit and rank but different IDs. The duplicate ID check from TR-001 would catch the most obvious case (same card submitted twice), but there is no explicit validation that two cards with different IDs but same suit/rank actually exist as a pair. This is a minor concern since the caller is responsible for submitting valid cards, and the hand check ensures cards are physically held. However, for robustness, an explicit pair-uniqueness check (e.g., all card IDs are distinct) would be more defensive.
- Decision Reason: Resolved by TR-001 fix. The duplicate ID check ensures cards are physically distinct. Cards with same suit+rank but different IDs (from different decks) are correctly handled as a valid pair.

## Code Quality Issues

### CQ-001: reveal() does not validate len(cards) == event.count for trump_rank bids
- Status: Resolved
- Description: In `server/sm/deal_bid.py:155-163`, the `reveal()` function validates `event.count in (1, 2)` for trump_rank bids but never checks that `len(event.cards) == event.count`. This allows a player to submit a bid with `count=2` but only 1 card (or vice versa). The `bid_value()` function in `server/sm/comparator.py:162` computes value based on `len(cards)`, not `event.count`, so the comparison logic works on the actual card count. However, `bid_winner.count` is set to the inconsistent `event.count` value. This creates a state where `bid_winner.count` says "single" but the actual bid value is "pair" level (e.g., 1 card submitted with count=2 gets stored as count=2 but bid_value=102 because len(cards)=1). A subsequent valid bid would be compared against the wrong internal representation. The joker path correctly validates both `event.count == 2` and `len(event.cards) == 2`, but the trump_rank path lacks the equivalent `len(cards) == event.count` check.
- Decision Reason: Fixed by adding `if len(event.cards) != event.count: return state` check after the `event.count in (1, 2)` validation in reveal().

### CQ-002: Missing test coverage for count/cards length mismatch
- Status: Resolved
- Description: There is no test in `server/sm/deal_bid_tests.py` that verifies `reveal()` rejects bids where `event.count` does not match `len(event.cards)` for trump_rank bids. The existing test suite does not cover this edge case, which is why CQ-001 was not caught during implementation. Tests should verify that: (1) a bid with count=2 but only 1 card is rejected, (2) a bid with count=1 but 2 cards is rejected or accepted consistently.
- Decision Reason: Added two tests: test_reveal_count_cards_mismatch_rejected (count=2 with 1 card) and test_reveal_count_one_with_two_cards_rejected (count=1 with 2 cards). Both verify bid_events count is unchanged after the rejected bid.

### CQ-003: Misleading test names and docstrings for deal-all-card tests
- Status: Resolved
- Description: `test_deal_next_card_all_dealt_with_bid` (line 149) has a docstring "After 100 cards dealt with a bid, phase = COMPLETE" but never places any bid. It asserts `state.phase in ("COMPLETE", "NO_BID")` which always resolves to `NO_BID` in practice, making the assertion weaker than intended. Similarly, `test_deal_bid_full_flow_with_bids` (line 505) has a docstring "Complete flow: deal all cards, someone bids, result has winner" but also never places any bid and asserts `state.phase == "NO_BID"`. The COMPLETE path IS covered by `test_reveal_joker_pair_sets_no_trump`, but the test that claims to cover "full flow with bids" does not actually test a bid scenario. This makes the test suite harder to reason about and could give false confidence in coverage of the COMPLETE transition path.
- Decision Reason: Fixed `test_deal_next_card_all_dealt_with_bid` to actually place a bid mid-deal using deterministic deck (deals 5 cards, reveals ♠TWO, deals remaining 95) and assert phase == COMPLETE. Renamed `test_deal_bid_full_flow_with_bids` to `test_deal_bid_full_flow_no_bid` with corrected docstring.

### CQ-004: reveal() does not validate player index bounds
- Status: Resolved
- Description: In `server/sm/deal_bid.py:145-146`, `reveal()` accesses `state.players_hand[event.player]` without checking that `event.player` is in range [0, 4). An out-of-range player index would cause an unhandled IndexError at runtime. While the state machine architecture assumes valid inputs from callers, defensive bounds checking (returning state unchanged for invalid players) would prevent runtime crashes from malformed input and make the module more robust as a standalone unit.
- Decision Reason: Added bounds check `if event.player < 0 or event.player >= 4: return state` as the first validation in reveal(), before any other precondition checks. Returns the state unchanged for invalid player indices, consistent with how other invalid inputs are handled.
