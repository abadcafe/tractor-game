# Test Case Changes: Task-011

## test_stir_during_stirring
- File: server/sm/round_sm_tests.py::TestStirringPhase::test_stir_during_stirring
- Changes: Dynamically finds a valid trump-rank pair in the current player's hand using Counter instead of fabricating card IDs with hardcoded suit/rank.
- Reason: Fabricated card IDs may not exist in the dealt hand, causing test to fail non-deterministically.

## test_playing_all_tricks_to_scoring
- File: server/sm/round_sm_tests.py::TestPlayingPhase::test_playing_all_tricks_to_scoring
- Changes: Assertion relaxed from `state.phase == "SCORING"` to `state.phase in ("SCORING", "COMPLETE")`.
- Reason: SCORING is a transient phase that immediately transitions to COMPLETE during play().

## test_playing_trick_resolved_starts_next
- File: server/sm/round_sm_tests.py::TestPlayingPhase::test_playing_trick_resolved_starts_next
- Changes: Added assertions for trick_history length, defender_points, and new trick lead_player matching previous winner. Changed from dict access to attribute access for CompletedTrickSlot.
- Reason: Original test had no post-condition assertions to verify trick resolution and next trick start.

## test_deal_bid_to_stirring_with_winner
- File: server/sm/round_sm_tests.py::TestDealBidPhase::test_deal_bid_to_stirring_with_winner
- Changes: Added `random.seed(42)` before create_round. Replaced conditional assertions with unconditional assertions.
- Reason: Without fixed seed, _complete_deal_bid_with_reveal might not produce a winner, causing test to pass vacuously.

## test_round_first_round_declarer_from_bid
- File: server/sm/round_sm_tests.py::TestRoundDeclarer::test_round_first_round_declarer_from_bid
- Changes: Added `random.seed(42)` and replaced conditional assertion with unconditional assertion.
- Reason: Same as test_deal_bid_to_stirring_with_winner -- prevents vacuous pass.

## _play_first_legal (helper)
- File: server/sm/round_sm_tests.py::_play_first_legal
- Changes: Changed from dict access (`lead_slot["cards"]`) to attribute access (`lead_slot.cards`). Uses `get_legal_plays` to find valid plays instead of blindly playing hand[0].
- Reason: CompletedTrickSlot uses attribute access not dict access. Blind play violates follow-suit rules.

## _complete_exchange (helper)
- File: server/sm/round_sm_tests.py::_complete_exchange
- Changes: Fixed discard to use `hand_after_pickup[:count]` from exchange state.
- Reason: Corrected to match exchange module API.

## test_scoring_produces_round_result
- File: server/sm/round_sm_tests.py::TestScoringPhase::test_scoring_produces_round_result
- Changes: Changed from conditional `if state.phase == "SCORING"` to unconditional `assert state.phase == "COMPLETE"`. Added assertions for result fields.
- Reason: SCORING is transient; test should verify complete round result.

## test_round_full_round_flow
- File: server/sm/round_sm_tests.py::TestRoundFullFlow::test_round_full_round_flow
- Changes: Replaced conditional `if is_round_complete(state):` with unconditional `assert state.phase == "COMPLETE"` and unconditional result assertions.
- Reason: Test purpose is to validate complete round flow; assertions must be unconditional.

## test_stir_cards_not_in_hand_rejected
- File: server/sm/round_sm_tests.py::TestStirringPhase::test_stir_cards_not_in_hand_rejected
- Changes: Changed regex match from `"hand|not in"` to `"not in hand"`.
- Reason: More precise regex matches only the specific error message.
