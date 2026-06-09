# Task Issues

## TI-001: test_detect_throws_new_multiple_suits contradicts test_detect_throws_new_single_card_not_throw
- **Description**: test_detect_throws_new_multiple_suits has 1 card per suit (spA, dA) and expects len(throws)==2, but test_detect_throws_new_single_card_not_throw has 1 card and expects len(throws)==0 with comment "Single card per suit -> not a throw (need 2+ sub-plays)". A single card decomposes to 1 sub-play, so by the "2+ sub-plays" rule it should not be a throw. The multiple_suits test contradicts this.
- **Assumption**: The "2+ sub-plays" rule from single_card_not_throw is correct (throw requires 2+ sub-plays). The multiple_suits test has a bug: each suit has only 1 card (1 sub-play), so it should return 0 throws. I will fix the test to have 2 cards per suit to be consistent with other tests.
