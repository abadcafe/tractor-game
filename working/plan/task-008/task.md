# Task 008: Implement new get_legal_plays in play_rules.py

## Project Overview

- **Goal:** Implement the new SubPlay-based play rules system for the Tractor card game
- **Architecture:** get_legal_plays is the main entry point for UI and AI to enumerate legal plays.
- **Tech Stack:** Python 3.12+, Pydantic, pytest

## Task Objective

Implement the new `get_legal_plays(hand, is_leading, lead_cards, trump_suit, trump_rank, other_hands) -> list[list[Card]]` per spec section 9.3. The new signature replaces the old `lead_action: PlayAction` parameter with `lead_cards: list[Card]` and adds `other_hands: list[Card]`. Returns `list[list[Card]]` instead of `list[PlayAction]`.

This is Task 8 of 13.

---

## Module Design

```
module: play_rules
responsibilities: legal play enumeration
public operations: decompose, is_legal_lead, is_legal_follow, compare_plays_new, can_win, detect_throws_new, get_legal_plays_new
data entities: SubPlay
tests: test_get_legal_plays_new_leading_*, test_get_legal_plays_new_following_*
```

## Files

- Modify: `server/sm/play_rules.py`
- Modify: `server/sm/play_rules_tests.py`

Dependencies: Tasks 001-007 must be complete (decompose from Task-003, is_legal_lead/_is_biggest from Task-004, is_legal_follow from Task-005, detect_throws_new from Task-007).

## Steps

- [x] **Step 1: Write complete test code in play_rules_tests.py**

Add a new test class `TestGetLegalPlaysNew`:

```python
from server.sm.play_rules import get_legal_plays_new


class TestGetLegalPlaysNew:
    # --- Leading ---
    def test_leading_returns_singles(self) -> None:
        """Leading: each card is a single option."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.HEARTS, Rank.KING)]
        plays = get_legal_plays_new(hand, True, None, Suit.SPADES, Rank.TWO, [])
        singles = [p for p in plays if len(p) == 1]
        assert len(singles) >= 2

    def test_leading_returns_pairs(self) -> None:
        """Leading: pairs are options."""
        hand = [_card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2)]
        plays = get_legal_plays_new(hand, True, None, Suit.SPADES, Rank.TWO, [])
        pairs = [p for p in plays if len(p) == 2]
        assert len(pairs) >= 1

    def test_leading_returns_tractors(self) -> None:
        """Leading: tractors are options."""
        hand = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
        ]
        plays = get_legal_plays_new(hand, True, None, Suit.SPADES, Rank.TWO, [])
        tractors = [p for p in plays if len(p) == 4]
        assert len(tractors) >= 1

    def test_leading_returns_valid_throws(self) -> None:
        """Leading: valid throws (all sub-plays biggest) are options."""
        hand = [_card(Suit.SPADES, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        plays = get_legal_plays_new(hand, True, None, Suit.HEARTS, Rank.TWO, [])
        throws = [p for p in plays if len(p) == 2]
        assert len(throws) >= 1

    # --- Following ---
    def test_following_single_must_follow(self) -> None:
        """Following single: must play same suit if possible."""
        hand = [_card(Suit.HEARTS, Rank.ACE), _card(Suit.SPADES, Rank.KING)]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        plays = get_legal_plays_new(hand, False, lead, Suit.SPADES, Rank.TWO, [])
        # All plays must be heart
        for p in plays:
            assert all(c.suit == Suit.HEARTS for c in p)

    def test_following_single_no_suit(self) -> None:
        """Following single with no matching suit: can play anything."""
        hand = [_card(Suit.SPADES, Rank.KING)]
        lead = [_card(Suit.HEARTS, Rank.QUEEN)]
        plays = get_legal_plays_new(hand, False, lead, Suit.SPADES, Rank.TWO, [])
        assert len(plays) >= 1

    def test_following_pair_must_follow(self) -> None:
        """Following pair: must play pair of same suit if available."""
        hand = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.SPADES, Rank.KING, 1), _card(Suit.SPADES, Rank.KING, 2),
        ]
        lead = [_card(Suit.HEARTS, Rank.QUEEN, 1), _card(Suit.HEARTS, Rank.QUEEN, 2)]
        plays = get_legal_plays_new(hand, False, lead, Suit.SPADES, Rank.TWO, [])
        # Must include hA pair option
        has_heart_pair = any(
            len(p) == 2 and all(c.suit == Suit.HEARTS for c in p)
            for p in plays
        )
        assert has_heart_pair

    def test_following_tractor_priority(self) -> None:
        """Following tractor: must use higher-level sub-plays first."""
        # Hand: tractor h3-3-4-4-5-5 + pair hK-K. Lead: 2-pair tractor.
        hand = [
            _card(Suit.HEARTS, Rank.THREE, 1), _card(Suit.HEARTS, Rank.THREE, 2),
            _card(Suit.HEARTS, Rank.FOUR, 1), _card(Suit.HEARTS, Rank.FOUR, 2),
            _card(Suit.HEARTS, Rank.FIVE, 1), _card(Suit.HEARTS, Rank.FIVE, 2),
            _card(Suit.HEARTS, Rank.KING, 1), _card(Suit.HEARTS, Rank.KING, 2),
        ]
        lead = [
            _card(Suit.HEARTS, Rank.ACE, 1), _card(Suit.HEARTS, Rank.ACE, 2),
            _card(Suit.HEARTS, Rank.QUEEN, 1), _card(Suit.HEARTS, Rank.QUEEN, 2),
        ]
        plays = get_legal_plays_new(hand, False, lead, Suit.SPADES, Rank.TWO, [])
        # All plays must be 4 cards
        for p in plays:
            assert len(p) == 4

    def test_following_empty_lead_cards(self) -> None:
        """Following with empty lead_cards -> returns empty (wait for lead)."""
        hand = [_card(Suit.HEARTS, Rank.ACE)]
        plays = get_legal_plays_new(hand, False, [], Suit.SPADES, Rank.TWO, [])
        assert plays == []

    def test_following_lead_cards_none(self) -> None:
        """Following with lead_cards=None -> returns empty."""
        hand = [_card(Suit.HEARTS, Rank.ACE)]
        plays = get_legal_plays_new(hand, False, None, Suit.SPADES, Rank.TWO, [])
        assert plays == []
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd /home/lfw/works/tractor-game && python -m pytest server/sm/play_rules_tests.py::TestGetLegalPlaysNew -x -v 2>&1 | tail -30`
Expected: FAIL -- new get_legal_plays not importable with new signature

- [x] **Step 3: Implement new get_legal_plays in play_rules.py**

Add `get_legal_plays_new(hand, is_leading, lead_cards, trump_suit, trump_rank, other_hands) -> list[list[Card]]`:

**Leading:**
1. Group hand by effective suit
2. For each group, decompose -> emit each SubPlay as an option (singles, pairs, tractors)
3. Call `detect_throws_new` for throw options
4. Return all options as `list[list[Card]]`

**Following:**
1. If lead_cards is None or empty -> return []
2. Compute lead_pair_count from decompose(lead_cards)
3. Separate hand into suit_cards (same effective suit as lead) and other_cards
4. Decompose suit_cards, sort by level descending
5. Extract pairs/tractors from highest level down to fill lead_pair_count. **Branching**: when extracting K pairs from an N-pair tractor (K < N), there are (N - K + 1) possible consecutive K-pair combinations (starting at pair 1, pair 2, ..., pair N-K+1). Each combination produces a separate option branch. Also, when there are multiple same-level alternative sub-plays (e.g., two independent pairs), each unused alternative can replace a used one, producing additional branches.
6. For each branch, fill remaining slots with same-suit singles, then other-suit cards
7. Return all valid options (each validated by is_legal_follow)

NOTE: Temporarily name it `get_legal_plays_new`. Will replace old function after integration.

- [x] **Step 4: Run tests to verify they pass**

Run: `cd /home/lfw/works/tractor-game && python -m pytest server/sm/play_rules_tests.py::TestGetLegalPlaysNew -x -v 2>&1 | tail -30`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/sm/play_rules.py server/sm/play_rules_tests.py
git commit -m "feat(play_rules): implement new get_legal_plays with SubPlay-based enumeration"
```
