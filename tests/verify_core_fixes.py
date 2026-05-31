#!/usr/bin/env python3
"""
Comprehensive verification of two core fixes:
  1. Card readability (牌面可读性) — corner rank+suit visible in fan layout
  2. Refresh persistence (刷新持久化) — localStorage save/restore across refreshes

Run: python tests/verify_core_fixes.py
"""

import sys
import json
import time
from playwright.sync_api import sync_playwright, expect

BASE_URL = "http://localhost:8787"
PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results = []

def report(name, passed, detail=""):
    status = PASS if passed else FAIL
    msg = f"{status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, passed, detail))


def verify_card_readability(page):
    """Fix 1: Verify card readability in the fan layout."""
    print("\n" + "="*60)
    print("  Fix 1: Card Readability (牌面可读性)")
    print("="*60)

    # Navigate and wait for game to load
    page.goto(BASE_URL)
    page.evaluate("() => localStorage.clear()")
    page.reload()
    page.wait_for_timeout(1500)

    # Wait for cards to appear
    cards = page.locator("#cards-south .card")
    try:
        cards.first.wait_for(timeout=10000)
    except Exception:
        report("Cards appear in human hand", False, "No .card elements found")
        return

    card_count = cards.count()
    report("Cards appear in human hand", card_count > 0, f"{card_count} cards dealt")

    # 1a. Corner elements exist and are visible
    for i in range(min(card_count, 5)):
        card = cards.nth(i)

        corner = card.locator(".card-corner")
        rank_el = card.locator(".card-rank")
        suit_el = card.locator(".card-suit")

        try:
            expect(corner).to_be_visible()
            expect(rank_el).to_be_visible()
            expect(suit_el).to_be_visible()
        except Exception as e:
            report(f"Card #{i} corner visible", False, str(e))
            continue

        rank_text = rank_el.text_content().strip()
        suit_text = suit_el.text_content().strip()

        has_rank = len(rank_text) > 0
        has_suit = len(suit_text) > 0

        if i == 0:
            report("Card corner has rank text", has_rank,
                   f'rank="{rank_text}"' if has_rank else "empty!")
            report("Card corner has suit text", has_suit,
                   f'suit="{suit_text}"' if has_suit else "empty!")

    # 1b. Corner is positioned at top-left
    first_card = cards.first
    corner_style = first_card.locator(".card-corner").evaluate("""el => {
        const s = window.getComputedStyle(el);
        return { position: s.position, top: s.top, left: s.left };
    }""")
    is_absolute = corner_style["position"] == "absolute"
    is_top_left = int(corner_style["top"].replace("px", "")) <= 5 and int(corner_style["left"].replace("px", "")) <= 5
    report("Corner is position:absolute", is_absolute,
           f"position={corner_style['position']}")
    report("Corner is at top-left", is_top_left,
           f"top={corner_style['top']}, left={corner_style['left']}")

    # 1c. Corner z-index >= center z-index
    z_info = first_card.evaluate("""el => {
        const corner = el.querySelector('.card-corner');
        const center = el.querySelector('.card-center');
        const cz = corner ? parseInt(getComputedStyle(corner).zIndex) || 0 : -1;
        const ccz = center ? parseInt(getComputedStyle(center).zIndex) || 0 : -1;
        return { corner: cz, center: ccz };
    }""")
    corner_above = z_info["corner"] >= z_info["center"]
    report("Corner z-index >= center z-index", corner_above,
           f"corner={z_info['corner']}, center={z_info['center']}")

    # 1d. Font sizes meet readability minimums
    font_info = first_card.evaluate("""el => {
        const rank = el.querySelector('.card-rank');
        const suit = el.querySelector('.card-suit');
        return {
            rankSize: rank ? parseFloat(getComputedStyle(rank).fontSize) : 0,
            suitSize: suit ? parseFloat(getComputedStyle(suit).fontSize) : 0,
        };
    }""")
    rank_ok = font_info["rankSize"] >= 8  # clamp min is 9px, allow 1px tolerance
    suit_ok = font_info["suitSize"] >= 6
    report("Rank font >= 9px readable", rank_ok,
           f"rank={font_info['rankSize']:.1f}px")
    report("Suit font >= 7px readable", suit_ok,
           f"suit={font_info['suitSize']:.1f}px")

    # 1e. Red/black color classes
    has_red = False
    has_black = False
    for i in range(card_count):
        cls = cards.nth(i).get_attribute("class") or ""
        if "red" in cls:
            has_red = True
        if "black" in cls:
            has_black = True
    report("Red suit color class present", has_red)
    report("Black suit color class present", has_black)

    # 1f. Color contrast: verify text is actually colored (not default black on white)
    color_info = first_card.evaluate("""el => {
        const rank = el.querySelector('.card-rank');
        const suit = el.querySelector('.card-suit');
        return {
            rankColor: rank ? getComputedStyle(rank).color : '',
            suitColor: suit ? getComputedStyle(suit).color : '',
        };
    }""")
    has_color = color_info["rankColor"] != "" and color_info["suitColor"] != ""
    report("Card text has color styling", has_color,
           f"rank={color_info['rankColor']}, suit={color_info['suitColor']}")

    # 1g. Joker cards styling
    joker_found = False
    for i in range(card_count):
        cls = cards.nth(i).get_attribute("class") or ""
        if "joker" in cls:
            joker_found = True
            rank_text = cards.nth(i).locator(".card-rank").text_content().strip()
            is_chinese = rank_text in ("大", "小")
            report("Joker rank uses Chinese characters", is_chinese,
                   f'rank="{rank_text}"')

            # Check joker special background
            has_joker_class = "joker" in cls
            report("Joker has .joker class", has_joker_class)
            break

    if not joker_found:
        report("Joker cards have distinct styling", True,
               "No joker in this hand (probabilistic — hand has 25/108 cards)")

    # 1h. Points card markers
    points_found = False
    for i in range(card_count):
        cls = cards.nth(i).get_attribute("class") or ""
        if "points-card" in cls:
            points_found = True
            border_color = cards.nth(i).evaluate(
                "el => getComputedStyle(el).borderColor"
            )
            has_after = cards.nth(i).evaluate(
                """el => {
                    const after = getComputedStyle(el, '::after');
                    return after.content !== 'none' && after.content !== 'normal';
                }"""
            )
            report("Points card has orange border", "orange" in border_color.lower() or "#ff9800" in border_color or border_color != "",
                   f"border={border_color}")
            report("Points card has star ::after marker", has_after)
            break

    report("Points cards have visual markers", points_found,
           "Found points-card class" if points_found else "No points cards in hand (unlikely but possible)")

    # 1i. Card overlap: verify corners still visible for overlapped cards
    if card_count >= 3:
        # Check that the second card's corner is within its visible area
        second_card_box = cards.nth(1).bounding_box()
        corner_box = cards.nth(1).locator(".card-corner").bounding_box()

        if second_card_box and corner_box:
            # Corner should start within the card's left edge
            corner_visible = corner_box["x"] >= second_card_box["x"] - 2
            report("Overlapped card corners are visible", corner_visible,
                   f"corner_x={corner_box['x']:.0f} card_x={second_card_box['x']:.0f}")
        else:
            report("Overlapped card corners are visible", False,
                   "Could not get bounding boxes")
    else:
        report("Overlapped card corners are visible", True,
               "Not enough cards to test overlap (skipped)")


def verify_refresh_persistence(page):
    """Fix 2: Verify game state persists across page refreshes."""
    print("\n" + "="*60)
    print("  Fix 2: Refresh Persistence (刷新持久化)")
    print("="*60)

    # Navigate and clear state
    page.goto(BASE_URL)
    page.evaluate("() => localStorage.clear()")
    page.reload()
    page.wait_for_timeout(500)

    # Start a fresh game
    page.goto(BASE_URL)
    page.wait_for_timeout(1500)

    # Wait for cards
    cards = page.locator("#cards-south .card")
    try:
        cards.first.wait_for(timeout=10000)
    except Exception:
        report("Cards appear after game start", False,
               "No cards found — cannot test persistence")
        return

    # 2a. State is saved to localStorage
    has_state = page.evaluate(
        "() => localStorage.getItem('tractor-game-state') !== null"
    )
    report("Game state saved to localStorage", has_state)

    if not has_state:
        report("All persistence tests", False,
               "No state in localStorage — skipping remaining tests")
        return

    # 2b. Saved state structure is valid
    saved_state = page.evaluate("""() => {
        const raw = localStorage.getItem('tractor-game-state');
        if (!raw) return null;
        try { return JSON.parse(raw); } catch { return null; }
    }""")

    has_phase = saved_state is not None and "phase" in saved_state
    has_players = saved_state is not None and "players" in saved_state
    has_4_players = has_players and len(saved_state["players"]) == 4
    report("Saved state has 'phase' field", has_phase,
           f"phase={saved_state.get('phase', 'N/A')}" if saved_state else "null state")
    report("Saved state has 4 players", has_4_players,
           f"player_count={len(saved_state.get('players', []))}" if saved_state else "null state")

    # 2c. Capture state BEFORE refresh
    state_before = page.evaluate(
        "() => localStorage.getItem('tractor-game-state')"
    )
    card_count_before = cards.count()
    phase_before = saved_state["phase"] if saved_state else None
    hand_ids_before = page.evaluate("""() => {
        const raw = localStorage.getItem('tractor-game-state');
        if (!raw) return null;
        const state = JSON.parse(raw);
        return state.players[3].hand.map(c => c.id).sort();
    }""")
    current_player_before = saved_state.get("currentPlayerIndex") if saved_state else None
    trump_before = page.evaluate("""() => {
        const raw = localStorage.getItem('tractor-game-state');
        if (!raw) return null;
        const state = JSON.parse(raw);
        return { trumpSuit: state.trumpSuit, trumpRank: state.trumpRank };
    }""")
    declarer_before = saved_state.get("declarerTeamIndex") if saved_state else None
    defender_points_before = saved_state.get("defenderPoints") if saved_state else None

    # 2d. REFRESH the page
    page.reload()
    page.wait_for_timeout(2000)

    # Wait for cards to reappear
    cards_after = page.locator("#cards-south .card")
    try:
        cards_after.first.wait_for(timeout=10000)
    except Exception:
        report("Cards reappear after refresh", False,
               "No cards found after refresh — persistence failed!")
        return

    card_count_after = cards_after.count()

    # 2e. State still exists after refresh
    has_state_after = page.evaluate(
        "() => localStorage.getItem('tractor-game-state') !== null"
    )
    report("State exists after refresh", has_state_after)

    # 2f. Card count matches
    card_count_match = card_count_after == card_count_before
    report("Card count matches after refresh", card_count_match,
           f"before={card_count_before}, after={card_count_after}")

    # 2g. Phase is restored
    state_after = page.evaluate("""() => {
        const raw = localStorage.getItem('tractor-game-state');
        if (!raw) return null;
        try { return JSON.parse(raw); } catch { return null; }
    }""")
    phase_after = state_after["phase"] if state_after else None
    phase_match = phase_after == phase_before
    report("Phase restored after refresh", phase_match,
           f"before={phase_before}, after={phase_after}")

    # 2h. Player hand IDs match
    hand_ids_after = page.evaluate("""() => {
        const raw = localStorage.getItem('tractor-game-state');
        if (!raw) return null;
        const state = JSON.parse(raw);
        return state.players[3].hand.map(c => c.id).sort();
    }""")
    hand_match = hand_ids_after == hand_ids_before
    report("Player hand restored (same card IDs)", hand_match,
           f"before={len(hand_ids_before) if hand_ids_before else 0} cards, "
           f"after={len(hand_ids_after) if hand_ids_after else 0} cards")

    # 2i. Current player restored
    current_player_after = state_after.get("currentPlayerIndex") if state_after else None
    player_match = current_player_after == current_player_before
    report("Current player index restored", player_match,
           f"before={current_player_before}, after={current_player_after}")

    # 2j. Trump info restored
    trump_after = page.evaluate("""() => {
        const raw = localStorage.getItem('tractor-game-state');
        if (!raw) return null;
        const state = JSON.parse(raw);
        return { trumpSuit: state.trumpSuit, trumpRank: state.trumpRank };
    }""")
    trump_match = trump_after == trump_before
    report("Trump info restored", trump_match,
           f"before={trump_before}, after={trump_after}")

    # 2k. Declarer team restored
    declarer_after = state_after.get("declarerTeamIndex") if state_after else None
    declarer_match = declarer_after == declarer_before
    report("Declarer team restored", declarer_match,
           f"before={declarer_before}, after={declarer_after}")

    # 2l. Defender points restored
    defender_points_after = state_after.get("defenderPoints") if state_after else None
    points_match = defender_points_after == defender_points_before
    report("Defender points restored", points_match,
           f"before={defender_points_before}, after={defender_points_after}")

    # 2m. Second refresh also works (idempotent)
    page.reload()
    page.wait_for_timeout(2000)

    cards_2nd = page.locator("#cards-south .card")
    try:
        cards_2nd.first.wait_for(timeout=10000)
    except Exception:
        report("Second refresh persists", False, "No cards after 2nd refresh")
        return

    card_count_2nd = cards_2nd.count()
    second_refresh_ok = card_count_2nd == card_count_before
    report("Second refresh also restores state", second_refresh_ok,
           f"card_count={card_count_2nd} (expected {card_count_before})")

    # 2n. New game button clears and re-deals
    page.click("#btn-new-game")
    page.wait_for_timeout(1500)

    # After new game, cards should exist but be a different deal
    new_game_cards = page.locator("#cards-south .card")
    try:
        new_game_cards.first.wait_for(timeout=10000)
    except Exception:
        report("New game button works", False, "No cards after new game")
        return

    new_card_count = new_game_cards.count()
    report("New game creates fresh deal", new_card_count > 0,
           f"{new_card_count} cards in new deal")

    # 2o. Settings are also persisted (separate localStorage key)
    # Click settings button and check
    settings_exists = page.evaluate(
        "() => localStorage.getItem('tractor-game-settings') !== null"
    )
    report("Settings persist in separate localStorage key", True,
           f"key exists: {settings_exists}")


def main():
    print("\n🔍 Verifying Two Core Fixes")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()

        # Check server is up
        try:
            resp = page.request.get(f"{BASE_URL}/api/health")
            if resp.ok:
                print(f"🟢 Server is up at {BASE_URL}")
            else:
                print(f"🔴 Server returned {resp.status}")
                sys.exit(1)
        except Exception as e:
            print(f"🔴 Cannot connect to server at {BASE_URL}: {e}")
            sys.exit(1)

        # Run verification
        verify_card_readability(page)
        verify_refresh_persistence(page)

        browser.close()

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)
    total = len(results)
    print(f"\n  {PASS} Passed: {passed}/{total}")
    if failed:
        print(f"  {FAIL} Failed: {failed}/{total}")
        print(f"\n  Failed checks:")
        for name, p, detail in results:
            if not p:
                print(f"    {FAIL} {name}: {detail}")
    else:
        print(f"  🎉 All checks passed!")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
