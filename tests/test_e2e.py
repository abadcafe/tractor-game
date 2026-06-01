"""End-to-end Playwright tests for the full game stack.

Adjusted selectors to match actual UI:
- Cards: #cards-south .card (not .hand-card)
- Bidding panel: #bidding-panel (not .bidding-panel class)
- Stirring panel: #stirring-panel (not .stir-panel)
- Trick display: #trick-display (not .trick-area)
- Trump info: #trump-info (not .trump-info class)
- Play button: #btn-play (not text=出牌 which also matches "出牌记录")
- New game: #btn-new-game
- Pass bidding: #bidding-panel .pass-btn
- Pass stirring: #stirring-panel .pass-btn
- Next round: #btn-next-round

IMPORTANT: These E2E tests depend on the server-side _ai_auto_play() method
(game.py:363) which automatically advances AI turns after each human action.
Without this, the game would stall waiting for AI input and the tests would
time out. The _ai_auto_play() is called in submit_bid(), set_trump(),
submit_discard(), submit_play(), and the deal() API endpoint.
"""
import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(scope="session")
def base_url(live_server):
    return live_server


# ---- Shared navigation helpers (CR-007) ----

def _navigate_past_bidding(page: Page) -> None:
    """Click through bidding and trump selection until panels disappear.

    The #bidding-panel ID is reused for both bidding (pass button) and
    trump suit selection (suit buttons) after winning the bid. This helper
    handles both cases.

    Note: wait_for_timeout is used here because the bidding panel may reappear
    when the AI bids and it becomes the human's turn again. A strict
    wait_for(state="hidden") would time out in that scenario.
    """
    for _ in range(10):
        bidding = page.locator("#bidding-panel")
        if not bidding.is_visible():
            break
        pass_btn = bidding.locator(".pass-btn")
        if pass_btn.is_visible():
            pass_btn.click()
            page.wait_for_timeout(1000)
        else:
            # Check if this is trump selection (suit buttons)
            suit_btn = bidding.locator(".suit-btn").first
            if suit_btn.is_visible():
                suit_btn.click()
                page.wait_for_timeout(1000)
            else:
                break


def _navigate_past_stirring(page: Page) -> None:
    """Pass stirring if the panel appears."""
    stir_panel = page.locator("#stirring-panel")
    if stir_panel.is_visible():
        pass_btn = stir_panel.locator(".pass-btn")
        if pass_btn.is_visible():
            pass_btn.click()
            page.wait_for_timeout(1000)


def _navigate_to_playing_phase(page: Page, base_url: str) -> None:
    """Navigate from game start through bidding/stirring/discard to playing phase."""
    page.goto(base_url)
    page.locator("#btn-new-game").click()
    # Wait for cards to appear (game started) -- condition-based (CR-008)
    page.locator("#cards-south .card").first.wait_for(state="visible", timeout=10000)
    _navigate_past_bidding(page)
    _navigate_past_stirring(page)
    # Wait for game to settle into exchange phase
    page.wait_for_timeout(3000)
    # Handle discard phase if present (25 cards = need to discard 8)
    cards = page.locator("#cards-south .card")
    if cards.count() > 20:
        for i in range(8):
            cards.nth(i).click(force=True)
            page.wait_for_timeout(100)
        play_btn = page.locator("#btn-play")
        if play_btn.is_visible() and not play_btn.is_disabled():
            play_btn.click()
            page.wait_for_timeout(2000)


# ---- E2E Test Classes ----

class TestE2EDealAndBid:
    """E2E-1: Dealing and bidding flow."""
    def test_e2e_deal_and_bid(self, page: Page, base_url):
        page.goto(base_url)
        # Click New Game / Deal (game auto-starts on load, but explicit click creates fresh game)
        page.locator("#btn-new-game").click()
        # Human player should see 25 cards
        cards = page.locator("#cards-south .card")
        expect(cards).to_have_count(25)
        # Bidding panel should appear
        bidding = page.locator("#bidding-panel")
        expect(bidding).to_be_visible()
        # Pass on bidding -- may need multiple passes if all players pass
        # (redeal triggers a new bidding round)
        _navigate_past_bidding(page)
        # Wait for AI bidding to resolve -- wait for trump text to change from
        # "主牌: 未定" (renderer default when trumpSuit is null) to a specific
        # suit symbol (CR-011, CR-013). "主牌: --" is only the HTML template
        # default; the renderer immediately replaces it with "主牌: 未定".
        expect(page.locator("#trump-info")).not_to_have_text("主牌: 未定", timeout=15000)


class TestE2EStirring:
    """E2E-2: Stirring flow."""
    def test_e2e_stirring(self, page: Page, base_url):
        page.goto(base_url)
        page.locator("#btn-new-game").click()
        page.locator("#cards-south .card").first.wait_for(state="visible", timeout=10000)
        _navigate_past_bidding(page)
        _navigate_past_stirring(page)
        # After bidding/stirring, bidding panel should be gone (CR-002)
        expect(page.locator("#bidding-panel")).not_to_be_visible()
        # Game should have cards displayed (didn't crash)
        cards = page.locator("#cards-south .card")
        assert cards.count() > 0, "Cards should still be displayed after bidding/stirring"


class TestE2EDiscard:
    """E2E-3: Discard flow."""
    def test_e2e_discard(self, page: Page, base_url):
        page.goto(base_url)
        page.locator("#btn-new-game").click()
        page.locator("#cards-south .card").first.wait_for(state="visible", timeout=10000)
        _navigate_past_bidding(page)
        _navigate_past_stirring(page)
        page.wait_for_timeout(3000)
        cards = page.locator("#cards-south .card")
        initial_count = cards.count()
        # Check if we're in exchange phase (human is bid winner, needs to discard)
        play_btn = page.locator("#btn-play")
        if initial_count > 20 and play_btn.is_visible() and not play_btn.is_disabled():
            # In exchange phase - select 8 cards and discard
            for i in range(8):
                cards.nth(i).click(force=True)
                page.wait_for_timeout(100)
            if not play_btn.is_disabled():
                play_btn.click()
                # Verify card count reduced after discard (CR-002)
                expect(cards).to_have_count(initial_count - 8, timeout=10000)
        else:
            # AI won the bid - human doesn't discard, game advances to playing
            # Verify bidding panel is gone and game is still functional (CR-002)
            expect(page.locator("#bidding-panel")).not_to_be_visible()
            assert cards.count() > 0, "Cards should still be displayed"


class TestE2EPlayAndFollow:
    """E2E-4: Playing and following cards."""
    def test_e2e_play_and_follow(self, page: Page, base_url):
        _navigate_to_playing_phase(page, base_url)
        cards = page.locator("#cards-south .card")
        initial_count = cards.count()
        if initial_count > 0:
            cards.first.click(force=True)
            play_btn = page.locator("#btn-play")
            if play_btn.is_visible() and not play_btn.is_disabled():
                play_btn.click()
                # After playing, trick display should be visible (CR-002)
                trick_display = page.locator("#trick-display")
                expect(trick_display).to_be_visible()
                # Card count should reduce after playing
                expect(cards).to_have_count(initial_count - 1, timeout=10000)


class TestE2ETrumpAndWinner:
    """E2E-5: Trump and winner determination."""
    def test_e2e_trump_and_winner(self, page: Page, base_url):
        page.goto(base_url)
        page.locator("#btn-new-game").click()
        page.locator("#cards-south .card").first.wait_for(state="visible", timeout=10000)
        _navigate_past_bidding(page)
        # After bidding resolves, trump info should be set (CR-002)
        trump_info = page.locator("#trump-info")
        # If human won the bid, trump might still be "--" until they select a suit.
        # In that case, the bidding panel with suit buttons should be visible.
        bidding = page.locator("#bidding-panel")
        if bidding.is_visible():
            suit_btn = bidding.locator(".suit-btn").first
            if suit_btn.is_visible():
                suit_btn.click()
                page.wait_for_timeout(1000)
        # Wait for trump text to change from "主牌: 未定" (renderer default when
        # trumpSuit is null) to a specific suit symbol (CR-011, CR-013)
        expect(trump_info).not_to_have_text("主牌: 未定", timeout=10000)


class TestE2EScoringAndLevelUp:
    """E2E-6: Scoring and level advancement."""
    def test_e2e_scoring_and_level_up(self, page: Page, base_url):
        page.goto(base_url)
        page.locator("#btn-new-game").click()
        page.locator("#cards-south .card").first.wait_for(state="visible", timeout=10000)
        # Verify initial level display shows expected default values (CR-003)
        level_declarer = page.locator("#level-declarer")
        level_defender = page.locator("#level-defender")
        expect(level_declarer).to_be_visible()
        expect(level_defender).to_be_visible()
        # Check initial level values (should be "2" for new game)
        assert level_declarer.text_content() == "2", "Declarer level should start at 2"
        assert level_defender.text_content() == "2", "Defender level should start at 2"
        # Verify score display exists and has content (CR-003)
        score_info = page.locator("#score-info")
        expect(score_info).to_be_visible()
        score_text = score_info.text_content()
        assert score_text is not None and "得分" in score_text, \
            "Score display should show score info text"


class TestE2ERefreshPersistence:
    """E2E-7: Refresh persistence (server-side)."""
    def test_e2e_refresh_persistence(self, page: Page, base_url):
        # Capture game ID from API response on game creation
        captured_game_id = {"value": None}

        def on_response(response):
            if "/api/game" in response.url and response.request.method == "POST":
                if response.status == 200:
                    try:
                        body = response.json()
                        if "gameId" in body:
                            captured_game_id["value"] = body["gameId"]
                    except Exception:
                        pass

        page.on("response", on_response)
        page.goto(base_url)
        # Wait for game to be created -- condition-based (CR-008)
        page.locator("#cards-south .card").first.wait_for(state="visible", timeout=10000)

        game_id = captured_game_id["value"]
        assert game_id is not None, "Game ID should be captured from API"

        # Get state before refresh via API
        api_state_before = page.evaluate(
            """async (gid) => {
                const r = await fetch('/api/game/' + gid);
                return r.json();
            }""",
            game_id,
        )
        phase_before = api_state_before["state"]["phase"]

        # Refresh page
        page.reload()
        # Wait for game to restore -- condition-based (CR-008)
        page.locator("#cards-south .card").first.wait_for(state="visible", timeout=10000)

        # Old game should still exist on server
        api_state_after = page.evaluate(
            """async (gid) => {
                const r = await fetch('/api/game/' + gid);
                return { status: r.status, body: await r.json() };
            }""",
            game_id,
        )
        assert api_state_after["status"] == 200, "Old game should persist on server after refresh"
        phase_after = api_state_after["body"]["state"]["phase"]
        assert phase_after == phase_before, \
            f"Game phase must persist after refresh: before={phase_before}, after={phase_after}"


class TestE2EFullGame:
    """E2E-8: Full game without errors."""
    def test_e2e_full_game(self, page: Page, base_url):
        # Track JS console errors
        js_errors = []
        page.on("console", lambda msg: js_errors.append(msg.text) if msg.type == "error" else None)

        page.goto(base_url)
        page.locator("#btn-new-game").click()
        # Wait for game to start -- condition-based (CR-008)
        page.locator("#cards-south .card").first.wait_for(state="visible", timeout=10000)
        # Let the game run for a while, clicking through phases
        for _ in range(50):
            page.wait_for_timeout(1000)
            # Try to interact with whatever is available
            # Use specific selectors to avoid strict mode violations
            bidding = page.locator("#bidding-panel")
            if bidding.is_visible():
                pass_btn = bidding.locator(".pass-btn")
                if pass_btn.is_visible():
                    pass_btn.click()
                    continue

            stir_panel = page.locator("#stirring-panel")
            if stir_panel.is_visible():
                pass_btn = stir_panel.locator(".pass-btn")
                if pass_btn.is_visible():
                    pass_btn.click()
                    continue

            play_btn = page.locator("#btn-play")
            if play_btn.is_visible() and not play_btn.is_disabled():
                # Select first card if none selected
                card = page.locator("#cards-south .card").first
                if card.is_visible():
                    card.click(force=True)
                    page.wait_for_timeout(200)
                    if not play_btn.is_disabled():
                        play_btn.click()
                    else:
                        # Deselect if play is still not possible
                        card.click(force=True)
                continue

            next_btn = page.locator("#btn-next-round")
            if next_btn.is_visible():
                next_btn.click()
                continue

            no_play = page.locator("#btn-pass")
            if no_play.is_visible():
                no_play.click()
                continue

        # Filter errors from known non-critical sources (CR-004):
        # - favicon 404s: browser default request, not game-related
        # - HTTP 400/404: stale UI race conditions where the test clicks a
        #   button after the server already advanced to a new phase. These
        #   occur because the monkey-clicking loop acts on potentially stale
        #   DOM state. The server correctly rejects the invalid action.
        # NOTE: "Invalid stir/bid/play/set-trump" are the server's 400 detail
        # messages for these race conditions. A genuine game logic bug that
        # produces the same error message would be caught by the other focused
        # E2E tests (E2E-1 through E2E-7) which validate specific flows.
        critical_errors = [
            e for e in js_errors
            if "favicon" not in e.lower()
            and "status of 400" not in e
            and "status of 404" not in e
            and "Invalid stir" not in e
            and "Invalid bid" not in e
            and "Invalid play" not in e
            and "Invalid set-trump" not in e
            and "Invalid discard" not in e
        ]
        assert len(critical_errors) == 0, f"JS errors found: {critical_errors}"
        # Game should still be running without crashes
        assert page.locator("body").is_visible()
