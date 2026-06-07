"""
测试完整的UI流程
"""

from playwright.sync_api import sync_playwright
import time


def test_ui_flow():
    """测试完整的UI流程"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8787")

        # 等待页面加载
        page.wait_for_timeout(2000)

        print("=== 页面加载 ===")
        print(f"标题: {page.title()}")

        # 检查游戏元素
        print("\n=== 检查游戏元素 ===")
        print(f"新游戏按钮: {page.locator('#btn-new-game').count()}")
        print(f"出牌按钮: {page.locator('#btn-play').count()}")
        print(f"手牌区域: {page.locator('#cards-south').count()}")

        # 点击新游戏
        print("\n=== 点击新游戏 ===")
        page.click("#btn-new-game")
        page.wait_for_timeout(3000)

        # 检查游戏状态
        print(f"阶段: {page.locator('#cards-south').count()}")

        # 检查是否有叫牌面板
        bidding_panel = page.locator("#bidding-panel")
        if bidding_panel.count() > 0:
            print("有叫牌面板")
        else:
            print("没有叫牌面板")

        # 检查手牌
        cards = page.locator("#cards-south > *")
        print(f"手牌数量: {cards.count()}")

        # 检查是否有"不出"按钮
        pass_btn = page.locator("#btn-pass")
        if pass_btn.count() > 0:
            print("有不出按钮（不应该有）")
        else:
            print("没有不出按钮（正确）")

        browser.close()


if __name__ == "__main__":
    test_ui_flow()
