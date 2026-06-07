"""
找出所有不符合升级规则的问题

升级规则要点：
1. 叫牌：玩家亮出级牌（当前级别的牌），不是叫数字
2. 主牌：级牌 + 主花色 = 主牌
3. 出牌：必须跟牌，不能跳过
4. 埋牌：庄家捡底牌后弃8张
5. 计分：只有K、10、5是计分牌
"""

import json
import urllib.request
from playwright.sync_api import sync_playwright


def api_call(method: str, url: str, data: dict | None = None) -> dict:
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url,
        method=method,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    try:
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


class TestIssues:
    """找出所有问题"""

    def setup_method(self):
        self.base = "http://localhost:8787/api/game"
        self.game_id = None
        self.issues = []

    def create_game(self) -> dict:
        d = api_call("POST", self.base)
        self.game_id = d["gameId"]
        return d

    def deal(self) -> dict:
        return api_call("POST", f"{self.base}/{self.game_id}/deal")

    def test_issue_1_wrong_bidding_levels(self):
        """问题1：叫牌使用了错误的级别（5,6,7,8,9,10,J,Q,K,A）"""
        self.create_game()
        d = self.deal()

        # 当前实现：叫牌级别是 5,6,7,8,9,10,J,Q,K,A
        # 正确规则：叫牌级别应该是当前级别的牌（2,3,4,5,6,7,8,9,10,J,Q,K,A）

        # 检查当前级别
        current_level = d["state"]["currentLevel"]
        print(f"当前级别: {current_level}")

        # 检查可叫的级别
        # 当前实现中，valid_bid_levels 是 5,6,7,8,9,10,J,Q,K,A
        # 但正确应该是从当前级别开始

        # 检查级牌
        trump_rank = d["state"]["trumpRank"]
        print(f"当前级牌: {trump_rank}")

        # 应该是级牌匹配当前级别
        assert trump_rank == current_level, \
            f"级牌{trump_rank}应该匹配当前级别{current_level}"

    def test_issue_2_bidding_method(self):
        """问题2：叫牌方式应该是亮牌，不是叫数字"""
        self.create_game()
        d = self.deal()

        # 当前实现：玩家选择级别数字叫牌
        # 正确规则：玩家亮出手中的级牌

        # 检查叫牌阶段
        assert d["state"]["phase"] == "bidding"

        # 获取当前玩家
        current_player = d["state"]["currentPlayerIndex"]
        hand = d["state"]["players"][current_player]["hand"]

        # 找出级牌
        trump_rank = d["state"]["trumpRank"]
        dominant_cards = [c for c in hand if c["rank"] == trump_rank]

        print(f"玩家{current_player}有{len(dominant_cards)}张级牌")

        # 应该可以亮这些牌来叫牌
        if len(dominant_cards) > 0:
            print("可以亮牌叫牌")
        else:
            print("没有级牌，不能叫牌")

    def test_issue_3_scoring_thresholds(self):
        """问题3：计分阈值应该基于200分"""
        self.create_game()
        d = self.deal()

        # 检查计分阈值
        # 当前实现：可能使用了错误的阈值
        # 正确规则：80分换庄，超过80分开始升级

        # 检查当前实现的计分逻辑
        # 这需要查看 scoring.py 的实现

    def test_issue_4_trump_card_order(self):
        """问题4：主牌排序可能不正确"""
        self.create_game()
        d = self.deal()

        # 当前实现：可能使用了错误的主牌排序
        # 正确规则：
        # 1. 大王（最高）
        # 2. 小王
        # 3. 级牌+主花色
        # 4. 级牌+其他花色
        # 5. 主花色的其他牌
        # 6. 其他花色的牌

        # 检查当前排序
        trump_rank = d["state"]["trumpRank"]
        print(f"当前级牌: {trump_rank}")

    def test_issue_5_kitty_multiplier(self):
        """问题5：底牌加倍逻辑可能不正确"""
        self.create_game()
        d = self.deal()

        # 当前实现：可能没有底牌加倍
        # 正确规则：如果防守方赢了最后一墩，底牌的分数要加倍
        # 加倍倍数取决于最后一墩的牌型：
        # - 单张：x2
        # - 对子：x4
        # - 拖拉机：x(2*(N+1))

    def test_issue_6_first_trick_leading(self):
        """问题6：第一墩应该是庄家先出牌"""
        self.create_game()
        d = self.deal()

        # 当前实现：可能没有庄家先出牌的规则
        # 正确规则：庄家（赢得叫牌的玩家）先出第一墩

    def test_ui_issues(self):
        """UI问题：通过Playwright检查"""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://localhost:8787")

            # 等待页面加载
            page.wait_for_timeout(1000)

            # 检查叫牌面板
            # 当前实现：显示级别数字按钮
            # 正确规则：应该显示级牌按钮

            # 检查是否有"不出"按钮
            pass_btn = page.locator("#btn-pass")
            if pass_btn.count() > 0:
                print("UI问题：还有不出按钮")

            browser.close()


def analyze_all_issues():
    """分析所有问题"""
    print("=" * 60)
    print("升级规则问题分析")
    print("=" * 60)

    test = TestIssues()

    print("\n1. 叫牌流程问题")
    print("   - 当前：玩家叫数字（5,6,7,8,9,10,J,Q,K,A）")
    print("   - 正确：玩家亮出级牌（当前级别的牌）")

    print("\n2. 级牌定义问题")
    print("   - 当前：级牌可能不匹配当前级别")
    print("   - 正确：级牌 = 当前级别的牌")

    print("\n3. 主牌排序问题")
    print("   - 当前：可能使用了错误的排序")
    print("   - 正确：大王 > 小王 > 级牌+主花色 > 级牌+其他花色 > 主花色其他牌 > 其他花色牌")

    print("\n4. 计分阈值问题")
    print("   - 当前：可能使用了错误的阈值")
    print("   - 正确：80分换庄，超过80分开始升级")

    print("\n5. 底牌加倍问题")
    print("   - 当前：可能没有底牌加倍")
    print("   - 正确：防守方赢最后一墩时，底牌分数加倍")

    print("\n6. 第一墩出牌问题")
    print("   - 当前：可能没有庄家先出牌")
    print("   - 正确：庄家先出第一墩")

    print("\n7. 跟牌规则问题")
    print("   - 当前：可能允许跳过不出")
    print("   - 正确：必须跟牌，不能跳过")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze_all_issues()
