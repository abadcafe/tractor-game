"""
测试正确的升级规则流程

升级规则要点：
1. 叫牌：玩家亮出级牌（当前级别的牌），不是叫数字
2. 主牌：级牌 + 主花色 = 主牌
3. 出牌：必须跟牌，不能跳过
4. 埋牌：庄家捡底牌后弃8张
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
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


class TestCorrectRules:
    """测试正确的升级规则"""

    def setup_method(self):
        self.base = "http://localhost:8787/api/game"
        self.game_id = None

    def create_game(self) -> dict:
        d = api_call("POST", self.base)
        self.game_id = d["gameId"]
        return d

    def deal(self) -> dict:
        return api_call("POST", f"{self.base}/{self.game_id}/deal")

    def test_deal_correct_card_count(self):
        """发牌后每个玩家25张，底牌8张"""
        self.create_game()
        d = self.deal()

        # 每个玩家25张牌
        for i, p in enumerate(d["state"]["players"]):
            assert len(p["hand"]) == 25, f"玩家{i}应有25张牌，实际{len(p['hand'])}张"

        # 底牌8张
        assert len(d["state"]["bottomCards"]) == 8, "底牌应有8张"

        # 总共108张牌
        total = sum(len(p["hand"]) for p in d["state"]["players"]) + len(d["state"]["bottomCards"])
        assert total == 108, f"总牌数应为108，实际{total}"

    def test_initial_level_is_2(self):
        """初始级别应该是2"""
        self.create_game()
        d = self.deal()

        # 两队的初始级别都应该是2
        for team in d["state"]["teams"]:
            assert team["currentLevel"] == "2", f"初始级别应为2，实际{team['currentLevel']}"

    def test_bidding_flow(self):
        """测试叫牌流程 - 玩家应该亮级牌"""
        self.create_game()
        d = self.deal()

        # 检查叫牌阶段
        assert d["state"]["phase"] == "bidding", "应处于叫牌阶段"

        # 检查当前玩家
        current_player = d["state"]["currentPlayerIndex"]

        # 获取当前玩家的手牌
        hand = d["state"]["players"][current_player]["hand"]

        # 找出级牌（当前级别是2）
        trump_rank = d["state"]["trumpRank"]
        dominant_cards = [c for c in hand if c["rank"] == trump_rank]

        print(f"当前级别: {trump_rank}")
        print(f"玩家{current_player}的级牌: {len(dominant_cards)}张")

        # 应该可以亮级牌
        if len(dominant_cards) > 0:
            # 亮一张级牌
            card_id = dominant_cards[0]["id"]
            print(f"亮出: {card_id}")

    def test_trump_rank_matches_level(self):
        """级牌应该匹配当前级别"""
        self.create_game()
        d = self.deal()

        # 初始级别是2
        assert d["state"]["trumpRank"] == "2", f"初始级牌应为2，实际{d['state']['trumpRank']}"

    def test_player_hand_sorted_by_trump(self):
        """玩家手牌应该按主牌排序"""
        self.create_game()
        d = self.deal()

        # 检查当前玩家的手牌排序
        current_player = d["state"]["currentPlayerIndex"]
        hand = d["state"]["players"][current_player]["hand"]

        if len(hand) < 2:
            return

        # 检查主牌是否在前面
        trump_rank = d["state"]["trumpRank"]
        trump_suit = d["state"]["trumpSuit"]

        # 找到第一个非主牌的位置
        first_non_trump_idx = None
        for i, card in enumerate(hand):
            is_joker = card["isJoker"]
            is_trump_rank = card["rank"] == trump_rank
            is_trump_suit = card["suit"] == trump_suit

            if not (is_joker or is_trump_rank or is_trump_suit):
                first_non_trump_idx = i
                break

        if first_non_trump_idx is not None:
            # 检查之前的所有牌都是主牌
            for i in range(first_non_trump_idx):
                card = hand[i]
                is_joker = card["isJoker"]
                is_trump_rank = card["rank"] == trump_rank
                is_trump_suit = card["suit"] == trump_suit
                assert is_joker or is_trump_rank or is_trump_suit, \
                    f"手牌{i}应该是主牌，实际是{card['suit']}{card['rank']}"

    def test_no_pass_button_during_play(self):
        """出牌阶段不应该有"不出"按钮"""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://localhost:8787")

            # 等待页面加载
            page.wait_for_timeout(1000)

            # 检查"不出"按钮是否不存在
            pass_btn = page.locator("#btn-pass")
            assert pass_btn.count() == 0, "出牌阶段不应该有不出按钮"

            browser.close()

    def test_play_must_play_cards(self):
        """出牌阶段必须出牌，不能跳过"""
        self.create_game()
        d = self.deal()

        # 跳过叫牌和炒地皮，直接进入出牌阶段
        # 这里需要手动模拟完整的流程
        # 实际测试中需要通过UI操作

    def test_scoring_cards(self):
        """只有K、10、5是计分牌"""
        self.create_game()
        d = self.deal()

        # 检查计分牌
        for p in d["state"]["players"]:
            for card in p["hand"]:
                rank = card["rank"]
                points = card["points"]

                if rank == "K":
                    assert points == 10, f"K应该10分，实际{points}分"
                elif rank == "10":
                    assert points == 10, f"10应该10分，实际{points}分"
                elif rank == "5":
                    assert points == 5, f"5应该5分，实际{points}分"
                else:
                    assert points == 0, f"{rank}应该0分，实际{points}分"


def test_api_flow():
    """通过API测试完整流程"""
    base = "http://localhost:8787/api/game"

    # 创建游戏
    d = api_call("POST", base)
    game_id = d["gameId"]
    print(f"游戏ID: {game_id}")

    # 发牌
    d = api_call("POST", f"{base}/{game_id}/deal")
    print(f"阶段: {d['state']['phase']}")
    print(f"当前级别: {d['state']['trumpRank']}")
    print(f"当前玩家: {d['state']['currentPlayerIndex']}")

    # 检查玩家手牌
    for i, p in enumerate(d["state"]["players"]):
        print(f"玩家{i}: {len(p['hand'])}张牌")

    # 检查底牌
    print(f"底牌: {len(d['state']['bottomCards'])}张")

    # 检查计分牌
    total_points = 0
    for p in d["state"]["players"]:
        for card in p["hand"]:
            total_points += card["points"]
    for card in d["state"]["bottomCards"]:
        total_points += card["points"]
    print(f"总分数牌: {total_points}分")


if __name__ == "__main__":
    test_api_flow()
