"""
QQ升级完整游戏流程端到端测试

根据QQ升级规则，测试完整的游戏流程：
1. 创建游戏
2. 发牌
3. 亮牌（叫牌）- 第一个亮级牌的人成为庄家
4. 炒地皮 - 其他玩家可以炒地皮
5. 埋牌 - 庄家捡起底牌，弃掉8张
6. 出牌 - 完整的一轮出牌
7. 计分 - 检查计分逻辑
"""

import json
import urllib.request
import pytest


BASE_URL = "http://localhost:8787/api/game"


def api(method, path, data=None):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, method=method, data=body,
        headers={"Content-Type": "application/json"} if body else {})
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


class TestQQUpgradeFullFlow:
    """QQ升级完整游戏流程测试"""

    def test_full_round_flow(self):
        """测试完整的一轮游戏流程"""
        # 1. 创建游戏
        d = api("POST", "")
        game_id = d["gameId"]
        assert d["state"]["phase"] == "dealing"

        # 2. 发牌
        d = api("POST", f"/{game_id}/deal")
        assert d["state"]["phase"] == "bidding"
        assert len(d["state"]["bottomCards"]) == 8
        assert len(d["state"]["players"][3]["hand"]) == 25

        # 3. 亮牌（叫牌）
        current_player = d["state"]["currentPlayerIndex"]
        hand = d["state"]["players"][current_player]["hand"]
        trump_rank = d["state"]["trumpRank"]
        trump_cards = [c for c in hand if c["rank"] == trump_rank]

        if trump_cards:
            card = trump_cards[0]
            d = api("POST", f"/{game_id}/bid", {
                "playerIndex": current_player,
                "cardIds": [card["id"]],
                "pass": False
            })
            assert d["state"]["phase"] == "stirring"
            assert d["state"]["trumpSuit"] is not None
            print(f"亮牌成功: {card['suit']}{card['rank']}")

            # 4. 炒地皮（所有玩家跳过）
            while d["awaitingAction"] == "stir":
                current_player = d["state"]["currentPlayerIndex"]
                d = api("POST", f"/{game_id}/stir", {
                    "playerIndex": current_player,
                    "pass": True
                })
                print(f"玩家{current_player}跳过炒地皮")

            # 5. 检查进入埋牌阶段
            assert d["state"]["phase"] == "exchange"
            assert d["awaitingAction"] == "discard"
            print("进入埋牌阶段")

            # 6. 埋牌（弃8张牌）
            declarer = d["state"]["currentPlayerIndex"]
            hand = d["state"]["players"][declarer]["hand"]
            assert len(hand) == 33, f"庄家应该有33张牌（25+8），实际{len(hand)}"

            discard_ids = [c["id"] for c in hand[:8]]
            d = api("POST", f"/{game_id}/discard", {
                "playerIndex": declarer,
                "cardIds": discard_ids
            })

            # 7. 检查进入出牌阶段
            assert d["state"]["phase"] == "playing"
            assert d["awaitingAction"] == "play"
            print("进入出牌阶段")

            # 8. 检查手牌数量
            hand = d["state"]["players"][declarer]["hand"]
            assert len(hand) == 25, f"弃牌后应该有25张牌，实际{len(hand)}"

            # 9. 出牌（出第一张牌）
            current_player = d["state"]["currentPlayerIndex"]
            hand = d["state"]["players"][current_player]["hand"]
            card = hand[0]
            d = api("POST", f"/{game_id}/play", {
                "playerIndex": current_player,
                "cardIds": [card["id"]]
            })
            print(f"玩家{current_player}出牌: {card['suit']}{card['rank']}")

            # 10. 检查出牌后状态
            assert d["state"]["phase"] == "playing"
            print("出牌成功")

    def test_bidding_flow(self):
        """测试亮牌流程"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 检查叫牌阶段
        assert d["state"]["phase"] == "bidding"
        current_player = d["state"]["currentPlayerIndex"]
        hand = d["state"]["players"][current_player]["hand"]
        trump_rank = d["state"]["trumpRank"]
        trump_cards = [c for c in hand if c["rank"] == trump_rank]

        if trump_cards:
            card = trump_cards[0]
            d = api("POST", f"/{game_id}/bid", {
                "playerIndex": current_player,
                "cardIds": [card["id"]],
                "pass": False
            })

            # 检查亮牌后状态
            assert d["state"]["phase"] == "stirring"
            assert d["state"]["trumpSuit"] is not None
            assert d["state"]["players"][current_player]["isDeclarer"]

    def test_stirring_flow(self):
        """测试炒地皮流程"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 亮牌
        current_player = d["state"]["currentPlayerIndex"]
        hand = d["state"]["players"][current_player]["hand"]
        trump_rank = d["state"]["trumpRank"]
        trump_cards = [c for c in hand if c["rank"] == trump_rank]

        if trump_cards:
            card = trump_cards[0]
            d = api("POST", f"/{game_id}/bid", {
                "playerIndex": current_player,
                "cardIds": [card["id"]],
                "pass": False
            })

            # 炒地皮
            while d["awaitingAction"] == "stir":
                current_player = d["state"]["currentPlayerIndex"]
                d = api("POST", f"/{game_id}/stir", {
                    "playerIndex": current_player,
                    "pass": True
                })

            # 检查进入埋牌阶段
            assert d["state"]["phase"] == "exchange"

    def test_exchange_flow(self):
        """测试埋牌流程"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 亮牌
        current_player = d["state"]["currentPlayerIndex"]
        hand = d["state"]["players"][current_player]["hand"]
        trump_rank = d["state"]["trumpRank"]
        trump_cards = [c for c in hand if c["rank"] == trump_rank]

        if trump_cards:
            card = trump_cards[0]
            d = api("POST", f"/{game_id}/bid", {
                "playerIndex": current_player,
                "cardIds": [card["id"]],
                "pass": False
            })

            # 炒地皮
            while d["awaitingAction"] == "stir":
                current_player = d["state"]["currentPlayerIndex"]
                d = api("POST", f"/{game_id}/stir", {
                    "playerIndex": current_player,
                    "pass": True
                })

            # 埋牌
            assert d["state"]["phase"] == "exchange"
            declarer = d["state"]["currentPlayerIndex"]
            hand = d["state"]["players"][declarer]["hand"]
            assert len(hand) == 33

            discard_ids = [c["id"] for c in hand[:8]]
            d = api("POST", f"/{game_id}/discard", {
                "playerIndex": declarer,
                "cardIds": discard_ids
            })

            # 检查进入出牌阶段
            assert d["state"]["phase"] == "playing"
            hand = d["state"]["players"][declarer]["hand"]
            assert len(hand) == 25

    def test_playing_flow(self):
        """测试出牌流程"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 亮牌
        current_player = d["state"]["currentPlayerIndex"]
        hand = d["state"]["players"][current_player]["hand"]
        trump_rank = d["state"]["trumpRank"]
        trump_cards = [c for c in hand if c["rank"] == trump_rank]

        if trump_cards:
            card = trump_cards[0]
            d = api("POST", f"/{game_id}/bid", {
                "playerIndex": current_player,
                "cardIds": [card["id"]],
                "pass": False
            })

            # 炒地皮
            while d["awaitingAction"] == "stir":
                current_player = d["state"]["currentPlayerIndex"]
                d = api("POST", f"/{game_id}/stir", {
                    "playerIndex": current_player,
                    "pass": True
                })

            # 埋牌
            declarer = d["state"]["currentPlayerIndex"]
            hand = d["state"]["players"][declarer]["hand"]
            discard_ids = [c["id"] for c in hand[:8]]
            d = api("POST", f"/{game_id}/discard", {
                "playerIndex": declarer,
                "cardIds": discard_ids
            })

            # 出牌
            assert d["state"]["phase"] == "playing"
            current_player = d["state"]["currentPlayerIndex"]
            hand = d["state"]["players"][current_player]["hand"]
            card = hand[0]
            d = api("POST", f"/{game_id}/play", {
                "playerIndex": current_player,
                "cardIds": [card["id"]]
            })

            # 检查出牌后状态
            assert d["state"]["phase"] == "playing"

    def test_scoring_rules(self):
        """测试计分规则"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 检查计分牌
        all_cards = []
        for p in d["state"]["players"]:
            all_cards.extend(p["hand"])
        all_cards.extend(d["state"]["bottomCards"])

        total_points = sum(c["points"] for c in all_cards)
        assert total_points == 200, f"总分应为200，实际{total_points}"

        # 检查每种计分牌
        for c in all_cards:
            if c["rank"] == "K":
                assert c["points"] == 10
            elif c["rank"] == "10":
                assert c["points"] == 10
            elif c["rank"] == "5":
                assert c["points"] == 5
            else:
                assert c["points"] == 0

    def test_card_distribution(self):
        """测试牌的分配"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 检查总牌数
        total = sum(len(p["hand"]) for p in d["state"]["players"]) + len(d["state"]["bottomCards"])
        assert total == 108, f"总牌数应为108，实际{total}"

        # 检查每个玩家的牌数
        for p in d["state"]["players"]:
            assert len(p["hand"]) == 25, f"玩家{p['index']}应有25张牌"

        # 检查底牌数
        assert len(d["state"]["bottomCards"]) == 8, "底牌应有8张"

    def test_trump_rank(self):
        """测试级牌"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 检查级牌
        assert d["state"]["trumpRank"] == "2"

    def test_team_structure(self):
        """测试队伍结构"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 检查队伍
        teams = {}
        for p in d["state"]["players"]:
            t = p["teamIndex"]
            teams.setdefault(t, []).append(p["index"])
        assert len(teams) == 2, "应该有两队"
        assert all(len(v) == 2 for v in teams.values()), "每队2人"

    def test_player_names(self):
        """测试玩家名称"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 检查玩家名称
        players = d["state"]["players"]
        assert players[0]["name"] == "同伴 (AI)"
        assert players[1]["name"] == "对手A (AI)"
        assert players[2]["name"] == "对手B (AI)"
        assert players[3]["name"] == "你"

    def test_human_player(self):
        """测试人类玩家"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 检查人类玩家
        players = d["state"]["players"]
        assert players[3]["isHuman"] == True
        assert players[0]["isHuman"] == False
        assert players[1]["isHuman"] == False
        assert players[2]["isHuman"] == False


def test_api_full_round():
    """测试API完整一轮游戏"""
    # 1. 创建游戏
    d = api("POST", "")
    game_id = d["gameId"]

    # 2. 发牌
    d = api("POST", f"/{game_id}/deal")
    assert d["state"]["phase"] == "bidding"

    # 3. 亮牌
    current_player = d["state"]["currentPlayerIndex"]
    hand = d["state"]["players"][current_player]["hand"]
    trump_rank = d["state"]["trumpRank"]
    trump_cards = [c for c in hand if c["rank"] == trump_rank]

    if trump_cards:
        card = trump_cards[0]
        d = api("POST", f"/{game_id}/bid", {
            "playerIndex": current_player,
            "cardIds": [card["id"]],
            "pass": False
        })
        assert d["state"]["phase"] == "stirring"
        assert d["state"]["trumpSuit"] is not None

        # 4. 炒地皮
        while d["awaitingAction"] == "stir":
            current_player = d["state"]["currentPlayerIndex"]
            d = api("POST", f"/{game_id}/stir", {
                "playerIndex": current_player,
                "pass": True
            })

        # 5. 埋牌
        assert d["state"]["phase"] == "exchange"
        declarer = d["state"]["currentPlayerIndex"]
        hand = d["state"]["players"][declarer]["hand"]
        assert len(hand) == 33

        discard_ids = [c["id"] for c in hand[:8]]
        d = api("POST", f"/{game_id}/discard", {
            "playerIndex": declarer,
            "cardIds": discard_ids
        })

        # 6. 出牌
        assert d["state"]["phase"] == "playing"
        current_player = d["state"]["currentPlayerIndex"]
        hand = d["state"]["players"][current_player]["hand"]
        card = hand[0]
        d = api("POST", f"/{game_id}/play", {
            "playerIndex": current_player,
            "cardIds": [card["id"]]
        })

        assert d["state"]["phase"] == "playing"


    # ============================================================
    # 【拖拉机的构成】
    # ============================================================

    def test_tractor_consecutive_pairs(self):
        """拖拉机：大小顺序相邻且花色相同的联对"""
        # KKQQ, JJ99, 554433 是拖拉机
        # 需要验证拖拉机构成规则
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 检查手牌中是否有拖拉机
        hand = d["state"]["players"][3]["hand"]
        # 拖拉机需要在出牌阶段验证

    def test_tractor_invalid_examples(self):
        """无效的拖拉机：554, 544, 5533, JJQ"""
        # 这些不是拖拉机
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证无效拖拉机

    # ============================================================
    # 【出牌规则】
    # ============================================================

    def test_follow_suit(self):
        """跟牌规则：必须跟同花色"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证跟牌规则

    def test_throw_cards(self):
        """甩牌规则：同门花色的大牌可以联出"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证甩牌规则

    def test_must_follow_pair(self):
        """对牌规则：首家出对牌时，其余家有对牌必须出对牌"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证对牌规则

    def test_must_follow_tractor(self):
        """拖拉机规则：首家出拖拉机时，其余家有拖拉机必须出拖拉机"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证拖拉机规则

    # ============================================================
    # 【毙牌规则】
    # ============================================================

    def test_trump_beat(self):
        """毙牌规则：首家出副牌时，其余家无此门花色时可出主牌"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证毙牌规则

    def test_trump_beat_with_pair(self):
        """毙牌时对牌规则：毙牌时所出的牌必须是主牌，且对牌数目不得少于"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证毙牌对牌规则

    def test_cover_beat(self):
        """盖毙规则：出现多家毙牌时，毙牌的大小以毙牌中的拖拉机和对牌大小为准"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证盖毙规则

    # ============================================================
    # 【轮庄规则】
    # ============================================================

    def test_dealer_rotation_win(self):
        """庄家升级时，下一副牌由其对家当庄家"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证轮庄规则

    def test_dealer_rotation_lose(self):
        """闲家上台时，下一副牌由此副牌的庄家的下家当庄家"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证轮庄规则

    # ============================================================
    # 【扣底规则】
    # ============================================================

    def test_kitty_no_multiplier(self):
        """最后一圈牌如果是庄家方大，不抠底"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证不抠底规则

    def test_kitty_multiplier_single(self):
        """单张扣底：2倍"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证单张扣底

    def test_kitty_multiplier_pair(self):
        """对子扣底：4倍"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证对子扣底

    def test_kitty_multiplier_tractor(self):
        """拖拉机扣底：2^(N+1)倍"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证拖拉机扣底

    def test_kitty_max_multiplier(self):
        """最大扣底倍数：64倍"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证最大扣底倍数

    # ============================================================
    # 【积分规则】
    # ============================================================

    def test_score_0_promote_3(self):
        """得分为0：庄家方连升3级"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证0分升3级

    def test_score_below_40_promote_2(self):
        """得分不满40：庄家连升2级"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证不满40分升2级

    def test_score_40_to_80_promote_1(self):
        """得分40-80：庄家方升1级"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证40-80分升1级

    def test_score_80_to_120_swap(self):
        """得分80-120：轮换庄家"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证80-120分换庄

    def test_score_120_to_160_promote_1(self):
        """得分120-160：抓分方升1级"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证120-160分抓分方升1级

    def test_score_160_to_200_promote_2(self):
        """得分160-200：抓分方连升2级"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证160-200分抓分方升2级

    def test_score_above_200_promote_per_40(self):
        """200分以上：每40分加一级"""
        d = api("POST", "")
        game_id = d["gameId"]
        d = api("POST", f"/{game_id}/deal")

        # 需要验证200分以上每40分升一级


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
