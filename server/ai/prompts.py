"""
Prompt templates for the AI player decision steps.
"""

# ---- System Prompt (base rules + role) ----

SYSTEM_PROMPT = """你是一个升级（拖拉机）扑克牌游戏的AI玩家。你会根据手牌和场上形势做出最优决策。

## 游戏规则
- 4人游戏，2副牌（108张），两队对抗（对家为同伴）
- 主牌排序：大王 > 小王 > 级牌主花色(主牌) > 级牌副花色(副级牌) > 主花色其他牌 > 副花色牌
- 出牌类型：单张、对子（2张相同）、拖拉机（2+连对）、甩牌（同花色多张最大牌）
- 跟牌规则：必须跟出相同花色和牌型，没有则可以用主牌毙或垫其他牌
- 分牌：5(5分)、10(10分)、K(10分)，共200分
- 防守方拿分决定升降级：0分(+3级), 5-35(+2), 40-75(+1), 80-115(换庄), 120-155(-1), 160-195(-2), 200(-3)
- 扣底：防守方赢得最后一墩，底牌分数翻倍
- 炒地皮：叫牌结束后，其他人可同级别换主或更高级别抢庄

## 你的身份
{role_description}

## 基本策略
- 庄家方：尽快出完副牌中的大牌，保护分牌，控制牌局节奏
- 防守方：尽量拿分，保留主牌用于毙牌，与同伴配合
- 注意记牌，推算对手手牌
- 优先出长套（同花色多的牌），逼对手消耗主牌
- 有分牌时考虑是否安全出，没有把握时先保留

## 输出格式
你必须返回一个JSON对象，包含以下字段：
```json
{{
  "reasoning": "你的推理过程（中文，简洁）",
  "action": "play" 或 "bid" 或 "stir" 或 "discard" 或 "pass",
  "card_ids": ["选中的牌的ID列表"],
  "strategy_note": "当前策略说明（可选）"
}}
```
只返回JSON，不要包含其他内容。
"""

# ---- Role descriptions ----

ROLE_DESCRIPTIONS = {
    "declarer_partner": "你是庄家的同伴。你和庄家一队，目标是帮助庄家保级升级。配合庄家出牌，保护庄家的分牌，用大牌帮庄家拿下关键墩。",
    "defender": "你是防守方。你的目标是尽可能多拿分（5、10、K），阻止庄家方保级。积极用主牌毙副牌，保护同伴的分牌，寻找机会出分。",
}


# ---- Step-specific prompts ----

def assess_situation_prompt(game_state: dict, hand: list[dict]) -> str:
    """Step 1: Situation assessment."""
    return f"""## 态势评估

请快速评估当前局势：

当前级别: {game_state.get('current_level')}
主牌: {game_state.get('trump_suit', '未定')} {game_state.get('trump_rank')}
防守方得分: {game_state.get('defender_points', 0)}/200
已打轮数: {game_state.get('trick_count', 0)}

当前回合出牌情况:
{_format_trick(game_state.get('current_trick', []))}

你的手牌:
{_format_hand(hand)}

请用简短中文描述当前局势（1-2句话）：
- 现在是什么阶段（早期/中期/后期）？
- 分数压力大吗？
- 主牌情况如何？
"""


def review_strategy_prompt(game_state: dict, hand: list[dict],
                           previous_strategy: dict | None,
                           key_memories: list[dict]) -> str:
    """Step 2: Strategy review."""
    prev = previous_strategy.get("name", "无") if previous_strategy else "无"

    memories_str = "\n".join(
        f"- {m.get('event', '')}" for m in (key_memories or [])[-5:]
    ) or "无"

    return f"""## 策略回顾

你之前的策略: {prev}
最近的记忆:
{memories_str}

当前级别: {game_state.get('current_level')}
主牌: {game_state.get('trump_suit', '未定')} {game_state.get('trump_rank')}
防守方得分: {game_state.get('defender_points', 0)}/200

你的手牌:
{_format_hand(hand)}

你的策略还适用吗？需要调整吗？请回答JSON：
```json
{{
  "previous_strategy": "{prev}",
  "still_valid": true或false,
  "reason": "原因",
  "new_strategy": "如果需要调整，新策略是什么？如：保底跑分/抢分压制/控牌消耗/配合同伴"
}}
```
"""


def formulate_strategy_prompt(game_state: dict, hand: list[dict],
                              strategy_name: str,
                              opponent_models: dict) -> str:
    """Step 3: Strategy formulation."""
    opp_info = ""
    for pid, info in (opponent_models or {}).items():
        opp_info += f"- {pid}: {info.get('note', '未知')}\n"

    return f"""## 策略制定

当前策略: {strategy_name}

对手信息:
{opp_info or "暂无对手信息"}

当前级别: {game_state.get('current_level')}
主牌: {game_state.get('trump_suit', '未定')} {game_state.get('trump_rank')}
防守方得分: {game_state.get('defender_points', 0)}/200

当前回合:
{_format_trick(game_state.get('current_trick', []))}
领出玩家: {game_state.get('lead_player', '?')}
领出牌型: {game_state.get('lead_play_type', '?')}

你的手牌:
{_format_hand(hand)}

请制定本轮的战术意图，回答JSON：
```json
{{
  "strategy": "策略名",
  "this_trick_goal": "本墩想达成什么",
  "tactics": "具体怎么做",
  "risk_level": "low/medium/high"
}}
```
"""


def select_cards_prompt(game_state: dict, hand: list[dict],
                        legal_actions: list[str],
                        strategy_plan: dict | None) -> str:
    """Step 4: Card selection."""
    strategy = strategy_plan.get("strategy", "未定") if strategy_plan else "未定"
    goal = strategy_plan.get("this_trick_goal", "未定") if strategy_plan else "未定"

    actions_str = "\n".join(f"  {a}" for a in legal_actions[:20])

    return f"""## 选牌

当前策略: {strategy}
本轮目标: {goal}

当前回合出牌情况:
{_format_trick(game_state.get('current_trick', []))}

你的手牌:
{_format_hand(hand)}

可选操作（部分）:
{actions_str}

请选择你要出的牌。返回你选中的牌的ID列表：
```json
{{
  "card_ids": ["card-id-1", "card-id-2"],
  "reasoning": "简短说明为什么选这些牌"
}}
```
"""


# ---- Helper ----

def _format_hand(hand: list[dict]) -> str:
    """Format hand cards grouped by suit."""
    by_suit: dict[str, list[str]] = {}
    for c in hand:
        suit = c.get("suit", "?")
        display = c.get("display", c.get("id", "?"))
        if suit not in by_suit:
            by_suit[suit] = []
        by_suit[suit].append(display)

    lines = []
    for suit, cards in sorted(by_suit.items()):
        suit_name = {"hearts": "♥", "spades": "♠", "diamonds": "♦",
                     "clubs": "♣", "joker": "🃏"}.get(suit, suit)
        lines.append(f"  {suit_name}: {', '.join(cards)}")
    return "\n".join(lines) if lines else "  (空)"


def _format_trick(trick: list[dict]) -> str:
    """Format current trick state."""
    if not trick:
        return "  尚无出牌"
    lines = []
    for s in trick:
        cards = s.get("cards", [])
        if isinstance(cards, list) and cards:
            cards_str = ", ".join(str(c) for c in cards)
        else:
            cards_str = str(cards)
        lines.append(f"  玩家{s.get('player', '?')}: {cards_str}")
    return "\n".join(lines)


def build_role_description(player_index: int, team_index: int,
                           is_declarer: bool) -> str:
    """Build role description for a player."""
    if is_declarer:
        if player_index == 0 or player_index == 3:
            # Partner (AI) of human declarer
            return ROLE_DESCRIPTIONS["declarer_partner"]
        else:
            return ROLE_DESCRIPTIONS["defender"]
    else:
        if player_index == 0 or player_index == 3:
            return ROLE_DESCRIPTIONS["defender"]
        else:
            return ROLE_DESCRIPTIONS["declarer_partner"]
