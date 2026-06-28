# 升级 AI 自我对战训练方案：单规则第一版

## 0. 项目目标

基于现有升级规则引擎，训练一个能在固定规则下自我对战提升的四人升级 AI。

第一版只聚焦一种规则，不考虑多规则泛化。规则引擎已经实现复杂规则，包括但不限于：

- 合法动作判断
- 抢主 / 反主
- 炒地皮
- 庄家埋牌
- 出牌阶段合法性
- 2、J、A 必打规则
- 主勾 / 副勾抠底退级
- A 枪毙规则
- 最后一墩主勾、副勾、主 A、副 A 的特殊大小
- 最终结算与等级变化

模型不负责理解规则细节，规则引擎是唯一权威裁判。模型只学习在当前规则环境下如何决策。

---

## 1. 总体架构

使用一个小型专用 Transformer，而不是 LLM。

```text
当前玩家可见信息
  - 公开全局状态
  - 自己手牌
  - 从开局到当前的全部公开历史
  - 当前 phase
  - 当前 partial action
        ↓
ObservationBuilder
        ↓
原子事实 token 序列
        ↓
UpgradePolicyModel / Transformer
        ↓
根据 phase 输出动作
        ↓
RuleEngine mask / 合法性校验
        ↓
RuleEngine step
        ↓
整局结束后按真实规则 reward 训练
```

模型主干：

```text
UpgradePolicyModel
├── FactTokenEncoder
├── TransformerEncoder
├── DeclareTrumpHead
├── ChaoDiPiHead
├── BuryBottomDecoder
├── PlayCardDecoder
└── ValueHead
```

MLP 不作为主策略核心。Transformer 是主要决策结构。最后的 Linear / 小 MLP
只作为输出 head。

---

## 2. 四人自我对战设计

升级是四人二打二：

```text
P0 + P2 一队
P1 + P3 一队
```

第一版使用 **共享 policy**：

```text
P0, P1, P2, P3 都调用同一个模型
但 observation 必须转换成当前玩家视角
```

相对角色：

```text
self
partner
left_enemy
right_enemy
```

示例：

```text
当前玩家 = P1

P1 = self
P3 = partner
P2 = left_enemy
P0 = right_enemy
```

所有 token 中凡是涉及玩家身份，都优先使用相对身份，而不是绝对
P0/P1/P2/P3。

---

## 3. phase 定义

模型需要覆盖整局，而不只是出牌阶段。

必须支持这些 phase：

```python
class Phase(Enum):
    DECLARE_TRUMP = "declare_trump"   # 抢主 / 反主
    CHAO_DIPI = "chao_dipi"           # 炒地皮 / 反底
    BURY_BOTTOM = "bury_bottom"       # 埋牌
    PLAY_CARD = "play_card"           # 正式出牌
```

不同 phase 用同一个 Transformer 主干，但不同输出 head。

---

## 4. 规则引擎适配接口

不要改规则引擎核心逻辑。新增一个 wrapper / adapter。

建议文件：

```text
upgrade_ai/env_adapter.py
```

接口：

```python
class UpgradeEnvAdapter:
    def reset(self, seed: int | None = None) -> None:
        ...

    def clone(self) -> "UpgradeEnvAdapter":
        ...

    def current_player(self) -> int:
        ...

    def phase(self) -> Phase:
        ...

    def team_of(self, player_id: int) -> int:
        ...

    def relative_role(self, viewer_id: int, actor_id: int) -> str:
        """
        Return one of:
        self / partner / left_enemy / right_enemy
        """
        ...

    def is_terminal(self) -> bool:
        ...

    def get_private_hand(self, viewer_id: int) -> list[CardInstance]:
        """
        Only current viewer's hand.
        """
        ...

    def get_public_state(self, viewer_id: int) -> PublicState:
        """
        Only public/current-player-visible state.
        No hidden hands.
        """
        ...

    def get_public_history(self, viewer_id: int) -> list[PublicEvent]:
        """
        All public events from start to current decision point.
        Includes declare, chao, bottom-public-events, played cards.
        No future events.
        """
        ...

    def get_discrete_legal_actions(self, player_id: int, phase: Phase) -> list[DiscreteAction]:
        """
        Used for DECLARE_TRUMP and CHAO_DIPI if action space is enumerable.
        """
        ...

    def legal_next_card_mask(
        self,
        player_id: int,
        phase: Phase,
        selected_cards: list[CardInstance],
    ) -> dict[CardInstance, bool]:
        """
        Used by BURY_BOTTOM and PLAY_CARD autoregressive decoders.
        Tells which remaining physical hand cards may be selected next.
        """
        ...

    def can_stop_action(
        self,
        player_id: int,
        phase: Phase,
        selected_cards: list[CardInstance],
    ) -> bool:
        """
        Whether current selected card group can be finalized.
        For BURY_BOTTOM this usually means selected count == bottom count.
        For PLAY_CARD this means selected_cards forms a legal completed play.
        """
        ...

    def build_action_from_selected_cards(
        self,
        player_id: int,
        phase: Phase,
        selected_cards: list[CardInstance],
    ) -> GameAction:
        ...

    def is_legal_action(self, player_id: int, action: GameAction) -> bool:
        ...

    def step(self, action: GameAction) -> None:
        ...

    def final_rewards(self) -> dict[int, float]:
        """
        Return reward per player, or at least per team.
        Reward must reflect real level progression and special punishments.
        """
        ...
```

如果当前规则引擎只有完整动作合法性判断，第一版可以先实现保守版本：

```text
legal_next_card_mask:
  允许当前玩家手里还没选过的牌

can_stop_action:
  调用 rule_engine.is_legal_action(selected_cards)
```

如果模型生成到最大长度仍不能 STOP，则触发安全
fallback：重新采样或调用一个简单合法动作 fallback。后续再逐步增强 prefix
mask。

---

## 5. token 设计总原则

核心原则：

```text
token 表示原子事实，不表示复合事件。
```

不要使用这种复合 token：

```text
PLAY_EVENT = 第 3 墩 P1 跟出红桃AA，是对子，20分，跟上花色
```

应该拆成原子事实 token：

```text
scope=play_group_17, field=actor, value=left_enemy
scope=play_group_17, field=order_in_current_trick, value=2
scope=played_card_41, field=belongs_to_group, value=play_group_17
scope=played_card_41, field=card, value=H_A
scope=played_card_42, field=belongs_to_group, value=play_group_17
scope=played_card_42, field=card, value=H_A
```

每个 token 是一个 `field=value` 事实，但带有
scope，用于绑定属于同一张牌、同一次出牌、同一个阶段。

---

## 6. FactToken 数据结构

建议文件：

```text
upgrade_ai/tokens.py
```

数据结构：

```python
@dataclass(frozen=True)
class FactToken:
    token_type: str
    scope_type: str
    scope_id: int
    field: str
    value_type: str
    value: str | int | float | bool
    actor_role: str | None = None
    numeric_value: float | None = None
```

解释：

```text
token_type:
  GLOBAL_FACT
  PRIVATE_CARD_FACT
  PLAYED_CARD_FACT
  DECLARE_FACT
  BOTTOM_FACT
  CURRENT_TRICK_FACT
  PARTIAL_ACTION_FACT

scope_type:
  global
  private_card
  played_card
  play_group
  declare_event
  bottom_event
  current_trick
  partial_action

scope_id:
  当前 observation 内部的局部 id。
  用于让模型知道哪些 token 属于同一对象/同一次出牌。
```

注意：`scope_id`
不是策略语义，只是绑定关系。不要把“第几墩”作为核心策略输入。

---

## 7. 不要把“第几墩”作为核心 token

不要给模型显式强调：

```text
当前是第 13 墩
```

因为真实打牌时更重要的是：

```text
我还剩几张牌
每人还剩几张牌
当前是否最后一墩
这一墩第几个出牌
当前要跟几张
当前这一墩有多少分
当前谁最大
```

保留：

```text
remaining_cards_self
remaining_cards_partner
remaining_cards_left_enemy
remaining_cards_right_enemy
is_last_trick
order_in_current_trick
required_play_count
current_trick_points
current_trick_best_owner
```

不要依赖绝对 trick index。

历史顺序由 token 序列位置编码表示。

---

## 8. ObservationBuilder

建议文件：

```text
upgrade_ai/observation.py
```

接口：

```python
@dataclass
class Observation:
    viewer_id: int
    phase: Phase
    tokens: list[FactToken]
    hand_card_instances: list[CardInstance]
    partial_action_cards: list[CardInstance]
```

构造函数：

```python
class ObservationBuilder:
    def build(
        self,
        env: UpgradeEnvAdapter,
        viewer_id: int,
        partial_action_cards: list[CardInstance] | None = None,
    ) -> Observation:
        ...
```

每个决策点都构造当前玩家视角的 observation。

输入必须包含：

```text
1. 当前 phase
2. 当前公开全局状态
3. 当前玩家自己的私有手牌
4. 从开局到当前为止的全部公开历史
5. 当前这一墩公开状态
6. partial action，即本次动作已经选了哪些牌
```

绝不能包含：

```text
其他玩家真实手牌
未来出牌
最终结果
底牌内容，除非当前玩家实际看过
```

---

## 9. 必须包含的 GLOBAL_FACT tokens

示例：

```text
PHASE = PLAY_CARD
TRUMP_SUIT = spade
LEVEL_RANK = J
DEALER_ROLE = partner
CURRENT_PLAYER_ROLE = self
CURRENT_LEADER_ROLE = left_enemy
ORDER_IN_CURRENT_TRICK = 2
REQUIRED_PLAY_COUNT = 2
REMAINING_CARDS_SELF = 6
REMAINING_CARDS_PARTNER = 6
REMAINING_CARDS_LEFT_ENEMY = 6
REMAINING_CARDS_RIGHT_ENEMY = 6
IS_LAST_TRICK = false
CURRENT_TRICK_POINTS = 20
CURRENT_TRICK_BEST_OWNER = right_enemy
CURRENT_SCORE = normalized score
```

特殊规则相关公开状态也要加入，例如：

```text
CURRENT_LEVEL = A
MUST_PLAY_2_STATUS
MUST_PLAY_J_STATUS
MUST_PLAY_A_STATUS
SPECIAL_HOOK_RULE_ENABLED = true
SPECIAL_A_EXECUTION_RULE_ENABLED = true
```

第一版固定单规则，也可以不放规则配置，但必须放当前局面会影响策略的公开状态。

---

## 10. 手牌 token：每张物理牌拆成多个原子事实

每张当前玩家手里的物理牌都要有自己的 scope。

示例：

```text
scope=private_card_0, field=card_id, value=S_J
scope=private_card_0, field=suit, value=spade
scope=private_card_0, field=rank, value=J
scope=private_card_0, field=is_trump, value=true
scope=private_card_0, field=is_point, value=false
scope=private_card_0, field=is_level_card, value=false
scope=private_card_0, field=is_joker, value=false
scope=private_card_0, field=is_main_hook, value=true
scope=private_card_0, field=is_side_hook, value=false
scope=private_card_0, field=is_main_A_killer, value=false
scope=private_card_0, field=is_side_A_killer, value=false
scope=private_card_0, field=last_trick_power_class, value=beats_big_joker
```

两副牌里两张相同牌必须是两个不同 physical card instances。

---

## 11. 历史出牌 token：每张已出物理牌一个 token scope

历史里每张公开已出的物理牌都要独立表示。

例如某人出红桃AA：

```text
scope=play_group_17, field=actor_role, value=left_enemy
scope=play_group_17, field=order_in_current_trick, value=2
scope=play_group_17, field=lead_suit, value=heart

scope=played_card_41, field=belongs_to_group, value=play_group_17
scope=played_card_41, field=card_id, value=H_A
scope=played_card_41, field=actor_role, value=left_enemy
scope=played_card_41, field=card_index_in_group, value=0
scope=played_card_41, field=is_trump, value=false
scope=played_card_41, field=is_point, value=true
scope=played_card_41, field=is_main_A_killer, value=false
scope=played_card_41, field=is_side_A_killer, value=true_or_false

scope=played_card_42, field=belongs_to_group, value=play_group_17
scope=played_card_42, field=card_id, value=H_A
scope=played_card_42, field=actor_role, value=left_enemy
scope=played_card_42, field=card_index_in_group, value=1
...
```

`play_group` 只用于绑定同一次出牌，不要做成复合事件 token。

---

## 12. 抢主 / 反主 / 炒地皮 / 埋牌公开事件

这些也拆成原子事实。

抢主示例：

```text
scope=declare_3, field=actor_role, value=right_enemy
scope=declare_3, field=declare_type, value=call_trump
scope=declare_3, field=declare_suit, value=heart
scope=declare_3, field=shown_card, value=H_J
scope=declare_3, field=declare_strength, value=single
```

炒地皮示例：

```text
scope=chao_1, field=actor_role, value=left_enemy
scope=chao_1, field=event_type, value=chao_dipi_success
scope=chao_1, field=bottom_transferred, value=true
scope=chao_1, field=new_controller_role, value=left_enemy
```

埋牌公开事件示例：

```text
scope=bottom_2, field=event_type, value=bury_bottom
scope=bottom_2, field=actor_role, value=self
scope=bottom_2, field=buried_count, value=8
```

如果当前玩家看过底牌，底牌进入其 private hand
tokens。其他玩家不能看到底牌具体内容。

---

## 13. partial action tokens

出牌和埋牌是自回归生成的，所以每一步 observation
还要包含已经选入本次动作的牌。

示例：

```text
scope=partial_action, field=phase, value=PLAY_CARD
scope=partial_card_0, field=card_id, value=D_A
scope=partial_card_0, field=belongs_to, value=partial_action
scope=partial_card_1, field=card_id, value=D_A
scope=partial_card_1, field=belongs_to, value=partial_action
```

模型根据当前 partial action 决定下一张选什么，或 STOP。

---

## 14. 模型结构

建议文件：

```text
upgrade_ai/model.py
```

模型：

```python
class UpgradePolicyModel(nn.Module):
    def __init__(self, config: ModelConfig):
        ...

    def encode_observation(self, obs_batch: BatchObservation) -> EncodedContext:
        ...

    def forward_value(self, obs_batch: BatchObservation) -> torch.Tensor:
        ...

    def forward_discrete_actions(
        self,
        obs_batch: BatchObservation,
        legal_actions: list[list[DiscreteAction]],
    ) -> torch.Tensor:
        """
        Used for DECLARE_TRUMP and CHAO_DIPI.
        Return logits per legal action.
        """
        ...

    def forward_card_pointer(
        self,
        obs_batch: BatchObservation,
        hand_card_mask: torch.Tensor,
        stop_mask: torch.Tensor,
    ) -> CardPointerOutput:
        """
        Used for BURY_BOTTOM and PLAY_CARD.
        Return logits over current physical hand cards + STOP.
        """
        ...
```

第一版模型配置：

```python
@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    max_tokens: int = 512
```

RK3588 部署友好，先从小模型开始。后面再扩到：

```text
d_model = 256
layers = 4
heads = 8
```

---

## 15. FactTokenEncoder

将 FactToken 转成 embedding。

建议每个 token embedding 由这些部分组成：

```text
token_type_embedding
+ scope_type_embedding
+ field_embedding
+ categorical_value_embedding
+ actor_role_embedding
+ local_scope_embedding
+ numeric_projection
```

数值字段不要直接作为类别，用归一化数值过 Linear：

```python
numeric_emb = numeric_mlp(torch.tensor([numeric_value]))
```

categorical value 用 embedding。

---

## 16. 动作输出方式

### 16.1 抢主 / 反主 / 炒地皮

动作空间较小，由规则引擎枚举 legal discrete actions。

```text
pass
call_trump
redeclare
chao_dipi
...
```

模型对 legal actions 打分。

```python
legal_actions = env.get_discrete_legal_actions(player, phase)
logits = model.forward_discrete_actions(obs, legal_actions)
action = sample_or_argmax(logits)
```

### 16.2 埋牌

埋牌是从手牌中选固定数量 N 张牌。

使用自回归 decoder：

```text
Step 1: 选一张要埋的牌
Step 2: 选一张要埋的牌
...
Step N: 选够 N 张
```

每一步调用：

```python
mask = env.legal_next_card_mask(player, Phase.BURY_BOTTOM, selected_cards)
```

### 16.3 正式出牌

正式出牌也是自回归选牌：

```text
Step 1: 选一张牌
Step 2: 选一张牌
...
Step K: STOP
```

每一步调用：

```python
mask = env.legal_next_card_mask(player, Phase.PLAY_CARD, selected_cards)
stop_allowed = env.can_stop_action(player, Phase.PLAY_CARD, selected_cards)
```

输出空间：

```text
当前手牌里的每张物理牌 + STOP
```

---

## 17. 自我对战采样

建议文件：

```text
upgrade_ai/self_play.py
```

核心流程：

```python
def play_one_game(env, policy, temperature: float) -> GameTrajectory:
    trajectory = []

    while not env.is_terminal():
        player = env.current_player()
        phase = env.phase()

        if phase in [DECLARE_TRUMP, CHAO_DIPI]:
            obs = obs_builder.build(env, player)
            action, log_prob, value, entropy = sample_discrete_action(
                env, policy, obs, player, phase, temperature
            )

        elif phase in [BURY_BOTTOM, PLAY_CARD]:
            action, log_prob, value, entropy = sample_card_sequence_action(
                env, policy, player, phase, temperature
            )

        assert env.is_legal_action(player, action)
        env.step(action)

        trajectory.append(DecisionStep(
            player_id=player,
            team_id=env.team_of(player),
            phase=phase,
            obs=obs,
            action=action,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
        ))

    rewards = env.final_rewards()

    for step in trajectory:
        step.final_reward = rewards[step.player_id]

    return GameTrajectory(steps=trajectory)
```

---

## 18. Reward 设计

第一版 reward 必须来自规则引擎真实结算。

不要只用：

```text
赢 +1，输 -1
```

必须反映真实等级损益，尤其是：

```text
正常上台
正常保庄
升几级
退回打 8
退回打 2
A 枪毙到 J
主勾抠底
副勾抠底
```

推荐：

```python
reward = rule_engine.level_progress_delta_for_player_team(player)
```

也就是：本局前后，我方在真实规则下的等级/进度变化。

这样模型才能学到：

```text
牌不好时，宁可让对方正常上台，也不要让对方主勾抠底。
```

---

## 19. 训练算法第一版

使用简单 on-policy actor-critic。

建议文件：

```text
upgrade_ai/train.py
```

一批 self-play games 后训练：

```python
advantage = final_reward - value_prediction.detach()

policy_loss = -log_prob_sum * advantage
value_loss = mse(value_prediction, final_reward)
entropy_loss = -entropy

loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

推荐初始超参：

```python
learning_rate = 3e-4
value_coef = 0.5
entropy_coef = 0.01
batch_games = 32
gamma = 1.0
```

如果一个动作由多步选牌组成：

```python
action_log_prob = sum(step_log_probs)
action_entropy = sum(step_entropies)
```

每个决策点都保存自己的 log_prob、value、entropy。

后续可以升级到 PPO，但第一版先用 actor-critic 跑通闭环。

---

## 20. 模型池与评估

不要永远让 current model 只和自己打。

维护模型池：

```text
random_policy
best_v1
best_v2
best_v3
current_best
```

第一阶段可以所有玩家都用 current policy。
第二阶段开始，每局随机选择对手或队友策略。

评估 candidate 是否晋级：

```text
candidate 控制 P0/P2
best 控制 P1/P3
打 N 局

交换座位：
best 控制 P0/P2
candidate 控制 P1/P3
再打 N 局
```

晋级条件示例：

```text
candidate 平均 reward 明显高于 best
或胜率 > 53% / 55%
且非法动作数 = 0
```

跟踪指标：

```text
illegal_action_count
average_reward
win_rate
level_delta
main_hook_bottom_capture_count
side_hook_bottom_capture_count
main_A_execution_count
side_A_execution_count
normal_up_count
reset_to_2_count
reset_to_8_count
```

---

## 21. 文件结构建议

```text
upgrade_ai/
  __init__.py

  env_adapter.py          # 规则引擎适配层
  types.py                # CardInstance, Phase, GameAction, DecisionStep
  tokens.py               # FactToken, token enums
  observation.py          # ObservationBuilder
  batching.py             # token padding / tensorization

  model.py                # UpgradePolicyModel
  action_sampling.py      # discrete action + card pointer sampling
  self_play.py            # 四人自我对战
  train.py                # actor-critic training loop
  evaluate.py             # candidate vs best model
  checkpoints.py          # save/load model + metadata
  config.py               # ModelConfig, TrainConfig

  tests/
    test_observation_no_hidden_info.py
    test_tokenizer_shapes.py
    test_env_adapter_masks.py
    test_random_game_completion.py
    test_model_forward.py
    test_self_play_one_game.py
```

---

## 22. 第一阶段任务清单

### Task 1：实现 EnvAdapter

目标：

```text
能 reset
能 current_player
能 phase
能拿当前玩家手牌
能拿公开历史
能 step
能 final_rewards
能判断合法动作
```

验收：

```text
随机动作策略能完整打完 100 局
非法动作数为 0 或 fallback 后为 0
```

---

### Task 2：实现 ObservationBuilder

目标：

```text
给定 env + viewer_id + partial_action
构造当前玩家视角 token 序列
```

验收：

```text
不能包含其他玩家真实手牌
不能包含未来出牌
不能包含最终结果
当前玩家自己的手牌必须完整
历史必须只包含公开事件
```

---

### Task 3：实现 FactTokenEncoder + batching

目标：

```text
FactToken list → tensor batch
支持 padding mask
支持 categorical embedding
支持 numeric projection
```

验收：

```text
不同长度 observation 能 batch
模型 forward 不报错
```

---

### Task 4：实现 UpgradePolicyModel

目标：

```text
Transformer 编码 token
ValueHead 输出 value
DiscreteHead 能处理抢主/炒地皮 legal actions
CardPointerHead 能在手牌 card tokens + STOP 上输出 logits
```

验收：

```text
随机 observation forward 正常
logits shape 正确
padding mask 生效
```

---

### Task 5：实现动作采样

目标：

```text
DECLARE_TRUMP / CHAO_DIPI:
  从 legal discrete actions 里采样

BURY_BOTTOM / PLAY_CARD:
  自回归从手牌里选牌
  每步应用 legal_next_card_mask
  STOP 由 can_stop_action 控制
```

验收：

```text
采样动作全部通过 env.is_legal_action
```

---

### Task 6：实现 self-play

目标：

```text
一个共享模型控制四个玩家
每个玩家使用相对视角 observation
完整打一局，保存 trajectory
```

验收：

```text
能跑 100 局 self-play
无非法动作
trajectory 每步包含 obs, action, log_prob, value, entropy, final_reward
```

---

### Task 7：实现 actor-critic 训练

目标：

```text
收集 batch_games 局
计算 policy_loss + value_loss + entropy_loss
更新模型
保存 checkpoint
```

验收：

```text
loss 能下降或至少稳定
value 输出不全是 NaN
模型能持续 self-play
```

---

### Task 8：实现评估与晋级

目标：

```text
candidate vs best
双向座位交换
输出胜率、平均 reward、特殊灾难事件次数
达标才晋级
```

验收：

```text
能自动保存 best checkpoint
能保留 model pool
```

---

## 23. 第一版不要做的事情

第一版不要做：

```text
多规则泛化
rule config token
大模型 / LLM
纯 MLP 策略网络
完整候选动作枚举
人工手写复杂牌理特征
PPO
MCTS
复杂 belief head
复杂 risk auxiliary head
```

第一版只做：

```text
单规则
原子事实 token
小 Transformer
规则引擎 mask
自回归动作生成
四人自我对战
actor-critic
评估晋级
```

---

## 24. 后续增强方向

第一版稳定后再加：

```text
BeliefHead:
  预测其他玩家剩余手牌分布

RiskAuxHeads:
  预测主勾抠底风险
  预测副勾抠底风险
  预测主 A / 副 A 枪毙风险

ScenarioSampler:
  专门采样危险残局
  比如主勾未出、底分重、最后几手风险高

ModelPool:
  更多旧模型混合

PPO:
  替代简单 actor-critic

Rule adaptation:
  从当前单规则 best model 微调到其他规则
```

---

## 25. 最终验收目标

第一版成功标准：

```text
1. 模型能在固定规则下完整打完一局
2. 任何阶段不出非法动作
3. 四人 self-play 能稳定跑批量训练
4. candidate 模型能通过评估晋级
5. 模型能打赢 random policy
6. 模型能逐步减少明显灾难事件，例如主勾/副勾/A 导致的严重负收益
7. 所有训练输入严格符合当前玩家可见信息
```

实现时优先保证正确性，再考虑速度。
