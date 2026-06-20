# Player 协议规范

这份文档总结当前 player 和 server 之间的协议。

这里的 player 包括：

- `HumanPlayer`
- `AutoPlayer`
- `AIPlayer`

所有 player 都必须通过同一套协议和 `Game.receive()` 交互，不能绕过
`Game` 直接调用底层状态机。

## 通信模型

协议只有两个方向。

Player 发给 server：

```text
PlayerMessage(seq, raw)
```

Server 发给 player：

```text
StateMessage(seq, awaiting, state, error?)
```

浏览器 WebSocket 传输的 JSON 和 Python 内部的 `PlayerMessage` /
`StateMessage` 是同一套语义。

区别只在于：

- `HumanPlayer` 从 WebSocket 收 JSON，再包装成 `PlayerMessage`
- `AutoPlayer` 直接构造 `PlayerMessage`
- `AIPlayer` 也应该直接构造 `PlayerMessage`

## PlayerMessage

Player 发给 server 的消息结构是：

```python
PlayerMessage(
    seq=int,
    raw=dict[str, object],
)
```

浏览器发出的 JSON 也是同样结构：

```json
{
  "seq": 12,
  "type": "play",
  "cards": ["D1-hearts-A"]
}
```

`seq` 是协议序号。  
`raw` 是动作内容。

## StateMessage

Server 发给 player 的状态结构是：

```python
StateMessage(
    seq=int,
    awaiting=str | None,
    state=StateSnapshot,
    error=str | None = None,
)
```

WebSocket JSON 形式是：

```json
{
  "type": "state",
  "seq": 12,
  "awaiting": "play",
  "state": {},
  "error": "可选错误信息"
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `type` | 固定为 `"state"` |
| `seq` | 当前状态序号 |
| `awaiting` | 当前 player 是否需要行动 |
| `state` | 当前 player 可见的完整状态 |
| `error` | 上一次动作被拒绝时的原因 |

`error` 只发给刚才动作被拒绝的 player。

## 序号规则

游戏状态序号从 `1` 开始。

如果 player 不知道当前状态，必须发：

```json
{ "seq": 0 }
```

server 收到消息后，先检查 `seq`。

| 情况 | server 行为 |
| --- | --- |
| `seq == 0` | 返回当前状态，不解析其他字段 |
| `seq != 当前状态序号` | 返回当前状态，不解析其他字段 |
| `seq == 当前状态序号` | 才解析动作 |

因此：

- server 启动后不主动推初始状态
- player 必须主动用 `seq=0` 拉状态
- 浏览器崩溃、AI 内存丢失、连接重建后，都可以重新发 `seq=0`
- seq 不匹配时，server 不看 `type`、`cards`、`pass` 等字段

状态变化成功后：

- server 广播新状态给所有 player
- `seq` 增加

动作被拒绝后：

- server 只把错误状态发给行动的 player
- `seq` 不增加
- 游戏状态不变化

## Player 可以发送的动作

除了拉状态的 `{ "seq": 0 }`，所有动作都必须带当前状态的 `seq`。

| 场景 | 消息 |
| --- | --- |
| 拉当前状态 | `{ "seq": 0 }` |
| 下一轮/开始确认 | `{ "type": "next_round", "seq": 12 }` |
| 抢主 | `{ "type": "bid", "seq": 12, "cards": ["..."] }` |
| 不抢 | `{ "type": "bid", "seq": 12, "pass": true }` |
| 反主 | `{ "type": "stir", "seq": 12, "cards": ["..."] }` |
| 不反 | `{ "type": "stir", "seq": 12, "pass": true }` |
| 埋底牌 | `{ "type": "discard", "seq": 12, "cards": ["..."] }` |
| 出牌 | `{ "type": "play", "seq": 12, "cards": ["..."] }` |

`cards` 里放 card id，不放牌面文字。

server 会用当前 player 的手牌解析这些 id。  
如果某个 id 不在当前 player 的手牌里，动作会被拒绝。

## StateSnapshot

`state` 是当前 player 能看到的信息。它不应该包含其他 player 的完整手牌。

### 基本局面

| 字段 | 含义 |
| --- | --- |
| `phase` | 当前阶段 |
| `awaiting_action` | 当前 player 应该做什么 |
| `player_hand` | 当前 player 的手牌 |
| `player_hand_counts` | 四个 player 的手牌数量 |
| `bottom_cards` | 当前可见的底牌 |

### 主牌和庄家

| 字段 | 含义 |
| --- | --- |
| `trump_rank` | 当前级牌 |
| `trump_suit` | 当前主花色，`null` 表示无主 |
| `declarer_team` | 庄家队 |
| `declarer_player` | 庄家 player |
| `bid_events` | 抢主阶段的历史 |
| `bid_winner` | 当前有效的主牌声明 |

`bid_winner` 表示当前有效的抢主/反主结果。  
它不是庄家字段。

### 出牌信息

| 字段 | 含义 |
| --- | --- |
| `trick` | 当前墩进行中的出牌 |
| `last_completed_trick` | 最近完成的一墩，未完成过则为 `null` |
| `failed_throw` | 甩牌失败后的公开惩罚信息 |
| `defender_points` | 防守方当前捡分 |
| `defender_point_cards` | 防守方按捡分顺序累计拿到的分牌 |

### 其他信息

| 字段 | 含义 |
| --- | --- |
| `action_hints` | server 给当前动作的候选牌组 |
| `stirring_state` | 炒地皮/反主阶段状态 |
| `scoring` | 一轮结束后的结算信息 |
| `winning_team` | 游戏结束时的获胜队 |
| `team0_level` | 0 队等级 |
| `team1_level` | 1 队等级 |
| `next_round_confirmed` | 已确认下一轮的 player |

## action_hints

`action_hints` 是提示，不是协议权限。

| 情况 | 含义 |
| --- | --- |
| 非空 | server 提供了当前动作完整的可呈现候选牌组集合 |
| 空数组 | server 没有提供封闭 hint 集合，不等于没有合法动作 |

例子：

- 抢主阶段：列出当前可抢的逻辑主牌选项
- 反主阶段：列出当前可反的牌
- 跟牌阶段：列出完整合法跟牌集合；如果数量超过上限，则返回空数组
- 首出阶段：通常没有 hint，因为首出可以主动甩牌，组合空间很大

非空 `action_hints` 可以被 UI/AI 当作封闭候选集使用。
因此 server 不能只返回前几个 hint；如果无法完整返回，就必须返回 `[]`。

抢主 hint 是“逻辑动作”的完整集合，牌 ID 只用作 canonical 表示。
例如手里两张等价的黑桃级牌时，单张黑桃抢主只会用其中一个 card id 表示这一种逻辑抢主动作，而不是枚举两个等价 card id。

最终是否合法永远由 server 校验。

## 阶段协议

### WAITING

用途：

- 游戏开始前等待四个 player 确认
- 一轮结束后等待四个 player 确认下一轮

如果当前 player 需要确认：

```text
awaiting_action == "next_round"
```

则发送：

```json
{ "type": "next_round", "seq": 12 }
```

### DEAL_BID

抓牌和抢主阶段。

只有刚抓到牌的 player 会看到：

```text
awaiting_action == "bid"
```

这个 player 必须响应抢主或不抢。  
响应后，server 才继续发下一张牌。

抢主：

```json
{ "type": "bid", "seq": 12, "cards": ["..."] }
```

不抢：

```json
{ "type": "bid", "seq": 12, "pass": true }
```

### STIRRING

炒地皮/反主阶段。

这里有两类动作。

如果当前 player 要埋底牌：

```text
awaiting_action == "discard"
```

则发送：

```json
{ "type": "discard", "seq": 12, "cards": ["..."] }
```

如果当前 player 要反主或不反：

```text
awaiting_action == "stir"
```

反主：

```json
{ "type": "stir", "seq": 12, "cards": ["..."] }
```

不反：

```json
{ "type": "stir", "seq": 12, "pass": true }
```

### PLAYING

出牌阶段。

如果当前 player 要出牌：

```text
awaiting_action == "play"
```

则发送：

```json
{ "type": "play", "seq": 12, "cards": ["..."] }
```

首出时，所有出牌从协议上都可以理解成甩牌。  
正常出牌只是只甩一组牌，所以不会产生额外惩罚展示。

如果甩牌失败：

- server 记录 `failed_throw`
- 实际打出的牌变成被“捡小”的那组牌
- 状态仍然推进

### GAME_OVER

游戏结束。  
player 不需要再发动作。

## 拒绝动作

动作可能被拒绝，例如：

- seq 不匹配
- 阶段不允许
- 不是当前 player 行动
- card id 不在手牌里
- 抢主/反主优先级不足
- 埋底数量不对
- 出牌不符合规则

其中 seq 不匹配比较特殊：

- 不算动作错误
- 不解析动作内容
- 只返回当前状态

其他动作拒绝会通过 `StateMessage.error` 返回给行动 player。

## 协议边界

协议层不应该读取 `Game` 的内部当前序号。  
player 如果不知道状态，就发 `seq=0`。

player 不应该直接操作状态机。  
所有行为都必须变成 `PlayerMessage`，交给 `Game.receive()`。

server 不应该在创建 player 或启动 game 后主动推初始状态。  
初始状态由 player 主动请求。
