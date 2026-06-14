# Task Issues

## TI-001: Task spec _interleave_error assertions use wrong error keywords
- **Description**: The task spec's _interleave_error function has two assertion bugs: (1) The unknown action type assertion checks for Chinese keywords ("无效的操作", "不能在", "未知") but the server returns English "unknown action type: {action_type}" from _parse_action(). (2) The DEAL_BID bid error assertion checks for ("不在手牌中", "不是主牌等级", "不一致", "价值为零") but the server can return "王叫牌必须出对子" (joker pair required), "优先级不足" (priority insufficient), or "非庄家方不能叫牌" (non-declarer team cannot bid).
- **Assumption**: Added "unknown" to unknown action keywords. Added "王叫牌必须出对子", "优先级不足", "非庄家方不能叫牌" to bid error keywords. Fixed in code.

## TI-002: Disconnector thread + _interleave_error causes test hang
- **Description**: The test spec's test_full_game includes a disconnector thread that randomly closes WS connections every 0.5-2s. The _interleave_error function sends 4+ do_action() calls per phase. Each do_action() can be disrupted by the disconnector, triggering reconnection. The reconnection sends {"type":"next_round","seq":0} which produces extra state pushes. These extra pushes cause the test's receive_state() loop to spin indefinitely, re-entering _interleave_error or skipping it but still receiving stale state messages that prevent the game from advancing.
- **Assumption**: Added tested_interleave flags and do_bid_pass fallback. The test still hangs. The fundamental issue is the reconnection strategy producing state pushes that interfere with the test loop. This requires task redesign to fix properly.

## TI-003: Task spec code references _ws_cm as object type causing pyright errors
- **Description**: Task spec defines `self._ws_cm: object | None = None` but then calls `self._ws_cm.__exit__()` on it. Pyright strict mode reports "Cannot access attribute '__exit__' for class 'object'". The actual type returned by `websocket_connect()` is `WebSocketTestSession`.
- **Assumption**: Changed type to `WebSocketTestSession | None`. Fixed in code.
