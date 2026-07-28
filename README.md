# 拖拉机

一个包含纯函数游戏状态机、自对战训练、训练监控和模型推理玩家的
拖拉机项目。

服务器唯一启动入口：

```bash
uv run python -m server.web --host 127.0.0.1 --port 8000
```

游戏大厅可用两类自动玩家：

- `AUTO`：只使用规则的随机合法策略；
- `AI`：加载训练 checkpoint，以位置无关 observation 建立隐藏牌
  粒子分布，并执行 Gumbel 逐次减半策略改进搜索。

AI 默认加载
`$TRAINING_RUN_DIR/checkpoints/latest.json`。可用以下环境变量配置：

- `TRACTOR_AI_CHECKPOINT`
- `TRACTOR_AI_DEVICE`（`cpu`、`cuda` 或 `mps`）
- `TRACTOR_AI_PARTICLES`
- `TRACTOR_AI_ROOT_CANDIDATES`
- `TRACTOR_AI_SIMULATIONS`

训练与推理共用同一 checkpoint 合约和模型恢复路径；checkpoint 中的
模型配置决定实际网络结构，不存在按 schema 编号命名的专用推理加载器。

AI 部署模式由以下环境变量控制：

- `TRACTOR_AI_MODE`（`local` 或 `remote`）
- `TRACTOR_AI_ENDPOINT`（remote 模式的完整服务器地址）
- `TRACTOR_AI_REQUEST_TIMEOUT`

`remote` 模式把完整 AI controller 放在另一台运行同一
`python -m server.web` 的 CUDA 主机上。游戏进程只向远端
`/api/ai/decision` 发送连续玩家快照并接收命令；belief、搜索、
模拟和模型推理全部留在远端，不会逐次远程调用模型 forward，
也不会在失败时退回较弱玩家。
