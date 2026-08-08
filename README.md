# 拖拉机

一个包含纯函数游戏状态机、自对战训练、训练监控和模型推理玩家的
拖拉机项目。

## PyTorch 后端

安装时必须明确选择且只选择一个 PyTorch 后端：

```bash
# macOS MPS
uv sync --extra training-mps

# Linux CPU
uv sync --extra training-cpu

# Linux CUDA
uv sync --extra training-cuda
```

`training-mps` 和 `training-cuda` 使用 PyPI 上对应平台的官方
PyTorch wheel；`training-cpu` 使用 PyTorch 官方 CPU-only
wheel。三个 extra 互斥。

服务器唯一启动入口；`<training-backend>` 必须替换为上述三个 extra
之一：

```bash
uv run --extra <training-backend> python -m server.web \
  --host 127.0.0.1 --port 8000
```

游戏大厅可用两类自动玩家：

- `AUTO`：只使用规则的随机合法策略；
- `AI`：加载训练 checkpoint，按 Policy 的合法动作概率分布直接
  采样完整动作。

AI 默认加载
`$TRAINING_RUN_DIR/checkpoints/latest.json`。可用以下环境变量配置：

- `TRACTOR_AI_CHECKPOINT`
- `TRACTOR_AI_DEVICE`（`cpu`、`cuda` 或 `mps`）

训练与推理共用同一 checkpoint 合约和模型恢复路径；checkpoint 中的
模型配置决定实际网络结构，不存在按 schema 编号命名的专用推理加载器。

当前 checkpoint schema 为 25，只承载一个 `PolicyModel`：

- 模型只包含一个 Observation Encoder 和一个自回归 Action Decoder；
- Policy 直接使用归一化终局回报做 PPO advantage；
- 训练与推理使用同一个完整动作采样器，不存在独立动作评分分支；
- 一个 AdamW optimizer 保存并更新全部 Policy 参数。

AI 部署模式由以下环境变量控制：

- `TRACTOR_AI_MODE`（`local` 或 `remote`）
- `TRACTOR_AI_ENDPOINT`（remote 模式的完整服务器地址）
- `TRACTOR_AI_REQUEST_TIMEOUT`

`remote` 模式把完整 AI controller 放在另一台运行同一
`python -m server.web` 的 CUDA 主机上。游戏进程只向远端
`/api/ai/decision` 发送连续玩家快照并接收命令；完整模型决策留在
远端，不会逐次远程调用模型 forward，也不会在失败时退回较弱玩家。
