# 拖拉机游戏 (Tractor Game) — 开发铁律

## 类型系统铁律（不可违反）

### 1. Python：100% 类型注解，pyright strict 零错误

- **所有 Python 代码**（包括 `*_tests.py`）必须有完整的类型注解
- 禁止 `Any`、禁止裸 `list` / `dict` / `tuple`（必须写泛型参数如 `list[str]`）
- 必须通过 `pyright --pythonversion 3.14` **strict 模式**，**0 errors, 0 warnings**
- 目标：像 Rust 一样强类型，编译期捕获所有类型错误
- Python 版本：3.14，必须使用 `type` 类型别名语法和泛型函数语法

### 2. TypeScript：禁止 any

- **所有 TypeScript 代码**禁止出现 `any`
- 接口必须精确，不允许 `object` 模糊类型
- 禁止隐式 `any`（开启 `noImplicitAny`）

### 3. 状态机：Result 类型驱动

- **状态机操作**（stir/play/discard/exchange）返回 `StateResult[T] = Ok[T] | Rejected`
- 绝不通过异常控制正常业务流（异常只用于代码 bug 哨兵）
- **测试状态机的代码**：用 `assert isinstance(result, Ok)` / `assert isinstance(result, Rejected)` 替代 `pytest.raises`
- `game.act()` 永远在最后执行 `_push_state_to_all()`，任何路径都不能跳过

### 4. 代码修改后必须立即验证

- 修改任何 `.py` 文件后，运行 `pyright <path>` 确认无任何警告和错误，包括测试文件

## 项目技术栈

- **Backend**: Python 3.14, FastAPI, Pydantic v2, WebSocket
- **Frontend**: TypeScript, 原生 DOM（无框架）
- **Game Logic**: 纯函数状态机（SM），不可变状态模式
- **Testing**: pytest, pytest-asyncio
