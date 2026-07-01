# 拖拉机游戏 (Tractor Game)

## 开发铁律（不可违反）

- 所有语言都要像 Rust 一样强类型，编译期解决所有类型错误，禁止忽略和跳过
- 任何代码修改后都必须全量验证（含格式化，检查，单测，集测等）

### 1. TypeScript：禁止 any，0 编译警告和错误

- **所有 TypeScript 代码**禁止出现 `any`
- 接口必须精确，不允许 `object` 模糊类型
- 禁止隐式 `any`（开启 `noImplicitAny`）

### 2. Python：100% 类型注解，0 编译警告和错误

- **所有 Python 代码**（包括 `*_tests.py`）必须有完整的类型注解
- 必须使用 `type` 类型别名语法和泛型函数语法
- 禁止 `Any`、禁止裸 `list` / `dict` / `tuple`（必须写泛型参数如 `list[str]`）
- 禁止使用 `# pyright: ignore` 和 `# type: ignore` 掩盖错误
- 优先使用 `Pydantic` 而不是 `cast`
- 必须通过 `pyright` **strict 模式**，**0 errors, 0 warnings**
- 必须通过 `ruff check` 和 `ruff format --check`，**0 errors, 0 warnings**
- 全量验证必须包含所有依赖：
  - `uv run --extra dev --extra training pyright`
  - `uv run --extra dev --extra training pytest`
  - `uv run --extra dev --extra training ruff check`
  - `uv run --extra dev --extra training ruff format --check`

### 3. 错误处理

- **禁止抛异常**：所有业务错误必须用 Result/Optional 机制处理（如 `Ok[T] | Rejected`），像 Rust 一样
- **编程错误必须崩溃**：**禁止捕获** `AssertionError`、`IndexError`、`KeyError`、`TypeError`、`RuntimeError` 等编程错误，就让它崩溃暴露 bug
- **第三方异常处理**：第三方库抛出的异常如果是正常流程控制（例如网络错误处理等），则必须在调用点尽量窄地捕获并正确处理, **其他情况禁止捕获**，就让它崩溃暴露 bug
- **测试代码**：用 `assert isinstance(result, Ok)` / `assert isinstance(result, Rejected)` 替代 `pytest.raises`

## 项目技术栈

- **Backend**: Python 3.14, FastAPI, Pydantic v2, WebSocket
- **Frontend**: TypeScript, 原生 DOM（无框架）
- **Game Logic**: 纯函数状态机（SM），不可变状态模式
- **Testing**: pytest, pytest-asyncio
- **Training**: PyTorch（可选依赖 `training` extra，仅训练/全量验证安装）

## 启动游戏

- 启动服务器时必须加 `--ws websockets-sansio` 参数，例如：`uvicorn app:app --ws websockets-sansio`
