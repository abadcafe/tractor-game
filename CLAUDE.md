# 拖拉机游戏 (Tractor Game) — 开发铁律

## 类型系统铁律（不可违反）

目标：所有语言都要像 Rust 一样强类型，编译期捕获所有类型错误

### 1. Python：100% 类型注解，pyright strict 零错误

- **所有 Python 代码**（包括 `*_tests.py`）必须有完整的类型注解
- 禁止 `Any`、禁止裸 `list` / `dict` / `tuple`（必须写泛型参数如 `list[str]`）
- 禁止使用 `# pyright: ignore` 和 `# type: ignore` 掩盖错误
- 优先使用 `Pydantic` 而不是 `cast`
- 必须通过 `pyright --pythonversion 3.14` **strict 模式**，**0 errors, 0 warnings**
- Python 版本：3.14，必须使用 `type` 类型别名语法和泛型函数语法

### 2. TypeScript：禁止 any

- **所有 TypeScript 代码**禁止出现 `any`
- 接口必须精确，不允许 `object` 模糊类型
- 禁止隐式 `any`（开启 `noImplicitAny`）

### 3. 错误处理

- **禁止抛异常**：所有业务错误必须用 Result/Optional 机制处理（如 `Ok[T] | Rejected`），像 Rust 一样
- **编程错误必须崩溃**：**禁止捕获** `AssertionError`、`IndexError`、`KeyError`、`TypeError`、`RuntimeError` 等编程错误，就让它崩溃暴露 bug
- **第三方异常处理**：第三方库抛出的异常如果是正常流程控制（例如网络错误处理等），则必须在调用点尽量窄地捕获并正确处理, **其他情况禁止捕获**，就让它崩溃暴露 bug
- **测试代码**：用 `assert isinstance(result, Ok)` / `assert isinstance(result, Rejected)` 替代 `pytest.raises`

### 4. 代码修改后必须立即验证

- 修改任何 `.py` 文件（包括测试文件）后，必须不带参数运行 `pyright` 确认无任何警告和错误
- 修改任何 `.ts` 文件后，必须运行 `deno task build` 确认 tsc 编译零错误（含类型检查）

## 项目技术栈

- **Backend**: Python 3.14, FastAPI, Pydantic v2, WebSocket
- **Frontend**: TypeScript, 原生 DOM（无框架）
- **Game Logic**: 纯函数状态机（SM），不可变状态模式
- **Testing**: pytest, pytest-asyncio

## 启动游戏

- 启动服务器时必须加 `--ws websockets-sansio` 参数，例如：`uvicorn app:app --ws websockets-sansio`
