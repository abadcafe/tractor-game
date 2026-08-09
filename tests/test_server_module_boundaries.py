"""Enforce dependency direction between server packages."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_SERVER_ROOT = Path(__file__).parents[1] / "server"


def test_training_control_is_process_only() -> None:
    imports = _package_imports("training_control")

    assert not _matching(
        imports,
        (
            "server.training",
            "server.training_cli",
            "server.training_events",
            "server.training_metrics",
            "server.training_artifacts",
            "server.game",
        ),
    )


def test_training_and_game_do_not_depend_on_control_or_cli() -> None:
    forbidden = ("server.training_control", "server.training_cli")

    assert not _matching(_package_imports("training"), forbidden)
    assert not _matching(
        _package_imports("game"),
        (
            *forbidden,
            "server.training",
            "server.game_bots",
            "server.game_runtime",
            "server.web",
        ),
    )


def test_game_runtime_and_bots_follow_dependency_direction() -> None:
    assert not _matching(
        _package_imports("game_runtime"),
        (
            "server.game_bots",
            "server.training",
            "server.web",
        ),
    )
    assert not _matching(
        _package_imports("game_bots"),
        ("server.training", "server.web"),
    )
    assert not _matching(
        _package_imports("training"),
        (
            "server.game_bots",
            "server.game_runtime",
            "server.web",
        ),
    )
    assert not _matching(
        _package_imports("game_ai"),
        (
            "server.game_bots",
            "server.game_runtime",
            "server.training",
            "server.training_control",
            "server.training_cli",
            "server.web",
        ),
    )
    assert not _matching(
        _package_imports("game_auto"),
        (
            "server.game_ai",
            "server.game_bots",
            "server.game_runtime",
            "server.training",
            "server.web",
        ),
    )


def test_game_domain_has_no_io_or_synchronization_dependencies() -> (
    None
):
    imports = _package_imports("game")

    assert not _matching(
        imports,
        (
            "asyncio",
            "fastapi",
            "httpx",
            "os",
            "pathlib",
            "socket",
            "sqlite3",
            "threading",
            "time",
            "server.game_runtime",
            "server.game_bots",
            "server.web",
        ),
    )


def test_process_local_game_runtime_has_no_locks() -> None:
    forbidden_calls = {
        "Lock",
        "RLock",
        "Semaphore",
        "BoundedSemaphore",
        "Condition",
    }

    assert not _called_attributes("game_runtime", forbidden_calls)


def test_game_internals_stay_behind_public_facades() -> None:
    imports = _imports_outside_package("game")
    game_imports = {
        imported
        for imported in imports
        if imported == "server.game"
        or imported.startswith("server.game.")
    }
    allowed = {
        "server.game",
        "server.game.rules",
        "server.game.rules.bidding",
        "server.game.rules.cards",
        "server.game.rules.cards.faces",
        "server.game.rules.play",
        "server.game.rules.progression",
        "server.game.rules.scoring",
        "server.game.snapshots",
        "server.game.snapshots.contract",
        "server.game.snapshots.events",
        "server.game.snapshots.player",
        "server.game.snapshots.review",
        "server.game.snapshots.tricks",
    }

    assert game_imports <= allowed


def test_runtime_and_bot_internals_stay_behind_facades() -> None:
    imports = _imports_outside_package("game_runtime")
    runtime_imports = {
        imported
        for imported in imports
        if imported == "server.game_runtime"
        or imported.startswith("server.game_runtime.")
    }
    assert runtime_imports <= {
        "server.game_runtime",
        "server.game_runtime.player",
        "server.game_runtime.registry",
    }

    imports = _imports_outside_package("game_bots")
    bot_imports = {
        imported
        for imported in imports
        if imported == "server.game_bots"
        or imported.startswith("server.game_bots.")
    }
    assert bot_imports <= {
        "server.game_bots",
    }

    imports = _imports_outside_package("game_ai")
    ai_imports = {
        imported
        for imported in imports
        if imported == "server.game_ai"
        or imported.startswith("server.game_ai.")
    }
    assert ai_imports <= {"server.game_ai"}


def test_checkpoint_contract_has_no_application_dependencies() -> None:
    assert all(
        not imported.startswith("server.")
        for imported in _package_imports("checkpoint_contract")
    )


def test_events_metrics_and_artifacts_follow_read_model_dag() -> None:
    assert not _matching(
        _package_imports("training_events"),
        (
            "server.training",
            "server.training_control",
            "server.training_metrics",
            "server.training_artifacts",
            "server.game",
        ),
    )
    assert not _matching(
        _package_imports("training_metrics"),
        (
            "server.training",
            "server.training_control",
            "server.training_artifacts",
            "server.game",
        ),
    )
    assert not _matching(
        _package_imports("training_artifacts"),
        (
            "server.training",
            "server.training_control",
            "server.training_metrics",
            "server.game",
        ),
    )


def test_training_cli_uses_only_training_public_interface() -> None:
    imports = _package_imports("training_cli")
    training_imports = {
        imported
        for imported in imports
        if imported == "server.training"
        or imported.startswith("server.training.")
    }

    assert training_imports == {"server.training"}


def test_web_never_imports_training_implementation() -> None:
    assert not _matching(
        _package_imports("web"),
        ("server.training", "server.training_cli"),
    )


def test_policy_model_has_no_application_dependencies() -> None:
    assert not _matching(
        _package_imports("policy_model"),
        (
            "server.game_ai",
            "server.game_bots",
            "server.game_runtime",
            "server.training",
            "server.training_artifacts",
            "server.training_cli",
            "server.training_control",
            "server.training_events",
            "server.training_metrics",
            "server.web",
        ),
    )


def test_policy_model_internals_stay_behind_facades() -> None:
    imports = _imports_outside_package("policy_model")
    policy_imports = {
        imported
        for imported in imports
        if imported == "server.policy_model"
        or imported.startswith("server.policy_model.")
    }

    assert policy_imports <= {
        "server.policy_model",
        "server.policy_model.actions",
        "server.policy_model.actions.decoding",
        "server.policy_model.checkpoint",
        "server.policy_model.inference",
        "server.policy_model.inference.runtime",
        "server.policy_model.network",
        "server.policy_model.observation",
        "server.policy_model.observation.tensor",
        "server.policy_model.observation.tokenization",
        "server.policy_model.return_target",
    }


def test_policy_inference_contracts_import_loads_no_torch() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import server.policy_model.inference; "
            + "assert 'torch' not in sys.modules",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_policy_inference_runtime_loads_no_training_modules() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import server.policy_model.inference.runtime; "
            + "assert not [name for name in sys.modules "
            + "if name == 'server.training' "
            + "or name.startswith('server.training.')]",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _package_imports(package: str) -> set[str]:
    imports: set[str] = set()
    for path in (_SERVER_ROOT / package).rglob("*.py"):
        if path.name.endswith("_tests.py"):
            continue
        module = ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
            ):
                imports.add(node.module)
    return imports


def _imports_outside_package(package: str) -> set[str]:
    imports: set[str] = set()
    excluded_root = _SERVER_ROOT / package
    for path in _SERVER_ROOT.rglob("*.py"):
        if path.name.endswith("_tests.py") or path.is_relative_to(
            excluded_root
        ):
            continue
        module = ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
            ):
                imports.add(node.module)
    return imports


def _matching(
    imports: set[str], forbidden_prefixes: tuple[str, ...]
) -> set[str]:
    return {
        imported
        for imported in imports
        if any(
            imported == prefix or imported.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    }


def _called_attributes(
    package: str,
    forbidden_names: set[str],
) -> set[str]:
    called: set[str] = set()
    for path in (_SERVER_ROOT / package).rglob("*.py"):
        if path.name.endswith("_tests.py"):
            continue
        module = ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Name)
                and function.id in forbidden_names
            ):
                called.add(function.id)
            elif (
                isinstance(function, ast.Attribute)
                and function.attr in forbidden_names
            ):
                called.add(function.attr)
    return called
