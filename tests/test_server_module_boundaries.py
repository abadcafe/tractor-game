"""Enforce dependency direction between server packages."""

from __future__ import annotations

import ast
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
        (*forbidden, "server.training"),
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


def test_training_model_internals_stay_behind_package_boundary() -> (
    None
):
    imports = _package_imports_outside_subpackage("training", "model")
    internal_model_imports = {
        imported
        for imported in imports
        if imported.startswith("server.training.model.")
    }

    assert not internal_model_imports


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


def _package_imports_outside_subpackage(
    package: str, excluded_subpackage: str
) -> set[str]:
    imports: set[str] = set()
    package_root = _SERVER_ROOT / package
    excluded_root = package_root / excluded_subpackage
    for path in package_root.rglob("*.py"):
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
