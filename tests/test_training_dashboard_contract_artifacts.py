"""Deployment-level consistency for generated dashboard contracts."""

from scripts.generate_training_dashboard_contracts import (
    generated_artifacts,
)


def test_generated_training_dashboard_contracts_are_current() -> None:
    for path, expected in generated_artifacts():
        assert path.read_text(encoding="utf-8") == expected
