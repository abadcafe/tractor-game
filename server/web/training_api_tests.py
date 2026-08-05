"""Black-box tests for the training HTTP API."""

import os
import time
from pathlib import Path

from server.training_control.config import training_control_config
from server.training_control.process_inspection import pid_file_path
from server.web.application_test_client import (
    SyncServerClient,
)
from server.web.application_test_client import (
    is_dict as _is_dict,
)
from server.web.application_test_client import (
    json_object_from as _json_object_from,
)

pytest_plugins: tuple[str, ...] = (
    "server.web.application_test_fixtures",
)


def test_training_api_exposes_distinct_init_and_resume_schemas(
    sync_client: SyncServerClient,
) -> None:
    init_response = sync_client.get("/api/training/init/schema")
    resume_response = sync_client.get("/api/training/resume/schema")

    assert init_response.status_code == 200
    assert resume_response.status_code == 200
    init_document = _json_object_from(init_response)
    resume_document = _json_object_from(resume_response)
    assert _is_dict(init_document)
    assert _is_dict(resume_document)
    init_properties = init_document["properties"]
    resume_properties = resume_document["properties"]
    assert _is_dict(init_properties)
    assert _is_dict(resume_properties)
    assert "d_model" in init_properties
    assert "checkpoint" not in init_properties
    assert "checkpoint" in resume_properties
    assert "d_model" not in resume_properties


def test_training_config_returns_server_default_directory(
    sync_client: SyncServerClient,
) -> None:
    response = sync_client.get("/api/training/config")

    assert response.status_code == 200
    document = _json_object_from(response)
    assert _is_dict(document)
    assert document["default_run_dir"] == str(
        training_control_config().default_run_dir
    )


def test_training_init_requires_yes_before_replacement(
    sync_client: SyncServerClient,
    tmp_path: Path,
) -> None:
    request: dict[str, object] = {
        "run_dir": str(tmp_path),
        "d_model": 8,
        "layers": 1,
        "heads": 1,
    }

    initialized = sync_client.post("/api/training/init", json=request)
    _ = (tmp_path / "stdout.log").write_text(
        "old output\n", encoding="utf-8"
    )
    (tmp_path / "runtime").mkdir()
    _ = (tmp_path / "runtime" / "stale").write_text(
        "old", encoding="utf-8"
    )
    _ = pid_file_path(tmp_path).write_text(
        f"{os.getpid()}\n", encoding="ascii"
    )
    rejected = sync_client.post("/api/training/init", json=request)
    request["replace_existing"] = "yes"
    replaced = sync_client.post("/api/training/init", json=request)

    assert initialized.status_code == 204
    assert rejected.status_code == 412
    assert _json_object_from(rejected) == {
        "detail": "type yes to replace existing training artifacts"
    }
    assert replaced.status_code == 204
    assert not (tmp_path / "stdout.log").exists()
    assert not (tmp_path / "runtime").exists()
    assert not pid_file_path(tmp_path).exists()


def test_training_resume_returns_before_cli_preflight_failure(
    sync_client: SyncServerClient,
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoints").mkdir()

    response = sync_client.post(
        "/api/training/resume",
        json={"run_dir": str(tmp_path), "checkpoint": "latest.json"},
    )

    assert response.status_code == 204
    deadline = time.monotonic() + 15.0
    log_path = tmp_path / "training-cli.log"
    while not log_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert log_path.exists()
    while log_path.stat().st_size == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    output = log_path.read_text(encoding="utf-8")
    assert "latest.json" in output
    stopped = sync_client.post(
        "/api/training/stop", json={"run_dir": str(tmp_path)}
    )
    assert stopped.status_code == 200
    assert not pid_file_path(tmp_path).exists()
