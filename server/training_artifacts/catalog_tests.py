"""Black-box tests for the read-only checkpoint catalog."""

import hashlib
import subprocess
import sys
from pathlib import Path

from server.foundation.result import Ok
from server.training.config import TrainConfig
from server.training.model import ModelConfig
from server.training.torch_checkpoints.manifest import (
    write_checkpoint_manifest,
)
from server.training.torch_checkpoints.schema import (
    CheckpointManifest,
    TorchCheckpointMetadata,
)
from server.training_artifacts.catalog import (
    read_checkpoint_catalog,
)


def test_read_checkpoint_catalog_lists_manifest_object_and_orphan(
    tmp_path: Path,
) -> None:
    current_id = "a" * 32
    orphan_id = "b" * 32
    _write_checkpoint(tmp_path, "latest.json", current_id, b"state")
    orphan = tmp_path / "checkpoints" / "objects" / orphan_id
    orphan.mkdir(parents=True)
    orphan.joinpath("state.pt").write_bytes(b"orphan")

    result = read_checkpoint_catalog(tmp_path)

    assert isinstance(result, Ok)
    assert [item.name for item in result.value.manifests] == [
        "latest.json"
    ]
    assert result.value.manifests[0].valid is True
    assert [item.checkpoint_id for item in result.value.objects] == [
        current_id,
        orphan_id,
    ]
    assert result.value.objects[1].orphan is True
    assert result.value.total_unique_state_bytes == 11


def test_invalid_object_id_is_visible_but_never_valid(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "checkpoints" / "objects" / "foo"
    invalid.mkdir(parents=True)
    invalid.joinpath("state.pt").write_bytes(b"state")

    result = read_checkpoint_catalog(tmp_path)

    assert isinstance(result, Ok)
    assert len(result.value.objects) == 1
    assert result.value.objects[0].checkpoint_id == "foo"
    assert result.value.objects[0].valid is False
    assert result.value.objects[0].error is not None
    assert result.value.total_unique_state_bytes == 0


def test_read_checkpoint_catalog_keeps_invalid_manifest_visible(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_dir.joinpath("broken.json").write_text("not json")

    result = read_checkpoint_catalog(tmp_path)

    assert isinstance(result, Ok)
    assert len(result.value.manifests) == 1
    assert result.value.manifests[0].valid is False
    assert result.value.manifests[0].kind == "invalid"


def test_read_checkpoint_catalog_rejects_mismatched_schema_manifest(
    tmp_path: Path,
) -> None:
    checkpoint_id = "a" * 32
    _write_checkpoint(tmp_path, "latest.json", checkpoint_id, b"state")
    manifest_path = tmp_path / "checkpoints" / "latest.json"
    current = manifest_path.read_text(encoding="utf-8")
    stale = current.replace(
        '"schema_version": 22', '"schema_version": 0'
    )
    assert stale != current
    manifest_path.write_text(stale, encoding="utf-8")

    result = read_checkpoint_catalog(tmp_path)

    assert isinstance(result, Ok)
    assert len(result.value.manifests) == 1
    manifest = result.value.manifests[0]
    assert manifest.valid is False
    assert manifest.error is not None
    assert "Input should be 22" in manifest.error


def test_web_application_import_does_not_load_torch() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import server.web.app; "
            "assert 'torch' not in sys.modules",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _write_checkpoint(
    run_dir: Path,
    manifest_name: str,
    checkpoint_id: str,
    state: bytes,
) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    state_path = Path("objects") / checkpoint_id / "state.pt"
    absolute_state = checkpoint_dir / state_path
    absolute_state.parent.mkdir(parents=True)
    absolute_state.write_bytes(state)
    result = write_checkpoint_manifest(
        path=checkpoint_dir / manifest_name,
        manifest=CheckpointManifest(
            checkpoint_id=checkpoint_id,
            state_path=state_path,
            state_sha256=hashlib.sha256(state).hexdigest(),
            metadata=TorchCheckpointMetadata(
                model_config=ModelConfig(),
                train_config=TrainConfig(),
                total_rounds=10,
                total_samples=20,
                total_updates=2,
            ),
        ),
    )
    assert isinstance(result, Ok)
