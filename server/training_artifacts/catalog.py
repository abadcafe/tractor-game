"""Torch-free checkpoint artifact catalog."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from server.checkpoint_contract import (
    CHECKPOINT_OBJECTS_DIR,
    CHECKPOINT_POLICY_FILENAME,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_TRAINER_FILENAME,
)
from server.foundation import result as _result
from server.foundation.json_value import JsonObject

_CHECKPOINT_ID_PATTERN = r"^[0-9a-f]{32}$"


class _ManifestDocument(BaseModel):
    """Torch-free representation of the on-disk manifest contract."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    schema_version: int
    checkpoint_id: str = Field(pattern=_CHECKPOINT_ID_PATTERN)
    policy_path: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trainer_path: str = Field(min_length=1)
    trainer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_config_values: JsonObject = Field(alias="model_config")
    train_config_values: JsonObject = Field(alias="train_config")
    total_rounds: int = Field(ge=0)
    total_trainable_decisions: int = Field(ge=0)
    total_updates: int = Field(ge=0)

    @field_validator("schema_version")
    @classmethod
    def _require_current_schema(cls, value: int) -> int:
        if value != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Input should be {CHECKPOINT_SCHEMA_VERSION}"
            )
        return value


class CheckpointManifestView(BaseModel):
    """One valid or invalid manifest shown by the dashboard."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    name: str
    kind: Literal["latest", "archive", "invalid"]
    valid: bool
    error: str | None
    checkpoint_id: str | None
    policy_path: str | None
    policy_exists: bool
    policy_size_bytes: int | None = Field(ge=0)
    trainer_path: str | None
    trainer_exists: bool
    trainer_size_bytes: int | None = Field(ge=0)
    modified_at_ms: int | None = Field(ge=0)
    policy_modified_at_ms: int | None = Field(ge=0)
    trainer_modified_at_ms: int | None = Field(ge=0)
    policy_sha256: str | None
    trainer_sha256: str | None
    total_rounds: int | None = Field(ge=0)
    total_trainable_decisions: int | None = Field(ge=0)
    total_updates: int | None = Field(ge=0)
    model_config_values: JsonObject | None
    train_config_values: JsonObject | None


class CheckpointObjectView(BaseModel):
    """One immutable checkpoint object and its manifest references."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    checkpoint_id: str
    policy_path: str
    trainer_path: str
    valid: bool
    error: str | None
    policy_size_bytes: int | None = Field(ge=0)
    trainer_size_bytes: int | None = Field(ge=0)
    policy_modified_at_ms: int | None = Field(ge=0)
    trainer_modified_at_ms: int | None = Field(ge=0)
    referenced_by: tuple[str, ...]
    orphan: bool


class CheckpointCatalog(BaseModel):
    """Complete managed checkpoint directory inventory."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    checkpoint_directory: Path
    manifests: tuple[CheckpointManifestView, ...]
    objects: tuple[CheckpointObjectView, ...]
    total_unique_payload_bytes: int = Field(ge=0)


def read_checkpoint_catalog(
    run_dir: Path,
) -> _result.Ok[CheckpointCatalog] | _result.Rejected:
    """Catalog manifests and immutable objects without loading torch."""
    checkpoint_dir = run_dir.resolve() / "checkpoints"
    try:
        if checkpoint_dir.is_symlink():
            return _result.Rejected(
                reason=(
                    "checkpoint directory must not be a symlink: "
                    f"{checkpoint_dir}"
                )
            )
        if not checkpoint_dir.exists():
            return _result.Ok(
                value=CheckpointCatalog(
                    checkpoint_directory=checkpoint_dir,
                    manifests=(),
                    objects=(),
                    total_unique_payload_bytes=0,
                )
            )
        if not checkpoint_dir.is_dir():
            return _result.Rejected(
                reason=(
                    "checkpoint path is not a directory: "
                    f"{checkpoint_dir}"
                )
            )
        children = tuple(checkpoint_dir.iterdir())
    except OSError:
        return _result.Rejected(
            reason=(
                f"checkpoint directory is unreadable: {checkpoint_dir}"
            )
        )
    manifest_paths = sorted(
        (path for path in children if path.suffix == ".json"),
        key=_manifest_sort_key,
    )
    manifests = tuple(_manifest_view(path) for path in manifest_paths)
    references: dict[str, list[str]] = {}
    for manifest in manifests:
        if manifest.valid and manifest.checkpoint_id is not None:
            references.setdefault(manifest.checkpoint_id, []).append(
                manifest.name
            )
    objects_result = _object_views(checkpoint_dir, references)
    if isinstance(objects_result, _result.Rejected):
        return objects_result
    objects = objects_result.value
    return _result.Ok(
        value=CheckpointCatalog(
            checkpoint_directory=checkpoint_dir,
            manifests=manifests,
            objects=objects,
            total_unique_payload_bytes=sum(
                (item.policy_size_bytes or 0)
                + (item.trainer_size_bytes or 0)
                for item in objects
            ),
        )
    )


def _manifest_view(path: Path) -> CheckpointManifestView:
    kind = _manifest_kind(path)
    result = _read_manifest(path)
    if isinstance(result, _result.Rejected):
        return CheckpointManifestView(
            name=path.name,
            kind=kind,
            valid=False,
            error=result.reason,
            checkpoint_id=None,
            policy_path=None,
            policy_exists=False,
            policy_size_bytes=None,
            trainer_path=None,
            trainer_exists=False,
            trainer_size_bytes=None,
            modified_at_ms=_mtime_ms(path),
            policy_modified_at_ms=None,
            trainer_modified_at_ms=None,
            policy_sha256=None,
            trainer_sha256=None,
            total_rounds=None,
            total_trainable_decisions=None,
            total_updates=None,
            model_config_values=None,
            train_config_values=None,
        )
    manifest = result.value
    policy_path = path.parent / manifest.policy_path
    trainer_path = path.parent / manifest.trainer_path
    policy_error = _payload_path_error(policy_path)
    trainer_error = _payload_path_error(trainer_path)
    policy_stat = (
        None if policy_error is not None else _file_stat(policy_path)
    )
    trainer_stat = (
        None if trainer_error is not None else _file_stat(trainer_path)
    )
    error = policy_error if policy_error is not None else trainer_error
    if error is None and policy_stat is None:
        error = f"checkpoint policy is missing: {policy_path}"
    if error is None and trainer_stat is None:
        error = f"checkpoint trainer is missing: {trainer_path}"
    return CheckpointManifestView(
        name=path.name,
        kind=kind,
        valid=error is None,
        error=error,
        checkpoint_id=manifest.checkpoint_id,
        policy_path=manifest.policy_path,
        policy_exists=policy_stat is not None,
        policy_size_bytes=(
            None if policy_stat is None else policy_stat[0]
        ),
        trainer_path=manifest.trainer_path,
        trainer_exists=trainer_stat is not None,
        trainer_size_bytes=(
            None if trainer_stat is None else trainer_stat[0]
        ),
        modified_at_ms=_mtime_ms(path),
        policy_modified_at_ms=(
            None if policy_stat is None else policy_stat[1]
        ),
        trainer_modified_at_ms=(
            None if trainer_stat is None else trainer_stat[1]
        ),
        policy_sha256=manifest.policy_sha256,
        trainer_sha256=manifest.trainer_sha256,
        total_rounds=manifest.total_rounds,
        total_trainable_decisions=manifest.total_trainable_decisions,
        total_updates=manifest.total_updates,
        model_config_values=manifest.model_config_values,
        train_config_values=manifest.train_config_values,
    )


def _object_views(
    checkpoint_dir: Path,
    references: dict[str, list[str]],
) -> _result.Ok[tuple[CheckpointObjectView, ...]] | _result.Rejected:
    objects_dir = checkpoint_dir / CHECKPOINT_OBJECTS_DIR
    try:
        if objects_dir.is_symlink():
            return _result.Rejected(
                reason=(
                    "checkpoint objects must not be a symlink: "
                    f"{objects_dir}"
                )
            )
        if not objects_dir.exists():
            return _result.Ok(value=())
        children = tuple(
            sorted(objects_dir.iterdir(), key=lambda path: path.name)
        )
    except OSError:
        return _result.Rejected(
            reason=f"checkpoint objects are unreadable: {objects_dir}"
        )
    views: list[CheckpointObjectView] = []
    for object_dir in children:
        policy_path = object_dir / CHECKPOINT_POLICY_FILENAME
        trainer_path = object_dir / CHECKPOINT_TRAINER_FILENAME
        error: str | None = None
        if (
            re.fullmatch(_CHECKPOINT_ID_PATTERN, object_dir.name)
            is None
        ):
            error = f"invalid checkpoint object id: {object_dir}"
        elif object_dir.is_symlink() or not object_dir.is_dir():
            error = f"invalid checkpoint object: {object_dir}"
        elif policy_path.is_symlink() or trainer_path.is_symlink():
            error = (
                "checkpoint payload must not be a symlink: "
                + f"{object_dir}"
            )
        policy_stat = (
            None if error is not None else _file_stat(policy_path)
        )
        trainer_stat = (
            None if error is not None else _file_stat(trainer_path)
        )
        if error is None and policy_stat is None:
            error = f"checkpoint policy is missing: {policy_path}"
        if error is None and trainer_stat is None:
            error = f"checkpoint trainer is missing: {trainer_path}"
        names = tuple(sorted(references.get(object_dir.name, [])))
        views.append(
            CheckpointObjectView(
                checkpoint_id=object_dir.name,
                policy_path=policy_path.relative_to(
                    checkpoint_dir
                ).as_posix(),
                trainer_path=trainer_path.relative_to(
                    checkpoint_dir
                ).as_posix(),
                valid=error is None,
                error=error,
                policy_size_bytes=(
                    None if policy_stat is None else policy_stat[0]
                ),
                trainer_size_bytes=(
                    None if trainer_stat is None else trainer_stat[0]
                ),
                policy_modified_at_ms=(
                    None if policy_stat is None else policy_stat[1]
                ),
                trainer_modified_at_ms=(
                    None if trainer_stat is None else trainer_stat[1]
                ),
                referenced_by=names,
                orphan=not names,
            )
        )
    return _result.Ok(value=tuple(views))


def _file_stat(path: Path) -> tuple[int, int] | None:
    try:
        if not path.is_file():
            return None
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns // 1_000_000


def _mtime_ms(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns // 1_000_000
    except OSError:
        return None


def _read_manifest(
    path: Path,
) -> _result.Ok[_ManifestDocument] | _result.Rejected:
    if (
        path.name != "latest.json"
        and _managed_update_number(path) is None
    ):
        return _checkpoint_rejection(path, "invalid manifest file name")
    if path.is_symlink():
        return _checkpoint_rejection(
            path, "manifest must not be a symlink"
        )
    try:
        manifest = _ManifestDocument.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return _checkpoint_rejection(path, "manifest file is missing")
    except UnicodeDecodeError:
        return _checkpoint_rejection(
            path, "manifest is not valid UTF-8"
        )
    except ValidationError as error:
        return _checkpoint_rejection(path, str(error))
    except OSError:
        return _checkpoint_rejection(
            path, "manifest file is not readable"
        )
    policy_path = Path(manifest.policy_path)
    trainer_path = Path(manifest.trainer_path)
    expected_dir = Path(CHECKPOINT_OBJECTS_DIR) / manifest.checkpoint_id
    if (
        policy_path.is_absolute()
        or ".." in policy_path.parts
        or policy_path != expected_dir / CHECKPOINT_POLICY_FILENAME
        or trainer_path.is_absolute()
        or ".." in trainer_path.parts
        or trainer_path != expected_dir / CHECKPOINT_TRAINER_FILENAME
    ):
        return _checkpoint_rejection(
            path, "manifest payload paths do not match checkpoint id"
        )
    return _result.Ok(value=manifest)


def _checkpoint_rejection(path: Path, reason: str) -> _result.Rejected:
    return _result.Rejected(
        reason=f"checkpoint corruption: {path}: {reason}"
    )


def _payload_path_error(path: Path) -> str | None:
    if path.parent.is_symlink() or path.is_symlink():
        return f"checkpoint payload must not traverse a symlink: {path}"
    return None


def _managed_update_number(path: Path) -> int | None:
    if path.suffix != ".json":
        return None
    update_text = path.stem.removeprefix("update-")
    if update_text == path.stem or not update_text.isdecimal():
        return None
    update_number = int(update_text)
    if (
        update_number <= 0
        or path.name != f"update-{update_number}.json"
    ):
        return None
    return update_number


def _manifest_kind(
    path: Path,
) -> Literal["latest", "archive", "invalid"]:
    if path.name == "latest.json":
        return "latest"
    if _managed_update_number(path) is not None:
        return "archive"
    return "invalid"


def _manifest_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name == "latest.json":
        return (0, 0, path.name)
    update = _managed_update_number(path)
    if update is not None:
        return (1, -update, path.name)
    return (2, 0, path.name)
