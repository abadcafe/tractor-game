"""Typed SQLite result boundaries shared by tests."""

from __future__ import annotations

import sqlite3

from pydantic import ConfigDict, TypeAdapter

type SqlRow = tuple[object, ...]

_OPTIONAL_ROW: TypeAdapter[SqlRow | None] = TypeAdapter(
    SqlRow | None,
    config=ConfigDict(strict=True),
)
_ROWS: TypeAdapter[list[SqlRow]] = TypeAdapter(
    list[SqlRow],
    config=ConfigDict(strict=True),
)


def fetch_optional_row(cursor: sqlite3.Cursor) -> SqlRow | None:
    """Fetch and validate one dynamically typed SQLite row."""
    return _OPTIONAL_ROW.validate_python(cursor.fetchone())


def fetch_rows(cursor: sqlite3.Cursor) -> list[SqlRow]:
    """Fetch and validate all dynamically typed SQLite rows."""
    return _ROWS.validate_python(cursor.fetchall())


__all__ = ("SqlRow", "fetch_optional_row", "fetch_rows")
