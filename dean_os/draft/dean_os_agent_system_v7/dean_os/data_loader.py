from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

_DB_PATH = Path("data/trading_data.duckdb")


def _normalize_frame_key(base_dir: Path, path: Path) -> str:
    relative = path.relative_to(base_dir).with_suffix("")
    key = str(relative).replace(path.anchor, "").replace("\\", "_").replace("/", "_")
    return key.lstrip("_")


def _read_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".feather", ".ipc"}:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported tabular file format: {path}")


def load_local_tabular_data(
    *,
    directory: str | Path,
    extensions: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load local tabular files from a directory into pandas DataFrames."""
    base_dir = Path(directory)
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Local data directory not found: {base_dir}")

    extensions = {".csv", ".parquet", ".feather", ".ipc"} if extensions is None else {e.lower() for e in extensions}
    exclude_patterns = set(exclude_patterns or [])
    frames: dict[str, pd.DataFrame] = {}

    for path in sorted(base_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if any(pattern and pattern in str(path) for pattern in exclude_patterns):
            continue

        key = _normalize_frame_key(base_dir, path)
        if not key:
            key = path.stem
        if key in frames:
            suffix = 1
            while f"{key}_{suffix}" in frames:
                suffix += 1
            key = f"{key}_{suffix}"

        frames[key] = _read_tabular_file(path)
    return frames


def load_duckdb_tables(
    *,
    db_path: str | Path = _DB_PATH,
    tables: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load tables from DuckDB into DataFrames.

    Args:
        db_path: Path to DuckDB database file.
        tables: List of table names to load. If None, loads all.
        limit: Max rows per table (None = all).

    Returns:
        Dict mapping table_name → DataFrame.
    """
    import duckdb

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB not found: {db_path}")

    con = duckdb.connect(str(db_path))
    result: dict[str, pd.DataFrame] = {}

    try:
        all_tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
        table_names = [t[0] for t in all_tables]

        for name in table_names:
            if tables and name not in tables:
                continue
            if limit:
                df = con.execute(f'SELECT * FROM "{name}" LIMIT {limit}').fetchdf()
            else:
                df = con.execute(f'SELECT * FROM "{name}"').fetchdf()
            result[name] = df

    finally:
        con.close()

    return result


def get_table_sizes(*, db_path: str | Path = _DB_PATH) -> dict[str, int]:
    """Get row counts for all tables without loading data."""
    import duckdb

    db_path = Path(db_path)
    if not db_path.exists():
        return {}

    con = duckdb.connect(str(db_path))
    result: dict[str, int] = {}

    try:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
        for t in tables:
            count = con.execute(f'SELECT count(*) FROM "{t[0]}"').fetchone()[0]
            result[t[0]] = count
    finally:
        con.close()

    return result


def get_table_preview(
    table_name: str,
    *,
    db_path: str | Path = _DB_PATH,
    rows: int = 5,
) -> pd.DataFrame:
    """Preview a table's first rows."""
    import duckdb

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB not found: {db_path}")

    con = duckdb.connect(str(db_path))
    try:
        df = con.execute(f'SELECT * FROM "{table_name}" LIMIT {rows}').fetchdf()
    finally:
        con.close()

    return df


__all__ = [
    "get_table_preview",
    "get_table_sizes",
    "load_duckdb_tables",
    "load_local_tabular_data",
]
