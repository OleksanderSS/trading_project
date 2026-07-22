"""Build an immutable, point-in-time saved-news snapshot from local shards.

The main pipeline keeps news in several DuckDB tables.  A later collector run
may legitimately produce a smaller convenience parquet, so that parquet must
not silently become the only analyst source.  This builder reads an allowlist
of local tables in read-only mode, optionally unions existing saved parquets,
normalizes only provenance fields, and writes one hash-bound parquet plus a
review manifest.  It performs no collection or network access.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso


CONTRACT = "dean_saved_news_shard_snapshot_v1"
ALLOWED_TABLES = ("google_news", "newsapi_articles", "rss_news")


class SavedNewsShardSnapshotBuilder:
    """Materialize a deterministic saved-news union without mutating sources."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/saved_news_shard_snapshot_current"
        ),
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        database_path: str | Path,
        output_parquet_path: str | Path,
        as_of: str,
        include_parquet_paths: list[str | Path] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError("saved-news snapshot as_of must be timezone-aware")
        database = Path(database_path)
        if not database.exists():
            raise FileNotFoundError(database)
        database_sha_before = _sha256_file(database)

        frames, table_counts = _read_allowed_tables(database)
        parquet_inputs: list[dict[str, Any]] = []
        for raw_path in include_parquet_paths or []:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_parquet(path)
            normalized = _normalize_saved_parquet(frame, path)
            frames.append(normalized)
            parquet_inputs.append(
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "row_count": len(frame),
                }
            )

        database_sha_after = _sha256_file(database)
        if database_sha_after != database_sha_before:
            raise RuntimeError("saved-news database changed during snapshot read")

        combined = (
            pd.concat(frames, ignore_index=True)
            if frames
            else _empty_frame()
        )
        normalized = _finalize(combined, as_of=as_of_dt)
        output_path = Path(output_parquet_path)
        _atomic_write_parquet(normalized, output_path)
        output_sha = _sha256_file(output_path)

        run_id = _run_id()
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "snapshot_contract": CONTRACT,
            "status": (
                "saved_news_shard_snapshot_ready"
                if len(normalized)
                else "blocked_no_saved_news_rows"
            ),
            "inputs": {
                "database_path": str(database),
                "database_sha256": database_sha_after,
                "as_of": as_of_dt.isoformat(),
                "allowed_tables": list(ALLOWED_TABLES),
                "include_parquets": parquet_inputs,
            },
            "table_row_counts": table_counts,
            "snapshot": {
                "path": str(output_path),
                "sha256": output_sha,
                "row_count": len(normalized),
                "source_table_counts": {
                    str(key): int(value)
                    for key, value in normalized["source_table"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "earliest_published_at": (
                    normalized["published_date"].min()
                    if len(normalized)
                    else None
                ),
                "latest_published_at": (
                    normalized["published_date"].max()
                    if len(normalized)
                    else None
                ),
            },
            "integration_boundary": {
                "read_only_database": True,
                "allowlisted_tables_only": True,
                "point_in_time_filtered": True,
                "source_content_rewritten": False,
                "automatic_lane_promotion": False,
                "automatic_collection": False,
                "can_train": False,
                "can_trade": False,
            },
            "explicit_non_actions": [
                "No network collector was started.",
                "No source database table was written or altered.",
                "No evidence lane was promoted by this snapshot builder.",
                "No model, replay task, learning memory, order, or trade was created.",
            ],
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_markdown(payload),
                run_id=run_id,
            )
        return payload


def _read_allowed_tables(
    database: Path,
) -> tuple[list[pd.DataFrame], dict[str, int]]:
    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    connection = duckdb.connect(str(database), read_only=True)
    try:
        present = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        for table in ALLOWED_TABLES:
            if table not in present:
                counts[table] = 0
                continue
            if table in {"google_news", "rss_news"}:
                query = f"""
                    SELECT title,
                           content AS summary,
                           source,
                           published_date,
                           link,
                           hash AS source_record_hash,
                           '{table}' AS source_table
                    FROM {table}
                """
            else:
                query = """
                    SELECT title,
                           COALESCE(description, content) AS summary,
                           source.name AS source,
                           publishedAt AS published_date,
                           url AS link,
                           hash AS source_record_hash,
                           'newsapi_articles' AS source_table
                    FROM newsapi_articles
                """
            frame = connection.execute(query).fetchdf()
            counts[table] = len(frame)
            frames.append(frame)
    finally:
        connection.close()
    return frames, counts


def _normalize_saved_parquet(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    def column(*names: str) -> pd.Series:
        for name in names:
            if name in frame.columns:
                return frame[name]
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")

    return pd.DataFrame(
        {
            "title": column("title"),
            "summary": column("summary", "description", "content"),
            "source": column("source"),
            "published_date": column(
                "published_date", "publishedAt", "timestamp"
            ),
            "link": column("link", "url"),
            "source_record_hash": column("hash", "source_record_hash"),
            "source_table": f"parquet:{path.as_posix()}",
        }
    )


def _finalize(frame: pd.DataFrame, *, as_of: Any) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame()
    result = frame.copy()
    published = pd.to_datetime(
        result["published_date"], utc=True, errors="coerce"
    )
    result = result.loc[published.notna() & (published <= as_of)].copy()
    published = published.loc[result.index]
    result["published_date"] = published.map(lambda value: value.isoformat())
    for column in ("title", "summary", "source", "link", "source_record_hash"):
        result[column] = result[column].fillna("").astype(str).str.strip()
    result = result.loc[
        (result["title"] != "") & (result["source"] != "")
    ].copy()
    missing_hash = result["source_record_hash"] == ""
    result.loc[missing_hash, "source_record_hash"] = result.loc[
        missing_hash
    ].apply(
        lambda row: _canonical_sha256(
            {
                "title": row["title"],
                "source": row["source"],
                "published_date": row["published_date"],
                "link": row["link"],
            }
        ),
        axis=1,
    )
    result["snapshot_row_sha256"] = result.apply(
        lambda row: _canonical_sha256(
            {
                "title": row["title"],
                "summary": row["summary"],
                "source": row["source"],
                "published_date": row["published_date"],
                "link": row["link"],
                "source_table": row["source_table"],
                "source_record_hash": row["source_record_hash"],
            }
        ),
        axis=1,
    )
    result = result.drop_duplicates("snapshot_row_sha256", keep="first")
    return result.sort_values(
        ["published_date", "source_table", "source_record_hash"],
        kind="stable",
    ).reset_index(drop=True)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "title",
            "summary",
            "source",
            "published_date",
            "link",
            "source_record_hash",
            "source_table",
            "snapshot_row_sha256",
        ]
    )


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.parquet", dir=str(path.parent)
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id() -> str:
    return "saved_news_shard_snapshot_" + utc_now_iso().replace(
        ":", ""
    ).replace("+", "Z")


def render_markdown(payload: dict[str, Any]) -> str:
    snapshot = payload.get("snapshot", {})
    lines = [
        "# DEAN-OS Saved News Shard Snapshot",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- As of: `{payload.get('inputs', {}).get('as_of')}`",
        f"- Snapshot rows: {snapshot.get('row_count', 0)}",
        f"- Snapshot SHA-256: `{snapshot.get('sha256')}`",
        f"- Can trade: {payload.get('integration_boundary', {}).get('can_trade')}",
        "",
        "## Source tables",
        "",
    ]
    lines.extend(
        f"- `{name}`: {count} rows"
        for name, count in payload.get("table_row_counts", {}).items()
    )
    lines.extend(["", "## Explicit non-actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"
