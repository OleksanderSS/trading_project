from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from dean_os.saved_news_shard_snapshot import (
    CONTRACT,
    SavedNewsShardSnapshotBuilder,
)


def _database(path: Path) -> None:
    connection = duckdb.connect(str(path))
    connection.execute(
        "CREATE TABLE google_news(title VARCHAR, link VARCHAR, "
        "published_date TIMESTAMPTZ, source VARCHAR, content VARCHAR, hash VARCHAR)"
    )
    connection.execute(
        "INSERT INTO google_news VALUES "
        "('AI demand expands', 'https://example.com/a', "
        "TIMESTAMPTZ '2026-06-01 00:00:00+00', 'Reuters', 'chip demand', 'a'),"
        "('Future item', 'https://example.com/future', "
        "TIMESTAMPTZ '2026-08-01 00:00:00+00', 'Reuters', 'future', 'future')"
    )
    connection.execute(
        "CREATE TABLE rss_news(title VARCHAR, link VARCHAR, "
        "published_date TIMESTAMPTZ, source VARCHAR, content VARCHAR, hash VARCHAR)"
    )
    connection.execute(
        "INSERT INTO rss_news VALUES "
        "('Foundry capacity expands', 'https://example.com/b', "
        "TIMESTAMPTZ '2026-06-02 00:00:00+00', 'Bloomberg', "
        "'advanced packaging capacity expansion', 'b')"
    )
    connection.close()


def test_snapshot_unions_allowlisted_tables_and_filters_future(tmp_path: Path):
    database = tmp_path / "news.duckdb"
    _database(database)
    output = tmp_path / "snapshot.parquet"
    payload = SavedNewsShardSnapshotBuilder(tmp_path / "reports").build(
        database_path=database,
        output_parquet_path=output,
        as_of="2026-06-30T21:00:00Z",
    )

    assert payload["snapshot_contract"] == CONTRACT
    assert payload["status"] == "saved_news_shard_snapshot_ready"
    assert payload["snapshot"]["row_count"] == 2
    assert payload["integration_boundary"]["can_trade"] is False
    frame = pd.read_parquet(output)
    assert set(frame["source_table"]) == {"google_news", "rss_news"}
    assert "Future item" not in set(frame["title"])
    latest = json.loads(
        Path(payload["saved_paths"]["latest_json"]).read_text(encoding="utf-8")
    )
    assert latest["snapshot"]["sha256"] == payload["snapshot"]["sha256"]


def test_snapshot_can_include_existing_saved_parquet(tmp_path: Path):
    database = tmp_path / "news.duckdb"
    _database(database)
    extra = tmp_path / "extra.parquet"
    pd.DataFrame(
        [
            {
                "title": "Extra saved item",
                "summary": "semiconductor capex",
                "source": "CNBC",
                "timestamp": "2026-06-03T00:00:00Z",
                "url": "https://example.com/c",
            }
        ]
    ).to_parquet(extra, index=False)
    output = tmp_path / "snapshot.parquet"
    payload = SavedNewsShardSnapshotBuilder(tmp_path / "reports").build(
        database_path=database,
        output_parquet_path=output,
        as_of="2026-06-30T21:00:00Z",
        include_parquet_paths=[extra],
    )
    assert payload["snapshot"]["row_count"] == 3
    assert len(payload["inputs"]["include_parquets"]) == 1
