"""Build DuckDB FTS (Full-Text Search) indexes on news/sec tables.
Run once after data load. Subsequent orchestrator runs use the indexes."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import duckdb

FTS_TABLES: dict[str, list[str]] = {
    "huggingface_data": ["text"],
    "google_news": ["title", "content"],
    "rss_news": ["title", "content"],
    "newsapi_articles": ["title", "description", "content"],
    "sec_filings": ["primaryDocDescription"],
}


def build_fts_indexes(
    db_path: str | Path = "data/trading_data.duckdb",
    tables: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL fts; LOAD fts;")
    results: dict[str, float] = {}
    target = tables or FTS_TABLES
    for table, cols in target.items():
        try:
            cnt = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            if cnt == 0:
                results[table] = 0.0
                print(f"  {table}: empty, skipping")
                continue
        except Exception:
            results[table] = 0.0
            print(f"  {table}: not found, skipping")
            continue
        col_list = ", ".join(cols)
        index_name = f"{table}_fts_idx"
        try:
            t0 = time.time()
            con.execute(
                f'PRAGMA create_fts_index("{table}", "{index_name}", "{col_list}", overwrite=1)'
            )
            elapsed = time.time() - t0
            results[table] = elapsed
            print(f"  {table} ({cnt:,} rows, cols={col_list}): {elapsed:.1f}s")
        except Exception as e:
            print(f"  {table}: error - {e}")
            results[table] = -1.0
    con.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DuckDB FTS indexes")
    parser.add_argument("--db", default="data/trading_data.duckdb", help="DuckDB path")
    args = parser.parse_args()

    print(f"Building FTS indexes on {args.db} ...")
    results = build_fts_indexes(args.db)
    total = sum(v for v in results.values() if v > 0)
    print(f"\nDone. {len(results)} tables indexed in {total:.1f}s")
    for table, elapsed in results.items():
        status = f"{elapsed:.1f}s" if elapsed > 0 else ("skipped" if elapsed == 0 else "error")
        print(f"  {table}: {status}")


if __name__ == "__main__":
    main()
