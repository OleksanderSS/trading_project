"""Build keyword frequency index in DuckDB — small/medium tables only.
Larger tables (huggingface_data ~1M rows) use precomputed row counts.

Usage:
    python -m dean_os.build_keyword_index
    python -m dean_os.build_keyword_index --rebuild
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd


KEYWORD_INDEX_TABLE = "keyword_index"

TEXT_SOURCES: list[tuple[str, str]] = [
    ("google_news", "title"),
    ("google_news", "content"),
    ("rss_news", "title"),
    ("rss_news", "content"),
    ("newsapi_articles", "title"),
    ("newsapi_articles", "description"),
    ("newsapi_articles", "content"),
    ("sec_filings", "primaryDocDescription"),
]

DEFAULT_KEYWORDS = [
    "ai", "artificial intelligence", "semiconductor", "gpu", "chip",
    "data center", "cloud",
    "interest rate", "inflation", "cpi", "fed", "federal reserve", "treasury",
    "revenue", "earnings", "guidance", "outlook",
    "volatility", "correction", "recession", "bear market", "bull market",
    "dividend", "buyback", "merger", "acquisition",
    "energy", "oil", "gas", "renewable",
    "defense", "government", "regulation",
    "nvda", "amd", "intc", "tsm", "aapl", "msft", "amzn", "googl", "meta",
    "tsla", "spy", "qqq", "iwm",
    "pe ratio", "pb ratio", "roe", "fcf", "debt", "equity",
    "tariff", "trade war", "sanction",
]


def _scan_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    kw_list: list[str],
) -> list[tuple[str, str, str, int]]:
    parts = []
    for kw in kw_list:
        escaped = kw.replace("'", "''")
        parts.append(
            f'SUM(CASE WHEN "{column}" ILIKE \'%{escaped}%\' THEN 1 ELSE 0 END) AS "{escaped}"'
        )
    row = con.execute(
        f'SELECT {", ".join(parts)} FROM "{table}"'
    ).fetchone()
    return [
        (table, column, kw, int(row[i]))
        for i, kw in enumerate(kw_list)
        if row[i] and int(row[i]) > 0
    ]


def build_keyword_index(
    db_path: str | Path = "data/trading_data.duckdb",
    rebuild: bool = False,
    keywords: list[str] | None = None,
) -> int:
    con = duckdb.connect(str(db_path))
    kw_list = keywords or DEFAULT_KEYWORDS

    if rebuild:
        con.execute(f"DROP TABLE IF EXISTS {KEYWORD_INDEX_TABLE}")

    exists = con.execute(
        f"SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = '{KEYWORD_INDEX_TABLE}'"
    ).fetchone()[0] > 0
    if exists:
        row = con.execute(f"SELECT COUNT(*) FROM {KEYWORD_INDEX_TABLE}").fetchone()[0]
        if row > 0:
            print(f"Keyword index exists: {row} entries")
            con.close()
            return row
        con.execute(f"DROP TABLE IF EXISTS {KEYWORD_INDEX_TABLE}")

    con.execute(f"""
        CREATE TABLE {KEYWORD_INDEX_TABLE} (
            source_table VARCHAR,
            source_column VARCHAR,
            keyword VARCHAR,
            row_count BIGINT
        )
    """)

    total = 0
    for table, col in TEXT_SOURCES:
        try:
            cnt = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            if cnt == 0:
                continue
        except Exception:
            continue
        non_null = con.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IS NOT NULL'
        ).fetchone()[0]
        if non_null == 0:
            continue

        t0 = time.time()
        rows = _scan_table(con, table, col, kw_list)
        elapsed = time.time() - t0
        print(f"  {table}.{col} ({non_null:,} rows): {elapsed:.1f}s -> {len(rows)} keyword matches")

        if rows:
            con.executemany(
                f"INSERT INTO {KEYWORD_INDEX_TABLE} VALUES (?, ?, ?, ?)",
                rows,
            )
            total += len(rows)

    con.execute(f"CREATE INDEX IF NOT EXISTS idx_kw_lookup ON {KEYWORD_INDEX_TABLE}(keyword, source_table)")
    con.close()
    print(f"\nIndexed {total} keyword-table pairs across {len(TEXT_SOURCES)} text sources")
    return total


def lookup_keyword(
    db_path: str | Path = "data/trading_data.duckdb",
    keyword: str | list[str] | None = None,
    table: str | None = None,
) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    kw_list = [keyword] if isinstance(keyword, str) else (keyword or [])
    if kw_list:
        like_clauses = " OR ".join("keyword ILIKE ?" for _ in kw_list)
        params = [f"%{k}%" for k in kw_list]
    else:
        like_clauses = "1=1"
        params = []
    table_clause = f"AND source_table = '{table}'" if table else ""
    df = con.execute(
        f"SELECT source_table, source_column, keyword, row_count "
        f"FROM {KEYWORD_INDEX_TABLE} "
        f"WHERE ({like_clauses}) {table_clause} "
        f"ORDER BY row_count DESC LIMIT 200",
        params,
    ).fetchdf()
    con.close()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build keyword frequency index in DuckDB")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--db", default="data/trading_data.duckdb")
    args = parser.parse_args()
    print(f"Building keyword index on {args.db} ...")
    build_keyword_index(args.db, rebuild=args.rebuild)
    print("Done.")


if __name__ == "__main__":
    main()
