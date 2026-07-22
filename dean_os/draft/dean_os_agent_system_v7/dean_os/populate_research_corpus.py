"""Populate the ResearchCorpus SQLite database from DuckDB news tables.
Run once to seed the corpus; agents query it via SpecialistResearchAgent.

Usage:
    python -m dean_os.draft.dean_os_agent_system_v7.dean_os.populate_research_corpus
    python -m dean_os.draft.dean_os_agent_system_v7.dean_os.populate_research_corpus --rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime
from pathlib import Path

import duckdb

from dean_os.draft.dean_os_agent_system_v7.dean_os.research_corpus import ResearchCorpus
from dean_os.schemas import ResearchDocument, SourceCitation


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _make_doc(source_type: str, title: str, text: str,
              uri: str | None = None, published_at: str | None = None,
              tickers: list[str] | None = None) -> ResearchDocument:
    doc_id = _hash_text(text[:500])
    return ResearchDocument(
        document_id=doc_id,
        title=title or f"{source_type} {doc_id[:8]}",
        source_type=source_type,
        text=text,
        uri=uri or "",
        published_at=published_at or "",
        tickers=tickers or [],
        sectors=[],
        tags=[source_type],
        citations=[],
    )


def populate_from_duckdb(
    duckdb_path: str = "data/trading_data.duckdb",
    corpus_path: str = "data/dean_os/research_corpus.sqlite",
    rebuild: bool = False,
    max_per_table: int = 5000,
) -> dict[str, int]:
    con = duckdb.connect(duckdb_path)
    corpus = ResearchCorpus(corpus_path)

    if rebuild:
        import sqlite3
        Path(corpus_path).unlink(missing_ok=True)
        corpus = ResearchCorpus(corpus_path)

    existing = len(corpus.list_documents())
    if existing > 0 and not rebuild:
        print(f"Research corpus already has {existing} documents, skipping.")
        con.close()
        return {"existing": existing}

    sources = [
        ("huggingface_data", "text", "news", None, None),
        ("google_news", "content", "news", "title", "link"),
        ("rss_news", "content", "news", "title", "link"),
        ("newsapi_articles", "content", "news", "title", "url"),
    ]

    total = 0
    for table, text_col, source_type, title_col, uri_col in sources:
        try:
            cnt = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            if cnt == 0:
                continue
        except Exception:
            continue

        query = f'SELECT "{text_col}"'
        if title_col:
            query += f', "{title_col}"'
        if uri_col:
            query += f', "{uri_col}"'
        query += f' FROM "{table}" WHERE "{text_col}" IS NOT NULL'
        if max_per_table:
            query += f" LIMIT {max_per_table}"

        t0 = time.time()
        rows = con.execute(query).fetchall()
        elapsed = time.time() - t0
        print(f"  {table}: {len(rows)} rows loaded in {elapsed:.1f}s")

        docs = []
        for row in rows:
            text = row[0] if row[0] else ""
            if len(text) < 20:
                continue
            title = row[1] if len(row) > 1 and row[1] else ""
            uri = row[2] if len(row) > 2 and row[2] else ""
            docs.append(_make_doc(
                source_type=source_type,
                title=str(title)[:200],
                text=str(text),
                uri=str(uri) if uri else None,
            ))

        for doc in docs:
            corpus.add_document(doc, chunk_size=1200)
        total += len(docs)
        print(f"    -> {len(docs)} documents added")

    con.close()
    print(f"\nTotal: {total} documents added to research corpus")
    return {"added": total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate ResearchCorpus from DuckDB")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--duckdb", default="data/trading_data.duckdb")
    parser.add_argument("--corpus", default="data/dean_os/research_corpus.sqlite")
    parser.add_argument("--max-per-table", type=int, default=5000)
    args = parser.parse_args()
    print(f"Populating research corpus {args.corpus} from {args.duckdb} ...")
    populate_from_duckdb(
        duckdb_path=args.duckdb,
        corpus_path=args.corpus,
        rebuild=args.rebuild,
        max_per_table=args.max_per_table,
    )
    print("Done.")


if __name__ == "__main__":
    main()
