from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dean_os.schemas import ResearchChunk, ResearchDocument, ResearchNote, SourceCitation
from dean_os.utils import json_ready


class ResearchCorpus:
    """Small local research store for documents, chunks, and notes."""

    def __init__(self, db_path: str | Path = "data/dean_os/research_corpus.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_document(self, document: ResearchDocument, chunk_size: int = 1200) -> list[ResearchChunk]:
        chunks = chunk_document(document, chunk_size=chunk_size)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (document_id, title, source_type, uri, published_at, tickers, sectors, tags, metadata, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.title,
                    document.source_type,
                    document.uri,
                    document.published_at,
                    json.dumps(document.tickers, ensure_ascii=True),
                    json.dumps(document.sectors, ensure_ascii=True),
                    json.dumps(document.tags, ensure_ascii=True),
                    json.dumps(json_ready(document.metadata), ensure_ascii=True),
                    document.text,
                ),
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, document_id, chunk_index, text, token_estimate, citations, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.token_estimate,
                        json.dumps(json_ready(chunk.citations), ensure_ascii=True),
                        json.dumps(json_ready(chunk.metadata), ensure_ascii=True),
                    )
                    for chunk in chunks
                ],
            )
        return chunks

    def add_note(self, note: ResearchNote) -> str:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO notes
                (note_id, agent_name, topic, thesis, patterns, tickers, sectors, confidence, data_quality, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.note_id,
                    note.agent_name,
                    note.topic,
                    note.thesis,
                    json.dumps(note.patterns, ensure_ascii=True),
                    json.dumps(note.tickers, ensure_ascii=True),
                    json.dumps(note.sectors, ensure_ascii=True),
                    note.confidence,
                    note.data_quality,
                    json.dumps(note.model_dump(mode="json"), ensure_ascii=True),
                ),
            )
        return note.note_id

    def ingest_path(
        self,
        path: str | Path,
        source_type: str | None = None,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
        tags: list[str] | None = None,
        chunk_size: int = 1200,
        recursive: bool = True,
    ) -> dict:
        from dean_os.draft.dean_os_agent_system_v7.dean_os.material_loaders import ingest_research_path

        return ingest_research_path(
            path=path,
            corpus=self,
            source_type=source_type,
            tickers=tickers,
            sectors=sectors,
            tags=tags,
            chunk_size=chunk_size,
            recursive=recursive,
        )

    def list_documents(self) -> list[ResearchDocument]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY rowid").fetchall()
        return [self._document_from_row(row) for row in rows]

    def search_chunks(self, query: str, limit: int = 20) -> list[ResearchChunk]:
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return []
        sql = "SELECT * FROM chunks WHERE " + " OR ".join(["LOWER(text) LIKE ?" for _ in terms]) + " ORDER BY rowid LIMIT ?"
        params = [f"%{term}%" for term in terms]
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def list_notes(self, agent_name: str | None = None) -> list[ResearchNote]:
        if agent_name:
            sql = "SELECT payload FROM notes WHERE agent_name = ? ORDER BY rowid"
            params: tuple[Any, ...] = (agent_name,)
        else:
            sql = "SELECT payload FROM notes ORDER BY rowid"
            params = ()
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [ResearchNote(**json.loads(row["payload"])) for row in rows]

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    uri TEXT,
                    published_at TEXT,
                    tickers TEXT NOT NULL,
                    sectors TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    text TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_estimate INTEGER NOT NULL,
                    citations TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    note_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    thesis TEXT NOT NULL,
                    patterns TEXT NOT NULL,
                    tickers TEXT NOT NULL,
                    sectors TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    data_quality TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _document_from_row(self, row) -> ResearchDocument:
        metadata = json.loads(row["metadata"])
        quarantine_flags = _quarantine_flags_from_metadata(metadata)
        return ResearchDocument(
            document_id=row["document_id"],
            title=row["title"],
            source_type=row["source_type"],
            text=row["text"],
            uri=row["uri"],
            published_at=row["published_at"],
            tickers=json.loads(row["tickers"]),
            sectors=json.loads(row["sectors"]),
            tags=json.loads(row["tags"]),
            metadata=metadata,
            quarantine_flags=quarantine_flags,
            quality_precheck=metadata.get("quality_precheck") or ("quarantine_detected" if quarantine_flags else None),
        )

    def _chunk_from_row(self, row) -> ResearchChunk:
        metadata = json.loads(row["metadata"])
        return ResearchChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            token_estimate=row["token_estimate"],
            citations=[SourceCitation(**item) for item in json.loads(row["citations"])],
            metadata=metadata,
            quarantine_flags=list(metadata.get("quarantine_flags", [])),
            quality_precheck=metadata.get("quality_precheck"),
        )


def chunk_document(document: ResearchDocument, chunk_size: int = 1200) -> list[ResearchChunk]:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.intake_normalizer import normalize_and_chunk
    return normalize_and_chunk(document, chunk_size)


def _quarantine_flags_from_metadata(metadata: dict[str, Any]) -> list[str]:
    if isinstance(metadata.get("quarantine_flags"), list):
        return sorted(str(flag) for flag in metadata["quarantine_flags"])
    flags = {
        flag
        for block in metadata.get("quarantine_blocks", [])
        for flag in block.get("quarantine_flags", [])
    }
    return sorted(flags)
