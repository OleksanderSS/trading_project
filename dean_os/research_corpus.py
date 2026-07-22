from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from dean_os.schemas import ResearchChunk, ResearchDocument, ResearchNote, SourceCitation, utc_now_iso
from dean_os.utils import json_ready

# Compatibility boundary: the historical corpus store is a module while the
# governed hypothesis lifecycle lives in the sibling ``research_corpus/``
# directory. Expose that directory as the module's submodule search path so
# both public surfaces remain importable without copying the corpus store or
# maintaining a second ledger implementation.
__path__ = [str(Path(__file__).with_suffix(""))]


class ResearchCorpus:
    """Small local research store for documents, chunks, and notes."""

    def __init__(self, db_path: str | Path = "data/dean_os/research_corpus.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_document(self, document: ResearchDocument, chunk_size: int = 1200) -> list[ResearchChunk]:
        chunks = chunk_document(document, chunk_size=chunk_size)
        doc_payload = json.dumps(json_ready(document.model_dump(mode="json")), ensure_ascii=True)
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM documents WHERE document_id = ?",
                (document.document_id,),
            )
            doc_rev = (cur.fetchone()[0] or 0) + 1
            conn.execute(
                """
                INSERT INTO documents
                (document_id, revision, title, source_type, uri, published_at, tickers, sectors, tags, metadata, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    doc_rev,
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
            conn.execute(
                """
                INSERT INTO document_events
                (event_id, document_id, revision, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (uuid4().hex, document.document_id, doc_rev, utc_now_iso()),
            )
            for chunk in chunks:
                cur = conn.execute(
                    "SELECT COALESCE(MAX(revision), 0) FROM chunks WHERE chunk_id = ?",
                    (chunk.chunk_id,),
                )
                chunk_rev = (cur.fetchone()[0] or 0) + 1
                conn.execute(
                    """
                    INSERT INTO chunks
                    (chunk_id, revision, document_id, chunk_index, text, token_estimate, citations, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk_rev,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.token_estimate,
                        json.dumps(json_ready(chunk.citations), ensure_ascii=True),
                        json.dumps(json_ready(chunk.metadata), ensure_ascii=True),
                    ),
                )
        return chunks

    def add_note(self, note: ResearchNote) -> str:
        payload = json.dumps(note.model_dump(mode="json"), ensure_ascii=True)
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM notes WHERE note_id = ?",
                (note.note_id,),
            )
            next_revision = (cur.fetchone()[0] or 0) + 1
            conn.execute(
                """
                INSERT INTO notes
                (note_id, revision, agent_name, topic, thesis, patterns, tickers, sectors, confidence, data_quality, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.note_id,
                    next_revision,
                    note.agent_name,
                    note.topic,
                    note.thesis,
                    json.dumps(note.patterns, ensure_ascii=True),
                    json.dumps(note.tickers, ensure_ascii=True),
                    json.dumps(note.sectors, ensure_ascii=True),
                    note.confidence,
                    note.data_quality,
                    payload,
                ),
            )
            conn.execute(
                """
                INSERT INTO note_events
                (event_id, note_id, revision, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (uuid4().hex, note.note_id, next_revision, utc_now_iso()),
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
        from dean_os.material_loaders import ingest_research_path

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
        latest_sql = """
            SELECT d.*
            FROM documents d
            INNER JOIN (
                SELECT document_id, MAX(revision) AS max_rev
                FROM documents
                GROUP BY document_id
            ) latest ON d.document_id = latest.document_id AND d.revision = latest.max_rev
            ORDER BY d.rowid
        """
        with self._connect() as conn:
            rows = conn.execute(latest_sql).fetchall()
        return [self._document_from_row(row) for row in rows]

    def search_chunks(self, query: str, limit: int = 20) -> list[ResearchChunk]:
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return []
        latest_sql = """
            SELECT c.*
            FROM chunks c
            INNER JOIN (
                SELECT chunk_id, MAX(revision) AS max_rev
                FROM chunks
                GROUP BY chunk_id
            ) latest ON c.chunk_id = latest.chunk_id AND c.revision = latest.max_rev
        """
        sql = latest_sql + " WHERE " + " OR ".join(["LOWER(c.text) LIKE ?" for _ in terms]) + " ORDER BY c.rowid LIMIT ?"
        params = [f"%{term}%" for term in terms]
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def list_notes(self, agent_name: str | None = None) -> list[ResearchNote]:
        latest_sql = """
            SELECT n.payload
            FROM notes n
            INNER JOIN (
                SELECT note_id, MAX(revision) AS max_rev
                FROM notes
                GROUP BY note_id
            ) latest ON n.note_id = latest.note_id AND n.revision = latest.max_rev
        """
        if agent_name:
            sql = latest_sql + " WHERE n.agent_name = ? ORDER BY n.rowid"
            params: tuple[Any, ...] = (agent_name,)
        else:
            sql = latest_sql + " ORDER BY n.rowid"
            params = ()
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [ResearchNote(**json.loads(row["payload"])) for row in rows]

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
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
                "CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(document_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_events (
                    event_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
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
                "CREATE INDEX IF NOT EXISTS idx_chunks_id ON chunks(chunk_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_id ON notes(note_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS note_events (
                    event_id TEXT PRIMARY KEY,
                    note_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL
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
    from dean_os.intake_normalizer import normalize_and_chunk
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
