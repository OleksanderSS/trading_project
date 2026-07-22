from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


EVIDENCE_CATALOG_SCHEMA_VERSION = "dean_evidence_catalog_v2"
EVIDENCE_RUN_SCHEMA_VERSION = "dean_evidence_acquisition_run_v1"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class CatalogEvidenceRecord(BaseModel):
    """Immutable normalized evidence descriptor.

    The catalog stores provenance and content identity, not necessarily the full
    copyrighted source body. Full text may remain in the external artifact store.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = EVIDENCE_CATALOG_SCHEMA_VERSION
    evidence_id: str = Field(default_factory=lambda: f"evidence_{uuid4().hex}")
    domain_id: str
    source_type: Literal[
        "news", "article", "book", "report", "filing", "transcript", "metric",
        "dataset", "document", "research_note", "pipeline_artifact", "unknown"
    ] = "unknown"
    source_name: str
    title: str
    locator: str | None = None
    published_at: str | None = None
    observed_at: str | None = None
    available_at: str
    ingested_at: str = Field(default_factory=_utc_now_iso)
    content_hash: str
    metadata_hash: str
    sectors: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    evidence_lanes: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_tier: str = "unknown"
    credibility_score: float = Field(default=0.4, ge=0.0, le=1.0)
    credibility_decision_use: str = "lead_only"
    credibility_reasons: list[str] = Field(default_factory=list)
    duplicate_cluster_id: str | None = None
    duplicate_status: Literal["unique", "exact_duplicate", "near_duplicate", "independent_corroboration"] = "unique"
    duplicate_of: str | None = None
    point_in_time_status: Literal["valid", "review_only", "invalid"] = "review_only"
    quarantine_flags: list[str] = Field(default_factory=list)
    external_artifact_ref: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "CatalogEvidenceRecord":
        available = _parse_timestamp(self.available_at, field_name="available_at")
        ingested = _parse_timestamp(self.ingested_at, field_name="ingested_at")
        if self.published_at:
            _parse_timestamp(self.published_at, field_name="published_at")
        if self.observed_at:
            observed = _parse_timestamp(self.observed_at, field_name="observed_at")
            if available < observed:
                raise ValueError("available_at cannot be earlier than observed_at")
        if ingested < available:
            # Ingestion before availability is logically impossible for a valid catalog record.
            raise ValueError("ingested_at cannot be earlier than available_at")
        if not self.domain_id.strip() or not self.source_name.strip() or not self.title.strip():
            raise ValueError("domain_id, source_name, and title are required")
        if len(self.content_hash) != 64 or len(self.metadata_hash) != 64:
            raise ValueError("content_hash and metadata_hash must be SHA-256 hex digests")
        return self


class EvidenceCatalogAppendResult(BaseModel):
    status: Literal["stored", "already_exists"]
    evidence_id: str
    content_hash: str
    backend: str


class EvidenceAcquisitionRunManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = EVIDENCE_RUN_SCHEMA_VERSION
    acquisition_run_id: str = Field(default_factory=lambda: f"evidence_run_{uuid4().hex}")
    domain_id: str
    as_of: str
    knowledge_cutoff: str
    started_at: str
    completed_at: str
    source_counts: dict[str, int] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    rejected_items: list[dict[str, Any]] = Field(default_factory=list)
    suppressed_items: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: Literal["completed", "partial", "failed"] = "completed"
    content_hash: str

    @model_validator(mode="after")
    def _validate(self) -> "EvidenceAcquisitionRunManifest":
        as_of = _parse_timestamp(self.as_of, field_name="as_of")
        cutoff = _parse_timestamp(self.knowledge_cutoff, field_name="knowledge_cutoff")
        started = _parse_timestamp(self.started_at, field_name="started_at")
        completed = _parse_timestamp(self.completed_at, field_name="completed_at")
        if cutoff > as_of:
            raise ValueError("knowledge_cutoff cannot exceed as_of")
        if completed < started:
            raise ValueError("completed_at cannot be earlier than started_at")
        expected = _sha256({
            "domain_id": self.domain_id,
            "as_of": self.as_of,
            "knowledge_cutoff": self.knowledge_cutoff,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "source_counts": self.source_counts,
            "evidence_ids": self.evidence_ids,
            "rejected_items": self.rejected_items,
            "suppressed_items": self.suppressed_items,
            "warnings": self.warnings,
            "status": self.status,
        })
        if self.content_hash != expected:
            raise ValueError("evidence acquisition manifest content hash mismatch")
        return self


class EvidenceCatalogProtocol(Protocol):
    def append(self, record: CatalogEvidenceRecord) -> EvidenceCatalogAppendResult: ...
    def get(self, evidence_id: str) -> CatalogEvidenceRecord | None: ...
    def list_records(self, *, domain_id: str, available_before: str | None = None, limit: int = 1000) -> list[CatalogEvidenceRecord]: ...


class SQLiteEvidenceCatalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS evidence_catalog (
                    evidence_id TEXT PRIMARY KEY,
                    domain_id TEXT NOT NULL,
                    available_at_epoch REAL NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_catalog_domain_time
                    ON evidence_catalog(domain_id, available_at_epoch DESC);
                CREATE TRIGGER IF NOT EXISTS evidence_catalog_no_update
                    BEFORE UPDATE ON evidence_catalog
                    BEGIN SELECT RAISE(ABORT, 'evidence_catalog is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS evidence_catalog_no_delete
                    BEFORE DELETE ON evidence_catalog
                    BEGIN SELECT RAISE(ABORT, 'evidence_catalog is append-only'); END;
            """)

    def append(self, record: CatalogEvidenceRecord) -> EvidenceCatalogAppendResult:
        payload = record.model_dump(mode="json")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT evidence_id, content_hash, payload_json FROM evidence_catalog WHERE evidence_id = ? OR content_hash = ?",
                (record.evidence_id, record.content_hash),
            ).fetchone()
            if existing:
                existing_payload = json.loads(existing["payload_json"])
                if existing_payload != payload:
                    raise ValueError("evidence identity/content conflict")
                return EvidenceCatalogAppendResult(
                    status="already_exists",
                    evidence_id=existing["evidence_id"],
                    content_hash=existing["content_hash"],
                    backend="sqlite",
                )
            connection.execute(
                "INSERT INTO evidence_catalog(evidence_id, domain_id, available_at_epoch, content_hash, payload_json, created_at) VALUES(?,?,?,?,?,?)",
                (
                    record.evidence_id,
                    record.domain_id,
                    _parse_timestamp(record.available_at, field_name="available_at").timestamp(),
                    record.content_hash,
                    _canonical_json(payload),
                    _utc_now_iso(),
                ),
            )
        return EvidenceCatalogAppendResult(status="stored", evidence_id=record.evidence_id, content_hash=record.content_hash, backend="sqlite")

    def get(self, evidence_id: str) -> CatalogEvidenceRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload_json FROM evidence_catalog WHERE evidence_id = ?", (evidence_id,)).fetchone()
        return CatalogEvidenceRecord.model_validate(json.loads(row["payload_json"])) if row else None

    def list_records(self, *, domain_id: str, available_before: str | None = None, limit: int = 1000) -> list[CatalogEvidenceRecord]:
        clauses = ["domain_id = ?"]
        values: list[Any] = [domain_id]
        if available_before:
            clauses.append("available_at_epoch <= ?")
            values.append(_parse_timestamp(available_before, field_name="available_before").timestamp())
        values.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM evidence_catalog WHERE " + " AND ".join(clauses) + " ORDER BY available_at_epoch DESC LIMIT ?",
                values,
            ).fetchall()
        return [CatalogEvidenceRecord.model_validate(json.loads(row["payload_json"])) for row in rows]


class EvidenceCatalogBuilder:
    def build_record(
        self,
        payload: dict[str, Any],
        *,
        domain_id: str,
        as_of: str,
        knowledge_cutoff: str | None = None,
    ) -> CatalogEvidenceRecord:
        available_at = str(payload.get("available_at") or payload.get("published_at") or payload.get("timestamp") or as_of)
        raw_content = payload.get("text") or payload.get("content") or payload.get("value") or payload.get("summary") or ""
        content_hash = _sha256(raw_content)
        metadata_payload = {key: value for key, value in payload.items() if key not in {"text", "content"}}
        metadata_hash = _sha256(metadata_payload)
        observed_at = payload.get("observed_at") or payload.get("published_at") or payload.get("timestamp")
        ingested_at = str(payload.get("ingested_at") or max(_parse_timestamp(available_at, field_name="available_at"), _parse_timestamp(as_of, field_name="as_of")).isoformat())
        cutoff = knowledge_cutoff or as_of
        point_in_time_status = "valid" if _parse_timestamp(available_at, field_name="available_at") <= _parse_timestamp(cutoff, field_name="knowledge_cutoff") else "invalid"
        from dean_os.draft.dean_os_agent_system_v7.dean_os.source_credibility import SourceCredibilityRegistry
        credibility = SourceCredibilityRegistry.from_domain_profile(domain_id).assess(
            payload,
            point_in_time_status=point_in_time_status,
        )
        return CatalogEvidenceRecord(
            evidence_id=str(payload.get("evidence_id") or payload.get("document_id") or f"evidence_{content_hash[:24]}"),
            domain_id=domain_id,
            source_type=_normalize_source_type(payload.get("source_type")),
            source_name=str(payload.get("source") or payload.get("source_name") or "unknown"),
            title=str(payload.get("title") or payload.get("key") or "untitled evidence"),
            locator=payload.get("uri") or payload.get("locator"),
            published_at=payload.get("published_at"),
            observed_at=observed_at,
            available_at=available_at,
            ingested_at=ingested_at,
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            sectors=_string_list(payload.get("sectors")),
            regions=_string_list(payload.get("regions")),
            entities=_string_list(payload.get("entities") or payload.get("tickers")),
            evidence_lanes=_string_list(payload.get("evidence_lanes") or payload.get("tags")),
            quality_score=float(payload.get("quality_score", credibility.credibility_score)),
            source_tier=credibility.source_tier,
            credibility_score=credibility.credibility_score,
            credibility_decision_use=credibility.decision_use,
            credibility_reasons=credibility.reasons,
            duplicate_cluster_id=payload.get("duplicate_cluster_id"),
            duplicate_status=str(payload.get("duplicate_status") or "unique"),
            duplicate_of=payload.get("duplicate_of"),
            point_in_time_status=point_in_time_status,
            quarantine_flags=_string_list(payload.get("quarantine_flags")),
            external_artifact_ref=payload.get("external_artifact_ref"),
        )

    def build_manifest(
        self,
        *,
        domain_id: str,
        as_of: str,
        knowledge_cutoff: str,
        started_at: str,
        completed_at: str,
        records: Iterable[CatalogEvidenceRecord],
        rejected_items: list[dict[str, Any]] | None = None,
        suppressed_items: list[dict[str, Any]] | None = None,
        warnings: list[str] | None = None,
    ) -> EvidenceAcquisitionRunManifest:
        record_list = list(records)
        source_counts: dict[str, int] = {}
        for item in record_list:
            source_counts[item.source_type] = source_counts.get(item.source_type, 0) + 1
        rejected = list(rejected_items or [])
        suppressed = list(suppressed_items or [])
        warning_list = sorted(set(warnings or []))
        status = "partial" if rejected or warning_list else "completed"
        payload = {
            "domain_id": domain_id,
            "as_of": as_of,
            "knowledge_cutoff": knowledge_cutoff,
            "started_at": started_at,
            "completed_at": completed_at,
            "source_counts": source_counts,
            "evidence_ids": [item.evidence_id for item in record_list],
            "rejected_items": rejected,
            "suppressed_items": suppressed,
            "warnings": warning_list,
            "status": status,
        }
        return EvidenceAcquisitionRunManifest(**payload, content_hash=_sha256(payload))


def _normalize_source_type(value: Any) -> str:
    allowed = {"news", "article", "book", "report", "filing", "transcript", "metric", "dataset", "document", "research_note", "pipeline_artifact", "unknown"}
    normalized = str(value or "unknown").strip().lower().replace(" ", "_")
    return normalized if normalized in allowed else "unknown"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]
