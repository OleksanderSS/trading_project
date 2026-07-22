from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


DAILY_RUN_RECORD_SCHEMA_VERSION = "dean_daily_run_record_v1"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class DailyRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = DAILY_RUN_RECORD_SCHEMA_VERSION
    daily_run_id: str
    domain_id: str
    as_of: str
    knowledge_cutoff: str
    status: str
    evidence_acquisition_run_id: str
    evidence_manifest_hash: str
    system_run_id: str
    world_state_snapshot_id: str | None = None
    world_state_content_hash: str | None = None
    briefing_id: str
    briefing_hash: str
    evidence_dedup_hash: str | None = None
    evidence_gap_plan_hash: str | None = None
    review_inbox_item_ids: list[str] = Field(default_factory=list)
    review_inbox_item_hashes: list[str] = Field(default_factory=list)
    rendered_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    replay_task_ids: list[str] = Field(default_factory=list)
    due_replay_task_ids: list[str] = Field(default_factory=list)
    authority_boundary: dict[str, bool]
    content_hash: str

    @model_validator(mode="after")
    def _validate(self) -> "DailyRunRecord":
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if _sha256(payload) != self.content_hash:
            raise ValueError("daily run record content hash mismatch")
        if self.authority_boundary.get("can_write_learning_memory") is not False:
            raise ValueError("daily run record must not authorize learning-memory writes")
        if self.authority_boundary.get("can_trade") is not False:
            raise ValueError("daily run record must not authorize trading")
        return self


class DailyRunAppendResult(BaseModel):
    status: Literal["stored", "already_exists"]
    daily_run_id: str
    content_hash: str
    backend: str


class DailyRunStoreProtocol(Protocol):
    def append(self, record: DailyRunRecord) -> DailyRunAppendResult: ...
    def get(self, daily_run_id: str) -> DailyRunRecord | None: ...


class SQLiteDailyRunStore:
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
                CREATE TABLE IF NOT EXISTS daily_run_records (
                    daily_run_id TEXT PRIMARY KEY,
                    domain_id TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_daily_run_domain_asof
                    ON daily_run_records(domain_id, as_of DESC);
                CREATE TRIGGER IF NOT EXISTS daily_run_records_no_update
                    BEFORE UPDATE ON daily_run_records
                    BEGIN SELECT RAISE(ABORT, 'daily_run_records is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS daily_run_records_no_delete
                    BEFORE DELETE ON daily_run_records
                    BEGIN SELECT RAISE(ABORT, 'daily_run_records is append-only'); END;
            """)

    def append(self, record: DailyRunRecord) -> DailyRunAppendResult:
        payload = record.model_dump(mode="json")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT daily_run_id, content_hash, payload_json FROM daily_run_records WHERE daily_run_id = ? OR content_hash = ?",
                (record.daily_run_id, record.content_hash),
            ).fetchone()
            if existing:
                if json.loads(existing["payload_json"]) != payload:
                    raise ValueError("daily-run identity/content conflict")
                return DailyRunAppendResult(status="already_exists", daily_run_id=existing["daily_run_id"], content_hash=existing["content_hash"], backend="sqlite")
            connection.execute(
                "INSERT INTO daily_run_records(daily_run_id, domain_id, as_of, content_hash, payload_json) VALUES(?,?,?,?,?)",
                (record.daily_run_id, record.domain_id, record.as_of, record.content_hash, _canonical_json(payload)),
            )
        return DailyRunAppendResult(status="stored", daily_run_id=record.daily_run_id, content_hash=record.content_hash, backend="sqlite")

    def get(self, daily_run_id: str) -> DailyRunRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload_json FROM daily_run_records WHERE daily_run_id = ?", (daily_run_id,)).fetchone()
        return DailyRunRecord.model_validate(json.loads(row["payload_json"])) if row else None


class DailyRunRecordBuilder:
    def build(self, daily_result: Any) -> DailyRunRecord:
        result = daily_result.model_dump(mode="json") if hasattr(daily_result, "model_dump") else dict(daily_result)
        system = result.get("system_result", {}) or {}
        snapshot = system.get("world_state_snapshot", {}) or {}
        evidence_manifest = result.get("evidence_manifest", {}) or {}
        briefing = result.get("briefing", {}) or {}
        authority = dict(result.get("safety", {}) or {})
        payload = {
            "schema_version": DAILY_RUN_RECORD_SCHEMA_VERSION,
            "daily_run_id": str(result.get("daily_run_id")),
            "domain_id": str(result.get("domain_id")),
            "as_of": str(result.get("as_of")),
            "knowledge_cutoff": str(result.get("knowledge_cutoff")),
            "status": str(result.get("status")),
            "evidence_acquisition_run_id": str(evidence_manifest.get("acquisition_run_id")),
            "evidence_manifest_hash": str(evidence_manifest.get("content_hash")),
            "system_run_id": str(system.get("run_id")),
            "world_state_snapshot_id": snapshot.get("snapshot_id"),
            "world_state_content_hash": (snapshot.get("integrity", {}) or {}).get("content_hash"),
            "briefing_id": str(briefing.get("briefing_id")),
            "briefing_hash": _sha256(briefing),
            "evidence_dedup_hash": _sha256(result.get("evidence_dedup")) if result.get("evidence_dedup") else None,
            "evidence_gap_plan_hash": _sha256(result.get("evidence_gap_plan")) if result.get("evidence_gap_plan") else None,
            "review_inbox_item_ids": [str(item.get("item_id")) for item in result.get("review_inbox_items", []) or []],
            "review_inbox_item_hashes": [str(item.get("content_hash")) for item in result.get("review_inbox_items", []) or []],
            "rendered_artifact_hashes": _artifact_hashes(result.get("rendered_artifacts", {}) or {}),
            "replay_task_ids": [str(item.get("task_id")) for item in result.get("replay_schedule", []) or []],
            "due_replay_task_ids": [str(item.get("task_id")) for item in result.get("due_replay_tasks", []) or []],
            "authority_boundary": authority,
        }
        return DailyRunRecord(**payload, content_hash=_sha256(payload))


def _artifact_hashes(paths: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, value in paths.items():
        path = Path(str(value))
        if path.is_file():
            hashes[str(name)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes
