from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dean_os.analyst_core import ScenarioOutcomeGraph
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_grids import ContextGrid, IndicatorObservation, IndicatorStateGrid


WORLD_STATE_SCHEMA_VERSION = "dean_world_state_snapshot_v1"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class LeakageViolation(BaseModel):
    code: str
    path: str
    observed_timestamp: str | None = None
    knowledge_cutoff: str
    detail: str


class LeakageAudit(BaseModel):
    schema_version: str = "dean_world_state_leakage_audit_v1"
    status: str
    as_of: str
    knowledge_cutoff: str
    point_in_time_valid: bool
    violations: list[LeakageViolation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WorldStateIntegrity(BaseModel):
    point_in_time_valid: bool
    component_alignment_valid: bool
    atomic_components: list[str] = Field(default_factory=list)
    leakage_audit: LeakageAudit
    content_hash: str


class WorldStateSnapshot(BaseModel):
    """Immutable, point-in-time world-state snapshot.

    A snapshot atomically binds the qualitative Context Grid, quantitative
    Indicator State Grid, and optional Scenario Outcome Graph. It is append-only:
    a correction or later update creates another snapshot instead of mutating
    the historical record.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = WORLD_STATE_SCHEMA_VERSION
    snapshot_id: str
    domain_id: str
    as_of: str
    knowledge_cutoff: str
    created_at: str = Field(default_factory=_utc_now_iso)
    run_id: str | None = None
    parent_snapshot_id: str | None = None
    context_grid: ContextGrid
    indicator_state_grid: IndicatorStateGrid
    scenario_outcome_graph: ScenarioOutcomeGraph | None = None
    world_model_summary: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    integrity: WorldStateIntegrity
    authority_boundary: dict[str, bool] = Field(
        default_factory=lambda: {
            "review_only": True,
            "can_trade": False,
            "can_write_production_config": False,
            "can_promote_model": False,
            "can_write_learning_memory": False,
        }
    )

    @model_validator(mode="after")
    def _validate_identity(self) -> "WorldStateSnapshot":
        _parse_timestamp(self.as_of, field_name="as_of")
        _parse_timestamp(self.knowledge_cutoff, field_name="knowledge_cutoff")
        _parse_timestamp(self.created_at, field_name="created_at")
        if self.context_grid.domain_id != self.domain_id:
            raise ValueError("context_grid domain_id does not match snapshot domain_id")
        if self.indicator_state_grid.domain_id != self.domain_id:
            raise ValueError("indicator_state_grid domain_id does not match snapshot domain_id")
        if self.context_grid.as_of != self.as_of:
            raise ValueError("context_grid as_of must exactly match snapshot as_of")
        if self.indicator_state_grid.as_of != self.as_of:
            raise ValueError("indicator_state_grid as_of must exactly match snapshot as_of")
        if self.scenario_outcome_graph and self.scenario_outcome_graph.as_of != self.as_of:
            raise ValueError("scenario_outcome_graph as_of must exactly match snapshot as_of")
        if not self.integrity.point_in_time_valid:
            raise ValueError("invalid point-in-time snapshot cannot be persisted")
        expected_hash = _sha256(
            _world_state_content_payload(
                domain_id=self.domain_id,
                as_of=self.as_of,
                knowledge_cutoff=self.knowledge_cutoff,
                parent_snapshot_id=self.parent_snapshot_id,
                context_grid=self.context_grid,
                indicator_state_grid=self.indicator_state_grid,
                scenario_outcome_graph=self.scenario_outcome_graph,
                world_model_summary=self.world_model_summary,
                evidence_ids=self.evidence_ids,
                evidence_gaps=self.evidence_gaps,
            )
        )
        if self.integrity.content_hash != expected_hash:
            raise ValueError("world-state content hash does not match payload")
        if self.snapshot_id != f"world_state_{expected_hash[:24]}":
            raise ValueError("world-state snapshot_id does not match content hash")
        return self


class WorldStateSnapshotBuilder:
    """Build and validate one atomic world-state snapshot."""

    def build(
        self,
        *,
        domain_id: str,
        as_of: str,
        knowledge_cutoff: str,
        context_grid: ContextGrid | dict[str, Any],
        indicator_state_grid: IndicatorStateGrid | dict[str, Any],
        scenario_outcome_graph: ScenarioOutcomeGraph | dict[str, Any] | None = None,
        world_model_summary: dict[str, Any] | None = None,
        run_id: str | None = None,
        parent_snapshot_id: str | None = None,
        evidence_gaps: Iterable[str] | None = None,
    ) -> WorldStateSnapshot:
        as_of_dt = _parse_timestamp(as_of, field_name="as_of")
        cutoff_dt = _parse_timestamp(knowledge_cutoff, field_name="knowledge_cutoff")
        if cutoff_dt > as_of_dt:
            raise ValueError("knowledge_cutoff cannot be later than as_of")

        context_model = (
            context_grid
            if isinstance(context_grid, ContextGrid)
            else ContextGrid.model_validate(context_grid)
        )
        indicator_model = (
            indicator_state_grid
            if isinstance(indicator_state_grid, IndicatorStateGrid)
            else IndicatorStateGrid.model_validate(indicator_state_grid)
        )
        scenario_model = None
        if scenario_outcome_graph:
            scenario_model = (
                scenario_outcome_graph
                if isinstance(scenario_outcome_graph, ScenarioOutcomeGraph)
                else ScenarioOutcomeGraph.model_validate(scenario_outcome_graph)
            )

        leakage_audit = audit_world_state_point_in_time(
            as_of=as_of,
            knowledge_cutoff=knowledge_cutoff,
            context_grid=context_model,
            indicator_state_grid=indicator_model,
            scenario_outcome_graph=scenario_model,
        )
        if not leakage_audit.point_in_time_valid:
            summary = "; ".join(
                f"{item.code}@{item.path}" for item in leakage_audit.violations[:8]
            )
            raise ValueError(f"world-state point-in-time validation failed: {summary}")

        evidence_ids = _snapshot_evidence_ids(
            context_model,
            indicator_model,
            scenario_model,
        )
        gap_values = sorted(
            {
                str(value).strip()
                for value in (
                    list(context_model.evidence_gaps)
                    + list(scenario_model.evidence_gaps if scenario_model else [])
                    + list(evidence_gaps or [])
                )
                if str(value).strip()
            }
        )
        content_payload = _world_state_content_payload(
            domain_id=domain_id,
            as_of=as_of,
            knowledge_cutoff=knowledge_cutoff,
            parent_snapshot_id=parent_snapshot_id,
            context_grid=context_model,
            indicator_state_grid=indicator_model,
            scenario_outcome_graph=scenario_model,
            world_model_summary=dict(world_model_summary or {}),
            evidence_ids=evidence_ids,
            evidence_gaps=gap_values,
        )
        content_hash = _sha256(content_payload)
        snapshot_id = f"world_state_{content_hash[:24]}"
        integrity = WorldStateIntegrity(
            point_in_time_valid=True,
            component_alignment_valid=True,
            atomic_components=[
                "context_grid",
                "indicator_state_grid",
                *(["scenario_outcome_graph"] if scenario_model else []),
            ],
            leakage_audit=leakage_audit,
            content_hash=content_hash,
        )
        return WorldStateSnapshot(
            snapshot_id=snapshot_id,
            domain_id=domain_id,
            as_of=as_of,
            knowledge_cutoff=knowledge_cutoff,
            run_id=run_id,
            parent_snapshot_id=parent_snapshot_id,
            context_grid=context_model,
            indicator_state_grid=indicator_model,
            scenario_outcome_graph=scenario_model,
            world_model_summary=dict(world_model_summary or {}),
            evidence_ids=evidence_ids,
            evidence_gaps=gap_values,
            integrity=integrity,
        )


class WorldStateAppendResult(BaseModel):
    status: str
    snapshot_id: str
    content_hash: str
    backend: str


class WorldStateStoreProtocol(Protocol):
    def append(self, snapshot: WorldStateSnapshot) -> WorldStateAppendResult: ...

    def get(self, snapshot_id: str) -> WorldStateSnapshot | None: ...

    def get_as_of(
        self,
        *,
        domain_id: str,
        as_of: str,
        knowledge_cutoff: str | None = None,
    ) -> WorldStateSnapshot | None: ...

    def list_snapshots(
        self,
        *,
        domain_id: str,
        before_as_of: str | None = None,
        knowledge_cutoff: str | None = None,
        limit: int = 100,
    ) -> list[WorldStateSnapshot]: ...


class SQLiteWorldStateStore:
    """Append-only SQLite persistence for world-state snapshots.

    The complete snapshot payload is inserted in one transaction and one row,
    so Context Grid, Indicator State Grid and Scenario Graph cannot be observed
    as partially updated components.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        required_columns = {
            "snapshot_id",
            "domain_id",
            "as_of",
            "as_of_epoch",
            "knowledge_cutoff",
            "knowledge_cutoff_epoch",
            "created_at",
            "parent_snapshot_id",
            "content_hash",
            "payload_json",
        }
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS world_state_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    domain_id TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    as_of_epoch REAL NOT NULL,
                    knowledge_cutoff TEXT NOT NULL,
                    knowledge_cutoff_epoch REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    parent_snapshot_id TEXT,
                    content_hash TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(world_state_snapshots)"
                ).fetchall()
            }
            missing = sorted(required_columns - columns)
            if missing:
                raise RuntimeError(
                    "world_state_snapshots schema is incompatible; explicit "
                    f"migration required for columns: {missing}"
                )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_world_state_domain_as_of
                    ON world_state_snapshots(domain_id, as_of_epoch DESC);
                CREATE INDEX IF NOT EXISTS idx_world_state_knowledge_cutoff
                    ON world_state_snapshots(domain_id, knowledge_cutoff_epoch DESC);
                CREATE TRIGGER IF NOT EXISTS world_state_snapshots_no_update
                BEFORE UPDATE ON world_state_snapshots
                BEGIN
                    SELECT RAISE(ABORT, 'world_state_snapshots is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS world_state_snapshots_no_delete
                BEFORE DELETE ON world_state_snapshots
                BEGIN
                    SELECT RAISE(ABORT, 'world_state_snapshots is append-only');
                END;
                """
            )

    def append(self, snapshot: WorldStateSnapshot) -> WorldStateAppendResult:
        payload_json = _canonical_json(snapshot.model_dump(mode="json"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT snapshot_id, content_hash FROM world_state_snapshots "
                "WHERE snapshot_id = ? OR content_hash = ?",
                (snapshot.snapshot_id, snapshot.integrity.content_hash),
            ).fetchone()
            if existing:
                if (
                    existing["snapshot_id"] == snapshot.snapshot_id
                    and existing["content_hash"] == snapshot.integrity.content_hash
                ):
                    connection.commit()
                    return WorldStateAppendResult(
                        status="already_exists",
                        snapshot_id=snapshot.snapshot_id,
                        content_hash=snapshot.integrity.content_hash,
                        backend="sqlite",
                    )
                raise ValueError("world-state identity/content-hash conflict")
            connection.execute(
                """
                INSERT INTO world_state_snapshots (
                    snapshot_id, domain_id, as_of, as_of_epoch,
                    knowledge_cutoff, knowledge_cutoff_epoch, created_at,
                    parent_snapshot_id, content_hash, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.domain_id,
                    snapshot.as_of,
                    _parse_timestamp(snapshot.as_of, field_name="snapshot.as_of").timestamp(),
                    snapshot.knowledge_cutoff,
                    _parse_timestamp(
                        snapshot.knowledge_cutoff,
                        field_name="snapshot.knowledge_cutoff",
                    ).timestamp(),
                    snapshot.created_at,
                    snapshot.parent_snapshot_id,
                    snapshot.integrity.content_hash,
                    payload_json,
                ),
            )
            connection.commit()
        return WorldStateAppendResult(
            status="stored",
            snapshot_id=snapshot.snapshot_id,
            content_hash=snapshot.integrity.content_hash,
            backend="sqlite",
        )

    def get(self, snapshot_id: str) -> WorldStateSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM world_state_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        return _row_snapshot(row)

    def get_as_of(
        self,
        *,
        domain_id: str,
        as_of: str,
        knowledge_cutoff: str | None = None,
    ) -> WorldStateSnapshot | None:
        resolved_cutoff = knowledge_cutoff or as_of
        _parse_timestamp(as_of, field_name="as_of")
        _parse_timestamp(resolved_cutoff, field_name="knowledge_cutoff")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM world_state_snapshots
                WHERE domain_id = ?
                  AND as_of_epoch <= ?
                  AND knowledge_cutoff_epoch <= ?
                ORDER BY as_of_epoch DESC, created_at DESC
                LIMIT 1
                """,
                (
                    domain_id,
                    _parse_timestamp(as_of, field_name="as_of").timestamp(),
                    _parse_timestamp(
                        resolved_cutoff,
                        field_name="knowledge_cutoff",
                    ).timestamp(),
                ),
            ).fetchone()
        return _row_snapshot(row)

    def list_snapshots(
        self,
        *,
        domain_id: str,
        before_as_of: str | None = None,
        knowledge_cutoff: str | None = None,
        limit: int = 100,
    ) -> list[WorldStateSnapshot]:
        if limit < 1:
            return []
        clauses = ["domain_id = ?"]
        values: list[Any] = [domain_id]
        if before_as_of is not None:
            _parse_timestamp(before_as_of, field_name="before_as_of")
            clauses.append("as_of_epoch < ?")
            values.append(
                _parse_timestamp(before_as_of, field_name="before_as_of").timestamp()
            )
        if knowledge_cutoff is not None:
            _parse_timestamp(knowledge_cutoff, field_name="knowledge_cutoff")
            clauses.append("knowledge_cutoff_epoch <= ?")
            values.append(
                _parse_timestamp(
                    knowledge_cutoff,
                    field_name="knowledge_cutoff",
                ).timestamp()
            )
        values.append(limit)
        query = (
            "SELECT payload_json FROM world_state_snapshots WHERE "
            + " AND ".join(clauses)
            + " ORDER BY as_of_epoch DESC, created_at DESC LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [snapshot for row in rows if (snapshot := _row_snapshot(row)) is not None]


class InMemoryWorldStateStore:
    """Deterministic test/development store with the same append-only contract."""

    def __init__(self):
        self._snapshots: dict[str, WorldStateSnapshot] = {}

    def append(self, snapshot: WorldStateSnapshot) -> WorldStateAppendResult:
        existing = self._snapshots.get(snapshot.snapshot_id)
        if existing:
            if existing.integrity.content_hash != snapshot.integrity.content_hash:
                raise ValueError("world-state identity/content-hash conflict")
            return WorldStateAppendResult(
                status="already_exists",
                snapshot_id=snapshot.snapshot_id,
                content_hash=snapshot.integrity.content_hash,
                backend="memory",
            )
        if any(
            item.integrity.content_hash == snapshot.integrity.content_hash
            for item in self._snapshots.values()
        ):
            raise ValueError("duplicate content hash with a different snapshot identity")
        self._snapshots[snapshot.snapshot_id] = snapshot
        return WorldStateAppendResult(
            status="stored",
            snapshot_id=snapshot.snapshot_id,
            content_hash=snapshot.integrity.content_hash,
            backend="memory",
        )

    def get(self, snapshot_id: str) -> WorldStateSnapshot | None:
        return self._snapshots.get(snapshot_id)

    def get_as_of(
        self,
        *,
        domain_id: str,
        as_of: str,
        knowledge_cutoff: str | None = None,
    ) -> WorldStateSnapshot | None:
        resolved_cutoff = knowledge_cutoff or as_of
        as_of_dt = _parse_timestamp(as_of, field_name="as_of")
        cutoff_dt = _parse_timestamp(resolved_cutoff, field_name="knowledge_cutoff")
        candidates = [
            item
            for item in self._snapshots.values()
            if item.domain_id == domain_id
            and _parse_timestamp(item.as_of, field_name="snapshot.as_of") <= as_of_dt
            and _parse_timestamp(item.knowledge_cutoff, field_name="snapshot.knowledge_cutoff") <= cutoff_dt
        ]
        return max(candidates, key=lambda item: (item.as_of, item.created_at), default=None)

    def list_snapshots(
        self,
        *,
        domain_id: str,
        before_as_of: str | None = None,
        knowledge_cutoff: str | None = None,
        limit: int = 100,
    ) -> list[WorldStateSnapshot]:
        before_dt = (
            _parse_timestamp(before_as_of, field_name="before_as_of")
            if before_as_of is not None
            else None
        )
        cutoff_dt = (
            _parse_timestamp(knowledge_cutoff, field_name="knowledge_cutoff")
            if knowledge_cutoff is not None
            else None
        )
        candidates = []
        for item in self._snapshots.values():
            if item.domain_id != domain_id:
                continue
            if before_dt and _parse_timestamp(item.as_of, field_name="snapshot.as_of") >= before_dt:
                continue
            if cutoff_dt and _parse_timestamp(
                item.knowledge_cutoff,
                field_name="snapshot.knowledge_cutoff",
            ) > cutoff_dt:
                continue
            candidates.append(item)
        candidates.sort(key=lambda item: (item.as_of, item.created_at), reverse=True)
        return candidates[: max(0, limit)]


class HistoricalAnalogMatch(BaseModel):
    snapshot_id: str
    domain_id: str
    as_of: str
    knowledge_cutoff: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    component_scores: dict[str, float] = Field(default_factory=dict)
    matched_context_dimensions: list[str] = Field(default_factory=list)
    common_indicator_keys: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    retrieval_method: str = "deterministic_world_state_similarity_v1"
    review_only: bool = True
    eligible_for_learning: bool = False
    false_analogy_risk: str = "requires_human_or_replay_validation"


class HistoricalWorldStateRetriever:
    """Deterministic seed retrieval over prior world-state snapshots.

    This is intentionally a transparent baseline, not the final learned KNN or
    cluster model. It gives later training code a stable contract and auditable
    feature comparison while preventing future-state candidates.
    """

    def __init__(self, store: WorldStateStoreProtocol):
        self.store = store

    def find_analogs(
        self,
        target: WorldStateSnapshot,
        *,
        limit: int = 5,
        min_similarity: float = 0.0,
        candidate_limit: int = 500,
    ) -> list[HistoricalAnalogMatch]:
        candidates = self.store.list_snapshots(
            domain_id=target.domain_id,
            before_as_of=target.as_of,
            knowledge_cutoff=target.knowledge_cutoff,
            limit=candidate_limit,
        )
        matches: list[HistoricalAnalogMatch] = []
        for candidate in candidates:
            match = compare_world_states(target, candidate)
            if match.similarity_score >= min_similarity:
                matches.append(match)
        matches.sort(
            key=lambda item: (item.similarity_score, item.as_of),
            reverse=True,
        )
        return matches[: max(0, limit)]


def audit_world_state_point_in_time(
    *,
    as_of: str,
    knowledge_cutoff: str,
    context_grid: ContextGrid,
    indicator_state_grid: IndicatorStateGrid,
    scenario_outcome_graph: ScenarioOutcomeGraph | None,
) -> LeakageAudit:
    as_of_dt = _parse_timestamp(as_of, field_name="as_of")
    cutoff_dt = _parse_timestamp(knowledge_cutoff, field_name="knowledge_cutoff")
    violations: list[LeakageViolation] = []
    warnings: list[str] = []

    if cutoff_dt > as_of_dt:
        violations.append(
            LeakageViolation(
                code="knowledge_cutoff_after_as_of",
                path="knowledge_cutoff",
                observed_timestamp=knowledge_cutoff,
                knowledge_cutoff=knowledge_cutoff,
                detail="Knowledge cutoff cannot be later than the decision as_of time.",
            )
        )

    for node_index, node in enumerate(context_grid.nodes):
        for dimension_name, dimension in node.dimensions.items():
            path = f"context_grid.nodes[{node_index}].dimensions.{dimension_name}.as_of"
            if dimension.as_of is None:
                if dimension.evidence_ids:
                    warnings.append(f"{path}: evidence-backed dimension has no timestamp")
                continue
            _append_future_violation(
                violations,
                timestamp=dimension.as_of,
                cutoff_dt=as_of_dt,
                cutoff_text=as_of,
                path=path,
                code="context_dimension_after_snapshot_as_of",
            )

    for index, observation in enumerate(indicator_state_grid.observations):
        path = f"indicator_state_grid.observations[{index}].available_at"
        if observation.available_at is None:
            if observation.evidence_status == "point_in_time":
                violations.append(
                    LeakageViolation(
                        code="point_in_time_indicator_missing_available_at",
                        path=path,
                        observed_timestamp=None,
                        knowledge_cutoff=knowledge_cutoff,
                        detail="Point-in-time indicators require available_at.",
                    )
                )
            else:
                warnings.append(f"{path}: review-only indicator has no available_at")
        else:
            _append_future_violation(
                violations,
                timestamp=observation.available_at,
                cutoff_dt=cutoff_dt,
                cutoff_text=knowledge_cutoff,
                path=path,
                code="future_indicator_observation",
            )

    if scenario_outcome_graph is not None:
        _append_future_violation(
            violations,
            timestamp=scenario_outcome_graph.as_of,
            cutoff_dt=as_of_dt,
            cutoff_text=as_of,
            path="scenario_outcome_graph.as_of",
            code="scenario_graph_after_snapshot_as_of",
        )
        for index, node in enumerate(scenario_outcome_graph.nodes):
            _append_future_violation(
                violations,
                timestamp=node.as_of,
                cutoff_dt=as_of_dt,
                cutoff_text=as_of,
                path=f"scenario_outcome_graph.nodes[{index}].as_of",
                code="scenario_node_after_snapshot_as_of",
            )

    return LeakageAudit(
        status="valid" if not violations else "invalid",
        as_of=as_of,
        knowledge_cutoff=knowledge_cutoff,
        point_in_time_valid=not violations,
        violations=violations,
        warnings=sorted(set(warnings)),
    )


def compare_world_states(
    target: WorldStateSnapshot,
    candidate: WorldStateSnapshot,
) -> HistoricalAnalogMatch:
    if target.domain_id != candidate.domain_id:
        raise ValueError("world-state analog comparison requires the same domain_id")
    if _parse_timestamp(candidate.as_of, field_name="candidate.as_of") >= _parse_timestamp(
        target.as_of,
        field_name="target.as_of",
    ):
        raise ValueError("historical analog candidate must precede target as_of")
    if _parse_timestamp(
        candidate.knowledge_cutoff,
        field_name="candidate.knowledge_cutoff",
    ) > _parse_timestamp(target.knowledge_cutoff, field_name="target.knowledge_cutoff"):
        raise ValueError("historical analog candidate exceeds target knowledge_cutoff")

    context_score, matched_dimensions = _context_similarity(
        target.context_grid,
        candidate.context_grid,
    )
    indicator_score, common_indicators = _indicator_similarity(
        target.indicator_state_grid,
        candidate.indicator_state_grid,
    )
    scenario_score = _scenario_shape_similarity(
        target.scenario_outcome_graph,
        candidate.scenario_outcome_graph,
    )
    available_components: list[tuple[str, float, float]] = [
        ("context", context_score, 0.65),
    ]
    if common_indicators:
        available_components.append(("indicators", indicator_score, 0.30))
    if target.scenario_outcome_graph and candidate.scenario_outcome_graph:
        available_components.append(("scenario_shape", scenario_score, 0.05))
    weight_sum = sum(weight for _, _, weight in available_components)
    similarity = sum(score * weight for _, score, weight in available_components) / weight_sum
    warnings: list[str] = []
    if not common_indicators:
        warnings.append("no_common_indicator_keys")
    return HistoricalAnalogMatch(
        snapshot_id=candidate.snapshot_id,
        domain_id=candidate.domain_id,
        as_of=candidate.as_of,
        knowledge_cutoff=candidate.knowledge_cutoff,
        similarity_score=max(0.0, min(1.0, similarity)),
        component_scores={name: score for name, score, _ in available_components},
        matched_context_dimensions=matched_dimensions,
        common_indicator_keys=common_indicators,
        warnings=warnings,
    )


def _append_future_violation(
    violations: list[LeakageViolation],
    *,
    timestamp: str,
    cutoff_dt: datetime,
    cutoff_text: str,
    path: str,
    code: str,
) -> None:
    try:
        observed = _parse_timestamp(timestamp, field_name=path)
    except ValueError as exc:
        violations.append(
            LeakageViolation(
                code=f"invalid_timestamp:{code}",
                path=path,
                observed_timestamp=timestamp,
                knowledge_cutoff=cutoff_text,
                detail=str(exc),
            )
        )
        return
    if observed > cutoff_dt:
        violations.append(
            LeakageViolation(
                code=code,
                path=path,
                observed_timestamp=timestamp,
                knowledge_cutoff=cutoff_text,
                detail="Timestamp exceeds the admissible temporal boundary for this field.",
            )
        )


def _snapshot_evidence_ids(
    context_grid: ContextGrid,
    indicator_state_grid: IndicatorStateGrid,
    scenario_outcome_graph: ScenarioOutcomeGraph | None,
) -> list[str]:
    values: set[str] = set()
    for node in context_grid.nodes:
        values.update(node.evidence_ids)
        for dimension in node.dimensions.values():
            values.update(dimension.evidence_ids)
    for edge in context_grid.edges:
        values.update(edge.evidence_ids)
    for observation in indicator_state_grid.observations:
        values.add(observation.indicator_id)
        provenance_id = observation.provenance.get("evidence_id")
        if provenance_id:
            values.add(str(provenance_id))
    if scenario_outcome_graph:
        for node in scenario_outcome_graph.nodes:
            values.update(node.evidence_ids)
        for edge in scenario_outcome_graph.edges:
            values.update(edge.evidence_ids)
    return sorted(value for value in values if value)


def _world_state_content_payload(
    *,
    domain_id: str,
    as_of: str,
    knowledge_cutoff: str,
    parent_snapshot_id: str | None,
    context_grid: ContextGrid,
    indicator_state_grid: IndicatorStateGrid,
    scenario_outcome_graph: ScenarioOutcomeGraph | None,
    world_model_summary: dict[str, Any],
    evidence_ids: list[str],
    evidence_gaps: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": WORLD_STATE_SCHEMA_VERSION,
        "domain_id": domain_id,
        "as_of": as_of,
        "knowledge_cutoff": knowledge_cutoff,
        "parent_snapshot_id": parent_snapshot_id,
        "context_grid": context_grid.model_dump(mode="json"),
        "indicator_state_grid": indicator_state_grid.model_dump(mode="json"),
        "scenario_outcome_graph": (
            scenario_outcome_graph.model_dump(mode="json")
            if scenario_outcome_graph
            else None
        ),
        "world_model_summary": dict(world_model_summary),
        "evidence_ids": list(evidence_ids),
        "evidence_gaps": list(evidence_gaps),
    }


def _row_snapshot(row: sqlite3.Row | None) -> WorldStateSnapshot | None:
    if row is None:
        return None
    return WorldStateSnapshot.model_validate_json(row["payload_json"])


def _context_features(grid: ContextGrid) -> dict[str, str]:
    features: dict[str, str] = {}
    for node in grid.nodes:
        for dimension_name, dimension in node.dimensions.items():
            normalized_state = dimension.state.strip().lower()
            if normalized_state in {"", "unknown", "missing", "not_available"}:
                continue
            key = f"{node.level}|{node.node_id}|{dimension_name}"
            features[key] = f"{normalized_state}|{dimension.direction}"
    return features


def _context_similarity(
    left: ContextGrid,
    right: ContextGrid,
) -> tuple[float, list[str]]:
    left_features = _context_features(left)
    right_features = _context_features(right)
    keys = sorted(set(left_features) | set(right_features))
    if not keys:
        return 0.0, []
    matched = [
        key
        for key in keys
        if key in left_features
        and key in right_features
        and left_features[key] == right_features[key]
    ]
    return len(matched) / len(keys), matched


def _indicator_key(item: IndicatorObservation) -> str:
    return "|".join(
        [
            item.family.strip().lower(),
            item.name.strip().lower(),
            str(item.unit or "").strip().lower(),
        ]
    )


def _indicator_similarity(
    left: IndicatorStateGrid,
    right: IndicatorStateGrid,
) -> tuple[float, list[str]]:
    left_values = {_indicator_key(item): item.value for item in left.observations}
    right_values = {_indicator_key(item): item.value for item in right.observations}
    common = sorted(set(left_values) & set(right_values))
    if not common:
        return 0.0, []
    scores = [_value_similarity(left_values[key], right_values[key]) for key in common]
    return sum(scores) / len(scores), common


def _value_similarity(left: Any, right: Any) -> float:
    if isinstance(left, bool) or isinstance(right, bool):
        return 1.0 if left == right else 0.0
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return 1.0 if str(left).strip().lower() == str(right).strip().lower() else 0.0
    if not math.isfinite(left_number) or not math.isfinite(right_number):
        return 0.0
    scale = max(abs(left_number), abs(right_number), 1e-9)
    return 1.0 / (1.0 + abs(left_number - right_number) / scale)


def _scenario_shape_similarity(
    left: ScenarioOutcomeGraph | None,
    right: ScenarioOutcomeGraph | None,
) -> float:
    if left is None or right is None:
        return 0.0
    left_labels = {
        f"{node.node_type}|{node.label.strip().lower()}" for node in left.nodes
    }
    right_labels = {
        f"{node.node_type}|{node.label.strip().lower()}" for node in right.nodes
    }
    union = left_labels | right_labels
    if not union:
        return 1.0
    return len(left_labels & right_labels) / len(union)


__all__ = [
    "HistoricalAnalogMatch",
    "HistoricalWorldStateRetriever",
    "InMemoryWorldStateStore",
    "LeakageAudit",
    "LeakageViolation",
    "SQLiteWorldStateStore",
    "WORLD_STATE_SCHEMA_VERSION",
    "WorldStateAppendResult",
    "WorldStateIntegrity",
    "WorldStateSnapshot",
    "WorldStateSnapshotBuilder",
    "WorldStateStoreProtocol",
    "audit_world_state_point_in_time",
    "compare_world_states",
]
