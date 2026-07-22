from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dean_os.analyst_core import OUTCOME_HORIZONS
from dean_os.draft.dean_os_agent_system_v7.dean_os.world_state_store import HistoricalAnalogMatch, WorldStateSnapshot


OUTCOME_SCHEMA_VERSION = "dean_world_state_outcome_v1"
OUTCOME_REVIEW_SCHEMA_VERSION = "dean_world_state_outcome_review_v1"
CALIBRATION_PROPOSAL_SCHEMA_VERSION = "dean_outcome_calibration_proposal_v1"
LEARNING_PROMOTION_SCHEMA_VERSION = "dean_learning_promotion_packet_v1"


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


def _normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().lower().replace("_", " ").split())
    return normalized or None


class ScenarioResolutionStatus(str, Enum):
    REALIZED = "realized"
    REJECTED = "rejected"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


class HypothesisResolutionStatus(str, Enum):
    CONFIRMED = "confirmed"
    WEAKENED = "weakened"
    FALSIFIED = "falsified"
    UNRESOLVED = "unresolved"


class ReviewDecisionStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class OutcomeEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    source: str
    observed_at: str
    available_at: str
    metric_name: str
    value: Any
    unit: str | None = None
    scope: str = "domain"
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_evidence(self) -> "OutcomeEvidence":
        observed = _parse_timestamp(self.observed_at, field_name="observed_at")
        available = _parse_timestamp(self.available_at, field_name="available_at")
        if available < observed:
            raise ValueError("available_at cannot be earlier than observed_at")
        if not self.evidence_id.strip():
            raise ValueError("evidence_id cannot be empty")
        if not self.source.strip():
            raise ValueError("source cannot be empty")
        if not self.metric_name.strip():
            raise ValueError("metric_name cannot be empty")
        return self


class ScenarioResolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_node_id: str
    status: ScenarioResolutionStatus
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    outcome_class: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def _normalize(self) -> "ScenarioResolution":
        if not self.scenario_node_id.strip():
            raise ValueError("scenario_node_id cannot be empty")
        if self.status != ScenarioResolutionStatus.UNRESOLVED and not (
            self.supporting_evidence_ids or self.contradicting_evidence_ids
        ):
            raise ValueError(
                "resolved scenario nodes require supporting or contradicting evidence"
            )
        return self


class HypothesisResolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis_id: str
    status: HypothesisResolutionStatus
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def _normalize(self) -> "HypothesisResolution":
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id cannot be empty")
        if self.status != HypothesisResolutionStatus.UNRESOLVED and not (
            self.supporting_evidence_ids or self.contradicting_evidence_ids
        ):
            raise ValueError(
                "resolved hypotheses require supporting or contradicting evidence"
            )
        return self


class ScenarioProbabilityScore(BaseModel):
    scored: bool = False
    score_status: str = "unresolved"
    realized_scenario_node_id: str | None = None
    realized_outcome_class: str | None = None
    top_probability_scenario_node_id: str | None = None
    top_scenario_correct: bool | None = None
    winner_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    brier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    logarithmic_loss: float | None = Field(default=None, ge=0.0)
    unresolved_scenario_node_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OutcomeIntegrity(BaseModel):
    point_in_time_valid: bool
    component_alignment_valid: bool
    content_hash: str
    evidence_hashes: dict[str, str] = Field(default_factory=dict)


class WorldStateOutcomeSnapshot(BaseModel):
    """Immutable fixed-horizon evaluation of one World State.

    This is an outcome observation and scorecard, not a learning-memory write.
    A review decision is stored separately so the outcome itself remains
    immutable and auditable.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = OUTCOME_SCHEMA_VERSION
    outcome_snapshot_id: str
    world_state_snapshot_id: str
    domain_id: str
    world_state_as_of: str
    world_state_knowledge_cutoff: str
    scenario_graph_id: str | None = None
    horizon_days: int
    due_at: str
    evaluation_as_of: str
    created_at: str = Field(default_factory=_utc_now_iso)
    status: str
    evidence: list[OutcomeEvidence] = Field(default_factory=list)
    scenario_resolutions: list[ScenarioResolution] = Field(default_factory=list)
    hypothesis_resolutions: list[HypothesisResolution] = Field(default_factory=list)
    probability_score: ScenarioProbabilityScore = Field(
        default_factory=ScenarioProbabilityScore
    )
    evidence_gaps: list[str] = Field(default_factory=list)
    integrity: OutcomeIntegrity
    authority_boundary: dict[str, bool] = Field(
        default_factory=lambda: {
            "review_only": True,
            "can_trade": False,
            "can_write_production_config": False,
            "can_promote_model": False,
            "can_write_learning_memory": False,
            "human_review_required": True,
        }
    )

    @model_validator(mode="after")
    def _validate_identity(self) -> "WorldStateOutcomeSnapshot":
        world_as_of = _parse_timestamp(
            self.world_state_as_of, field_name="world_state_as_of"
        )
        cutoff = _parse_timestamp(
            self.world_state_knowledge_cutoff,
            field_name="world_state_knowledge_cutoff",
        )
        due = _parse_timestamp(self.due_at, field_name="due_at")
        evaluated = _parse_timestamp(
            self.evaluation_as_of, field_name="evaluation_as_of"
        )
        _parse_timestamp(self.created_at, field_name="created_at")
        if cutoff > world_as_of:
            raise ValueError("world-state knowledge cutoff cannot exceed world-state as_of")
        if self.horizon_days not in OUTCOME_HORIZONS:
            raise ValueError(
                f"horizon_days must be one of the fixed horizons {OUTCOME_HORIZONS}"
            )
        expected_due = world_as_of + timedelta(days=self.horizon_days)
        if abs((due - expected_due).total_seconds()) > 1e-6:
            raise ValueError("due_at does not match world_state_as_of + horizon_days")
        if evaluated < due:
            raise ValueError("an outcome cannot be evaluated before its fixed horizon is due")
        for item in self.evidence:
            if _parse_timestamp(item.available_at, field_name="evidence.available_at") > evaluated:
                raise ValueError("outcome evidence available after evaluation_as_of is future leakage")
            if _parse_timestamp(item.observed_at, field_name="evidence.observed_at") > evaluated:
                raise ValueError("outcome evidence observed after evaluation_as_of is future leakage")
        expected_hash = _sha256(
            _outcome_content_payload(
                world_state_snapshot_id=self.world_state_snapshot_id,
                domain_id=self.domain_id,
                world_state_as_of=self.world_state_as_of,
                world_state_knowledge_cutoff=self.world_state_knowledge_cutoff,
                scenario_graph_id=self.scenario_graph_id,
                horizon_days=self.horizon_days,
                due_at=self.due_at,
                evaluation_as_of=self.evaluation_as_of,
                status=self.status,
                evidence=self.evidence,
                scenario_resolutions=self.scenario_resolutions,
                hypothesis_resolutions=self.hypothesis_resolutions,
                probability_score=self.probability_score,
                evidence_gaps=self.evidence_gaps,
            )
        )
        if self.integrity.content_hash != expected_hash:
            raise ValueError("outcome content hash does not match payload")
        if self.outcome_snapshot_id != f"world_outcome_{expected_hash[:24]}":
            raise ValueError("outcome_snapshot_id does not match content hash")
        return self


class OutcomeSnapshotBuilder:
    def build(
        self,
        *,
        world_state: WorldStateSnapshot,
        horizon_days: int,
        evaluation_as_of: str,
        evidence: Iterable[OutcomeEvidence | dict[str, Any]],
        scenario_resolutions: Iterable[ScenarioResolution | dict[str, Any]] = (),
        hypothesis_resolutions: Iterable[HypothesisResolution | dict[str, Any]] = (),
        evidence_gaps: Iterable[str] = (),
    ) -> WorldStateOutcomeSnapshot:
        if horizon_days not in OUTCOME_HORIZONS:
            raise ValueError(
                f"horizon_days must be one of the fixed horizons {OUTCOME_HORIZONS}"
            )
        world_as_of_dt = _parse_timestamp(world_state.as_of, field_name="world_state.as_of")
        evaluation_dt = _parse_timestamp(
            evaluation_as_of, field_name="evaluation_as_of"
        )
        due_dt = world_as_of_dt + timedelta(days=horizon_days)
        if evaluation_dt < due_dt:
            raise ValueError("an outcome cannot be evaluated before its fixed horizon is due")

        evidence_models = [
            item if isinstance(item, OutcomeEvidence) else OutcomeEvidence.model_validate(item)
            for item in evidence
        ]
        evidence_ids = {item.evidence_id for item in evidence_models}
        if len(evidence_ids) != len(evidence_models):
            raise ValueError("duplicate outcome evidence_id values are not allowed")
        for item in evidence_models:
            if _parse_timestamp(item.available_at, field_name="evidence.available_at") > evaluation_dt:
                raise ValueError("outcome evidence available after evaluation_as_of is future leakage")
            if _parse_timestamp(item.observed_at, field_name="evidence.observed_at") > evaluation_dt:
                raise ValueError("outcome evidence observed after evaluation_as_of is future leakage")

        scenario_models = [
            item
            if isinstance(item, ScenarioResolution)
            else ScenarioResolution.model_validate(item)
            for item in scenario_resolutions
        ]
        hypothesis_models = [
            item
            if isinstance(item, HypothesisResolution)
            else HypothesisResolution.model_validate(item)
            for item in hypothesis_resolutions
        ]
        self._validate_resolution_evidence(
            scenario_models,
            hypothesis_models,
            evidence_ids,
        )
        self._validate_scenario_ids(world_state, scenario_models)

        score = _score_scenarios(world_state, scenario_models)
        gaps = sorted({str(item).strip() for item in evidence_gaps if str(item).strip()})
        if not evidence_models:
            gaps.append("outcome_evidence_missing")
        if not score.scored:
            gaps.append("scenario_probability_score_unresolved")
        gaps = sorted(set(gaps))
        if not evidence_models:
            status = "insufficient_evidence"
        elif score.scored:
            status = "evaluated_pending_review"
        else:
            status = "partial_pending_review"

        due_at = due_dt.isoformat()
        evidence_hashes = {
            item.evidence_id: _sha256(item.model_dump(mode="json"))
            for item in evidence_models
        }
        content_payload = _outcome_content_payload(
            world_state_snapshot_id=world_state.snapshot_id,
            domain_id=world_state.domain_id,
            world_state_as_of=world_state.as_of,
            world_state_knowledge_cutoff=world_state.knowledge_cutoff,
            scenario_graph_id=(
                world_state.scenario_outcome_graph.scenario_graph_id
                if world_state.scenario_outcome_graph
                else None
            ),
            horizon_days=horizon_days,
            due_at=due_at,
            evaluation_as_of=evaluation_as_of,
            status=status,
            evidence=evidence_models,
            scenario_resolutions=scenario_models,
            hypothesis_resolutions=hypothesis_models,
            probability_score=score,
            evidence_gaps=gaps,
        )
        content_hash = _sha256(content_payload)
        return WorldStateOutcomeSnapshot(
            outcome_snapshot_id=f"world_outcome_{content_hash[:24]}",
            world_state_snapshot_id=world_state.snapshot_id,
            domain_id=world_state.domain_id,
            world_state_as_of=world_state.as_of,
            world_state_knowledge_cutoff=world_state.knowledge_cutoff,
            scenario_graph_id=(
                world_state.scenario_outcome_graph.scenario_graph_id
                if world_state.scenario_outcome_graph
                else None
            ),
            horizon_days=horizon_days,
            due_at=due_at,
            evaluation_as_of=evaluation_as_of,
            status=status,
            evidence=evidence_models,
            scenario_resolutions=scenario_models,
            hypothesis_resolutions=hypothesis_models,
            probability_score=score,
            evidence_gaps=gaps,
            integrity=OutcomeIntegrity(
                point_in_time_valid=True,
                component_alignment_valid=True,
                content_hash=content_hash,
                evidence_hashes=evidence_hashes,
            ),
        )

    @staticmethod
    def _validate_resolution_evidence(
        scenario_resolutions: list[ScenarioResolution],
        hypothesis_resolutions: list[HypothesisResolution],
        evidence_ids: set[str],
    ) -> None:
        for path, items in (
            ("scenario_resolutions", scenario_resolutions),
            ("hypothesis_resolutions", hypothesis_resolutions),
        ):
            seen: set[str] = set()
            for item in items:
                item_id = getattr(item, "scenario_node_id", None) or getattr(
                    item, "hypothesis_id"
                )
                if item_id in seen:
                    raise ValueError(f"duplicate {path} id: {item_id}")
                seen.add(item_id)
                referenced = set(item.supporting_evidence_ids) | set(
                    item.contradicting_evidence_ids
                )
                missing = sorted(referenced - evidence_ids)
                if missing:
                    raise ValueError(
                        f"{path} references unknown evidence IDs: {missing}"
                    )

    @staticmethod
    def _validate_scenario_ids(
        world_state: WorldStateSnapshot,
        resolutions: list[ScenarioResolution],
    ) -> None:
        graph = world_state.scenario_outcome_graph
        if not resolutions:
            return
        if graph is None:
            raise ValueError("scenario resolutions require a scenario graph")
        scenario_ids = {
            node.node_id for node in graph.nodes if node.node_type == "scenario"
        }
        unknown = sorted(
            {item.scenario_node_id for item in resolutions} - scenario_ids
        )
        if unknown:
            raise ValueError(f"scenario resolutions reference unknown nodes: {unknown}")


class OutcomeAppendResult(BaseModel):
    status: str
    outcome_snapshot_id: str
    content_hash: str
    backend: str


class OutcomeReviewDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = OUTCOME_REVIEW_SCHEMA_VERSION
    review_decision_id: str
    outcome_snapshot_id: str
    decision: ReviewDecisionStatus
    reviewer: str
    decided_at: str
    rationale: str
    content_hash: str

    @model_validator(mode="after")
    def _validate_review(self) -> "OutcomeReviewDecision":
        _parse_timestamp(self.decided_at, field_name="decided_at")
        if not self.reviewer.strip():
            raise ValueError("reviewer cannot be empty")
        if not self.rationale.strip():
            raise ValueError("review rationale cannot be empty")
        payload = _review_content_payload(
            outcome_snapshot_id=self.outcome_snapshot_id,
            decision=self.decision,
            reviewer=self.reviewer,
            decided_at=self.decided_at,
            rationale=self.rationale,
        )
        expected_hash = _sha256(payload)
        if self.content_hash != expected_hash:
            raise ValueError("review content hash does not match payload")
        if self.review_decision_id != f"outcome_review_{expected_hash[:24]}":
            raise ValueError("review_decision_id does not match content hash")
        return self


class OutcomeReviewDecisionBuilder:
    def build(
        self,
        *,
        outcome_snapshot_id: str,
        decision: ReviewDecisionStatus | str,
        reviewer: str,
        rationale: str,
        decided_at: str | None = None,
    ) -> OutcomeReviewDecision:
        decided_at = decided_at or _utc_now_iso()
        decision_model = ReviewDecisionStatus(decision)
        payload = _review_content_payload(
            outcome_snapshot_id=outcome_snapshot_id,
            decision=decision_model,
            reviewer=reviewer,
            decided_at=decided_at,
            rationale=rationale,
        )
        content_hash = _sha256(payload)
        return OutcomeReviewDecision(
            review_decision_id=f"outcome_review_{content_hash[:24]}",
            outcome_snapshot_id=outcome_snapshot_id,
            decision=decision_model,
            reviewer=reviewer,
            decided_at=decided_at,
            rationale=rationale,
            content_hash=content_hash,
        )


class OutcomeStoreProtocol(Protocol):
    def append(self, snapshot: WorldStateOutcomeSnapshot) -> OutcomeAppendResult: ...

    def get(self, outcome_snapshot_id: str) -> WorldStateOutcomeSnapshot | None: ...

    def list_outcomes(
        self,
        *,
        domain_id: str | None = None,
        horizon_days: int | None = None,
        world_state_snapshot_id: str | None = None,
        limit: int = 100,
    ) -> list[WorldStateOutcomeSnapshot]: ...

    def append_review(self, review: OutcomeReviewDecision) -> OutcomeAppendResult: ...

    def latest_review(
        self, outcome_snapshot_id: str
    ) -> OutcomeReviewDecision | None: ...


class SQLiteOutcomeStore:
    """Append-only outcome and review persistence.

    This can use the same SQLite file as `SQLiteWorldStateStore`. It creates
    separate append-only tables and never mutates the world-state table.
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
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS world_state_outcomes (
                    outcome_snapshot_id TEXT PRIMARY KEY,
                    world_state_snapshot_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    due_at TEXT NOT NULL,
                    due_at_epoch REAL NOT NULL,
                    evaluation_as_of TEXT NOT NULL,
                    evaluation_as_of_epoch REAL NOT NULL,
                    status TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_world_outcome_domain_horizon
                    ON world_state_outcomes(domain_id, horizon_days, evaluation_as_of_epoch DESC);
                CREATE INDEX IF NOT EXISTS idx_world_outcome_world_state
                    ON world_state_outcomes(world_state_snapshot_id, horizon_days);
                CREATE TRIGGER IF NOT EXISTS world_state_outcomes_no_update
                BEFORE UPDATE ON world_state_outcomes
                BEGIN
                    SELECT RAISE(ABORT, 'world_state_outcomes is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS world_state_outcomes_no_delete
                BEFORE DELETE ON world_state_outcomes
                BEGIN
                    SELECT RAISE(ABORT, 'world_state_outcomes is append-only');
                END;

                CREATE TABLE IF NOT EXISTS world_state_outcome_reviews (
                    review_decision_id TEXT PRIMARY KEY,
                    outcome_snapshot_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    decided_at_epoch REAL NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_world_outcome_review_latest
                    ON world_state_outcome_reviews(outcome_snapshot_id, decided_at_epoch DESC);
                CREATE TRIGGER IF NOT EXISTS world_state_outcome_reviews_no_update
                BEFORE UPDATE ON world_state_outcome_reviews
                BEGIN
                    SELECT RAISE(ABORT, 'world_state_outcome_reviews is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS world_state_outcome_reviews_no_delete
                BEFORE DELETE ON world_state_outcome_reviews
                BEGIN
                    SELECT RAISE(ABORT, 'world_state_outcome_reviews is append-only');
                END;
                """
            )

    def append(self, snapshot: WorldStateOutcomeSnapshot) -> OutcomeAppendResult:
        payload_json = _canonical_json(snapshot.model_dump(mode="json"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT outcome_snapshot_id, content_hash FROM world_state_outcomes "
                "WHERE outcome_snapshot_id = ? OR content_hash = ?",
                (snapshot.outcome_snapshot_id, snapshot.integrity.content_hash),
            ).fetchone()
            if existing:
                if (
                    existing["outcome_snapshot_id"] == snapshot.outcome_snapshot_id
                    and existing["content_hash"] == snapshot.integrity.content_hash
                ):
                    connection.commit()
                    return OutcomeAppendResult(
                        status="already_exists",
                        outcome_snapshot_id=snapshot.outcome_snapshot_id,
                        content_hash=snapshot.integrity.content_hash,
                        backend="sqlite",
                    )
                raise ValueError("outcome identity/content-hash conflict")
            connection.execute(
                """
                INSERT INTO world_state_outcomes (
                    outcome_snapshot_id, world_state_snapshot_id, domain_id,
                    horizon_days, due_at, due_at_epoch, evaluation_as_of,
                    evaluation_as_of_epoch, status, content_hash, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.outcome_snapshot_id,
                    snapshot.world_state_snapshot_id,
                    snapshot.domain_id,
                    snapshot.horizon_days,
                    snapshot.due_at,
                    _parse_timestamp(snapshot.due_at, field_name="due_at").timestamp(),
                    snapshot.evaluation_as_of,
                    _parse_timestamp(
                        snapshot.evaluation_as_of, field_name="evaluation_as_of"
                    ).timestamp(),
                    snapshot.status,
                    snapshot.integrity.content_hash,
                    payload_json,
                ),
            )
            connection.commit()
        return OutcomeAppendResult(
            status="stored",
            outcome_snapshot_id=snapshot.outcome_snapshot_id,
            content_hash=snapshot.integrity.content_hash,
            backend="sqlite",
        )

    def get(self, outcome_snapshot_id: str) -> WorldStateOutcomeSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM world_state_outcomes WHERE outcome_snapshot_id = ?",
                (outcome_snapshot_id,),
            ).fetchone()
        if row is None:
            return None
        return WorldStateOutcomeSnapshot.model_validate_json(row["payload_json"])

    def list_outcomes(
        self,
        *,
        domain_id: str | None = None,
        horizon_days: int | None = None,
        world_state_snapshot_id: str | None = None,
        limit: int = 100,
    ) -> list[WorldStateOutcomeSnapshot]:
        where: list[str] = []
        params: list[Any] = []
        if domain_id is not None:
            where.append("domain_id = ?")
            params.append(domain_id)
        if horizon_days is not None:
            where.append("horizon_days = ?")
            params.append(horizon_days)
        if world_state_snapshot_id is not None:
            where.append("world_state_snapshot_id = ?")
            params.append(world_state_snapshot_id)
        sql = "SELECT payload_json FROM world_state_outcomes"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY evaluation_as_of_epoch DESC LIMIT ?"
        params.append(max(0, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            WorldStateOutcomeSnapshot.model_validate_json(row["payload_json"])
            for row in rows
        ]

    def append_review(self, review: OutcomeReviewDecision) -> OutcomeAppendResult:
        if self.get(review.outcome_snapshot_id) is None:
            raise ValueError("cannot review an unknown outcome snapshot")
        payload_json = _canonical_json(review.model_dump(mode="json"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT review_decision_id, content_hash FROM world_state_outcome_reviews "
                "WHERE review_decision_id = ? OR content_hash = ?",
                (review.review_decision_id, review.content_hash),
            ).fetchone()
            if existing:
                if (
                    existing["review_decision_id"] == review.review_decision_id
                    and existing["content_hash"] == review.content_hash
                ):
                    connection.commit()
                    return OutcomeAppendResult(
                        status="already_exists",
                        outcome_snapshot_id=review.review_decision_id,
                        content_hash=review.content_hash,
                        backend="sqlite_review",
                    )
                raise ValueError("outcome review identity/content-hash conflict")
            connection.execute(
                """
                INSERT INTO world_state_outcome_reviews (
                    review_decision_id, outcome_snapshot_id, decision, reviewer,
                    decided_at, decided_at_epoch, content_hash, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review.review_decision_id,
                    review.outcome_snapshot_id,
                    review.decision.value,
                    review.reviewer,
                    review.decided_at,
                    _parse_timestamp(review.decided_at, field_name="decided_at").timestamp(),
                    review.content_hash,
                    payload_json,
                ),
            )
            connection.commit()
        return OutcomeAppendResult(
            status="stored",
            outcome_snapshot_id=review.review_decision_id,
            content_hash=review.content_hash,
            backend="sqlite_review",
        )

    def latest_review(
        self, outcome_snapshot_id: str
    ) -> OutcomeReviewDecision | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM world_state_outcome_reviews "
                "WHERE outcome_snapshot_id = ? "
                "ORDER BY decided_at_epoch DESC LIMIT 1",
                (outcome_snapshot_id,),
            ).fetchone()
        if row is None:
            return None
        return OutcomeReviewDecision.model_validate_json(row["payload_json"])


class FalseAnalogyAssessment(BaseModel):
    target_outcome_snapshot_id: str
    analog_outcome_snapshot_id: str
    target_world_state_snapshot_id: str
    analog_world_state_snapshot_id: str
    horizon_days: int
    world_state_similarity_score: float = Field(ge=0.0, le=1.0)
    target_outcome_class: str | None = None
    analog_outcome_class: str | None = None
    outcome_path_match: bool | None = None
    false_analogy_score: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_band: str
    warnings: list[str] = Field(default_factory=list)
    review_only: bool = True
    eligible_for_learning: bool = False


class FalseAnalogyEvaluator:
    def assess(
        self,
        *,
        target_outcome: WorldStateOutcomeSnapshot,
        analog_match: HistoricalAnalogMatch,
        analog_outcome: WorldStateOutcomeSnapshot,
    ) -> FalseAnalogyAssessment:
        if target_outcome.horizon_days != analog_outcome.horizon_days:
            raise ValueError("false-analogy comparison requires the same horizon")
        if analog_match.snapshot_id != analog_outcome.world_state_snapshot_id:
            raise ValueError("analog match does not reference the supplied analog outcome")
        if target_outcome.domain_id != analog_outcome.domain_id:
            raise ValueError("false-analogy comparison requires the same domain")

        target_class = _outcome_class(target_outcome)
        analog_class = _outcome_class(analog_outcome)
        warnings: list[str] = []
        if target_class is None or analog_class is None:
            outcome_match = None
            score = None
            risk = "unknown"
            warnings.append("outcome_class_unresolved")
        else:
            outcome_match = target_class == analog_class
            score = analog_match.similarity_score * (0.0 if outcome_match else 1.0)
            if score >= 0.65:
                risk = "high"
            elif score >= 0.35:
                risk = "medium"
            else:
                risk = "low"
            if not outcome_match:
                warnings.append("similar_world_state_different_realized_outcome")
        return FalseAnalogyAssessment(
            target_outcome_snapshot_id=target_outcome.outcome_snapshot_id,
            analog_outcome_snapshot_id=analog_outcome.outcome_snapshot_id,
            target_world_state_snapshot_id=target_outcome.world_state_snapshot_id,
            analog_world_state_snapshot_id=analog_outcome.world_state_snapshot_id,
            horizon_days=target_outcome.horizon_days,
            world_state_similarity_score=analog_match.similarity_score,
            target_outcome_class=target_class,
            analog_outcome_class=analog_class,
            outcome_path_match=outcome_match,
            false_analogy_score=score,
            risk_band=risk,
            warnings=warnings,
        )


class OutcomeCalibrationProposal(BaseModel):
    schema_version: str = CALIBRATION_PROPOSAL_SCHEMA_VERSION
    proposal_id: str = Field(default_factory=lambda: f"outcome_calibration_{uuid4().hex}")
    domain_id: str
    horizon_days: int
    generated_at: str = Field(default_factory=_utc_now_iso)
    status: str
    approved_scored_sample_count: int
    total_candidate_count: int
    mean_brier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_logarithmic_loss: float | None = Field(default=None, ge=0.0)
    top_scenario_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    blockers: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    proposed_actions: list[str] = Field(default_factory=list)
    source_outcome_snapshot_ids: list[str] = Field(default_factory=list)
    review_only: bool = True
    human_approval_required: bool = True
    can_write_learning_memory: bool = False
    can_change_probabilities: bool = False


class OutcomeCalibrationService:
    def propose(
        self,
        *,
        outcome_store: OutcomeStoreProtocol,
        domain_id: str,
        horizon_days: int,
        min_approved_samples: int = 20,
        limit: int = 1000,
    ) -> OutcomeCalibrationProposal:
        outcomes = outcome_store.list_outcomes(
            domain_id=domain_id,
            horizon_days=horizon_days,
            limit=limit,
        )
        approved: list[WorldStateOutcomeSnapshot] = []
        for outcome in outcomes:
            review = outcome_store.latest_review(outcome.outcome_snapshot_id)
            if (
                review is not None
                and review.decision == ReviewDecisionStatus.APPROVED
                and outcome.probability_score.scored
            ):
                approved.append(outcome)

        blockers: list[str] = []
        cautions: list[str] = []
        if len(approved) < min_approved_samples:
            blockers.append(
                f"approved_scored_sample_count_below_minimum:{len(approved)}<{min_approved_samples}"
            )
        brier_values = [
            item.probability_score.brier_score
            for item in approved
            if item.probability_score.brier_score is not None
        ]
        log_values = [
            item.probability_score.logarithmic_loss
            for item in approved
            if item.probability_score.logarithmic_loss is not None
        ]
        top_values = [
            item.probability_score.top_scenario_correct
            for item in approved
            if item.probability_score.top_scenario_correct is not None
        ]
        if approved and len({item.scenario_graph_id for item in approved}) == len(approved):
            cautions.append(
                "scenario_taxonomy_may_not_be_stable_across_cases; aggregate scores only"
            )
        status = "ready_for_human_calibration_review" if not blockers else "blocked"
        actions = [
            "inspect probability reliability by horizon and regime cluster",
            "compare calibrated and uncalibrated Brier score in shadow mode",
            "review high false-analogy cases before changing priors",
        ]
        if status == "blocked":
            actions.insert(0, "collect and approve more fixed-horizon outcomes")
        return OutcomeCalibrationProposal(
            domain_id=domain_id,
            horizon_days=horizon_days,
            status=status,
            approved_scored_sample_count=len(approved),
            total_candidate_count=len(outcomes),
            mean_brier_score=(sum(brier_values) / len(brier_values) if brier_values else None),
            mean_logarithmic_loss=(sum(log_values) / len(log_values) if log_values else None),
            top_scenario_accuracy=(
                sum(1.0 for value in top_values if value) / len(top_values)
                if top_values
                else None
            ),
            blockers=blockers,
            cautions=cautions,
            proposed_actions=actions,
            source_outcome_snapshot_ids=sorted(
                item.outcome_snapshot_id for item in approved
            ),
        )


class LearningPromotionPacket(BaseModel):
    schema_version: str = LEARNING_PROMOTION_SCHEMA_VERSION
    promotion_packet_id: str = Field(default_factory=lambda: f"learning_promotion_{uuid4().hex}")
    calibration_proposal_id: str
    generated_at: str = Field(default_factory=_utc_now_iso)
    reviewer_decision: str
    status: str
    eligible_for_manual_implementation: bool = False
    blockers: list[str] = Field(default_factory=list)
    implementation_requirements: list[str] = Field(default_factory=list)
    review_only: bool = True
    config_write_performed: bool = False
    learning_memory_write_performed: bool = False
    model_promotion_performed: bool = False


class LearningPromotionGate:
    def evaluate(
        self,
        *,
        calibration_proposal: OutcomeCalibrationProposal,
        reviewer_decision: str,
    ) -> LearningPromotionPacket:
        decision = str(reviewer_decision).strip().lower()
        blockers: list[str] = []
        if calibration_proposal.status != "ready_for_human_calibration_review":
            blockers.append("calibration_proposal_not_ready")
        if decision != "approved":
            blockers.append("explicit_human_approval_missing")
        eligible = not blockers
        return LearningPromotionPacket(
            calibration_proposal_id=calibration_proposal.proposal_id,
            reviewer_decision=decision or "not_provided",
            status=(
                "approved_for_separate_manual_shadow_implementation"
                if eligible
                else "blocked"
            ),
            eligible_for_manual_implementation=eligible,
            blockers=blockers,
            implementation_requirements=[
                "create a separate versioned calibration artifact",
                "run shadow replay against a held-out time range",
                "compare calibration metrics before and after the proposed change",
                "obtain a second review before any production or learning-memory write",
            ],
        )


def _score_scenarios(
    world_state: WorldStateSnapshot,
    resolutions: list[ScenarioResolution],
) -> ScenarioProbabilityScore:
    graph = world_state.scenario_outcome_graph
    if graph is None:
        return ScenarioProbabilityScore(
            score_status="scenario_graph_missing",
            warnings=["scenario_graph_missing"],
        )
    scenario_nodes = [node for node in graph.nodes if node.node_type == "scenario"]
    if not scenario_nodes:
        return ScenarioProbabilityScore(
            score_status="scenario_nodes_missing",
            warnings=["scenario_nodes_missing"],
        )
    resolution_by_id = {item.scenario_node_id: item for item in resolutions}
    realized = [
        item for item in resolutions if item.status == ScenarioResolutionStatus.REALIZED
    ]
    unresolved = [
        node.node_id
        for node in scenario_nodes
        if node.node_id not in resolution_by_id
        or resolution_by_id[node.node_id].status
        in {ScenarioResolutionStatus.UNRESOLVED, ScenarioResolutionStatus.PARTIAL}
    ]
    if len(realized) != 1:
        warning = (
            "no_realized_scenario" if not realized else "multiple_realized_scenarios"
        )
        return ScenarioProbabilityScore(
            score_status=warning,
            unresolved_scenario_node_ids=unresolved,
            warnings=[warning],
        )
    winner = realized[0]
    probabilities = {node.node_id: node.probability for node in scenario_nodes}
    if winner.scenario_node_id not in probabilities:
        return ScenarioProbabilityScore(
            score_status="realized_scenario_not_in_graph",
            warnings=["realized_scenario_not_in_graph"],
        )
    n = len(scenario_nodes)
    brier = sum(
        (probability - (1.0 if node_id == winner.scenario_node_id else 0.0)) ** 2
        for node_id, probability in probabilities.items()
    ) / n
    winner_probability = probabilities[winner.scenario_node_id]
    top_node_id = max(probabilities, key=probabilities.get)
    log_loss = -math.log(max(winner_probability, 1e-15))
    return ScenarioProbabilityScore(
        scored=True,
        score_status="scored",
        realized_scenario_node_id=winner.scenario_node_id,
        realized_outcome_class=_normalize_label(winner.outcome_class),
        top_probability_scenario_node_id=top_node_id,
        top_scenario_correct=top_node_id == winner.scenario_node_id,
        winner_probability=winner_probability,
        brier_score=brier,
        logarithmic_loss=log_loss,
        unresolved_scenario_node_ids=unresolved,
    )


def _outcome_class(snapshot: WorldStateOutcomeSnapshot) -> str | None:
    explicit = _normalize_label(snapshot.probability_score.realized_outcome_class)
    if explicit:
        return explicit
    realized_id = snapshot.probability_score.realized_scenario_node_id
    if realized_id is None:
        return None
    for item in snapshot.scenario_resolutions:
        if item.scenario_node_id == realized_id:
            explicit = _normalize_label(item.outcome_class)
            if explicit:
                return explicit
    return _normalize_label(realized_id)


def _outcome_content_payload(
    *,
    world_state_snapshot_id: str,
    domain_id: str,
    world_state_as_of: str,
    world_state_knowledge_cutoff: str,
    scenario_graph_id: str | None,
    horizon_days: int,
    due_at: str,
    evaluation_as_of: str,
    status: str,
    evidence: list[OutcomeEvidence],
    scenario_resolutions: list[ScenarioResolution],
    hypothesis_resolutions: list[HypothesisResolution],
    probability_score: ScenarioProbabilityScore,
    evidence_gaps: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "world_state_snapshot_id": world_state_snapshot_id,
        "domain_id": domain_id,
        "world_state_as_of": world_state_as_of,
        "world_state_knowledge_cutoff": world_state_knowledge_cutoff,
        "scenario_graph_id": scenario_graph_id,
        "horizon_days": horizon_days,
        "due_at": due_at,
        "evaluation_as_of": evaluation_as_of,
        "status": status,
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "scenario_resolutions": [
            item.model_dump(mode="json") for item in scenario_resolutions
        ],
        "hypothesis_resolutions": [
            item.model_dump(mode="json") for item in hypothesis_resolutions
        ],
        "probability_score": probability_score.model_dump(mode="json"),
        "evidence_gaps": list(evidence_gaps),
    }


def _review_content_payload(
    *,
    outcome_snapshot_id: str,
    decision: ReviewDecisionStatus,
    reviewer: str,
    decided_at: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "schema_version": OUTCOME_REVIEW_SCHEMA_VERSION,
        "outcome_snapshot_id": outcome_snapshot_id,
        "decision": decision.value,
        "reviewer": reviewer.strip(),
        "decided_at": decided_at,
        "rationale": rationale.strip(),
    }


__all__ = [
    "CALIBRATION_PROPOSAL_SCHEMA_VERSION",
    "LEARNING_PROMOTION_SCHEMA_VERSION",
    "OUTCOME_REVIEW_SCHEMA_VERSION",
    "OUTCOME_SCHEMA_VERSION",
    "FalseAnalogyAssessment",
    "FalseAnalogyEvaluator",
    "HypothesisResolution",
    "HypothesisResolutionStatus",
    "LearningPromotionGate",
    "LearningPromotionPacket",
    "OutcomeAppendResult",
    "OutcomeCalibrationProposal",
    "OutcomeCalibrationService",
    "OutcomeEvidence",
    "OutcomeIntegrity",
    "OutcomeReviewDecision",
    "OutcomeReviewDecisionBuilder",
    "OutcomeSnapshotBuilder",
    "OutcomeStoreProtocol",
    "ReviewDecisionStatus",
    "SQLiteOutcomeStore",
    "ScenarioProbabilityScore",
    "ScenarioResolution",
    "ScenarioResolutionStatus",
    "WorldStateOutcomeSnapshot",
]
